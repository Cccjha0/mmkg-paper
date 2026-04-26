from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_FEATURE_SETS = {"CA-S1", "CA-S2", "CA-S3"}
VALID_LOSSES = {"pointwise", "pairwise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candidate-aware soft routing MLPs.")
    parser.add_argument("--train-table", default="outputs/candidate_router/features/candidate_router_dev_top100.parquet")
    parser.add_argument("--feature-contract", default="outputs/candidate_router/features/feature_contract.json")
    parser.add_argument("--feature-set", required=True, choices=sorted(VALID_FEATURE_SETS))
    parser.add_argument("--loss", required=True, choices=sorted(VALID_LOSSES))
    parser.add_argument("--negatives-per-query", type=int, default=20)
    parser.add_argument("--hard-ratio", default="2:2:1", help="gate_hard:residual_hard:random negative ratio")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-query-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print metadata without loading/training.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_contract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("feature_sets", "forbidden_features"):
        if key not in payload:
            raise KeyError(f"Missing {key} in feature contract: {path}")
    return payload


def parse_hard_ratio(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 3 or sum(parts) <= 0:
        raise ValueError("--hard-ratio must look like gate:residual:random, e.g. 2:2:1")
    return parts[0], parts[1], parts[2]


class CandidateSoftRouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(16, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PreparedFeatures:
    x: torch.Tensor
    input_columns: list[str]
    mean: list[float]
    std: list[float]


def effective_input_columns(raw_features: list[str]) -> list[str]:
    columns: list[str] = []
    for name in raw_features:
        if name == "direction":
            columns.append("direction_is_tail")
        else:
            columns.append(name)
    return columns


def materialize_input_frame(frame: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in raw_features:
        if name == "direction":
            out["direction_is_tail"] = frame["direction"].astype(str).eq("tail").astype("float32")
        else:
            out[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype("float32")
    return out


def fit_transform_features(frame: pd.DataFrame, raw_features: list[str]) -> PreparedFeatures:
    input_frame = materialize_input_frame(frame, raw_features)
    values = input_frame.to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    values = (values - mean) / std
    return PreparedFeatures(
        x=torch.from_numpy(values.astype(np.float32)),
        input_columns=list(input_frame.columns),
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
    )


def train_val_query_split(query_ids: np.ndarray, val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    unique = np.array(sorted(set(str(q) for q in query_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_n = max(1, int(round(len(unique) * val_ratio))) if val_ratio > 0 else 0
    val = set(unique[:val_n])
    train = set(unique[val_n:])
    return train, val


def select_queries(frame: pd.DataFrame, max_queries: int | None, seed: int) -> pd.DataFrame:
    if max_queries is None:
        return frame
    unique = np.array(sorted(frame["query_id"].astype(str).unique()))
    if len(unique) <= max_queries:
        return frame
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(unique, size=max_queries, replace=False).tolist())
    return frame[frame["query_id"].astype(str).isin(chosen)].copy()


def sample_counts(total: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    weights = np.array(ratio, dtype=np.float64)
    raw = weights / weights.sum() * total
    base = np.floor(raw).astype(int)
    while base.sum() < total:
        base[int(np.argmax(raw - base))] += 1
    return int(base[0]), int(base[1]), int(base[2])


def sample_negative_indices(group: pd.DataFrame, total: int, ratio: tuple[int, int, int], rng: np.random.Generator) -> list[int]:
    negatives = group[group["is_target"].astype(int).eq(0)]
    if negatives.empty:
        return []
    gate_n, residual_n, random_n = sample_counts(total, ratio)

    selected: list[int] = []

    gate_pool = negatives[negatives["in_gate_topk"].astype(int).eq(1)].sort_values("gate_rank_in_union")
    selected.extend(gate_pool.index[:gate_n].tolist())

    residual_pool = negatives[negatives["in_residual_topk"].astype(int).eq(1)].sort_values("residual_rank_in_union")
    for idx in residual_pool.index[:residual_n].tolist():
        if idx not in selected:
            selected.append(idx)

    remaining = negatives.index.difference(pd.Index(selected))
    if len(remaining) > 0 and random_n > 0:
        take = min(random_n, len(remaining))
        selected.extend(rng.choice(remaining.to_numpy(), size=take, replace=False).tolist())

    if len(selected) < total:
        remaining = negatives.index.difference(pd.Index(selected))
        if len(remaining) > 0:
            take = min(total - len(selected), len(remaining))
            selected.extend(rng.choice(remaining.to_numpy(), size=take, replace=False).tolist())
    return selected[:total]


def build_pointwise_sample(frame: pd.DataFrame, negatives_per_query: int, ratio: tuple[int, int, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _qid, group in frame.groupby("query_id", sort=False):
        target_idx = group[group["is_target"].astype(int).eq(1)].index.tolist()
        selected.extend(target_idx)
        selected.extend(sample_negative_indices(group, negatives_per_query, ratio, rng))
    return frame.loc[selected].copy()


def build_pairwise_sample(
    frame: pd.DataFrame,
    negatives_per_query: int,
    ratio: tuple[int, int, int],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    row_indices: list[int] = []
    pairs: list[tuple[int, int]] = []
    for _qid, group in frame.groupby("query_id", sort=False):
        target_idx = group[group["is_target"].astype(int).eq(1)].index.tolist()
        if len(target_idx) != 1:
            continue
        pos_idx = target_idx[0]
        neg_indices = sample_negative_indices(group, negatives_per_query, ratio, rng)
        row_indices.append(pos_idx)
        row_indices.extend(neg_indices)
        pairs.extend((pos_idx, neg_idx) for neg_idx in neg_indices)

    unique_indices = list(dict.fromkeys(row_indices))
    local = {idx: i for i, idx in enumerate(unique_indices)}
    pos = np.array([local[pair[0]] for pair in pairs], dtype=np.int64)
    neg = np.array([local[pair[1]] for pair in pairs], dtype=np.int64)
    return frame.loc[unique_indices].copy(), pos, neg


def train_pointwise(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    logs: list[dict] = []

    n = x_train.size(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.to(device))
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val.to(device), pos_weight=pos_weight)
            val_pred = torch.sigmoid(val_logits).ge(0.5).float().cpu()
            val_acc = float(val_pred.eq(y_val).float().mean().item())
        logs.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": float(val_loss.detach().cpu().item()),
                "val_accuracy": val_acc,
            }
        )
        print(f"[epoch {epoch:03d}] train_loss={logs[-1]['train_loss']:.6f} val_loss={logs[-1]['val_loss']:.6f}")
    return logs


def mixed_score(logits: torch.Tensor, score_gate: torch.Tensor, score_residual: torch.Tensor) -> torch.Tensor:
    alpha = torch.sigmoid(logits)
    return alpha * score_gate + (1.0 - alpha) * score_residual


def train_pairwise(
    model: nn.Module,
    x_train: torch.Tensor,
    scores_train: torch.Tensor,
    pos_train: np.ndarray,
    neg_train: np.ndarray,
    x_val: torch.Tensor,
    scores_val: torch.Tensor,
    pos_val: np.ndarray,
    neg_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs: list[dict] = []
    n_pairs = len(pos_train)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(n_pairs)
        losses = []
        for start in range(0, n_pairs, args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            p_idx = torch.from_numpy(pos_train[batch_idx]).long()
            n_idx = torch.from_numpy(neg_train[batch_idx]).long()
            xp = x_train[p_idx].to(device)
            xn = x_train[n_idx].to(device)
            sp = scores_train[p_idx].to(device)
            sn = scores_train[n_idx].to(device)
            pos_score = mixed_score(model(xp), sp[:, 0], sp[:, 1])
            neg_score = mixed_score(model(xn), sn[:, 0], sn[:, 1])
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            p_idx = torch.from_numpy(pos_val).long()
            n_idx = torch.from_numpy(neg_val).long()
            val_pos = mixed_score(model(x_val[p_idx].to(device)), scores_val[p_idx, 0].to(device), scores_val[p_idx, 1].to(device))
            val_neg = mixed_score(model(x_val[n_idx].to(device)), scores_val[n_idx, 0].to(device), scores_val[n_idx, 1].to(device))
            val_loss = -F.logsigmoid(val_pos - val_neg).mean()
            val_pair_acc = float(val_pos.gt(val_neg).float().mean().detach().cpu().item())
        logs.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": float(val_loss.detach().cpu().item()),
                "val_pair_accuracy": val_pair_acc,
            }
        )
        print(
            f"[epoch {epoch:03d}] train_loss={logs[-1]['train_loss']:.6f} "
            f"val_loss={logs[-1]['val_loss']:.6f} val_pair_acc={val_pair_acc:.4f}"
        )
    return logs


def write_train_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.negatives_per_query <= 0:
        raise ValueError("--negatives-per-query must be positive.")
    set_seed(args.seed)

    train_table = Path(args.train_table)
    contract = load_feature_contract(Path(args.feature_contract))
    raw_features = contract["feature_sets"][args.feature_set]
    forbidden = set(contract.get("forbidden_features", []))
    leaked = sorted(set(raw_features) & forbidden)
    if leaked:
        raise RuntimeError(f"Forbidden features requested for {args.feature_set}: {leaked}")

    parquet_meta = pq.ParquetFile(train_table).metadata
    if args.dry_run:
        print(f"[OK] train table: {train_table.as_posix()}")
        print(f"[OK] rows={parquet_meta.num_rows} row_groups={parquet_meta.num_row_groups}")
        print(f"[OK] feature_set={args.feature_set} raw_features={len(raw_features)}")
        print(f"[OK] loss={args.loss}")
        return

    required_columns = sorted(set(raw_features + ["query_id", "is_target", "score_gate", "score_residual", "in_gate_topk", "in_residual_topk", "gate_rank_in_union", "residual_rank_in_union"]))
    frame = pd.read_parquet(train_table, columns=required_columns)
    frame = select_queries(frame, args.max_train_queries, args.seed)
    train_queries, val_queries = train_val_query_split(frame["query_id"].to_numpy(), args.val_query_ratio, args.seed)
    train_frame = frame[frame["query_id"].astype(str).isin(train_queries)].copy()
    val_frame = frame[frame["query_id"].astype(str).isin(val_queries)].copy()
    ratio = parse_hard_ratio(args.hard_ratio)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.loss == "pointwise":
        train_sample = build_pointwise_sample(train_frame, args.negatives_per_query, ratio, args.seed)
        val_sample = build_pointwise_sample(val_frame, args.negatives_per_query, ratio, args.seed + 1)
        prep = fit_transform_features(train_sample, raw_features)
        val_input = materialize_input_frame(val_sample, raw_features).to_numpy(dtype=np.float32)
        val_input = (val_input - np.array(prep.mean, dtype=np.float32)) / np.array(prep.std, dtype=np.float32)
        x_val = torch.from_numpy(val_input.astype(np.float32))
        y_train = torch.from_numpy(train_sample["is_target"].astype("float32").to_numpy())
        y_val = torch.from_numpy(val_sample["is_target"].astype("float32").to_numpy())

        model = CandidateSoftRouterMLP(prep.x.size(1), args.hidden_dim, args.dropout)
        logs = train_pointwise(model, prep.x, y_train, x_val, y_val, args, resolve_device(args.device))
        train_rows = len(train_sample)
        val_rows = len(val_sample)
        train_pairs = None
        val_pairs = None
    else:
        train_sample, pos_train, neg_train = build_pairwise_sample(train_frame, args.negatives_per_query, ratio, args.seed)
        val_sample, pos_val, neg_val = build_pairwise_sample(val_frame, args.negatives_per_query, ratio, args.seed + 1)
        prep = fit_transform_features(train_sample, raw_features)
        val_input = materialize_input_frame(val_sample, raw_features).to_numpy(dtype=np.float32)
        val_input = (val_input - np.array(prep.mean, dtype=np.float32)) / np.array(prep.std, dtype=np.float32)
        x_val = torch.from_numpy(val_input.astype(np.float32))
        scores_train = torch.from_numpy(train_sample[["score_gate", "score_residual"]].to_numpy(dtype=np.float32))
        scores_val = torch.from_numpy(val_sample[["score_gate", "score_residual"]].to_numpy(dtype=np.float32))

        model = CandidateSoftRouterMLP(prep.x.size(1), args.hidden_dim, args.dropout)
        logs = train_pairwise(
            model,
            prep.x,
            scores_train,
            pos_train,
            neg_train,
            x_val,
            scores_val,
            pos_val,
            neg_val,
            args,
            resolve_device(args.device),
        )
        train_rows = len(train_sample)
        val_rows = len(val_sample)
        train_pairs = int(len(pos_train))
        val_pairs = int(len(pos_val))

    model_path = out_dir / "model.pt"
    config_path = out_dir / "config.json"
    log_path = out_dir / "train_log.csv"
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "input_dim": len(prep.input_columns),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        model_path,
    )
    config = {
        "feature_set": args.feature_set,
        "loss": args.loss,
        "raw_features": raw_features,
        "input_columns": prep.input_columns,
        "feature_mean": prep.mean,
        "feature_std": prep.std,
        "negatives_per_query": args.negatives_per_query,
        "hard_ratio": args.hard_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_table": train_table.as_posix(),
        "feature_contract": str(args.feature_contract),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "max_train_queries": args.max_train_queries,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_train_log(log_path, logs)
    print(f"[OK] wrote model     -> {model_path.as_posix()}")
    print(f"[OK] wrote config    -> {config_path.as_posix()}")
    print(f"[OK] wrote train log -> {log_path.as_posix()}")


if __name__ == "__main__":
    main()
