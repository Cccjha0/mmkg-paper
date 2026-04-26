from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-aware soft routers with top-K reranking.")
    parser.add_argument("--test-table", default="outputs/candidate_router/features/candidate_router_test_top100.parquet")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-by-regime", default=None)
    parser.add_argument("--out-query-rows", default=None)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--inference-batch-size", type=int, default=262_144)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path, device: torch.device) -> tuple[CandidateSoftRouterMLP, dict]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    payload = torch.load(model_dir / "model.pt", map_location="cpu")
    model = CandidateSoftRouterMLP(
        input_dim=int(payload.get("input_dim", len(config["input_columns"]))),
        hidden_dim=int(payload.get("hidden_dim", config.get("hidden_dim", 128))),
        dropout=float(payload.get("dropout", config.get("dropout", 0.1))),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def materialize_input_frame(frame: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in raw_features:
        if name == "direction":
            out["direction_is_tail"] = frame["direction"].astype(str).eq("tail").astype("float32")
        else:
            out[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype("float32")
    return out


@torch.inference_mode()
def predict_alpha(
    model: CandidateSoftRouterMLP,
    frame: pd.DataFrame,
    config: dict,
    device: torch.device,
    inference_batch_size: int,
) -> np.ndarray:
    values = materialize_input_frame(frame, config["raw_features"]).to_numpy(dtype=np.float32)
    mean = np.array(config["feature_mean"], dtype=np.float32)
    std = np.array(config["feature_std"], dtype=np.float32)
    values = (values - mean) / std
    outputs = []
    for start in range(0, len(values), inference_batch_size):
        batch = torch.from_numpy(values[start : start + inference_batch_size]).to(device)
        alpha = torch.sigmoid(model(batch)).detach().cpu().numpy()
        outputs.append(alpha.astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def rank_from_scores(scores: np.ndarray, target_mask: np.ndarray) -> tuple[int, float]:
    target_positions = np.flatnonzero(target_mask)
    if len(target_positions) != 1:
        raise RuntimeError(f"Expected exactly one target row per query, found {len(target_positions)}")
    target_score = float(scores[target_positions[0]])
    rank = int(np.sum(scores > target_score) + 1)
    return rank, float(1.0 / rank)


def process_complete_queries(
    frame: pd.DataFrame,
    query_limit_remaining: int | None = None,
    flush_all: bool = False,
) -> tuple[list[dict], pd.DataFrame]:
    if frame.empty:
        return [], frame
    if flush_all:
        complete = frame
        carry = frame.iloc[0:0].copy()
    else:
        query_values = frame["query_id"].astype(str)
        last_query = query_values.iloc[-1]
        complete_mask = query_values.ne(last_query)
        complete = frame[complete_mask]
        carry = frame[~complete_mask]

    rows = []
    processed = 0
    for query_id, group in complete.groupby("query_id", sort=False):
        if query_limit_remaining is not None and processed >= query_limit_remaining:
            carry = pd.concat([group, carry], ignore_index=True)
            continue
        target_mask = group["is_target"].astype(int).to_numpy() == 1
        mixed_scores = group["mixed_score"].to_numpy(dtype=np.float64)
        gate_scores = group["score_gate"].to_numpy(dtype=np.float64)
        residual_scores = group["score_residual"].to_numpy(dtype=np.float64)
        mixed_rank, mixed_rr = rank_from_scores(mixed_scores, target_mask)
        gate_rank, gate_rr = rank_from_scores(gate_scores, target_mask)
        residual_rank, residual_rr = rank_from_scores(residual_scores, target_mask)
        target_row = group[target_mask].iloc[0]
        rows.append(
            {
                "query_id": str(query_id),
                "seed": int(target_row["seed"]),
                "split": str(target_row["split"]),
                "direction": str(target_row["direction"]),
                "relation_id": int(target_row["relation_id"]),
                "target_regime": str(target_row["target_regime"]),
                "n_candidates": int(len(group)),
                "mixed_rank": mixed_rank,
                "mixed_rr": mixed_rr,
                "gate_topk_rank": gate_rank,
                "gate_topk_rr": gate_rr,
                "residual_topk_rank": residual_rank,
                "residual_topk_rr": residual_rr,
                "target_alpha": float(target_row["alpha"]),
                "mean_alpha": float(group["alpha"].mean()),
                "target_score_gate": float(target_row["score_gate"]),
                "target_score_residual": float(target_row["score_residual"]),
                "target_mixed_score": float(target_row["mixed_score"]),
            }
        )
        processed += 1
    return rows, carry.reset_index(drop=True)


def metric_bundle(rows: list[dict], prefix: str) -> dict:
    ranks = np.array([float(row[f"{prefix}_rank"]) for row in rows], dtype=np.float64)
    rr = np.array([float(row[f"{prefix}_rr"]) for row in rows], dtype=np.float64)
    if len(rows) == 0:
        return {"count": 0, "mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0}
    return {
        "count": int(len(rows)),
        "mrr": float(rr.mean()),
        "hits1": float(np.mean(ranks <= 1)),
        "hits3": float(np.mean(ranks <= 3)),
        "hits10": float(np.mean(ranks <= 10)),
    }


def summarize_rows(rows: list[dict], model_name: str, config: dict, scope: str) -> dict:
    mixed = metric_bundle(rows, "mixed")
    gate = metric_bundle(rows, "gate_topk")
    residual = metric_bundle(rows, "residual_topk")
    return {
        "model": model_name,
        "scope": scope,
        "feature_set": config["feature_set"],
        "loss": config["loss"],
        "candidate_topk_rerank_mrr": mixed["mrr"],
        "candidate_topk_hits1": mixed["hits1"],
        "candidate_topk_hits3": mixed["hits3"],
        "candidate_topk_hits10": mixed["hits10"],
        "gate_topk_mrr": gate["mrr"],
        "residual_topk_mrr": residual["mrr"],
        "delta_vs_gate_topk": mixed["mrr"] - gate["mrr"],
        "delta_vs_residual_topk": mixed["mrr"] - residual["mrr"],
        "count": mixed["count"],
        "mean_target_alpha": float(np.mean([row["target_alpha"] for row in rows])) if rows else 0.0,
        "mean_alpha": float(np.mean([row["mean_alpha"] for row in rows])) if rows else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    test_table = Path(args.test_table)
    model_dir = Path(args.model_dir)
    device = resolve_device(args.device)
    model, config = load_model(model_dir, device)

    required_columns = sorted(
        set(
            config["raw_features"]
            + [
                "query_id",
                "seed",
                "split",
                "direction",
                "relation_id",
                "target_regime",
                "is_target",
                "score_gate",
                "score_residual",
            ]
        )
    )
    parquet = pq.ParquetFile(test_table)
    if args.dry_run:
        print(f"[OK] test table: {test_table.as_posix()}")
        print(f"[OK] rows={parquet.metadata.num_rows} row_groups={parquet.metadata.num_row_groups}")
        print(f"[OK] model_dir={model_dir.as_posix()} feature_set={config['feature_set']} loss={config['loss']}")
        print(f"[OK] device={device}")
        return [], config

    all_rows: list[dict] = []
    carry = pd.DataFrame()
    for batch in parquet.iter_batches(columns=required_columns, batch_size=args.batch_size):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        alpha = predict_alpha(model, frame, config, device, args.inference_batch_size)
        frame["alpha"] = alpha
        frame["mixed_score"] = alpha * frame["score_gate"].astype("float32") + (1.0 - alpha) * frame[
            "score_residual"
        ].astype("float32")
        remaining = None if args.max_queries is None else max(0, args.max_queries - len(all_rows))
        rows, carry = process_complete_queries(frame, remaining)
        all_rows.extend(rows)
        if args.max_queries is not None and len(all_rows) >= args.max_queries:
            break

    if args.max_queries is None and not carry.empty:
        alpha = predict_alpha(model, carry, config, device, args.inference_batch_size)
        carry["alpha"] = alpha
        carry["mixed_score"] = alpha * carry["score_gate"].astype("float32") + (1.0 - alpha) * carry[
            "score_residual"
        ].astype("float32")
        rows, carry = process_complete_queries(carry, flush_all=True)
        all_rows.extend(rows)

    model_name = model_dir.name
    summary_rows = [summarize_rows(all_rows, model_name, config, "overall")]
    by_regime = []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        buckets[str(row["target_regime"])].append(row)
    for regime, bucket in sorted(buckets.items()):
        by_regime.append(summarize_rows(bucket, model_name, config, regime))
    write_csv(Path(args.out_summary), summary_rows)
    if args.out_by_regime:
        write_csv(Path(args.out_by_regime), by_regime)
    if args.out_query_rows:
        write_csv(Path(args.out_query_rows), all_rows)
    return all_rows, config


def main() -> None:
    args = parse_args()
    rows, _config = evaluate(args)
    if not args.dry_run:
        print(f"[OK] evaluated queries -> {len(rows)}")
        print(f"[OK] wrote summary     -> {Path(args.out_summary).as_posix()}")
        if args.out_by_regime:
            print(f"[OK] wrote by-regime   -> {Path(args.out_by_regime).as_posix()}")
        if args.out_query_rows:
            print(f"[OK] wrote query rows  -> {Path(args.out_query_rows).as_posix()}")


if __name__ == "__main__":
    main()
