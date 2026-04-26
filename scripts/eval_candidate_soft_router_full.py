from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_candidate_scores import (
    build_filtered_indexes,
    filter_scores_,
    load_run,
    load_split_triples,
    resolve_device,
    target_regime,
)
from scripts.build_candidate_router_table import build_entity_feature_arrays, load_relation_priors


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
    parser = argparse.ArgumentParser(description="Full-ranking evaluation for candidate-aware soft routers.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gate-run-dir", required=True)
    parser.add_argument("--residual-run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--direction", default="both", choices=["head", "tail", "both"])
    parser.add_argument("--relation-priors", default="outputs/router/raw/dev_relation_priors.csv")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--alpha-batch-size", type=int, default=262_144)
    parser.add_argument("--device", default=None, help="cuda | cpu | mps | auto; defaults to run config")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-by-regime", default=None)
    parser.add_argument("--out-query-rows", required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_router(model_dir: Path, device: str) -> tuple[CandidateSoftRouterMLP, dict]:
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


def relation_prior_map(frame: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for row in frame.to_dict(orient="records"):
        out[int(row["relation_id"])] = {
            "relation_gain_prior": float(row["relation_gain_prior"]),
            "relation_fusion_win_rate": float(row["relation_fusion_win_rate"]),
            "relation_support": int(row["relation_support"]),
            "relation_is_visual_prior": int(row["relation_is_visual_prior"]),
        }
    return out


def default_prior() -> dict:
    return {
        "relation_gain_prior": 0.0,
        "relation_fusion_win_rate": 0.0,
        "relation_support": 0,
        "relation_is_visual_prior": 0,
    }


@torch.inference_mode()
def score_full_matrix(
    model,
    q_cpu: torch.Tensor,
    direction: str,
    true_index: dict,
    num_entities: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    q = q_cpu.to(device)
    bq = q.size(0)
    all_entities = torch.arange(num_entities, dtype=torch.long)
    parts = []
    h = q[:, 0]
    r = q[:, 1]
    t = q[:, 2]
    for start in range(0, num_entities, chunk_size):
        end = min(num_entities, start + chunk_size)
        cand = all_entities[start:end].to(device)
        c = cand.numel()
        if direction == "tail":
            h_g = h.unsqueeze(1).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            e_g = cand.unsqueeze(0).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), e_g.reshape(-1)], dim=1)
        else:
            e_g = cand.unsqueeze(0).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = t.unsqueeze(1).expand(bq, c)
            batch = torch.stack([e_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)
        scores = model.score(batch).view(bq, c)
        filter_scores_(scores, q_cpu, start, direction, true_index)
        parts.append(scores.detach().cpu())
    return torch.cat(parts, dim=1)


def ranks_from_scores(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    values = torch.arange(1, scores.size(1) + 1, dtype=torch.long).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, values)
    return ranks


def target_ranks_and_rr(scores: torch.Tensor, target_ids: torch.Tensor) -> tuple[list[int], list[float]]:
    target_scores = scores.gather(1, target_ids.unsqueeze(1))
    ranks = (scores > target_scores).sum(dim=1).to(dtype=torch.long) + 1
    ranks_list = [int(v) for v in ranks.tolist()]
    rr_list = [float(1.0 / v) for v in ranks_list]
    return ranks_list, rr_list


def observed_ids_for_direction(q_cpu: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "tail":
        return q_cpu[:, 0]
    return q_cpu[:, 2]


def target_ids_for_direction(q_cpu: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "tail":
        return q_cpu[:, 2]
    return q_cpu[:, 0]


def build_feature_matrix(
    *,
    raw_features: list[str],
    q_cpu: torch.Tensor,
    direction: str,
    relation_priors: dict[int, dict],
    entity_features: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    score_gate: np.ndarray,
    score_residual: np.ndarray,
    rank_gate: np.ndarray,
    rank_residual: np.ndarray,
) -> np.ndarray:
    bq, c = score_gate.shape
    observed_ids = observed_ids_for_direction(q_cpu, direction).numpy().astype(np.int64)
    relation_ids = q_cpu[:, 1].numpy().astype(np.int64)
    rows = bq * c
    columns: list[np.ndarray] = []
    candidate_grid = np.broadcast_to(candidate_ids.reshape(1, c), (bq, c))
    observed_grid = np.broadcast_to(observed_ids.reshape(bq, 1), (bq, c))

    prior_values = [relation_priors.get(int(rid), default_prior()) for rid in relation_ids]
    def finite_score_copy(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if finite.any():
            low = float(values[finite].min()) - 1.0
            high = float(values[finite].max()) + 1.0
        else:
            low = -100.0
            high = 100.0
        return np.nan_to_num(values, nan=low, posinf=high, neginf=low)

    score_gate_safe = finite_score_copy(score_gate)
    score_residual_safe = finite_score_copy(score_residual)
    score_diff = score_gate_safe - score_residual_safe
    score_mean = 0.5 * (score_gate_safe + score_residual_safe)
    score_abs_diff = np.abs(score_diff)
    score_max = np.maximum(score_gate, score_residual)

    for name in raw_features:
        if name == "direction":
            value = np.full((bq, c), 1.0 if direction == "tail" else 0.0, dtype=np.float32)
        elif name == "relation_id":
            value = np.broadcast_to(relation_ids.reshape(bq, 1), (bq, c)).astype(np.float32)
        elif name in {"relation_gain_prior", "relation_fusion_win_rate", "relation_support", "relation_is_visual_prior"}:
            value = np.array([prior[name] for prior in prior_values], dtype=np.float32)
            value = np.broadcast_to(value.reshape(bq, 1), (bq, c)).astype(np.float32)
        elif name == "observed_has_img":
            value = entity_features["has_img"][observed_ids]
            value = np.broadcast_to(value.reshape(bq, 1), (bq, c)).astype(np.float32)
        elif name == "observed_text_img_cosine":
            value = entity_features["text_img_cosine"][observed_ids]
            value = np.broadcast_to(value.reshape(bq, 1), (bq, c)).astype(np.float32)
        elif name == "observed_img_missing_replaced":
            value = entity_features["img_missing_replaced"][observed_ids]
            value = np.broadcast_to(value.reshape(bq, 1), (bq, c)).astype(np.float32)
        elif name == "candidate_has_img":
            value = entity_features["has_img"][candidate_ids]
            value = np.broadcast_to(value.reshape(1, c), (bq, c)).astype(np.float32)
        elif name == "candidate_text_img_cosine":
            value = entity_features["text_img_cosine"][candidate_ids]
            value = np.broadcast_to(value.reshape(1, c), (bq, c)).astype(np.float32)
        elif name == "candidate_img_missing_replaced":
            value = entity_features["img_missing_replaced"][candidate_ids]
            value = np.broadcast_to(value.reshape(1, c), (bq, c)).astype(np.float32)
        elif name == "candidate_text_norm":
            value = entity_features["text_norm"][candidate_ids]
            value = np.broadcast_to(value.reshape(1, c), (bq, c)).astype(np.float32)
        elif name == "candidate_img_norm":
            value = entity_features["img_norm"][candidate_ids]
            value = np.broadcast_to(value.reshape(1, c), (bq, c)).astype(np.float32)
        elif name == "candidate_is_observed_entity":
            value = (candidate_grid == observed_grid).astype(np.float32)
        elif name == "score_gate":
            value = score_gate_safe.astype(np.float32)
        elif name == "score_residual":
            value = score_residual_safe.astype(np.float32)
        elif name == "score_diff":
            value = score_diff.astype(np.float32)
        elif name == "score_mean":
            value = score_mean.astype(np.float32)
        elif name == "score_abs_diff":
            value = score_abs_diff.astype(np.float32)
        elif name == "score_max":
            value = score_max.astype(np.float32)
        elif name == "gate_rank_in_union":
            value = np.minimum(rank_gate, 101).astype(np.float32)
        elif name == "residual_rank_in_union":
            value = np.minimum(rank_residual, 101).astype(np.float32)
        elif name == "in_gate_topk":
            value = (rank_gate <= 100).astype(np.float32)
        elif name == "in_residual_topk":
            value = (rank_residual <= 100).astype(np.float32)
        else:
            raise KeyError(f"Unsupported router feature in full-ranking eval: {name}")
        columns.append(value.reshape(rows))
    return np.stack(columns, axis=1).astype(np.float32)


@torch.inference_mode()
def predict_alpha_matrix(
    router: CandidateSoftRouterMLP,
    config: dict,
    feature_matrix: np.ndarray,
    shape: tuple[int, int],
    device: str,
    alpha_batch_size: int,
) -> torch.Tensor:
    mean = np.array(config["feature_mean"], dtype=np.float32)
    std = np.array(config["feature_std"], dtype=np.float32)
    values = np.clip((feature_matrix - mean) / std, -20.0, 20.0)
    outputs = []
    for start in range(0, values.shape[0], alpha_batch_size):
        batch = torch.from_numpy(values[start : start + alpha_batch_size]).to(device)
        alpha = torch.sigmoid(router(batch))
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        outputs.append(alpha.detach().cpu())
    return torch.cat(outputs, dim=0).view(*shape)


def metric_bundle(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {"count": 0, "mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0}
    ranks = np.array([int(row[f"{prefix}_rank"]) for row in rows], dtype=np.int64)
    rr = 1.0 / ranks
    return {
        "count": int(len(rows)),
        "mrr": float(rr.mean()),
        "hits1": float(np.mean(ranks <= 1)),
        "hits3": float(np.mean(ranks <= 3)),
        "hits10": float(np.mean(ranks <= 10)),
    }


def summarize(rows: list[dict], model_name: str, config: dict, scope: str) -> dict:
    mixed = metric_bundle(rows, "mixed")
    gate = metric_bundle(rows, "gate")
    residual = metric_bundle(rows, "residual")
    return {
        "model": model_name,
        "scope": scope,
        "feature_set": config["feature_set"],
        "loss": config["loss"],
        "full_ranking_mrr": mixed["mrr"],
        "full_ranking_hits1": mixed["hits1"],
        "full_ranking_hits3": mixed["hits3"],
        "full_ranking_hits10": mixed["hits10"],
        "gate_full_mrr": gate["mrr"],
        "residual_full_mrr": residual["mrr"],
        "delta_vs_gate_full": mixed["mrr"] - gate["mrr"],
        "delta_vs_residual_full": mixed["mrr"] - residual["mrr"],
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


def main() -> None:
    args = parse_args()
    gate_cfg = json.loads((Path(args.gate_run_dir) / "config_merged.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device, gate_cfg.get("system", {}).get("device", "cuda"))
    gate_cfg, gate_model, gate_num_entities = load_run(args.gate_run_dir, device)
    residual_cfg, residual_model, residual_num_entities = load_run(args.residual_run_dir, device)
    if gate_num_entities != residual_num_entities:
        raise RuntimeError(f"num_entities mismatch: gate={gate_num_entities}, residual={residual_num_entities}")
    seed = int(gate_cfg.get("system", {}).get("seed", 1))
    residual_seed = int(residual_cfg.get("system", {}).get("seed", seed))
    if seed != residual_seed:
        raise RuntimeError(f"seed mismatch: gate={seed}, residual={residual_seed}")

    router, router_config = load_router(Path(args.model_dir), device)
    triples = load_split_triples(gate_cfg, args.split)
    true_tails_idx, true_heads_idx = build_filtered_indexes(gate_cfg)
    ev_cfg = gate_cfg.get("evaluation", {})
    chunk_size = int(args.chunk_size or ev_cfg.get("chunk_size", 4096))
    query_batch_size = int(args.query_batch_size or ev_cfg.get("query_batch_size", 8))
    cache_dir = args.cache_dir or gate_cfg["dataset"]["cache_dir"]
    entity_features = build_entity_feature_arrays(cache_dir)
    prior_map = relation_prior_map(load_relation_priors(Path(args.relation_priors)))
    has_img = getattr(gate_model, "has_img", None)
    if has_img is None:
        raise RuntimeError("Gate model does not expose has_img.")
    has_img = has_img.detach().cpu().to(dtype=torch.bool)

    directions = ["head", "tail"] if args.direction == "both" else [args.direction]
    max_queries_per_direction = None
    if args.max_queries is not None:
        max_queries_per_direction = max(1, args.max_queries // len(directions))

    if args.dry_run:
        print(f"[OK] model_dir={args.model_dir} feature_set={router_config['feature_set']} loss={router_config['loss']}")
        print(f"[OK] split={args.split} directions={directions} seed={seed} triples={len(triples)}")
        print(f"[OK] num_entities={gate_num_entities} chunk_size={chunk_size} query_batch_size={query_batch_size}")
        print(f"[OK] device={device}")
        return

    rows: list[dict] = []
    candidate_ids = np.arange(gate_num_entities, dtype=np.int64)
    for direction in directions:
        true_index = true_tails_idx if direction == "tail" else true_heads_idx
        triples_eval = triples[:max_queries_per_direction] if max_queries_per_direction else triples
        triples_t = torch.tensor(triples_eval, dtype=torch.long)
        for q_start in range(0, triples_t.size(0), query_batch_size):
            q_end = min(triples_t.size(0), q_start + query_batch_size)
            q_cpu = triples_t[q_start:q_end]
            gate_scores = score_full_matrix(gate_model, q_cpu, direction, true_index, gate_num_entities, chunk_size, device)
            residual_scores = score_full_matrix(
                residual_model, q_cpu, direction, true_index, residual_num_entities, chunk_size, device
            )
            gate_ranks_full = ranks_from_scores(gate_scores)
            residual_ranks_full = ranks_from_scores(residual_scores)
            features = build_feature_matrix(
                raw_features=router_config["raw_features"],
                q_cpu=q_cpu,
                direction=direction,
                relation_priors=prior_map,
                entity_features=entity_features,
                candidate_ids=candidate_ids,
                score_gate=gate_scores.numpy(),
                score_residual=residual_scores.numpy(),
                rank_gate=gate_ranks_full.numpy(),
                rank_residual=residual_ranks_full.numpy(),
            )
            alpha = predict_alpha_matrix(
                router,
                router_config,
                features,
                tuple(gate_scores.shape),
                device,
                args.alpha_batch_size,
            )
            gate_finite = gate_scores[torch.isfinite(gate_scores)]
            residual_finite = residual_scores[torch.isfinite(residual_scores)]
            gate_low = float(gate_finite.min().item()) - 1.0 if gate_finite.numel() else -100.0
            residual_low = float(residual_finite.min().item()) - 1.0 if residual_finite.numel() else -100.0
            gate_high = float(gate_finite.max().item()) + 1.0 if gate_finite.numel() else 100.0
            residual_high = float(residual_finite.max().item()) + 1.0 if residual_finite.numel() else 100.0
            gate_scores_safe = torch.nan_to_num(gate_scores, nan=gate_low, posinf=gate_high, neginf=gate_low)
            residual_scores_safe = torch.nan_to_num(
                residual_scores, nan=residual_low, posinf=residual_high, neginf=residual_low
            )
            mixed_scores = alpha * gate_scores_safe + (1.0 - alpha) * residual_scores_safe
            both_filtered = (~torch.isfinite(gate_scores)) & (~torch.isfinite(residual_scores))
            mixed_scores[both_filtered] = float("-inf")
            target_ids = target_ids_for_direction(q_cpu, direction)
            gate_rank, gate_rr = target_ranks_and_rr(gate_scores, target_ids)
            residual_rank, residual_rr = target_ranks_and_rr(residual_scores, target_ids)
            mixed_rank, mixed_rr = target_ranks_and_rr(mixed_scores, target_ids)

            for j in range(q_cpu.size(0)):
                h_id = int(q_cpu[j, 0].item())
                r_id = int(q_cpu[j, 1].item())
                t_id = int(q_cpu[j, 2].item())
                target_id = int(target_ids[j].item())
                target_has_img = bool(has_img[target_id].item())
                rows.append(
                    {
                        "query_id": f"{args.split}|{seed}|{direction}|r={r_id}|h={h_id}|t={t_id}|target={target_id}",
                        "seed": seed,
                        "split": args.split,
                        "direction": direction,
                        "relation_id": r_id,
                        "head_id": h_id,
                        "tail_id": t_id,
                        "target_entity_id": target_id,
                        "target_regime": target_regime(direction, target_has_img),
                        "gate_rank": gate_rank[j],
                        "gate_rr": gate_rr[j],
                        "residual_rank": residual_rank[j],
                        "residual_rr": residual_rr[j],
                        "mixed_rank": mixed_rank[j],
                        "mixed_rr": mixed_rr[j],
                        "target_alpha": float(alpha[j, target_id].item()),
                        "mean_alpha": float(alpha[j].mean().item()),
                    }
                )

    model_name = Path(args.model_dir).name
    summary_rows = [summarize(rows, model_name, router_config, "overall")]
    by_regime_rows = []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["target_regime"]].append(row)
    for regime, bucket in sorted(buckets.items()):
        by_regime_rows.append(summarize(bucket, model_name, router_config, regime))

    write_csv(Path(args.out_summary), summary_rows)
    if args.out_by_regime:
        write_csv(Path(args.out_by_regime), by_regime_rows)
    write_csv(Path(args.out_query_rows), rows)
    print(f"[OK] evaluated queries -> {len(rows)}")
    print(f"[OK] wrote summary     -> {Path(args.out_summary).as_posix()}")
    if args.out_by_regime:
        print(f"[OK] wrote by-regime   -> {Path(args.out_by_regime).as_posix()}")
    print(f"[OK] wrote query rows  -> {Path(args.out_query_rows).as_posix()}")


if __name__ == "__main__":
    main()
