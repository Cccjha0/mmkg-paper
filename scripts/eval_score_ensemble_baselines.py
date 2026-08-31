from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_candidate_soft_router_full import (
    score_full_matrix,
    target_ids_for_direction,
    target_ranks_and_rr,
)
from scripts.export_candidate_scores import (
    build_filtered_indexes,
    general_target_regime,
    load_split_triples,
    resolve_device,
    target_regime,
)
from scripts.build_candidate_router_paper_tables import markdown_table
from ml.training.src.models.build_model import build_model
from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from router.score_combination import (
    canonical_score_normalization,
    combine_expert_scores,
    shrink_relation_alpha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple Gate-only / Residual-only score-ensemble baselines."
    )
    parser.add_argument("--score-dir", default="outputs/candidate_router/scores")
    parser.add_argument("--output-dir", default="outputs/score_ensemble/eval")
    parser.add_argument("--paper-table-dir", default="docs/paper_tables")
    parser.add_argument(
        "--paper-figures-dir",
        default="docs/paper/figures",
        help="Optional mirror location for LaTeX table inputs used directly by docs/paper/manuscript_main.tex.",
    )
    parser.add_argument("--split", default="test", choices=["test"], help="Final reporting split.")
    parser.add_argument("--selection-split", default="dev", choices=["dev"], help="Split used for alpha selection.")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Evaluate and lock policies on DEV without loading or reporting test artifacts.",
    )
    parser.add_argument(
        "--alphas",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0",
    )
    parser.add_argument("--direction", default="both", choices=["both"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--relation-min-support",
        type=int,
        default=20,
        help="Minimum dev queries required before selecting a relation-specific alpha; lower-support relations fall back to global alpha.",
    )
    parser.add_argument(
        "--score-normalization",
        default="none",
        choices=["none", "query_zscore", "rank", "rank_based"],
        help="Answer-agnostic per-query expert score normalization before interpolation.",
    )
    parser.add_argument(
        "--relation-shrinkage-lambda",
        type=float,
        default=0.0,
        help="General-protocol shrinkage of relation alpha toward global alpha (validation-selected).",
    )
    parser.add_argument("--baseline-summary", default="outputs/router/eval/clean/baseline_locked_summary.csv")
    parser.add_argument("--candidate-main-results", default="outputs/candidate_router/eval/tables/candidate_router_main_results.csv")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for per-seed/direction score-ensemble checkpoints. Defaults to <output-dir>/checkpoints.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing per-seed/direction checkpoints.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Do not write per-seed/direction checkpoints.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N query batches while scoring. Use 0 to print only section start/end messages.",
    )
    parser.add_argument("--quiet-progress", action="store_true", help="Disable detailed progress output.")
    return parser.parse_args()


def parse_alpha_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("alpha grid is empty")
    return values


def load_run_pairs(score_dir: Path, split: str) -> list[dict]:
    pairs = []
    for path in sorted(score_dir.glob(f"{split}_seed*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"seed", "top_k", "gate_run_dir", "residual_run_dir"}
        missing = sorted(required - set(payload))
        if missing:
            continue
        pairs.append(
            {
                "seed": int(payload["seed"]),
                "gate_run_dir": payload["gate_run_dir"],
                "residual_run_dir": payload["residual_run_dir"],
                "summary_path": str(path),
                "dataset": str(payload.get("dataset", "openbg_img")),
                "protocol_version": str(payload.get("protocol_version", OPENBG_LEGACY_V1)),
                "top_k": int(payload["top_k"]),
            }
        )
    if not pairs:
        raise FileNotFoundError(f"No {split} score summaries found in {score_dir}")
    return sorted(pairs, key=lambda row: row["seed"])


def validate_run_pair_metadata(dev_pairs: list[dict], test_pairs: list[dict]) -> tuple[str, str]:
    combined = [*dev_pairs, *test_pairs]
    datasets = {str(pair["dataset"]) for pair in combined}
    protocols = {str(pair["protocol_version"]) for pair in combined}
    top_ks = {int(pair["top_k"]) for pair in combined}
    if len(datasets) != 1 or len(protocols) != 1:
        raise RuntimeError(
            f"Score summaries cannot mix datasets/protocols: datasets={sorted(datasets)}, "
            f"protocols={sorted(protocols)}"
        )
    if len(top_ks) != 1:
        raise RuntimeError(f"Score summaries cannot mix top_k values: {sorted(top_ks)}")
    for split_name, pairs in (("dev", dev_pairs), ("test", test_pairs)):
        seeds = [int(pair["seed"]) for pair in pairs]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(
                f"Duplicate {split_name} summaries for one or more seeds. "
                "Keep exactly one top_k artifact per seed in score-dir."
            )
    protocol = next(iter(protocols))
    if protocol not in {OPENBG_LEGACY_V1, MMKG_GENERAL_V1}:
        raise RuntimeError(f"Unsupported score-summary protocol: {protocol!r}")
    return next(iter(datasets)), protocol


def resolve_run_dir(raw_path: str | Path) -> Path:
    text = str(raw_path).strip().replace("\\", "/")
    path = Path(text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def read_run_config(run_dir: str | Path) -> dict:
    path = resolve_run_dir(run_dir) / "config_merged.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Could not read run config at {path!r}") from exc


def resolve_workspace_path(raw_path: str | Path) -> str:
    path = Path(str(raw_path).strip().replace("\\", "/"))
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def absolutize_dataset_paths(cfg: dict) -> dict:
    dataset = cfg.get("dataset", {})
    for key in ("train", "dev", "test", "cache_dir", "processed_dir"):
        if key in dataset:
            dataset[key] = resolve_workspace_path(dataset[key])
    return cfg


def load_run_stable(run_dir: str | Path, device: str):
    run_dir = resolve_run_dir(run_dir)
    cfg_path = run_dir / "config_merged.json"
    ckpt_path = run_dir / "best.ckpt"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = absolutize_dataset_paths(json.loads(cfg_path.read_text(encoding="utf-8")))
    cfg.setdefault("system", {})["device"] = device
    model, num_entities = build_model(cfg)
    state = torch.load(str(ckpt_path.resolve()), map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return cfg, model, num_entities


def safe_scores(scores: torch.Tensor) -> torch.Tensor:
    finite = scores[torch.isfinite(scores)]
    low = float(finite.min().item()) - 1.0 if finite.numel() else -100.0
    high = float(finite.max().item()) + 1.0 if finite.numel() else 100.0
    return torch.nan_to_num(scores, nan=low, posinf=high, neginf=low)


def reference_ranks_and_rr(
    candidate_scores: torch.Tensor,
    reference_target_scores: torch.Tensor,
) -> tuple[list[int], list[float]]:
    """Rank separately scored targets against candidate-score matrices.

    This mirrors the fixed-expert evaluator: the labeled target score is
    computed through the target-triple path, while candidate scores are
    computed through the chunked all-entity path. The separation matters for
    strict-``>`` ranking when numerically tied multimodal representations are
    evaluated with different batch shapes.
    """
    if candidate_scores.ndim != 2:
        raise ValueError("Candidate scores must be a [queries, entities] matrix.")
    reference_target_scores = reference_target_scores.reshape(-1).to(
        device=candidate_scores.device,
        dtype=candidate_scores.dtype,
    )
    if reference_target_scores.numel() != candidate_scores.size(0):
        raise ValueError("Reference target scores must contain one value per query.")
    ranks = (candidate_scores > reference_target_scores.unsqueeze(1)).sum(dim=1).to(torch.long) + 1
    ranks_list = [int(value) for value in ranks.tolist()]
    return ranks_list, [float(1.0 / value) for value in ranks_list]


def normalize_reference_target_scores(
    candidate_scores: torch.Tensor,
    reference_target_scores: torch.Tensor,
    mode: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply candidate-derived normalization to evaluation-only target scores.

    Normalization parameters depend only on each expert's candidate-score
    distribution. The reference target score is transformed afterward and is
    never exposed as a deployable router feature.
    """
    mode = canonical_score_normalization(mode)
    reference = reference_target_scores.reshape(-1).to(
        device=candidate_scores.device,
        dtype=candidate_scores.dtype,
    )
    if reference.numel() != candidate_scores.size(0):
        raise ValueError("Reference target scores must contain one value per query.")
    if mode == "none":
        return reference
    if mode == "query_zscore":
        finite = torch.isfinite(candidate_scores)
        finite_values = torch.where(finite, candidate_scores, torch.zeros_like(candidate_scores))
        count = finite.sum(dim=1).clamp_min(1)
        mean = finite_values.sum(dim=1) / count
        centered = torch.where(
            finite,
            candidate_scores - mean.unsqueeze(1),
            torch.zeros_like(candidate_scores),
        )
        variance = centered.square().sum(dim=1) / count
        return (reference - mean) / (variance.sqrt() + float(eps))

    # The target's reciprocal-rank score is derived from the original
    # candidate distribution with the same strict-greater tie rule.
    ranks = (candidate_scores > reference.unsqueeze(1)).sum(dim=1).to(candidate_scores.dtype) + 1.0
    return ranks.reciprocal()


def combine_with_reference_targets(
    gate_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    gate_target_scores: torch.Tensor,
    residual_target_scores: torch.Tensor,
    alpha: float | np.ndarray,
    score_normalization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine candidate matrices and separately scored evaluation targets."""
    score_normalization = canonical_score_normalization(score_normalization)
    mixed = combine_expert_scores(
        gate_scores,
        residual_scores,
        torch.as_tensor(alpha, dtype=gate_scores.dtype, device=gate_scores.device),
        normalization=score_normalization,
    )
    gate_reference = normalize_reference_target_scores(
        gate_scores,
        gate_target_scores,
        score_normalization,
    )
    residual_reference = normalize_reference_target_scores(
        residual_scores,
        residual_target_scores,
        score_normalization,
    )
    alpha_value = torch.as_tensor(alpha, dtype=gate_reference.dtype, device=gate_reference.device)
    if alpha_value.ndim == 0:
        alpha_value = alpha_value.expand(gate_reference.numel())
    else:
        alpha_value = alpha_value.reshape(-1)
    if alpha_value.numel() != gate_reference.numel():
        raise ValueError("alpha must be scalar or contain one value per query.")
    mixed_reference = alpha_value * gate_reference + (1.0 - alpha_value) * residual_reference

    # The mathematical endpoints are the original fixed experts. Preserve
    # their scores exactly instead of relying on an otherwise rank-preserving
    # floating-point transform; the latter can collapse near-equal values.
    gate_endpoint = alpha_value == 1.0
    residual_endpoint = alpha_value == 0.0
    if bool(gate_endpoint.any()):
        gate_raw_reference = gate_target_scores.reshape(-1).to(
            device=mixed_reference.device,
            dtype=mixed_reference.dtype,
        )
        mixed = torch.where(gate_endpoint.unsqueeze(1), gate_scores, mixed)
        mixed_reference = torch.where(gate_endpoint, gate_raw_reference, mixed_reference)
    if bool(residual_endpoint.any()):
        residual_raw_reference = residual_target_scores.reshape(-1).to(
            device=mixed_reference.device,
            dtype=mixed_reference.dtype,
        )
        mixed = torch.where(residual_endpoint.unsqueeze(1), residual_scores, mixed)
        mixed_reference = torch.where(residual_endpoint, residual_raw_reference, mixed_reference)
    return mixed, mixed_reference


def query_features(
    gate_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    direction: str,
    relations: torch.Tensor,
    protocol_version: str = OPENBG_LEGACY_V1,
) -> np.ndarray:
    def stats(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe = safe_scores(scores)
        top = torch.topk(safe, k=min(5, safe.size(1)), dim=1).values
        top1 = top[:, 0]
        top2 = top[:, 1] if top.size(1) > 1 else top[:, 0]
        return top1, top[:, : min(5, top.size(1))].mean(dim=1), top1 - top2, safe.std(dim=1)

    g1, g5, gm, gs = stats(gate_scores)
    r1, r5, rm, rs = stats(residual_scores)
    direction_tail = torch.full_like(g1, 1.0 if direction == "tail" else 0.0)
    rel = relations.to(dtype=torch.float32)
    score_features = [g1, g5, gm, gs, r1, r5, rm, rs, g1 - r1, g5 - r5, gm - rm, gs - rs]
    if protocol_version == OPENBG_LEGACY_V1:
        # Frozen C4 input order.  Raw relation id is retained only so existing
        # OpenBG artifacts and saved score-aware models remain reproducible.
        columns = [direction_tail, rel, *score_features]
    elif protocol_version == MMKG_GENERAL_V1:
        # Relation ids are categorical identifiers, not ordinal measurements.
        # Dataset-local relation priors can be joined by the caller separately.
        columns = [direction_tail, *score_features]
    else:
        raise ValueError(f"Unsupported protocol version: {protocol_version}")
    features = torch.stack(columns, dim=1)
    return features.detach().cpu().numpy().astype(np.float32)


def eval_mixed_rr(
    gate_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    target_ids: torch.Tensor,
    alpha: float | np.ndarray,
    score_normalization: str = "none",
    gate_target_scores: torch.Tensor | None = None,
    residual_target_scores: torch.Tensor | None = None,
) -> list[float]:
    if (gate_target_scores is None) != (residual_target_scores is None):
        raise ValueError("Gate and residual reference target scores must be supplied together.")
    if gate_target_scores is not None and residual_target_scores is not None:
        mixed, mixed_reference = combine_with_reference_targets(
            gate_scores,
            residual_scores,
            gate_target_scores,
            residual_target_scores,
            alpha,
            score_normalization,
        )
        _, rr = reference_ranks_and_rr(mixed, mixed_reference)
        return rr
    if canonical_score_normalization(score_normalization) == "none":
        gate_safe = safe_scores(gate_scores)
        residual_safe = safe_scores(residual_scores)
        alpha_value = torch.as_tensor(alpha, dtype=gate_safe.dtype, device=gate_safe.device)
        if alpha_value.ndim == 0:
            alpha_value = alpha_value.expand(gate_safe.size(0)).unsqueeze(1)
        else:
            alpha_value = alpha_value.view(-1, 1)
        mixed = alpha_value * gate_safe + (1.0 - alpha_value) * residual_safe
        both_filtered = (~torch.isfinite(gate_scores)) & (~torch.isfinite(residual_scores))
        mixed[both_filtered] = float("-inf")
    else:
        alpha_value = torch.as_tensor(alpha, dtype=gate_scores.dtype, device=gate_scores.device)
        mixed = combine_expert_scores(
            gate_scores,
            residual_scores,
            alpha_value,
            normalization=score_normalization,
        )
    _, rr = target_ranks_and_rr(mixed, target_ids)
    return rr


def eval_mixed_ranks_and_rr(
    gate_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    target_ids: torch.Tensor,
    alpha: float | np.ndarray,
    score_normalization: str = "none",
    gate_target_scores: torch.Tensor | None = None,
    residual_target_scores: torch.Tensor | None = None,
) -> tuple[list[int], list[float]]:
    if (gate_target_scores is None) != (residual_target_scores is None):
        raise ValueError("Gate and residual reference target scores must be supplied together.")
    if gate_target_scores is not None and residual_target_scores is not None:
        mixed, mixed_reference = combine_with_reference_targets(
            gate_scores,
            residual_scores,
            gate_target_scores,
            residual_target_scores,
            alpha,
            score_normalization,
        )
        return reference_ranks_and_rr(mixed, mixed_reference)
    if canonical_score_normalization(score_normalization) == "none":
        gate_safe = safe_scores(gate_scores)
        residual_safe = safe_scores(residual_scores)
        alpha_value = torch.as_tensor(alpha, dtype=gate_safe.dtype, device=gate_safe.device)
        if alpha_value.ndim == 0:
            alpha_value = alpha_value.expand(gate_safe.size(0)).unsqueeze(1)
        else:
            alpha_value = alpha_value.view(-1, 1)
        mixed = alpha_value * gate_safe + (1.0 - alpha_value) * residual_safe
        both_filtered = (~torch.isfinite(gate_scores)) & (~torch.isfinite(residual_scores))
        mixed[both_filtered] = float("-inf")
    else:
        alpha_value = torch.as_tensor(alpha, dtype=gate_scores.dtype, device=gate_scores.device)
        mixed = combine_expert_scores(
            gate_scores,
            residual_scores,
            alpha_value,
            normalization=score_normalization,
        )
    return target_ranks_and_rr(mixed, target_ids)


def alpha_column(alpha: float) -> str:
    return f"rr_alpha_{alpha:.2f}".replace(".", "_")


def evaluate_split(
    *,
    run_pairs: list[dict],
    split: str,
    alphas: list[float],
    device_arg: str | None,
    chunk_size_arg: int | None,
    query_batch_size_arg: int | None,
    max_queries: int | None,
    query_model=None,
    selected_global_alpha: float | None = None,
    selected_direction_alpha: dict[str, float] | None = None,
    selected_relation_alpha: dict[int, float] | None = None,
    selected_relation_fallback_alpha: float | None = None,
    progress_every: int = 25,
    quiet_progress: bool = False,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    write_checkpoints: bool = True,
    score_normalization: str = "none",
) -> dict:
    score_normalization = canonical_score_normalization(score_normalization)
    alpha_rr: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    alpha_direction_rr: dict[str, dict[float, list[float]]] = {
        "head": {alpha: [] for alpha in alphas},
        "tail": {alpha: [] for alpha in alphas},
    }
    alpha_relation_rr: dict[int, dict[float, list[float]]] = defaultdict(lambda: {alpha: [] for alpha in alphas})
    global_rows: list[dict] = []
    direction_rows: list[dict] = []
    relation_rows: list[dict] = []
    query_rows: list[dict] = []
    selected_policy_rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    gate_endpoint_rr: list[float] = []
    residual_endpoint_rr: list[float] = []
    evaluation_dataset: str | None = None
    evaluation_protocol: str | None = None
    started_at = time.time()
    total_batches = 0
    total_queries = 0
    pair_work: list[tuple[dict, int, int, list[str], int]] = []

    for pair in run_pairs:
        gate_cfg_for_count = read_run_config(pair["gate_run_dir"])
        triples_for_count = load_split_triples(gate_cfg_for_count, split)
        ev_cfg_for_count = gate_cfg_for_count.get("evaluation", {})
        query_batch_size = int(query_batch_size_arg or ev_cfg_for_count.get("query_batch_size", 8))
        directions = ["head", "tail"]
        max_queries_per_direction = None if max_queries is None else max(1, max_queries // len(directions))
        n_direction_queries = len(triples_for_count[:max_queries_per_direction] if max_queries_per_direction else triples_for_count)
        n_pair_batches = len(directions) * ((n_direction_queries + query_batch_size - 1) // query_batch_size)
        pair_work.append((pair, query_batch_size, n_direction_queries, directions, n_pair_batches))
        total_batches += n_pair_batches
        total_queries += len(directions) * n_direction_queries

    completed_batches = 0
    completed_queries = 0

    def fmt_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m{s:02d}s"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    def progress_line(seed: int, direction: str, q_end: int, n_direction_queries: int, force: bool = False) -> None:
        if quiet_progress:
            return
        if not force and progress_every <= 0:
            return
        if not force and completed_batches % progress_every != 0:
            return
        elapsed = time.time() - started_at
        frac = completed_batches / max(total_batches, 1)
        eta = elapsed * (1.0 / frac - 1.0) if frac > 0 else 0.0
        print(
            f"[PROGRESS] {split} {completed_batches}/{total_batches} batches "
            f"({frac * 100:5.1f}%) queries={completed_queries}/{total_queries} "
            f"seed={seed} direction={direction} direction_q={q_end}/{n_direction_queries} "
            f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}",
            flush=True,
        )

    def checkpoint_path(seed: int, direction: str) -> Path | None:
        if checkpoint_dir is None or selected_global_alpha is None:
            return None
        key = "full" if max_queries is None else f"max{max_queries}"
        base = checkpoint_dir / key
        if score_normalization != "none":
            base = base / score_normalization
        return base / f"{split}_seed{seed}_{direction}_score_ensemble_query_rows.csv"

    def ingest_selected_rows(rows: list[dict]) -> None:
        selected_policy_rows.extend(rows)
        for row in rows:
            gate_endpoint_rr.append(float(row["rr_gate"]))
            residual_endpoint_rr.append(float(row["rr_residual"]))
            rr_global = float(row["rr_global_interp"])
            rr_direction = float(row["rr_direction_interp"])
            rr_relation = float(row["rr_relation_interp"])
            global_rows.append({"mixed_rr": rr_global})
            direction_rows.append({"mixed_rr": rr_direction})
            relation_rows.append({"mixed_rr": rr_relation})
            if "rr_query_soft" in row and str(row["rr_query_soft"]) != "":
                query_rows.append({"mixed_rr": float(row["rr_query_soft"]), "alpha": float(row.get("alpha_query_soft", 0.0))})
            relation_id = int(row["relation_id"])
            direction = str(row["direction"])
            for alpha in alphas:
                value = float(row[alpha_column(alpha)])
                alpha_rr[alpha].append(value)
                alpha_direction_rr[direction][alpha].append(value)
                alpha_relation_rr[relation_id][alpha].append(value)

    if not quiet_progress:
        print(
            f"[PROGRESS] {split} starting: {len(run_pairs)} seeds, {total_queries} direction-queries, "
            f"{total_batches} batches",
            flush=True,
    )

    for pair, query_batch_size, _n_direction_queries, directions, _n_pair_batches in pair_work:
        gate_cfg_raw = read_run_config(pair["gate_run_dir"])
        device = resolve_device(device_arg, gate_cfg_raw.get("system", {}).get("device", "cuda"))
        gate_run_dir = resolve_run_dir(pair["gate_run_dir"])
        residual_run_dir = resolve_run_dir(pair["residual_run_dir"])
        gate_cfg, gate_model, gate_num_entities = load_run_stable(gate_run_dir, device)
        residual_cfg, residual_model, residual_num_entities = load_run_stable(residual_run_dir, device)
        protocol_version = str(gate_cfg.get("protocol", {}).get("version", OPENBG_LEGACY_V1))
        residual_protocol = str(residual_cfg.get("protocol", {}).get("version", OPENBG_LEGACY_V1))
        if protocol_version != residual_protocol:
            raise RuntimeError("Gate and Residual protocol versions differ.")
        dataset_name = str(gate_cfg.get("dataset", {}).get("name", "openbg_img"))
        residual_dataset = str(residual_cfg.get("dataset", {}).get("name", "openbg_img"))
        if dataset_name != residual_dataset:
            raise RuntimeError("Gate and Residual datasets differ.")
        if evaluation_dataset is not None and (dataset_name, protocol_version) != (
            evaluation_dataset,
            evaluation_protocol,
        ):
            raise RuntimeError("Score-aware alpha/model fitting cannot mix datasets or protocol versions.")
        evaluation_dataset, evaluation_protocol = dataset_name, protocol_version
        if gate_num_entities != residual_num_entities:
            raise RuntimeError("Gate and Residual entity counts differ.")
        seed = int(gate_cfg.get("system", {}).get("seed", pair["seed"]))
        if seed != int(residual_cfg.get("system", {}).get("seed", seed)):
            raise RuntimeError(f"Seed mismatch for pair {pair}")
        triples = load_split_triples(gate_cfg, split)
        true_tails_idx, true_heads_idx = build_filtered_indexes(gate_cfg)
        has_img = getattr(gate_model, "has_img", None)
        if has_img is None:
            raise RuntimeError("Gate model does not expose has_img.")
        has_img = has_img.detach().cpu().to(dtype=torch.bool)
        residual_has_img = getattr(residual_model, "has_img", None)
        if residual_has_img is not None and not torch.equal(has_img, residual_has_img.detach().cpu().bool()):
            raise RuntimeError("Gate and Residual image-availability masks differ.")
        if protocol_version == OPENBG_LEGACY_V1 and residual_has_img is None:
            raise RuntimeError("Legacy Residual model does not expose has_img.")
        has_text = getattr(gate_model, "has_text", None)
        if protocol_version == MMKG_GENERAL_V1 and has_text is None:
            raise RuntimeError("General Gate model does not expose has_text.")
        has_text = has_text.detach().cpu().bool() if has_text is not None else None
        residual_has_text = getattr(residual_model, "has_text", None)
        if protocol_version == MMKG_GENERAL_V1 and residual_has_text is not None:
            if not torch.equal(has_text, residual_has_text.detach().cpu().bool()):
                raise RuntimeError("Gate and Residual text-availability masks differ.")
        ev_cfg = gate_cfg.get("evaluation", {})
        chunk_size = int(chunk_size_arg or ev_cfg.get("chunk_size", 4096))
        max_queries_per_direction = None if max_queries is None else max(1, max_queries // len(directions))

        for direction in directions:
            true_index = true_tails_idx if direction == "tail" else true_heads_idx
            triples_eval = triples[:max_queries_per_direction] if max_queries_per_direction else triples
            triples_t = torch.tensor(triples_eval, dtype=torch.long)
            n_direction_queries = int(triples_t.size(0))
            ckpt_path = checkpoint_path(seed, direction)
            if resume and ckpt_path is not None and ckpt_path.exists():
                frame = pd.read_csv(ckpt_path)
                cached_rows = frame.to_dict(orient="records")
                if protocol_version == MMKG_GENERAL_V1:
                    cached_datasets = {str(row.get("dataset", "")) for row in cached_rows}
                    cached_protocols = {str(row.get("protocol_version", "")) for row in cached_rows}
                    cached_target_semantics = {
                        str(row.get("target_score_semantics", "")) for row in cached_rows
                    }
                    if (
                        cached_datasets != {dataset_name}
                        or cached_protocols != {protocol_version}
                        or cached_target_semantics != {"canonical_separate_target_score"}
                    ):
                        raise RuntimeError(
                            f"Cached score-ensemble rows do not match {dataset_name}/{protocol_version}: "
                            f"datasets={sorted(cached_datasets)}, protocols={sorted(cached_protocols)}"
                        )
                ingest_selected_rows(cached_rows)
                cached_batches = (n_direction_queries + query_batch_size - 1) // query_batch_size
                completed_batches += cached_batches
                completed_queries += n_direction_queries
                if not quiet_progress:
                    print(f"[PROGRESS] {split} seed={seed} direction={direction} resumed from {ckpt_path}", flush=True)
                progress_line(seed, direction, n_direction_queries, n_direction_queries, force=True)
                continue
            direction_selected_rows: list[dict] = []
            for q_start in range(0, triples_t.size(0), query_batch_size):
                q_end = min(triples_t.size(0), q_start + query_batch_size)
                q_cpu = triples_t[q_start:q_end]
                gate_scores = score_full_matrix(gate_model, q_cpu, direction, true_index, gate_num_entities, chunk_size, device)
                residual_scores = score_full_matrix(
                    residual_model, q_cpu, direction, true_index, residual_num_entities, chunk_size, device
                )
                target_ids = target_ids_for_direction(q_cpu, direction)
                gate_target_scores: torch.Tensor | None = None
                residual_target_scores: torch.Tensor | None = None
                if protocol_version == MMKG_GENERAL_V1:
                    # General score-aware evaluation must reproduce the fixed
                    # evaluator at alpha endpoints. OpenBG legacy intentionally
                    # retains its historical full-matrix target semantics.
                    target_triples = q_cpu.to(device)
                    gate_target_scores = gate_model.score(target_triples).detach().cpu()
                    residual_target_scores = residual_model.score(target_triples).detach().cpu()
                    _, gate_rr = reference_ranks_and_rr(gate_scores, gate_target_scores)
                    _, residual_rr = reference_ranks_and_rr(residual_scores, residual_target_scores)
                else:
                    _, gate_rr = target_ranks_and_rr(gate_scores, target_ids)
                    _, residual_rr = target_ranks_and_rr(residual_scores, target_ids)
                gate_endpoint_rr.extend(gate_rr)
                residual_endpoint_rr.extend(residual_rr)
                feats = query_features(
                    gate_scores,
                    residual_scores,
                    direction,
                    q_cpu[:, 1],
                    protocol_version=protocol_version,
                )
                feature_rows.append(feats)
                labels.extend([int(g > r) for g, r in zip(gate_rr, residual_rr)])

                alpha_rr_by_value: dict[float, list[float]] = {}
                for alpha in alphas:
                    rr = eval_mixed_rr(
                        gate_scores,
                        residual_scores,
                        target_ids,
                        alpha,
                        score_normalization=score_normalization,
                        gate_target_scores=gate_target_scores,
                        residual_target_scores=residual_target_scores,
                    )
                    alpha_rr_by_value[alpha] = rr
                    alpha_rr[alpha].extend(rr)
                    alpha_direction_rr[direction][alpha].extend(rr)
                    for relation_id, value in zip(q_cpu[:, 1].tolist(), rr):
                        alpha_relation_rr[int(relation_id)][alpha].append(value)

                global_rank: list[int] | None = None
                global_rr: list[float] | None = None
                direction_rank: list[int] | None = None
                direction_rr: list[float] | None = None
                relation_rank: list[int] | None = None
                relation_rr: list[float] | None = None
                relation_alpha: np.ndarray | None = None
                query_soft_rr: list[float] | None = None
                query_soft_alpha: np.ndarray | None = None
                if selected_global_alpha is not None:
                    global_rank, global_rr = eval_mixed_ranks_and_rr(
                        gate_scores,
                        residual_scores,
                        target_ids,
                        selected_global_alpha,
                        score_normalization=score_normalization,
                        gate_target_scores=gate_target_scores,
                        residual_target_scores=residual_target_scores,
                    )
                    global_rows.extend({"mixed_rr": value} for value in global_rr)
                if selected_direction_alpha is not None:
                    direction_rank, direction_rr = eval_mixed_ranks_and_rr(
                        gate_scores,
                        residual_scores,
                        target_ids,
                        selected_direction_alpha[direction],
                        score_normalization=score_normalization,
                        gate_target_scores=gate_target_scores,
                        residual_target_scores=residual_target_scores,
                    )
                    direction_rows.extend({"mixed_rr": value} for value in direction_rr)
                if selected_relation_alpha is not None:
                    fallback_alpha = selected_relation_fallback_alpha
                    if fallback_alpha is None:
                        fallback_alpha = selected_global_alpha if selected_global_alpha is not None else 0.0
                    relation_alpha = np.array(
                        [
                            selected_relation_alpha.get(int(relation_id), float(fallback_alpha))
                            for relation_id in q_cpu[:, 1].tolist()
                        ],
                        dtype=np.float32,
                    )
                    relation_rank, relation_rr = eval_mixed_ranks_and_rr(
                        gate_scores,
                        residual_scores,
                        target_ids,
                        relation_alpha,
                        score_normalization=score_normalization,
                        gate_target_scores=gate_target_scores,
                        residual_target_scores=residual_target_scores,
                    )
                    relation_rows.extend({"mixed_rr": value} for value in relation_rr)
                if query_model is not None:
                    query_soft_alpha = query_model.predict_proba(feats)[:, 1].astype(np.float32)
                    query_soft_rr = eval_mixed_rr(
                        gate_scores,
                        residual_scores,
                        target_ids,
                        query_soft_alpha,
                        score_normalization=score_normalization,
                        gate_target_scores=gate_target_scores,
                        residual_target_scores=residual_target_scores,
                    )
                    query_rows.extend({"mixed_rr": value, "alpha": float(a)} for value, a in zip(query_soft_rr, query_soft_alpha))

                if selected_global_alpha is not None:
                    if global_rank is None or global_rr is None or direction_rank is None or direction_rr is None:
                        raise RuntimeError("Selected interpolation rows were requested but selected ranks/RR are missing.")
                    if relation_rank is None or relation_rr is None or relation_alpha is None:
                        raise RuntimeError("Relation-specific interpolation rows were requested but relation ranks/RR are missing.")
                    for j in range(q_cpu.size(0)):
                        h_id = int(q_cpu[j, 0].item())
                        r_id = int(q_cpu[j, 1].item())
                        t_id = int(q_cpu[j, 2].item())
                        target_id = int(target_ids[j].item())
                        target_has_img = bool(has_img[target_id].item())
                        target_has_text = bool(has_text[target_id].item()) if has_text is not None else True
                        row = {
                            "query_id": f"{split}|{seed}|{direction}|r={r_id}|h={h_id}|t={t_id}|target={target_id}",
                            "split": split,
                            "seed": seed,
                            "direction": direction,
                            "relation_id": r_id,
                            "head_id": h_id,
                            "tail_id": t_id,
                            "target_entity_id": target_id,
                            "target_regime": (
                                target_regime(direction, target_has_img)
                                if protocol_version == OPENBG_LEGACY_V1
                                else general_target_regime(direction, target_has_text, target_has_img)
                            ),
                            "rr_gate": float(gate_rr[j]),
                            "rr_residual": float(residual_rr[j]),
                            "rr_global_interp": float(global_rr[j]),
                            "rr_direction_interp": float(direction_rr[j]),
                            "rr_relation_interp": float(relation_rr[j]),
                            "rank_global_interp": int(global_rank[j]),
                            "rank_direction_interp": int(direction_rank[j]),
                            "rank_relation_interp": int(relation_rank[j]),
                            "alpha_global": float(selected_global_alpha),
                            "alpha_direction": float(selected_direction_alpha[direction]) if selected_direction_alpha else float("nan"),
                            "alpha_relation": float(relation_alpha[j]),
                        }
                        if protocol_version == MMKG_GENERAL_V1:
                            row["dataset"] = dataset_name
                            row["protocol_version"] = protocol_version
                            row["target_score_semantics"] = "canonical_separate_target_score"
                            row["target_has_text"] = int(target_has_text)
                            row["target_has_img"] = int(target_has_img)
                        if query_soft_rr is not None and query_soft_alpha is not None:
                            row["rr_query_soft"] = float(query_soft_rr[j])
                            row["alpha_query_soft"] = float(query_soft_alpha[j])
                        for alpha, rr_values in alpha_rr_by_value.items():
                            row[alpha_column(alpha)] = float(rr_values[j])
                        direction_selected_rows.append(row)
                completed_batches += 1
                completed_queries += int(q_end - q_start)
                progress_line(seed, direction, q_end, n_direction_queries)
            if ckpt_path is not None:
                selected_policy_rows.extend(direction_selected_rows)
                if write_checkpoints:
                    write_csv_rows(ckpt_path, direction_selected_rows)

    progress_line(seed if run_pairs else -1, "done", total_queries, total_queries, force=True)
    if evaluation_protocol == MMKG_GENERAL_V1:
        for endpoint, expected in ((0.0, residual_endpoint_rr), (1.0, gate_endpoint_rr)):
            if endpoint not in alpha_rr:
                continue
            actual = alpha_rr[endpoint]
            if len(actual) != len(expected) or any(a != b for a, b in zip(actual, expected)):
                expert = "structural" if endpoint == 0.0 else "fusion"
                raise RuntimeError(
                    f"General score interpolation alpha={endpoint:g} does not reproduce the {expert} endpoint."
                )
    return {
        "alpha_rr": alpha_rr,
        "alpha_direction_rr": alpha_direction_rr,
        "alpha_relation_rr": {relation_id: dict(alpha_map) for relation_id, alpha_map in alpha_relation_rr.items()},
        "global_rows": global_rows,
        "direction_rows": direction_rows,
        "relation_rows": relation_rows,
        "query_rows": query_rows,
        "selected_policy_rows": selected_policy_rows,
        "features": np.concatenate(feature_rows, axis=0) if feature_rows else np.empty((0, 0), dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "gate_endpoint_rr": gate_endpoint_rr,
        "residual_endpoint_rr": residual_endpoint_rr,
    }


def best_alpha(alpha_rr: dict[float, list[float]]) -> tuple[float, float]:
    scored = [(alpha, float(np.mean(rr)) if rr else 0.0) for alpha, rr in alpha_rr.items()]
    return max(scored, key=lambda item: (item[1], -item[0]))


def select_relation_alphas(
    alpha_relation_rr: dict[int, dict[float, list[float]]],
    *,
    fallback_alpha: float,
    min_support: int,
    shrinkage_lambda: float = 0.0,
) -> tuple[dict[int, float], dict]:
    selected: dict[int, float] = {}
    summary_rows = []
    for relation_id, alpha_map in sorted(alpha_relation_rr.items()):
        support = max((len(values) for values in alpha_map.values()), default=0)
        if support >= min_support:
            raw_alpha, dev_mrr = best_alpha(alpha_map)
            alpha = shrink_relation_alpha(
                raw_alpha,
                support=support,
                global_alpha=fallback_alpha,
                shrinkage_lambda=shrinkage_lambda,
            )
            used_fallback = False
        else:
            raw_alpha = fallback_alpha
            alpha = fallback_alpha
            dev_mrr = float(np.mean(alpha_map.get(fallback_alpha, []))) if alpha_map.get(fallback_alpha) else 0.0
            used_fallback = True
        selected[int(relation_id)] = float(alpha)
        summary_rows.append(
            {
                "relation_id": int(relation_id),
                "support": int(support),
                "raw_alpha": float(raw_alpha),
                "alpha": float(alpha),
                "dev_mrr": float(dev_mrr),
                "used_fallback": bool(used_fallback),
            }
        )
    summary = {
        "min_support": int(min_support),
        "fallback_alpha": float(fallback_alpha),
        "shrinkage_lambda": float(shrinkage_lambda),
        "n_relations": len(summary_rows),
        "n_relation_specific": sum(1 for row in summary_rows if not row["used_fallback"]),
        "n_fallback": sum(1 for row in summary_rows if row["used_fallback"]),
        "relations": summary_rows,
    }
    return selected, summary


def rows_to_metrics(rows: list[dict]) -> dict:
    rr = np.array([float(row["mixed_rr"]) for row in rows], dtype=np.float64)
    ranks = np.rint(1.0 / np.maximum(rr, 1e-12)).astype(np.int64)
    return {
        "count": int(rr.size),
        "mrr": float(rr.mean()) if rr.size else 0.0,
        "hits1": float((ranks <= 1).mean()) if rr.size else 0.0,
        "hits3": float((ranks <= 3).mean()) if rr.size else 0.0,
        "hits10": float((ranks <= 10).mean()) if rr.size else 0.0,
    }


def load_reference_metrics(baseline_summary: Path, candidate_main: Path) -> dict:
    baseline = pd.read_csv(baseline_summary)
    by_method = {row["method"]: row for _, row in baseline.iterrows()}
    candidate = pd.read_csv(candidate_main)
    ca_s2 = candidate[candidate["Method"].eq("CA-S2 score-aware")]
    if ca_s2.empty:
        raise RuntimeError("Could not find CA-S2 row in candidate main results.")
    return {
        "residual": float(by_method["Residual-only"]["mrr"]),
        "e5": float(by_method["Regression-based clean router"]["mrr"]),
        "ca_s2": float(str(ca_s2.iloc[0]["MRR"]).split()[0]),
    }


def result_row(method: str, granularity: str, alpha_policy: str, metrics: dict, refs: dict, notes: str) -> dict:
    mrr = float(metrics["mrr"])
    row = {
        "method": method,
        "level": "ensemble",
        "granularity": granularity,
        "selected_on": "dev",
        "alpha_policy": alpha_policy,
        "mrr": mrr,
        "hits1": float(metrics["hits1"]),
        "hits3": float(metrics["hits3"]),
        "hits10": float(metrics["hits10"]),
        "delta_vs_residual": mrr - refs["residual"],
        "notes": notes,
    }
    if "e5" in refs and "ca_s2" in refs:
        row["delta_vs_e5"] = mrr - refs["e5"]
        row["delta_vs_ca_s2"] = mrr - refs["ca_s2"]
    return row


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def write_latex_table(path: Path, rows: list[dict], refs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paper_rows = []
    for row in rows:
        paper_rows.append(
            {
                "Method": row["method"],
                "Level": row["level"],
                "Granularity": row["granularity"],
                "MRR": fmt(row["mrr"]),
                "Delta vs E5": fmt_delta(row["delta_vs_e5"]),
                "Delta vs CA-S2": "--" if row["method"].startswith("CA-S2") else fmt_delta(row["delta_vs_ca_s2"]),
            }
        )
    paper_rows.append(
        {
            "Method": "CA-S2 score-aware candidate router",
            "Level": "router",
            "Granularity": "candidate",
            "MRR": fmt(refs["ca_s2"]),
            "Delta vs E5": fmt_delta(refs["ca_s2"] - refs["e5"]),
            "Delta vs CA-S2": "--",
        }
    )
    frame = pd.DataFrame(paper_rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Simple score-ensemble baselines compared with CA-S2 under full filtered ranking. Ensemble baselines use fixed Gate-only and Residual-only scores, select their policies on the development split, and are evaluated on the test split.}",
        r"\label{tab:score_ensemble_baselines}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.30\textwidth}p{0.12\textwidth}p{0.13\textwidth}ccc}",
        r"\toprule",
        "Method & Level & Granularity & MRR & Delta vs E5 & Delta vs CA-S2" + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(str(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}\caption*{\footnotesize The baselines test whether CA-S2 can be explained by fixed, direction-specific, relation-specific, or query-level score averaging alone.}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def alpha_curve_rows(split: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    for scope, alpha_map in [
        ("global", payload["alpha_rr"]),
        ("head", payload["alpha_direction_rr"]["head"]),
        ("tail", payload["alpha_direction_rr"]["tail"]),
    ]:
        for alpha, rr_values in sorted(alpha_map.items()):
            rows.append(
                {
                    "split": split,
                    "scope": scope,
                    "alpha": float(alpha),
                    "mrr": float(np.mean(rr_values)) if rr_values else 0.0,
                    "n_queries": int(len(rr_values)),
                }
            )
    return rows


def write_alpha_curve_outputs(output_dir: Path, paper_figures_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = output_dir / "score_ensemble_alpha_curve.csv"
    write_csv_rows(curve_csv, rows)

    frame = pd.DataFrame(rows)
    md_frame = frame.copy()
    md_frame["alpha"] = md_frame["alpha"].map(lambda value: f"{float(value):.2f}")
    md_frame["mrr"] = md_frame["mrr"].map(lambda value: f"{float(value):.4f}")
    (output_dir / "score_ensemble_alpha_curve.md").write_text(markdown_table(md_frame) + "\n", encoding="utf-8")

    best_rows = []
    for (split, scope), bucket in frame.groupby(["split", "scope"], sort=True):
        best = bucket.sort_values(["mrr", "alpha"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "split": split,
                "scope": scope,
                "best_alpha": float(best["alpha"]),
                "best_mrr": float(best["mrr"]),
                "n_queries": int(best["n_queries"]),
            }
        )
    best_frame = pd.DataFrame(best_rows)
    write_csv_rows(output_dir / "score_ensemble_alpha_curve_best.csv", best_rows)
    best_md = best_frame.copy()
    best_md["best_alpha"] = best_md["best_alpha"].map(lambda value: f"{float(value):.2f}")
    best_md["best_mrr"] = best_md["best_mrr"].map(lambda value: f"{float(value):.4f}")
    (output_dir / "score_ensemble_alpha_curve_best.md").write_text(markdown_table(best_md) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] skipped alpha curve plots because matplotlib is unavailable: {exc}")
        return

    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(frame["split"].unique()):
        split_frame = frame[frame["split"].eq(split)]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for scope, style in [("global", "-o"), ("head", "-s"), ("tail", "-^")]:
            series = split_frame[split_frame["scope"].eq(scope)].sort_values("alpha")
            ax.plot(series["alpha"], series["mrr"], style, linewidth=1.4, markersize=3.2, label=scope)
        ax.set_xlabel("Interpolation weight alpha for Gate-only")
        ax.set_ylabel(f"{split} MRR")
        ax.set_title(f"Score interpolation alpha sweep ({split})")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        png_path = paper_figures_dir / f"score_interpolation_alpha_curve_{split}.png"
        pdf_path = paper_figures_dir / f"score_interpolation_alpha_curve_{split}.pdf"
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    alphas = parse_alpha_grid(args.alphas)
    score_normalization = canonical_score_normalization(args.score_normalization)
    if args.relation_shrinkage_lambda < 0.0:
        raise ValueError("--relation-shrinkage-lambda must be non-negative.")
    dev_pairs = load_run_pairs(score_dir, args.selection_split)
    test_pairs = [] if args.selection_only else load_run_pairs(score_dir, args.split)
    dataset_name, protocol_version = validate_run_pair_metadata(dev_pairs, test_pairs)
    if protocol_version == OPENBG_LEGACY_V1 and args.relation_shrinkage_lambda != 0.0:
        raise ValueError("Relation-alpha shrinkage is general-protocol only; OpenBG legacy must use lambda=0.")
    refs = (
        load_reference_metrics(Path(args.baseline_summary), Path(args.candidate_main_results))
        if protocol_version == OPENBG_LEGACY_V1 and not args.selection_only
        else None
    )

    print("[INFO] evaluating development split for policy selection")
    dev = evaluate_split(
        run_pairs=dev_pairs,
        split=args.selection_split,
        alphas=alphas,
        device_arg=args.device,
        chunk_size_arg=args.chunk_size,
        query_batch_size_arg=args.query_batch_size,
        max_queries=args.max_queries,
        progress_every=args.progress_every,
        quiet_progress=args.quiet_progress,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        write_checkpoints=not args.no_checkpoint,
        score_normalization=score_normalization,
    )
    global_alpha, global_dev_mrr = best_alpha(dev["alpha_rr"])
    head_alpha, head_dev_mrr = best_alpha(dev["alpha_direction_rr"]["head"])
    tail_alpha, tail_dev_mrr = best_alpha(dev["alpha_direction_rr"]["tail"])
    relation_alpha, relation_summary = select_relation_alphas(
        dev["alpha_relation_rr"],
        fallback_alpha=global_alpha,
        min_support=args.relation_min_support,
        shrinkage_lambda=args.relation_shrinkage_lambda,
    )

    if args.selection_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        selection_payload = {
            "dataset": dataset_name,
            "protocol_version": protocol_version,
            "selection_split": args.selection_split,
            "score_normalization": score_normalization,
            "target_score_semantics": (
                "canonical_separate_target_score"
                if protocol_version == MMKG_GENERAL_V1
                else "legacy_full_matrix_target"
            ),
            "fusion_endpoint_dev_mrr": float(np.mean(dev["gate_endpoint_rr"])),
            "structural_endpoint_dev_mrr": float(np.mean(dev["residual_endpoint_rr"])),
            "global_alpha": global_alpha,
            "global_dev_mrr": global_dev_mrr,
            "head_alpha": head_alpha,
            "head_dev_mrr": head_dev_mrr,
            "tail_alpha": tail_alpha,
            "tail_dev_mrr": tail_dev_mrr,
            "relation": relation_summary,
            "alpha_grid": alphas,
        }
        (output_dir / "score_ensemble_validation_selection.json").write_text(
            json.dumps(selection_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        write_alpha_curve_outputs(
            output_dir,
            output_dir / "figures",
            alpha_curve_rows(args.selection_split, dev),
        )
        print(f"[OK] wrote {output_dir / 'score_ensemble_validation_selection.json'}")
        print("[INFO] selection-only mode: no test artifacts were loaded or evaluated")
        return

    query_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    query_model.fit(dev["features"], dev["labels"])

    print(
        "[INFO] selected policies: "
        f"global alpha={global_alpha:.2f} dev_mrr={global_dev_mrr:.4f}; "
        f"head alpha={head_alpha:.2f} dev_mrr={head_dev_mrr:.4f}; "
        f"tail alpha={tail_alpha:.2f} dev_mrr={tail_dev_mrr:.4f}; "
        f"relation-specific={relation_summary['n_relation_specific']} fallback={relation_summary['n_fallback']}"
    )
    print("[INFO] evaluating test split")
    test = evaluate_split(
        run_pairs=test_pairs,
        split=args.split,
        alphas=alphas,
        device_arg=args.device,
        chunk_size_arg=args.chunk_size,
        query_batch_size_arg=args.query_batch_size,
        max_queries=args.max_queries,
        query_model=query_model,
        selected_global_alpha=global_alpha,
        selected_direction_alpha={"head": head_alpha, "tail": tail_alpha},
        selected_relation_alpha=relation_alpha,
        selected_relation_fallback_alpha=global_alpha,
        progress_every=args.progress_every,
        quiet_progress=args.quiet_progress,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        write_checkpoints=not args.no_checkpoint,
        score_normalization=score_normalization,
    )
    if refs is None:
        selected_rows = test["selected_policy_rows"]
        if not selected_rows:
            raise RuntimeError("General score analysis produced no selected query rows.")
        refs = {
            "residual": float(np.mean([float(row["rr_residual"]) for row in selected_rows])),
        }

    curve_rows = alpha_curve_rows(args.selection_split, dev) + alpha_curve_rows(args.split, test)

    rows = [
        result_row(
            "Global score interpolation",
            "global",
            f"alpha={global_alpha:.2f}",
            rows_to_metrics(test["global_rows"]),
            refs,
            f"alpha selected by dev MRR ({global_dev_mrr:.4f}); normalization={score_normalization}",
        ),
        result_row(
            "Direction-specific score interpolation",
            "direction",
            f"alpha_head={head_alpha:.2f}; alpha_tail={tail_alpha:.2f}",
            rows_to_metrics(test["direction_rows"]),
            refs,
            f"head/tail alphas selected independently on dev MRR ({head_dev_mrr:.4f}/{tail_dev_mrr:.4f}); normalization={score_normalization}",
        ),
        result_row(
            "Relation-specific score interpolation",
            "relation",
            f"per-relation alpha; fallback alpha={global_alpha:.2f}; min_support={args.relation_min_support}; shrinkage_lambda={args.relation_shrinkage_lambda:g}",
            rows_to_metrics(test["relation_rows"]),
            refs,
            f"relation alphas selected on dev MRR; {relation_summary['n_relation_specific']} relations selected, {relation_summary['n_fallback']} used fallback; normalization={score_normalization}",
        ),
        result_row(
            "Query-level soft score weighting",
            "query",
            "logistic p(Gate beats Residual) from score-distribution features",
            rows_to_metrics(test["query_rows"]),
            refs,
            "query-level soft alpha uses non-answer-aware score-distribution features; labels are dev-only expert wins",
        ),
    ]

    write_csv_rows(output_dir / "score_ensemble_baselines.csv", rows)
    if test["selected_policy_rows"]:
        write_csv_rows(output_dir / "score_ensemble_selected_query_rows.csv", test["selected_policy_rows"])
    (output_dir / "score_ensemble_baselines.json").write_text(
        json.dumps(
            {
                "selection": {
                    "score_normalization": score_normalization,
                    "global_alpha": global_alpha,
                    "global_dev_mrr": global_dev_mrr,
                    "head_alpha": head_alpha,
                    "head_dev_mrr": head_dev_mrr,
                    "tail_alpha": tail_alpha,
                    "tail_dev_mrr": tail_dev_mrr,
                    "relation": relation_summary,
                    "alpha_grid": alphas,
                },
                "alpha_curves": curve_rows,
                "reference_metrics": refs,
                "dataset": dataset_name,
                "protocol_version": protocol_version,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_frame = pd.DataFrame(rows)
    for col in [
        "mrr",
        "hits1",
        "hits3",
        "hits10",
        "delta_vs_residual",
        "delta_vs_e5",
        "delta_vs_ca_s2",
    ]:
        if col not in md_frame:
            continue
        md_frame[col] = md_frame[col].map(lambda value: f"{float(value):.4f}")
    (output_dir / "score_ensemble_baselines.md").write_text(markdown_table(md_frame) + "\n", encoding="utf-8")
    if protocol_version == OPENBG_LEGACY_V1:
        paper_table_path = Path(args.paper_table_dir) / "table_score_ensemble_baselines.tex"
        write_latex_table(paper_table_path, rows, refs)
        paper_figures_path = Path(args.paper_figures_dir) / "table_score_ensemble_baselines.tex"
        if paper_figures_path != paper_table_path:
            write_latex_table(paper_figures_path, rows, refs)
        alpha_figure_dir = Path(args.paper_figures_dir)
    else:
        paper_table_path = None
        paper_figures_path = None
        alpha_figure_dir = output_dir / "figures"
    write_alpha_curve_outputs(output_dir, alpha_figure_dir, curve_rows)
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.csv'}")
    if test["selected_policy_rows"]:
        print(f"[OK] wrote {output_dir / 'score_ensemble_selected_query_rows.csv'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.json'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.md'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_alpha_curve.csv'}")
    if paper_table_path is not None:
        print(f"[OK] wrote {paper_table_path}")
    if paper_figures_path is not None and paper_figures_path != paper_table_path:
        print(f"[OK] wrote {paper_figures_path}")


if __name__ == "__main__":
    main()
