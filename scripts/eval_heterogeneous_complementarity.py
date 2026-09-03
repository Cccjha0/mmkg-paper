from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.dataset_loader import load_dataset_bundle
from ml.training.src.eval.filtered_ranking import (
    prepare_true_heads_index,
    prepare_true_tails_index,
)
from ml.training.src.models.build_model import build_model
from ml.training.src.utils.seed import set_seed
from router.query_geometry import QUERY_GEOMETRY_FIELDS, query_geometry_rows


DEFAULT_ALPHAS = tuple(round(index * 0.05, 2) for index in range(21))
BASE_FIELDS = (
    "pair_name",
    "dataset",
    "protocol_version",
    "expert_a_name",
    "expert_b_name",
    "query_key",
    "query_id",
    "split",
    "seed",
    "direction",
    "relation_id",
    "head_id",
    "tail_id",
    "target_entity_id",
    "rank_a",
    "rr_a",
    "rank_b",
    "rr_b",
    "rr_oracle",
    "rank_rrf",
    "rr_rrf",
    "rank_equal",
    "rr_equal",
    "rrf_k",
    *QUERY_GEOMETRY_FIELDS,
)


@dataclass
class LoadedExpert:
    name: str
    run_dir: Path
    cfg: dict
    bundle: object
    model: object
    num_entities: int
    seed: int
    chunk_size: int
    query_batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact full-ranking complementarity for any paired MMKGC checkpoints. "
            "DEV selects low-capacity policies; TEST requires the locked DEV selection."
        )
    )
    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--expert-a-name", required=True)
    parser.add_argument("--expert-b-name", required=True)
    parser.add_argument(
        "--run-pair",
        action="append",
        required=True,
        metavar="A_RUN::B_RUN",
        help="Paired same-seed checkpoint directories; repeat for every seed.",
    )
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:.2f}" for value in DEFAULT_ALPHAS),
        help="DEV-only alpha grid. Alpha is the weight on expert A.",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--relation-min-support",
        type=int,
        default=60,
        help="Pooled DEV seed-query observations required for a relation alpha.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-reference-check", action="store_true")
    parser.add_argument(
        "--export-alpha-grid",
        action="store_true",
        help=(
            "Also export exact reciprocal ranks for every locked alpha on TEST. "
            "Required when a DEV-locked per-query policy will be applied afterwards."
        ),
    )
    return parser.parse_args()


def parse_alphas(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("alpha grid is empty")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("all alpha values must lie in [0, 1]")
    values = tuple(sorted(set(round(value, 10) for value in values)))
    if any(value != round(value, 2) for value in values):
        raise ValueError("alpha grid values must use at most two decimal places")
    for required in (0.0, 0.5, 1.0):
        if required not in values:
            raise ValueError(f"alpha grid must contain {required:g}")
    return values


def parse_run_pairs(values: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for value in values:
        if "::" not in value:
            raise ValueError(f"Invalid --run-pair {value!r}; expected A_RUN::B_RUN")
        left, right = value.split("::", 1)
        if not left.strip() or not right.strip():
            raise ValueError(f"Invalid --run-pair {value!r}")
        pairs.append((Path(left.strip()), Path(right.strip())))
    return pairs


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError("device must be cuda, cpu, or auto")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_expert(name: str, run_dir: Path, device: str) -> LoadedExpert:
    run_dir = run_dir.resolve()
    cfg_path = run_dir / "config_merged.json"
    ckpt_path = run_dir / "best.ckpt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config/checkpoint under {run_dir}")
    cfg = read_json(cfg_path)
    seed = int(cfg.get("system", {}).get("seed", 1))
    set_seed(seed, deterministic=bool(cfg.get("system", {}).get("deterministic", False)))
    cfg.setdefault("system", {})["device"] = device
    bundle = load_dataset_bundle(cfg)
    model, num_entities = build_model(cfg, dataset_bundle=bundle)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    prepare_eval_cache = getattr(model, "prepare_eval_cache", None)
    if prepare_eval_cache is not None:
        prepare_eval_cache()
    evaluation = cfg.get("evaluation", {})
    return LoadedExpert(
        name=name,
        run_dir=run_dir,
        cfg=cfg,
        bundle=bundle,
        model=model,
        num_entities=int(num_entities),
        seed=seed,
        chunk_size=int(evaluation.get("chunk_size", 4096)),
        query_batch_size=int(evaluation.get("query_batch_size", 8)),
    )


def validate_pair(left: LoadedExpert, right: LoadedExpert) -> None:
    if left.seed != right.seed:
        raise RuntimeError(f"Seed mismatch: {left.name}={left.seed}, {right.name}={right.seed}")
    if left.num_entities != right.num_entities:
        raise RuntimeError("Expert entity counts differ")
    if left.bundle.name != right.bundle.name:
        raise RuntimeError(f"Dataset mismatch: {left.bundle.name} vs {right.bundle.name}")
    if left.bundle.protocol_version != right.bundle.protocol_version:
        raise RuntimeError("Protocol versions differ")
    if left.bundle.entity2id != right.bundle.entity2id:
        raise RuntimeError("Expert entity mappings differ")
    if left.bundle.relation2id != right.bundle.relation2id:
        raise RuntimeError("Expert relation mappings differ")
    for split_name, left_rows, right_rows in (
        ("train", left.bundle.train_triples, right.bundle.train_triples),
        ("dev", left.bundle.valid_triples, right.bundle.valid_triples),
        ("test", left.bundle.test_triples, right.bundle.test_triples),
    ):
        if left_rows != right_rows:
            raise RuntimeError(f"Expert {split_name} triples differ")


def direction_scorer(model, direction: str):
    scorer = getattr(model, "score_head" if direction == "head" else "score_tail", None)
    return scorer if scorer is not None else model.score


def filter_scores_(
    scores: torch.Tensor,
    q_cpu: torch.LongTensor,
    start: int,
    direction: str,
    true_index: dict,
) -> None:
    end = start + scores.size(1)
    row_chunks = []
    col_chunks = []
    for row in range(q_cpu.size(0)):
        if direction == "tail":
            key = (int(q_cpu[row, 0]), int(q_cpu[row, 1]))
            target = int(q_cpu[row, 2])
        else:
            key = (int(q_cpu[row, 1]), int(q_cpu[row, 2]))
            target = int(q_cpu[row, 0])
        excluded = true_index.get(key, torch.empty(0, dtype=torch.long))
        if excluded.numel():
            excluded = excluded[excluded != target]
        left = int(torch.searchsorted(excluded, start, right=False))
        right = int(torch.searchsorted(excluded, end, right=False))
        local = excluded[left:right]
        if local.numel():
            row_chunks.append(torch.full((local.numel(),), row, dtype=torch.long))
            col_chunks.append(local - start)
    if row_chunks:
        rows = torch.cat(row_chunks).to(scores.device)
        columns = torch.cat(col_chunks).to(scores.device)
        scores[rows, columns] = float("-inf")


@torch.inference_mode()
def score_expert_block(
    expert: LoadedExpert,
    q_cpu: torch.LongTensor,
    direction: str,
    true_index: dict,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one outer query block using the expert's original eval batch shapes."""
    scorer = direction_scorer(expert.model, direction)
    matrices = []
    references = []
    all_entities = torch.arange(expert.num_entities, dtype=torch.long, device=device)
    for q_start in range(0, q_cpu.size(0), expert.query_batch_size):
        q_sub_cpu = q_cpu[q_start : q_start + expert.query_batch_size]
        q_sub = q_sub_cpu.to(device)
        references.append(scorer(q_sub).detach().cpu())
        h, r, t = q_sub.unbind(dim=1)
        parts = []
        for start in range(0, expert.num_entities, expert.chunk_size):
            end = min(expert.num_entities, start + expert.chunk_size)
            candidates = all_entities[start:end]
            width = candidates.numel()
            if direction == "tail":
                batch = torch.stack(
                    [
                        h.unsqueeze(1).expand(-1, width).reshape(-1),
                        r.unsqueeze(1).expand(-1, width).reshape(-1),
                        candidates.unsqueeze(0).expand(q_sub.size(0), -1).reshape(-1),
                    ],
                    dim=1,
                )
            else:
                batch = torch.stack(
                    [
                        candidates.unsqueeze(0).expand(q_sub.size(0), -1).reshape(-1),
                        r.unsqueeze(1).expand(-1, width).reshape(-1),
                        t.unsqueeze(1).expand(-1, width).reshape(-1),
                    ],
                    dim=1,
                )
            scores = scorer(batch).view(q_sub.size(0), width)
            filter_scores_(scores, q_sub_cpu, start, direction, true_index)
            parts.append(scores.detach().cpu())
        matrices.append(torch.cat(parts, dim=1))
    return torch.cat(matrices, dim=0), torch.cat(references, dim=0)


def ranks_against_reference(scores: torch.Tensor, reference: torch.Tensor) -> torch.LongTensor:
    return (scores > reference.reshape(-1, 1)).sum(dim=1).to(torch.long) + 1


def reciprocal(ranks: torch.Tensor) -> torch.Tensor:
    return ranks.to(torch.float64).reciprocal()


def query_zscore_with_reference(
    scores: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(scores)
    values = torch.where(finite, scores, torch.zeros_like(scores))
    count = finite.sum(dim=1, keepdim=True).clamp_min(1)
    mean = values.sum(dim=1, keepdim=True) / count
    centered = torch.where(finite, scores - mean, torch.zeros_like(scores))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    scale = variance.sqrt() + float(eps)
    normalized = centered / scale
    normalized = normalized.masked_fill(~finite, float("-inf"))
    normalized_reference = (reference.reshape(-1, 1) - mean) / scale
    return normalized, normalized_reference.reshape(-1)


def competition_rank_scores(scores: torch.Tensor, k: float) -> torch.Tensor:
    finite = torch.isfinite(scores)
    safe = scores.masked_fill(~finite, float("-inf"))
    order = torch.argsort(safe, dim=1, descending=True, stable=True)
    sorted_scores = safe.gather(1, order)
    ordinal = torch.arange(1, scores.size(1) + 1, dtype=torch.long).expand_as(order)
    starts = torch.ones_like(sorted_scores, dtype=torch.bool)
    starts[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    sorted_ranks = torch.where(starts, ordinal, torch.zeros_like(ordinal)).cummax(dim=1).values
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, sorted_ranks)
    fused = 1.0 / (float(k) + ranks.to(torch.float64))
    return fused.masked_fill(~finite, float("-inf"))


def mixed_ranks(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.LongTensor:
    alpha_tensor = torch.as_tensor(alpha, dtype=scores_a.dtype).reshape(-1)
    if alpha_tensor.numel() == 1:
        alpha_tensor = alpha_tensor.expand(scores_a.size(0))
    if alpha_tensor.numel() != scores_a.size(0):
        raise ValueError("alpha must be scalar or contain one value per query")
    mixed = alpha_tensor.unsqueeze(1) * scores_a + (1.0 - alpha_tensor.unsqueeze(1)) * scores_b
    mixed_reference = alpha_tensor * reference_a + (1.0 - alpha_tensor) * reference_b
    both_filtered = (~torch.isfinite(scores_a)) & (~torch.isfinite(scores_b))
    mixed = mixed.masked_fill(both_filtered, float("-inf"))
    return ranks_against_reference(mixed, mixed_reference)


def endpoint_safe_mixed_ranks(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    alpha: float | torch.Tensor,
    rank_a: torch.Tensor,
    rank_b: torch.Tensor,
) -> torch.LongTensor:
    """Interpolate normalized scores while preserving the exact fixed endpoints."""
    alpha_tensor = torch.as_tensor(alpha, dtype=scores_a.dtype).reshape(-1)
    if alpha_tensor.numel() == 1:
        alpha_tensor = alpha_tensor.expand(scores_a.size(0))
    ranks = mixed_ranks(scores_a, scores_b, reference_a, reference_b, alpha_tensor)
    ranks = torch.where(alpha_tensor == 1.0, rank_a, ranks)
    return torch.where(alpha_tensor == 0.0, rank_b, ranks)


def rrf_ranks(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    rank_a: torch.Tensor,
    rank_b: torch.Tensor,
    k: float,
) -> torch.LongTensor:
    candidate = competition_rank_scores(raw_a, k) + competition_rank_scores(raw_b, k)
    reference = 1.0 / (float(k) + rank_a.to(torch.float64)) + 1.0 / (
        float(k) + rank_b.to(torch.float64)
    )
    return ranks_against_reference(candidate, reference)


def alpha_column(alpha: float) -> str:
    return f"rr_alpha_{alpha:.2f}".replace(".", "_")


def query_identifiers(
    split: str,
    seed: int,
    direction: str,
    h: int,
    r: int,
    t: int,
) -> tuple[str, str, int]:
    target = h if direction == "head" else t
    stable = f"{direction}|r={r}|h={h}|t={t}|target={target}"
    return stable, f"{split}|{seed}|{stable}", target


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty checkpoint: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_is_valid(
    rows: list[dict],
    *,
    split: str,
    seed: int,
    direction: str,
    expected_count: int,
    dev: bool,
    export_alpha_grid: bool,
    alphas: tuple[float, ...],
    rrf_k: float,
    selection: dict | None,
    pair_name: str,
    dataset: str,
    protocol_version: str,
    expert_a_name: str,
    expert_b_name: str,
) -> bool:
    if len(rows) != expected_count or not rows:
        return False
    required = set(BASE_FIELDS)
    required.update(alpha_column(alpha) for alpha in alphas if dev or export_alpha_grid)
    if not required.issubset(rows[0]):
        return False
    if not all(
        row.get("split") == split
        and int(row.get("seed", -1)) == seed
        and row.get("direction") == direction
        and row.get("pair_name") == pair_name
        and row.get("dataset") == dataset
        and row.get("protocol_version") == protocol_version
        and row.get("expert_a_name") == expert_a_name
        and row.get("expert_b_name") == expert_b_name
        and math.isclose(float(row.get("rrf_k", "nan")), rrf_k, rel_tol=0.0, abs_tol=1e-12)
        and all(math.isfinite(float(row.get(field, "nan"))) for field in QUERY_GEOMETRY_FIELDS)
        for row in rows
    ):
        return False
    if dev:
        return True
    if selection is None:
        return False
    required_test = {"alpha_global", "rank_global", "rr_global", "alpha_relation", "rank_relation", "rr_relation"}
    if not required_test.issubset(rows[0]):
        return False
    global_alpha = float(selection["global_alpha"])
    relation_map = {int(key): float(value) for key, value in selection["relation_alpha"].items()}
    return all(
        math.isclose(float(row["alpha_global"]), global_alpha, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            float(row["alpha_relation"]),
            relation_map.get(int(row["relation_id"]), global_alpha),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in rows
    )


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes:d}m{seconds:02d}s"


def evaluate_unit(
    *,
    expert_a: LoadedExpert,
    expert_b: LoadedExpert,
    triples: list[tuple[int, int, int]],
    split: str,
    direction: str,
    true_index: dict,
    alphas: tuple[float, ...],
    rrf_k: float,
    selection: dict | None,
    export_alpha_grid: bool,
    device: str,
    progress_every: int,
    pair_name: str,
) -> list[dict]:
    triples_t = torch.tensor(triples, dtype=torch.long)
    outer_batch = max(expert_a.query_batch_size, expert_b.query_batch_size)
    rows = []
    total_batches = math.ceil(len(triples) / outer_batch)
    started = time.time()
    for batch_index, start in enumerate(range(0, len(triples), outer_batch), start=1):
        end = min(len(triples), start + outer_batch)
        q_cpu = triples_t[start:end]
        raw_a, target_a = score_expert_block(expert_a, q_cpu, direction, true_index, device)
        raw_b, target_b = score_expert_block(expert_b, q_cpu, direction, true_index, device)
        rank_a = ranks_against_reference(raw_a, target_a)
        rank_b = ranks_against_reference(raw_b, target_b)
        rr_a = reciprocal(rank_a)
        rr_b = reciprocal(rank_b)
        rank_rrf = rrf_ranks(raw_a, raw_b, rank_a, rank_b, rrf_k)
        rr_rrf = reciprocal(rank_rrf)
        z_a, z_target_a = query_zscore_with_reference(raw_a, target_a)
        z_b, z_target_b = query_zscore_with_reference(raw_b, target_b)
        rank_equal = mixed_ranks(z_a, z_b, z_target_a, z_target_b, 0.5)
        rr_equal = reciprocal(rank_equal)
        geometry_rows = query_geometry_rows(raw_a, raw_b, direction)

        alpha_rr: dict[float, torch.Tensor] = {}
        if split == "dev" or export_alpha_grid:
            for alpha in alphas:
                if alpha == 1.0:
                    alpha_rr[alpha] = rr_a
                elif alpha == 0.0:
                    alpha_rr[alpha] = rr_b
                elif alpha == 0.5:
                    alpha_rr[alpha] = rr_equal
                else:
                    alpha_rr[alpha] = reciprocal(
                        endpoint_safe_mixed_ranks(
                            z_a,
                            z_b,
                            z_target_a,
                            z_target_b,
                            alpha,
                            rank_a,
                            rank_b,
                        )
                    )
        if split == "test":
            if selection is None:
                raise RuntimeError("TEST evaluation requires a locked DEV selection")
            global_alpha = float(selection["global_alpha"])
            relation_map = {int(key): float(value) for key, value in selection["relation_alpha"].items()}
            relation_ids = q_cpu[:, 1].tolist()
            relation_alpha = torch.tensor(
                [relation_map.get(int(rel), global_alpha) for rel in relation_ids],
                dtype=z_a.dtype,
            )
            rank_global = endpoint_safe_mixed_ranks(
                z_a, z_b, z_target_a, z_target_b, global_alpha, rank_a, rank_b
            )
            rank_relation = endpoint_safe_mixed_ranks(
                z_a, z_b, z_target_a, z_target_b, relation_alpha, rank_a, rank_b
            )
            rr_global = reciprocal(rank_global)
            rr_relation = reciprocal(rank_relation)

        for index in range(q_cpu.size(0)):
            h, relation, t = (int(value) for value in q_cpu[index].tolist())
            stable_id, query_id, target = query_identifiers(
                split, expert_a.seed, direction, h, relation, t
            )
            row = {
                "pair_name": pair_name,
                "dataset": expert_a.bundle.name,
                "protocol_version": expert_a.bundle.protocol_version,
                "expert_a_name": expert_a.name,
                "expert_b_name": expert_b.name,
                "query_key": stable_id,
                "query_id": query_id,
                "split": split,
                "seed": expert_a.seed,
                "direction": direction,
                "relation_id": relation,
                "head_id": h,
                "tail_id": t,
                "target_entity_id": target,
                "rank_a": int(rank_a[index]),
                "rr_a": float(rr_a[index]),
                "rank_b": int(rank_b[index]),
                "rr_b": float(rr_b[index]),
                "rr_oracle": float(max(rr_a[index], rr_b[index])),
                "rank_rrf": int(rank_rrf[index]),
                "rr_rrf": float(rr_rrf[index]),
                "rank_equal": int(rank_equal[index]),
                "rr_equal": float(rr_equal[index]),
                "rrf_k": float(rrf_k),
            }
            row.update(geometry_rows[index])
            if split == "dev" or export_alpha_grid:
                for alpha in alphas:
                    row[alpha_column(alpha)] = float(alpha_rr[alpha][index])
            if split == "test":
                row.update(
                    {
                        "alpha_global": global_alpha,
                        "rank_global": int(rank_global[index]),
                        "rr_global": float(rr_global[index]),
                        "alpha_relation": float(relation_alpha[index]),
                        "rank_relation": int(rank_relation[index]),
                        "rr_relation": float(rr_relation[index]),
                    }
                )
            rows.append(row)
        if progress_every > 0 and (batch_index % progress_every == 0 or end == len(triples)):
            elapsed = time.time() - started
            fraction = batch_index / total_batches
            eta = elapsed * (1.0 / fraction - 1.0)
            print(
                f"[PROGRESS] seed={expert_a.seed} direction={direction} "
                f"{batch_index}/{total_batches} batches queries={end}/{len(triples)} "
                f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}",
                flush=True,
            )
    return rows


def best_alpha(rows: list[dict], alphas: tuple[float, ...]) -> tuple[float, float]:
    scored = []
    for alpha in alphas:
        values = [float(row[alpha_column(alpha)]) for row in rows]
        scored.append((sum(values) / len(values), -abs(alpha - 0.5), -alpha, alpha))
    best = max(scored)
    return float(best[3]), float(best[0])


def select_policies(
    rows: list[dict],
    alphas: tuple[float, ...],
    relation_min_support: int,
) -> dict:
    global_alpha, global_mrr = best_alpha(rows, alphas)
    by_relation: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_relation[int(row["relation_id"])].append(row)
    relation_alpha = {}
    relation_details = {}
    for relation_id, relation_rows in sorted(by_relation.items()):
        if len(relation_rows) < relation_min_support:
            alpha = global_alpha
            source = "global_fallback"
        else:
            alpha, _ = best_alpha(relation_rows, alphas)
            source = "relation_dev"
        relation_alpha[str(relation_id)] = alpha
        relation_details[str(relation_id)] = {
            "support": len(relation_rows),
            "alpha": alpha,
            "source": source,
        }
    for row in rows:
        row["alpha_global"] = global_alpha
        row["rr_global"] = float(row[alpha_column(global_alpha)])
        alpha = float(relation_alpha[str(int(row["relation_id"]))])
        row["alpha_relation"] = alpha
        row["rr_relation"] = float(row[alpha_column(alpha)])
    return {
        "global_alpha": global_alpha,
        "global_dev_mrr": global_mrr,
        "relation_min_support": relation_min_support,
        "relation_alpha": relation_alpha,
        "relation_details": relation_details,
    }


def metric(values: list[float]) -> dict:
    ranks = [int(round(1.0 / value)) for value in values]
    return {
        "count": len(values),
        "mrr": sum(values) / len(values),
        "hits@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hits@3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "hits@10": sum(rank <= 10 for rank in ranks) / len(ranks),
    }


def method_columns(split: str, rrf_k: float) -> list[tuple[str, str, str]]:
    label = "DEV-selected" if split == "dev" else "DEV-locked"
    return [
        ("Expert A", "rr_a", "fixed expert"),
        ("Expert B", "rr_b", "fixed expert"),
        ("Equal RRF", "rr_rrf", f"fixed k={rrf_k:g}; answer-agnostic ranks"),
        ("Query-zscore 0.5", "rr_equal", "fixed equal weighting"),
        ("Global alpha", "rr_global", f"{label} shared alpha"),
        ("Relation alpha", "rr_relation", f"{label}; low-support fallback to global"),
        ("Oracle", "rr_oracle", "answer-aware upper bound"),
    ]


def summarize(
    rows: list[dict],
    *,
    split: str,
    expert_a_name: str,
    expert_b_name: str,
    rrf_k: float,
) -> tuple[list[dict], list[dict]]:
    methods = method_columns(split, rrf_k)
    overall = {}
    for method, column, notes in methods:
        values = [float(row[column]) for row in rows]
        overall[method] = {"method": method, **metric(values), "notes": notes}
    anchor = overall["Expert A"]["mrr"]
    oracle = overall["Oracle"]["mrr"]
    denominator = oracle - anchor
    overall_rows = []
    for method, _, _ in methods:
        item = dict(overall[method])
        item["delta_vs_a"] = item["mrr"] - anchor
        item["oracle_gap_recovery"] = (
            (item["mrr"] - anchor) / denominator if denominator > 0 else 0.0
        )
        item["method"] = item["method"].replace("Expert A", expert_a_name).replace(
            "Expert B", expert_b_name
        )
        overall_rows.append(item)

    by_seed = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for method, column, _ in methods:
            label = method.replace("Expert A", expert_a_name).replace("Expert B", expert_b_name)
            by_seed.append({"seed": seed, "method": label, **metric([float(row[column]) for row in seed_rows])})
    return overall_rows, by_seed


def stability_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        difference = float(row["rr_b"]) - float(row["rr_a"])
        grouped[row["query_key"]].append(1 if difference > 0 else -1 if difference < 0 else 0)
    seed_counts = sorted({len(labels) for labels in grouped.values()})
    pair_agreements = []
    unanimous = 0
    b_all = 0
    for labels in grouped.values():
        unanimous += len(set(labels)) == 1
        b_all += all(label == 1 for label in labels)
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                pair_agreements.append(int(labels[left] == labels[right]))
    return {
        "n_seed_stripped_queries": len(grouped),
        "observations_per_query": seed_counts,
        "unanimous_winner_label_rate": unanimous / len(grouped),
        "expert_b_wins_all_seeds_rate": b_all / len(grouped),
        "pairwise_seed_label_agreement": (
            sum(pair_agreements) / len(pair_agreements) if pair_agreements else 1.0
        ),
        "label_definition": "sign(RR_B - RR_A), with ties as a third label",
    }


def reference_mrr(run_dir: Path, split: str) -> float | None:
    if split == "test":
        path = run_dir / "test_metrics.json"
        return float(read_json(path)["mrr"]) if path.exists() else None
    metric_paths = sorted(run_dir.glob("metrics_seed*.csv"))
    if not metric_paths:
        return None
    values = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("mrr", "")).strip():
                    values.append(float(row["mrr"]))
    return max(values) if values else None


def validate_endpoint_reproduction(
    rows: list[dict],
    run_pairs: list[tuple[Path, Path]],
    split: str,
    tolerance: float = 5e-7,
) -> list[dict]:
    audit = []
    for left_path, right_path in run_pairs:
        left_cfg = read_json(left_path / "config_merged.json")
        seed = int(left_cfg.get("system", {}).get("seed", 1))
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for expert, path, column in (("A", left_path, "rr_a"), ("B", right_path, "rr_b")):
            expected = reference_mrr(path, split)
            actual = sum(float(row[column]) for row in seed_rows) / len(seed_rows)
            delta = None if expected is None else actual - expected
            if expected is not None and abs(delta) > tolerance:
                raise RuntimeError(
                    f"Expert {expert} seed={seed} endpoint mismatch: "
                    f"export={actual:.12f}, reference={expected:.12f}, delta={delta:.3e}"
                )
            audit.append(
                {
                    "expert": expert,
                    "seed": seed,
                    "run_dir": str(path),
                    "actual_mrr": actual,
                    "reference_mrr": expected,
                    "delta": delta,
                }
            )
    return audit


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], split: str) -> None:
    lines = [
        f"| Method | {split.upper()} MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs. A | Oracle gap recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mrr']:.6f} | {row['hits@1']:.6f} | "
            f"{row['hits@3']:.6f} | {row['hits@10']:.6f} | {row['delta_vs_a']:+.6f} | "
            f"{100.0 * row['oracle_gap_recovery']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.rrf_k <= 0 or args.relation_min_support < 1:
        raise ValueError("rrf-k and relation-min-support must be positive")
    device = resolve_device(args.device)
    alphas = parse_alphas(args.alphas)
    run_pairs = parse_run_pairs(args.run_pair)
    out_dir = Path(args.output_dir)
    checkpoint_dir = out_dir / "checkpoints"
    selection = None
    if args.split == "test":
        if not args.selection_json:
            raise ValueError("--selection-json is required for TEST")
        selection = read_json(Path(args.selection_json))
        if selection.get("pair_name") != args.pair_name:
            raise RuntimeError("Selection pair_name does not match this TEST run")
        if selection.get("expert_a_name") != args.expert_a_name or selection.get("expert_b_name") != args.expert_b_name:
            raise RuntimeError("Selection expert names do not match this TEST run")
        if not math.isclose(float(selection.get("rrf_k", float("nan"))), args.rrf_k, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("TEST rrf-k does not match the locked DEV selection")
        alphas = tuple(float(value) for value in selection["alpha_grid"])
    elif args.selection_json:
        raise ValueError("--selection-json is TEST-only")

    all_rows = []
    seen_seeds = set()
    dataset_name = None
    protocol_version = None
    for left_path, right_path in run_pairs:
        expert_a = load_expert(args.expert_a_name, left_path, device)
        expert_b = load_expert(args.expert_b_name, right_path, device)
        validate_pair(expert_a, expert_b)
        if expert_a.seed in seen_seeds:
            raise RuntimeError(f"Duplicate seed in run pairs: {expert_a.seed}")
        seen_seeds.add(expert_a.seed)
        current_dataset = expert_a.bundle.name
        current_protocol = expert_a.bundle.protocol_version
        if dataset_name is not None and (current_dataset, current_protocol) != (
            dataset_name,
            protocol_version,
        ):
            raise RuntimeError("Run pairs mix datasets or protocols")
        dataset_name, protocol_version = current_dataset, current_protocol
        if selection is not None:
            if selection.get("dataset") != dataset_name or selection.get("protocol_version") != protocol_version:
                raise RuntimeError("Locked DEV selection dataset/protocol does not match TEST checkpoints")
        triples = (
            expert_a.bundle.valid_triples
            if args.split == "dev"
            else expert_a.bundle.test_triples
        )
        if not triples:
            raise RuntimeError(f"No labeled triples for split={args.split}")
        true_tails, true_heads = build_true_facts(
            expert_a.bundle.train_triples
            + expert_a.bundle.valid_triples
            + expert_a.bundle.test_triples
        )
        true_indexes = {
            "tail": prepare_true_tails_index(true_tails),
            "head": prepare_true_heads_index(true_heads),
        }
        for direction in ("head", "tail"):
            checkpoint = checkpoint_dir / f"{args.split}_seed{expert_a.seed}_{direction}.csv"
            cached = read_rows(checkpoint) if checkpoint.exists() and not args.no_resume else []
            if checkpoint_is_valid(
                cached,
                split=args.split,
                seed=expert_a.seed,
                direction=direction,
                expected_count=len(triples),
                dev=args.split == "dev",
                export_alpha_grid=args.export_alpha_grid,
                alphas=alphas,
                rrf_k=args.rrf_k,
                selection=selection,
                pair_name=args.pair_name,
                dataset=current_dataset,
                protocol_version=current_protocol,
                expert_a_name=args.expert_a_name,
                expert_b_name=args.expert_b_name,
            ):
                print(f"[RESUME] {checkpoint}")
                unit_rows = cached
            else:
                print(
                    f"[START] {args.pair_name} split={args.split} seed={expert_a.seed} "
                    f"direction={direction} A={expert_a.chunk_size}/{expert_a.query_batch_size} "
                    f"B={expert_b.chunk_size}/{expert_b.query_batch_size}",
                    flush=True,
                )
                unit_rows = evaluate_unit(
                    expert_a=expert_a,
                    expert_b=expert_b,
                    triples=triples,
                    split=args.split,
                    direction=direction,
                    true_index=true_indexes[direction],
                    alphas=alphas,
                    rrf_k=args.rrf_k,
                    selection=selection,
                    export_alpha_grid=args.export_alpha_grid,
                    device=device,
                    progress_every=args.progress_every,
                    pair_name=args.pair_name,
                )
                write_rows(checkpoint, unit_rows)
                print(f"[CHECKPOINT] {checkpoint}")
            all_rows.extend(unit_rows)
        del expert_a.model
        del expert_b.model
        if device == "cuda":
            torch.cuda.empty_cache()

    all_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["direction"]),
            int(row["relation_id"]),
            int(row["head_id"]),
            int(row["tail_id"]),
        )
    )
    if selection is not None and sorted(seen_seeds) != sorted(
        int(value) for value in selection.get("seeds", [])
    ):
        raise RuntimeError("Locked DEV selection seeds do not match TEST checkpoint pairs")
    if args.split == "dev":
        selected = select_policies(all_rows, alphas, args.relation_min_support)
        selection = {
            "schema_version": 1,
            "pair_name": args.pair_name,
            "dataset": dataset_name,
            "protocol_version": protocol_version,
            "expert_a_name": args.expert_a_name,
            "expert_b_name": args.expert_b_name,
            "seeds": sorted(seen_seeds),
            "score_normalization": "query_zscore",
            "alpha_grid": list(alphas),
            "rrf_k": args.rrf_k,
            **selected,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        for row in all_rows:
            if "rr_global" not in row or "rr_relation" not in row:
                raise RuntimeError("TEST checkpoint is missing locked-policy columns")

    endpoint_audit = []
    if not args.no_reference_check:
        endpoint_audit = validate_endpoint_reproduction(all_rows, run_pairs, args.split)
    overall, by_seed = summarize(
        all_rows,
        split=args.split,
        expert_a_name=args.expert_a_name,
        expert_b_name=args.expert_b_name,
        rrf_k=args.rrf_k,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{args.split}_query_rows.csv", all_rows)
    write_csv(out_dir / f"{args.split}_results.csv", overall)
    write_csv(out_dir / f"{args.split}_results_by_seed.csv", by_seed)
    write_markdown(out_dir / f"{args.split}_results.md", overall, args.split)
    summary = {
        "schema_version": 1,
        "pair_name": args.pair_name,
        "dataset": dataset_name,
        "protocol_version": protocol_version,
        "split": args.split,
        "expert_a_name": args.expert_a_name,
        "expert_b_name": args.expert_b_name,
        "seeds": sorted(seen_seeds),
        "n_rows": len(all_rows),
        "rrf_k": args.rrf_k,
        "score_normalization": "query_zscore",
        "export_alpha_grid": bool(args.split == "dev" or args.export_alpha_grid),
        "query_geometry_fields": list(QUERY_GEOMETRY_FIELDS),
        "selection": selection,
        "results": overall,
        "stability": stability_summary(all_rows),
        "endpoint_reproduction": endpoint_audit,
        "information_boundaries": {
            "fixed_experts": "answer-agnostic",
            "rrf": "rank-aware, answer-agnostic, fixed",
            "equal": "score-aware, answer-agnostic, fixed",
            "global_alpha": "score-aware, answer-agnostic; selected on DEV",
            "relation_alpha": "score-aware, answer-agnostic; relation-conditioned and selected on DEV",
            "query_geometry": (
                "answer-agnostic candidate-score statistics; excludes target ids, "
                "target/reference scores, ranks, reciprocal ranks, and raw relation ids"
            ),
            "oracle": "answer-aware upper bound",
        },
    }
    (out_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_dir / f'{args.split}_results.md'}")
    if args.split == "dev":
        print(
            f"[LOCKED] global alpha={selection['global_alpha']:.2f}; "
            f"relation-specific={sum(v['source'] == 'relation_dev' for v in selection['relation_details'].values())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
