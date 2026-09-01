from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.eval.filtered_ranking import (
    prepare_true_heads_index,
    prepare_true_tails_index,
)
from scripts.eval_heterogeneous_complementarity import (
    endpoint_safe_mixed_ranks,
    load_expert,
    metric,
    parse_run_pairs,
    query_identifiers,
    query_zscore_with_reference,
    ranks_against_reference,
    reciprocal,
    resolve_device,
    score_expert_block,
    validate_pair,
)


SOURCE_REPOSITORY = "dair-iitd/KGC-Ensemble"
SOURCE_COMMIT = "48d66b915f64899798f736129fa8c4d0a40fdb78"
SOURCE_SELECTOR = "external/KGC-Ensemble/NBFNet/script/selector.py"
SOURCE_TRAINER = "external/KGC-Ensemble/NBFNet/script/train_selector.py"

# Frozen from the ACL 2024 paper and its released two-model configuration.
HIDDEN_DIM = 16
LEARNING_RATE = 5.0e-5
MARGIN = 2.0
NEGATIVE_COUNT = 9999
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 1
INIT_LOW = 0.0
INIT_HIGH = 2.0
FEATURE_DEFINITION = "released_code_one_minus_mean_plus_unbiased_variance"
NORMALIZATION = "filtered_query_minmax"
LEARNED_EXPERT = "M-Hyper"
FIXED_WEIGHT_EXPERT = "NativE"
FIXED_WEIGHT = 1.0

ROW_FIELDS = (
    "pair_name",
    "dataset",
    "protocol_version",
    "split",
    "seed",
    "direction",
    "query_key",
    "query_id",
    "head_id",
    "relation_id",
    "tail_id",
    "target_entity_id",
    "rank_a",
    "rr_a",
    "rank_b",
    "rr_b",
    "rank_equal",
    "rr_equal",
    "alpha_global",
    "rank_global",
    "rr_global",
    "rank_dynasemble",
    "rr_dynasemble",
    "rr_oracle",
    "weight_mhyper",
    "effective_alpha_mhyper",
    "feature_mhyper_one_minus_mean",
    "feature_mhyper_variance",
    "feature_native_one_minus_mean",
    "feature_native_variance",
    "selector_sha256",
    "baseline_selection_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faithfully adapt DynaSemble to the frozen OpenBG-IMG M-Hyper + NativE "
            "expert pair under the repository's exact filtered full-ranking protocol."
        )
    )
    parser.add_argument("--stage", required=True, choices=("dev", "test"))
    parser.add_argument(
        "--run-pair",
        action="append",
        required=True,
        metavar="MHYPER_RUN::NATIVE_RUN",
        help="Paired same-seed checkpoints; repeat exactly three times.",
    )
    parser.add_argument("--baseline-selection-json", required=True)
    parser.add_argument("--reference-query-rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_method_config() -> dict:
    return {
        "method": "DynaSemble",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_selector": SOURCE_SELECTOR,
        "source_trainer": SOURCE_TRAINER,
        "learned_weight_expert": LEARNED_EXPERT,
        "fixed_weight_expert": FIXED_WEIGHT_EXPERT,
        "fixed_weight": FIXED_WEIGHT,
        "normalization": NORMALIZATION,
        "features": FEATURE_DEFINITION,
        "mlp_topology": [4, HIDDEN_DIM, HIDDEN_DIM, 1],
        "output_activation": "relu",
        "released_code_initialization": {
            "second_linear_weight": [INIT_LOW, INIT_HIGH],
            "other_parameters": "torch_default",
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "margin_loss": "torch.nn.MultiMarginLoss",
        "margin": MARGIN,
        "strict_negative_count": NEGATIVE_COUNT,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "training_split": "dev_only",
        "training_direction_assignment": (
            "released-code behavior: shuffled triple batch, first half tail and second half head"
        ),
        "evaluation": "exact filtered full-entity ranking, both directions",
    }


class ReleasedDynaSembleSelector(nn.Module):
    """Two-model selector matching the released DynaSemble MLP topology."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
            nn.ReLU(),
        )
        nn.init.uniform_(self.layers[1].weight, INIT_LOW, INIT_HIGH)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def normalize_and_features(
    scores: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Min-max normalize finite candidates and return released-code features."""
    finite = torch.isfinite(scores)
    if not finite.any(dim=1).all():
        raise RuntimeError("A query has no finite candidate scores")
    safe_min = scores.masked_fill(~finite, float("inf")).amin(dim=1, keepdim=True)
    safe_max = scores.masked_fill(~finite, float("-inf")).amax(dim=1, keepdim=True)
    scale = safe_max - safe_min
    if (scale <= 0).any():
        raise RuntimeError("DynaSemble min-max normalization encountered a constant score row")
    normalized = (scores - safe_min) / scale
    normalized = normalized.masked_fill(~finite, float("-inf"))
    values = normalized.masked_fill(~finite, 0.0)
    count = finite.sum(dim=1, keepdim=True)
    mean = values.sum(dim=1, keepdim=True) / count
    centered = (values - mean).masked_fill(~finite, 0.0)
    denominator = (count - 1).clamp_min(1)
    variance = centered.square().sum(dim=1, keepdim=True) / denominator
    features = torch.cat((1.0 - mean, variance), dim=1)
    normalized_reference = None
    if reference is not None:
        normalized_reference = ((reference.reshape(-1, 1) - safe_min) / scale).reshape(-1)
    return normalized, normalized_reference, features


def sample_training_candidates(
    raw_a: torch.Tensor,
    target_a: torch.Tensor,
    raw_b: torch.Tensor,
    target_b: torch.Tensor,
    q_cpu: torch.LongTensor,
    direction: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_a = []
    sampled_b = []
    for row_index in range(q_cpu.size(0)):
        target = int(q_cpu[row_index, 0] if direction == "head" else q_cpu[row_index, 2])
        eligible = torch.isfinite(raw_a[row_index]) & torch.isfinite(raw_b[row_index])
        eligible[target] = False
        candidate_ids = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        if candidate_ids.numel() == 0:
            raise RuntimeError("No strict negative candidates remain for a DEV query")
        sampled_offsets = torch.randint(
            candidate_ids.numel(),
            (NEGATIVE_COUNT,),
            generator=generator,
        )
        negative_ids = candidate_ids[sampled_offsets]
        sampled_a.append(
            torch.cat((target_a[row_index].reshape(1), raw_a[row_index, negative_ids]))
        )
        sampled_b.append(
            torch.cat((target_b[row_index].reshape(1), raw_b[row_index, negative_ids]))
        )
    return torch.stack(sampled_a), torch.stack(sampled_b)


def true_indexes_for_expert(expert) -> dict:
    true_tails, true_heads = build_true_facts(
        expert.bundle.train_triples
        + expert.bundle.valid_triples
        + expert.bundle.test_triples
    )
    return {
        "tail": prepare_true_tails_index(true_tails),
        "head": prepare_true_heads_index(true_heads),
    }


def train_selector(
    expert_a,
    expert_b,
    true_indexes: dict,
    device: str,
    progress_every: int,
) -> tuple[ReleasedDynaSembleSelector, dict]:
    seed = int(expert_a.seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    selector = ReleasedDynaSembleSelector().to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MultiMarginLoss(margin=MARGIN)
    triples = list(expert_a.bundle.valid_triples)
    random.Random(seed).shuffle(triples)
    negative_generator = torch.Generator(device="cpu")
    negative_generator.manual_seed(seed + 104729)
    losses = []
    total_batches = math.ceil(len(triples) / TRAIN_BATCH_SIZE)
    started = time.time()
    selector.train()
    for batch_index, start in enumerate(range(0, len(triples), TRAIN_BATCH_SIZE), start=1):
        batch_triples = triples[start : start + TRAIN_BATCH_SIZE]
        midpoint = len(batch_triples) // 2
        directional = (
            ("tail", batch_triples[:midpoint]),
            ("head", batch_triples[midpoint:]),
        )
        normalized_parts_a = []
        normalized_parts_b = []
        feature_parts = []
        for direction, subset in directional:
            if not subset:
                continue
            q_cpu = torch.tensor(subset, dtype=torch.long)
            raw_a, target_a = score_expert_block(
                expert_a, q_cpu, direction, true_indexes[direction], device
            )
            raw_b, target_b = score_expert_block(
                expert_b, q_cpu, direction, true_indexes[direction], device
            )
            sampled_a, sampled_b = sample_training_candidates(
                raw_a,
                target_a,
                raw_b,
                target_b,
                q_cpu,
                direction,
                negative_generator,
            )
            normalized_a, _, features_a = normalize_and_features(sampled_a)
            normalized_b, _, features_b = normalize_and_features(sampled_b)
            normalized_parts_a.append(normalized_a)
            normalized_parts_b.append(normalized_b)
            feature_parts.append(torch.cat((features_a, features_b), dim=1))
        normalized_a = torch.cat(normalized_parts_a).to(device)
        normalized_b = torch.cat(normalized_parts_b).to(device)
        features = torch.cat(feature_parts).to(device)
        weights = selector(features)
        prediction = weights * normalized_a + FIXED_WEIGHT * normalized_b
        target = torch.zeros(prediction.size(0), dtype=torch.long, device=device)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if progress_every > 0 and (
            batch_index % progress_every == 0 or batch_index == total_batches
        ):
            elapsed = time.time() - started
            fraction = batch_index / total_batches
            eta = elapsed * (1.0 / fraction - 1.0)
            print(
                f"[TRAIN] seed={seed} epoch=1/1 batch={batch_index}/{total_batches} "
                f"loss={losses[-1]:.6f} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return selector, {
        "seed": seed,
        "num_dev_triples": len(triples),
        "num_training_directional_queries": len(triples),
        "num_batches": total_batches,
        "mean_training_loss": sum(losses) / len(losses),
        "final_training_loss": losses[-1],
    }


def save_selector(
    path: Path,
    selector: ReleasedDynaSembleSelector,
    seed: int,
    training_summary: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "seed": seed,
            "method_config": frozen_method_config(),
            "training_summary": training_summary,
            "state_dict": selector.state_dict(),
        },
        path,
    )
    return sha256_file(path)


def load_selector(path: Path, expected_seed: int, device: str) -> ReleasedDynaSembleSelector:
    payload = torch.load(path, map_location=device)
    if int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"Selector seed mismatch in {path}")
    if payload.get("method_config") != frozen_method_config():
        raise RuntimeError(f"Selector method configuration mismatch in {path}")
    selector = ReleasedDynaSembleSelector().to(device)
    selector.load_state_dict(payload["state_dict"])
    selector.eval()
    return selector


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_valid(
    rows: list[dict],
    *,
    split: str,
    seed: int,
    direction: str,
    expected_count: int,
    selector_hash: str,
    selection_hash: str,
) -> bool:
    return bool(rows) and len(rows) == expected_count and set(ROW_FIELDS).issubset(rows[0]) and all(
        row["split"] == split
        and int(row["seed"]) == seed
        and row["direction"] == direction
        and row["selector_sha256"] == selector_hash
        and row["baseline_selection_sha256"] == selection_hash
        for row in rows
    )


@torch.inference_mode()
def evaluate_direction(
    expert_a,
    expert_b,
    selector: ReleasedDynaSembleSelector,
    triples: list[tuple[int, int, int]],
    direction: str,
    true_index: dict,
    global_alpha: float,
    split: str,
    device: str,
    progress_every: int,
    selector_hash: str,
    selection_hash: str,
) -> list[dict]:
    selector.eval()
    triples_t = torch.tensor(triples, dtype=torch.long)
    outer_batch = max(expert_a.query_batch_size, expert_b.query_batch_size)
    rows = []
    total_batches = math.ceil(len(triples) / outer_batch)
    started = time.time()
    for batch_index, start in enumerate(range(0, len(triples), outer_batch), start=1):
        q_cpu = triples_t[start : start + outer_batch]
        raw_a, target_a = score_expert_block(expert_a, q_cpu, direction, true_index, device)
        raw_b, target_b = score_expert_block(expert_b, q_cpu, direction, true_index, device)
        rank_a = ranks_against_reference(raw_a, target_a)
        rank_b = ranks_against_reference(raw_b, target_b)
        z_a, z_target_a = query_zscore_with_reference(raw_a, target_a)
        z_b, z_target_b = query_zscore_with_reference(raw_b, target_b)
        rank_equal = endpoint_safe_mixed_ranks(
            z_a, z_b, z_target_a, z_target_b, 0.5, rank_a, rank_b
        )
        rank_global = endpoint_safe_mixed_ranks(
            z_a, z_b, z_target_a, z_target_b, global_alpha, rank_a, rank_b
        )
        normalized_a, normalized_target_a, features_a = normalize_and_features(raw_a, target_a)
        normalized_b, normalized_target_b, features_b = normalize_and_features(raw_b, target_b)
        features = torch.cat((features_a, features_b), dim=1)
        weights = selector(features.to(device)).reshape(-1).cpu()
        ensemble = weights.unsqueeze(1) * normalized_a + FIXED_WEIGHT * normalized_b
        ensemble_reference = weights * normalized_target_a + FIXED_WEIGHT * normalized_target_b
        both_filtered = (~torch.isfinite(normalized_a)) & (~torch.isfinite(normalized_b))
        ensemble = ensemble.masked_fill(both_filtered, float("-inf"))
        rank_dynasemble = ranks_against_reference(ensemble, ensemble_reference)
        rr_a = reciprocal(rank_a)
        rr_b = reciprocal(rank_b)
        rr_equal = reciprocal(rank_equal)
        rr_global = reciprocal(rank_global)
        rr_dynasemble = reciprocal(rank_dynasemble)
        for index in range(q_cpu.size(0)):
            h, relation, t = (int(value) for value in q_cpu[index].tolist())
            query_key, query_id, target = query_identifiers(
                split, expert_a.seed, direction, h, relation, t
            )
            weight = float(weights[index])
            rows.append(
                {
                    "pair_name": "openbg_mhyper_native_dynasemble",
                    "dataset": expert_a.bundle.name,
                    "protocol_version": expert_a.bundle.protocol_version,
                    "split": split,
                    "seed": expert_a.seed,
                    "direction": direction,
                    "query_key": query_key,
                    "query_id": query_id,
                    "head_id": h,
                    "relation_id": relation,
                    "tail_id": t,
                    "target_entity_id": target,
                    "rank_a": int(rank_a[index]),
                    "rr_a": float(rr_a[index]),
                    "rank_b": int(rank_b[index]),
                    "rr_b": float(rr_b[index]),
                    "rank_equal": int(rank_equal[index]),
                    "rr_equal": float(rr_equal[index]),
                    "alpha_global": global_alpha,
                    "rank_global": int(rank_global[index]),
                    "rr_global": float(rr_global[index]),
                    "rank_dynasemble": int(rank_dynasemble[index]),
                    "rr_dynasemble": float(rr_dynasemble[index]),
                    "rr_oracle": float(max(rr_a[index], rr_b[index])),
                    "weight_mhyper": weight,
                    "effective_alpha_mhyper": weight / (weight + FIXED_WEIGHT),
                    "feature_mhyper_one_minus_mean": float(features_a[index, 0]),
                    "feature_mhyper_variance": float(features_a[index, 1]),
                    "feature_native_one_minus_mean": float(features_b[index, 0]),
                    "feature_native_variance": float(features_b[index, 1]),
                    "selector_sha256": selector_hash,
                    "baseline_selection_sha256": selection_hash,
                }
            )
        if progress_every > 0 and (
            batch_index % progress_every == 0 or batch_index == total_batches
        ):
            elapsed = time.time() - started
            fraction = batch_index / total_batches
            eta = elapsed * (1.0 / fraction - 1.0)
            print(
                f"[EVAL] split={split} seed={expert_a.seed} direction={direction} "
                f"batch={batch_index}/{total_batches} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return rows


def validate_reference(rows: list[dict], reference_path: Path, split: str) -> dict:
    reference_rows = read_rows(reference_path)
    reference = {row["query_id"]: row for row in reference_rows}
    if len(reference) != len(reference_rows) or len(reference) != len(rows):
        raise RuntimeError("Reference query rows do not have a one-to-one query_id mapping")
    fields = ["rank_a", "rank_b", "rank_equal"]
    if split == "test":
        fields.append("rank_global")
    mismatches = defaultdict(int)
    max_rr_error = defaultdict(float)
    for row in rows:
        expected = reference.get(row["query_id"])
        if expected is None:
            raise RuntimeError(f"Reference rows are missing {row['query_id']}")
        for field in fields:
            if int(row[field]) != int(expected[field]):
                mismatches[field] += 1
        for field in ("rr_a", "rr_b", "rr_equal", "rr_oracle"):
            max_rr_error[field] = max(
                max_rr_error[field], abs(float(row[field]) - float(expected[field]))
            )
        if split == "dev":
            alpha_column = f"rr_alpha_{float(row['alpha_global']):.2f}".replace(".", "_")
            if alpha_column not in expected:
                raise RuntimeError(f"DEV reference rows are missing {alpha_column}")
            max_rr_error["rr_global"] = max(
                max_rr_error["rr_global"],
                abs(float(row["rr_global"]) - float(expected[alpha_column])),
            )
        else:
            max_rr_error["rr_global"] = max(
                max_rr_error["rr_global"],
                abs(float(row["rr_global"]) - float(expected["rr_global"])),
            )
    if any(mismatches.values()) or any(error > 5e-7 for error in max_rr_error.values()):
        raise RuntimeError(
            f"Matched baseline reproduction failed: mismatches={dict(mismatches)}, "
            f"max_rr_error={dict(max_rr_error)}"
        )
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "n_rows": len(rows),
        "rank_mismatches": dict(mismatches),
        "max_abs_rr_error": dict(max_rr_error),
    }


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    methods = (
        ("M-Hyper", "rr_a"),
        ("Query-zscore 0.5", "rr_equal"),
        ("DEV-locked Global alpha", "rr_global"),
        ("DynaSemble", "rr_dynasemble"),
        ("Oracle", "rr_oracle"),
    )
    anchor = metric([float(row["rr_a"]) for row in rows])["mrr"]
    oracle = metric([float(row["rr_oracle"]) for row in rows])["mrr"]
    gap = oracle - anchor

    def result(subset: list[dict], method: str, column: str) -> dict:
        output = metric([float(row[column]) for row in subset])
        output.update(
            {
                "method": method,
                "delta_vs_mhyper": output["mrr"] - metric(
                    [float(row["rr_a"]) for row in subset]
                )["mrr"],
            }
        )
        return output

    pooled = []
    for method, column in methods:
        output = result(rows, method, column)
        output["oracle_gap_recovery"] = (output["mrr"] - anchor) / gap if gap > 0 else 0.0
        pooled.append(output)
    by_seed = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        subset = [row for row in rows if int(row["seed"]) == seed]
        for method, column in methods:
            by_seed.append({"seed": seed, **result(subset, method, column)})
    by_direction = []
    for direction in ("head", "tail"):
        subset = [row for row in rows if row["direction"] == direction]
        for method, column in methods:
            by_direction.append({"direction": direction, **result(subset, method, column)})
    return pooled, by_seed, by_direction


def clustered_interval(rows: list[dict], column: str, reference: str) -> dict:
    clusters = defaultdict(list)
    for row in rows:
        key = (int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"]))
        clusters[key].append(float(row[column]) - float(row[reference]))
    values = [sum(items) / len(items) for items in clusters.values()]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    return {
        "n_original_triple_clusters": len(values),
        "mean_delta": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "note": "normal interval over original-triple cluster means; seeds and directions clustered",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], split: str) -> None:
    lines = [
        f"| Method | {split.upper()} MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs. M-Hyper | Oracle gap recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mrr']:.6f} | {row['hits@1']:.6f} | "
            f"{row['hits@3']:.6f} | {row['hits@10']:.6f} | "
            f"{row['delta_vs_mhyper']:+.6f} | "
            f"{100.0 * row['oracle_gap_recovery']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_baseline_selection(selection: dict) -> None:
    expected = {
        "pair_name": "openbg_mhyper_native",
        "dataset": "openbg_img",
        "expert_a_name": "M-Hyper",
        "expert_b_name": "NativE",
        "score_normalization": "query_zscore",
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"Baseline selection mismatch for {key}: {selection.get(key)!r}")
    if sorted(int(seed) for seed in selection.get("seeds", [])) != [1, 2, 3]:
        raise RuntimeError("Baseline selection must contain exactly expert seeds 1, 2, and 3")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_pairs = parse_run_pairs(args.run_pair)
    if len(run_pairs) != 3:
        raise RuntimeError("DynaSemble protocol requires exactly three paired expert seeds")
    output_dir = Path(args.output_dir)
    selector_dir = output_dir / "selectors"
    checkpoint_dir = output_dir / "checkpoints"
    lock_path = output_dir / "dev_lock.json"
    baseline_selection_path = Path(args.baseline_selection_json)
    baseline_selection = json.loads(baseline_selection_path.read_text(encoding="utf-8"))
    validate_baseline_selection(baseline_selection)
    baseline_selection_hash = sha256_file(baseline_selection_path)
    global_alpha = float(baseline_selection["global_alpha"])
    method_config = frozen_method_config()

    if args.stage == "test" and not lock_path.exists():
        raise RuntimeError("TEST is locked until DEV has produced dev_lock.json")
    if args.stage == "dev" and (output_dir / "test_summary.json").exists():
        raise RuntimeError("Refusing to retrain or relock DynaSemble after TEST output exists")
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    if lock is not None:
        if lock.get("method_config") != method_config:
            raise RuntimeError("Frozen DynaSemble configuration differs from dev_lock.json")
        if lock.get("baseline_selection_sha256") != baseline_selection_hash:
            raise RuntimeError("Baseline DEV selection differs from the locked file")

    all_rows = []
    training_summaries = {}
    selector_hashes = {}
    seen_seeds = set()
    run_manifest = []
    for left_path, right_path in run_pairs:
        expert_a = load_expert("M-Hyper", left_path, device)
        expert_b = load_expert("NativE", right_path, device)
        validate_pair(expert_a, expert_b)
        seed = int(expert_a.seed)
        if seed in seen_seeds:
            raise RuntimeError(f"Duplicate expert seed {seed}")
        seen_seeds.add(seed)
        true_indexes = true_indexes_for_expert(expert_a)
        selector_path = selector_dir / f"seed{seed}.pt"
        if args.stage == "dev" and lock is None and not selector_path.exists():
            print(f"[TRAIN START] DynaSemble seed={seed}", flush=True)
            selector, training_summary = train_selector(
                expert_a,
                expert_b,
                true_indexes,
                device,
                args.progress_every,
            )
            selector_hash = save_selector(
                selector_path, selector, seed, training_summary
            )
            print(f"[TRAIN LOCKED] {selector_path} sha256={selector_hash}", flush=True)
        else:
            if not selector_path.exists():
                raise RuntimeError(f"Locked selector is missing: {selector_path}")
            selector = load_selector(selector_path, seed, device)
            selector_hash = sha256_file(selector_path)
            training_summary = torch.load(selector_path, map_location="cpu")["training_summary"]
        if lock is not None:
            expected_hash = lock["selectors"][str(seed)]["sha256"]
            if selector_hash != expected_hash:
                raise RuntimeError(f"Selector hash mismatch for seed {seed}")
        selector_hashes[str(seed)] = selector_hash
        training_summaries[str(seed)] = training_summary
        run_manifest.append(
            {
                "seed": seed,
                "mhyper_run": str(Path(left_path)),
                "native_run": str(Path(right_path)),
            }
        )
        triples = (
            expert_a.bundle.valid_triples
            if args.stage == "dev"
            else expert_a.bundle.test_triples
        )
        for direction in ("head", "tail"):
            checkpoint_path = checkpoint_dir / f"{args.stage}_seed{seed}_{direction}.csv"
            cached = (
                read_rows(checkpoint_path)
                if checkpoint_path.exists() and not args.no_resume
                else []
            )
            if checkpoint_valid(
                cached,
                split=args.stage,
                seed=seed,
                direction=direction,
                expected_count=len(triples),
                selector_hash=selector_hash,
                selection_hash=baseline_selection_hash,
            ):
                print(f"[RESUME] {checkpoint_path}", flush=True)
                rows = cached
            else:
                print(
                    f"[EVAL START] split={args.stage} seed={seed} direction={direction}",
                    flush=True,
                )
                rows = evaluate_direction(
                    expert_a,
                    expert_b,
                    selector,
                    triples,
                    direction,
                    true_indexes[direction],
                    global_alpha,
                    args.stage,
                    device,
                    args.progress_every,
                    selector_hash,
                    baseline_selection_hash,
                )
                write_rows(checkpoint_path, rows)
                print(f"[CHECKPOINT] {checkpoint_path}", flush=True)
            all_rows.extend(rows)
        del selector
        del expert_a.model
        del expert_b.model
        if device == "cuda":
            torch.cuda.empty_cache()

    if sorted(seen_seeds) != [1, 2, 3]:
        raise RuntimeError(f"Expected expert seeds [1, 2, 3], got {sorted(seen_seeds)}")
    run_manifest.sort(key=lambda row: row["seed"])
    if lock is not None and lock.get("run_manifest") != run_manifest:
        raise RuntimeError("Expert run manifest differs from the DEV lock")
    pending_lock = None
    if args.stage == "dev" and lock is None:
        pending_lock = {
            "schema_version": 1,
            "lock_purpose": "DEV-only DynaSemble training and configuration lock before TEST",
            "pair_name": "openbg_mhyper_native_dynasemble",
            "dataset": "openbg_img",
            "protocol_version": "openbg_legacy_v1",
            "seeds": [1, 2, 3],
            "method_config": method_config,
            "baseline_selection_path": str(baseline_selection_path),
            "baseline_selection_sha256": baseline_selection_hash,
            "global_alpha": global_alpha,
            "run_manifest": run_manifest,
            "selectors": {
                seed: {
                    "path": str(selector_dir / f"seed{seed}.pt"),
                    "sha256": selector_hashes[seed],
                    "training_summary": training_summaries[seed],
                }
                for seed in sorted(selector_hashes, key=int)
            },
            "test_is_not_used_for_training_or_selection": True,
        }
    all_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            row["direction"],
            int(row["relation_id"]),
            int(row["head_id"]),
            int(row["tail_id"]),
        )
    )
    reference_audit = None
    if args.reference_query_rows:
        reference_audit = validate_reference(
            all_rows, Path(args.reference_query_rows), args.stage
        )
    if pending_lock is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(pending_lock, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        lock = pending_lock
        print(f"[DEV LOCK] {lock_path} sha256={sha256_file(lock_path)}", flush=True)
    pooled, by_seed, by_direction = summarize(all_rows)
    write_rows(output_dir / f"{args.stage}_query_rows.csv", all_rows)
    write_csv(output_dir / f"{args.stage}_results.csv", pooled)
    write_csv(output_dir / f"{args.stage}_results_by_seed.csv", by_seed)
    write_csv(output_dir / f"{args.stage}_results_by_direction.csv", by_direction)
    write_markdown(output_dir / f"{args.stage}_results.md", pooled, args.stage)
    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "pair_name": "openbg_mhyper_native_dynasemble",
        "dataset": "openbg_img",
        "protocol_version": "openbg_legacy_v1",
        "n_rows": len(all_rows),
        "seeds": [1, 2, 3],
        "method_config": method_config,
        "dev_lock_path": str(lock_path),
        "dev_lock_sha256": sha256_file(lock_path),
        "baseline_selection_sha256": baseline_selection_hash,
        "reference_audit": reference_audit,
        "results": pooled,
        "results_by_seed": by_seed,
        "results_by_direction": by_direction,
        "clustered_intervals": {
            "dynasemble_vs_mhyper": clustered_interval(
                all_rows, "rr_dynasemble", "rr_a"
            ),
            "dynasemble_vs_query_zscore_0_5": clustered_interval(
                all_rows, "rr_dynasemble", "rr_equal"
            ),
            "dynasemble_vs_global_alpha": clustered_interval(
                all_rows, "rr_dynasemble", "rr_global"
            ),
        },
        "information_boundary": (
            "DynaSemble selector parameters are fit on DEV only. TEST loads the immutable "
            "DEV lock and per-seed selector hashes; Oracle is reporting-only."
        ),
    }
    (output_dir / f"{args.stage}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {output_dir / f'{args.stage}_results.md'}", flush=True)


if __name__ == "__main__":
    main()
