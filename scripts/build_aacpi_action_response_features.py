from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.dataset_spec import DatasetBundle, MMKG_GENERAL_V1
from ml.training.src.data.feature_bundle import load_processed_feature_bundle
from ml.training.src.data.tsv_reader import read_integer_triples
from ml.training.src.models.build_model import build_model
from router.score_combination import normalize_candidate_scores
from scripts.aacpi_phase3a_common import (
    R1_ADDITIONS,
    R2_ADDITIONS,
    R3_ADDITIONS,
    action_response_features,
    cross_expert_features,
    feature_manifest,
    portable_path,
    sha256_file,
    stable_order,
    validate_reference_response,
)
from scripts.eval_heterogeneous_complementarity import direction_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build answer-agnostic AACPI Phase 3A action-response features from frozen checkpoints."
    )
    parser.add_argument("--utility-table", required=True)
    parser.add_argument("--full-ranking-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mapping(path: Path) -> dict[str, int]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping JSON: {path}")
    return {str(key): int(item) for key, item in value.items()}


def reject_test_path(path: Path) -> None:
    tokens = {part.lower() for part in path.parts}
    if "test" in tokens or path.name.lower().startswith("test"):
        raise RuntimeError(f"Phase 3A refuses TEST paths: {path}")


def load_dev_only_bundle(cfg: dict) -> tuple[DatasetBundle, list[Path]]:
    """Load model inputs without opening the canonical TEST split."""
    if cfg.get("dataset", {}).get("loader") != "openke_mmkg":
        raise ValueError("Phase 3A currently supports the frozen openke_mmkg runs only")
    if cfg.get("protocol", {}).get("version") != MMKG_GENERAL_V1:
        raise ValueError("Phase 3A requires mmkg_general_v1")
    directory = Path(cfg["dataset"]["processed_dir"])
    paths = {
        "manifest": directory / "manifest.json",
        "train": directory / "train.tsv",
        "valid": directory / "valid.tsv",
        "entity2id": directory / "entity2id.json",
        "relation2id": directory / "relation2id.json",
        "text_feat": directory / "text_feat.pt",
        "img_feat": directory / "img_feat.pt",
        "has_text": directory / "has_text.pt",
        "has_img": directory / "has_img.pt",
    }
    for path in paths.values():
        reject_test_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = load_json(paths["manifest"])
    train = read_integer_triples(paths["train"])
    valid = read_integer_triples(paths["valid"])
    expected = manifest.get("counts", {})
    if len(train) != int(expected.get("train", -1)) or len(valid) != int(expected.get("valid", -1)):
        raise RuntimeError("TRAIN/DEV counts do not match the canonical manifest")
    expected_hashes = manifest.get("hashes", {}).get("splits", {})
    if expected_hashes.get("train") and sha256_file(paths["train"]) != expected_hashes["train"]:
        raise RuntimeError("TRAIN hash does not match the canonical manifest")
    if expected_hashes.get("valid") and sha256_file(paths["valid"]) != expected_hashes["valid"]:
        raise RuntimeError("DEV hash does not match the canonical manifest")
    features = load_processed_feature_bundle(directory)
    bundle = DatasetBundle(
        name=str(manifest["dataset"]),
        protocol_version=MMKG_GENERAL_V1,
        train_triples=train,
        valid_triples=valid,
        test_triples=[],
        num_entities=int(manifest["counts"]["entities"]),
        num_relations=int(manifest["counts"]["relations"]),
        entity2id=load_mapping(paths["entity2id"]),
        relation2id=load_mapping(paths["relation2id"]),
        features=features,
        manifest={"phase3a_dev_only": True},
    )
    bundle.validate()
    return bundle, list(paths.values())


def load_frozen_model(run_dir: Path, device: str):
    cfg_path, checkpoint_path = run_dir / "config_merged.json", run_dir / "best.ckpt"
    if not cfg_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing frozen run assets under {run_dir}")
    cfg = load_json(cfg_path)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("system", {})["device"] = device
    bundle, dataset_sources = load_dev_only_bundle(cfg)
    model, num_entities = build_model(cfg, dataset_bundle=bundle)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    prepare = getattr(model, "prepare_eval_cache", None)
    if prepare is not None:
        prepare()
    return cfg, bundle, model, int(num_entities), [cfg_path, checkpoint_path, *dataset_sources]


@torch.inference_mode()
def score_unfiltered(
    model,
    queries: torch.LongTensor,
    *,
    direction: str,
    num_entities: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """Score every candidate without evaluating or retaining the correct target separately."""
    scorer = direction_scorer(model, direction)
    q = queries.to(device)
    h, r, t = q.unbind(dim=1)
    all_entities = torch.arange(num_entities, dtype=torch.long, device=device)
    parts = []
    for start in range(0, num_entities, chunk_size):
        candidates = all_entities[start : start + chunk_size]
        width = candidates.numel()
        if direction == "tail":
            batch = torch.stack(
                [
                    h.unsqueeze(1).expand(-1, width).reshape(-1),
                    r.unsqueeze(1).expand(-1, width).reshape(-1),
                    candidates.unsqueeze(0).expand(q.size(0), -1).reshape(-1),
                ],
                dim=1,
            )
        else:
            batch = torch.stack(
                [
                    candidates.unsqueeze(0).expand(q.size(0), -1).reshape(-1),
                    r.unsqueeze(1).expand(-1, width).reshape(-1),
                    t.unsqueeze(1).expand(-1, width).reshape(-1),
                ],
                dim=1,
            )
        parts.append(scorer(batch).view(q.size(0), width).detach().cpu())
    scores = torch.cat(parts, dim=1)
    if scores.shape != (queries.size(0), num_entities) or not torch.isfinite(scores).all():
        raise RuntimeError("Expert produced an invalid unfiltered candidate matrix")
    return scores


def train_context(bundle: DatasetBundle) -> dict:
    relation_frequency = Counter()
    entity_frequency = Counter()
    entity_head_frequency = Counter()
    entity_tail_frequency = Counter()
    entity_relations: dict[int, set[int]] = defaultdict(set)
    target_text = defaultdict(list)
    target_image = defaultdict(list)
    has_text, has_image = bundle.features.has_text, bundle.features.has_img
    for head, relation, tail in bundle.train_triples:
        relation_frequency[relation] += 1
        entity_frequency[head] += 1
        entity_frequency[tail] += 1
        entity_head_frequency[head] += 1
        entity_tail_frequency[tail] += 1
        entity_relations[head].add(relation)
        entity_relations[tail].add(relation)
        target_text[(relation, "tail")].append(float(has_text[tail]))
        target_image[(relation, "tail")].append(float(has_image[tail]))
        target_text[(relation, "head")].append(float(has_text[head]))
        target_image[(relation, "head")].append(float(has_image[head]))
    return {
        "relation_frequency": relation_frequency,
        "entity_frequency": entity_frequency,
        "entity_head_frequency": entity_head_frequency,
        "entity_tail_frequency": entity_tail_frequency,
        "entity_relations": entity_relations,
        "target_text": {key: float(np.mean(value)) for key, value in target_text.items()},
        "target_image": {key: float(np.mean(value)) for key, value in target_image.items()},
    }


def context_features(row, bundle: DatasetBundle, context: dict) -> dict[str, float]:
    direction = str(row.direction)
    observed = int(row.tail) if direction == "head" else int(row.head)
    relation = int(row.relation)
    directional = (
        context["entity_tail_frequency"][observed]
        if direction == "head"
        else context["entity_head_frequency"][observed]
    )
    return {
        "r3_train_relation_frequency_log1p": float(np.log1p(context["relation_frequency"][relation])),
        "r3_train_observed_entity_frequency_log1p": float(np.log1p(context["entity_frequency"][observed])),
        "r3_train_observed_entity_direction_frequency_log1p": float(np.log1p(directional)),
        "r3_train_observed_entity_unique_relation_count_log1p": float(
            np.log1p(len(context["entity_relations"].get(observed, set())))
        ),
        "r3_observed_entity_has_text": float(bundle.features.has_text[observed]),
        "r3_observed_entity_has_image": float(bundle.features.has_img[observed]),
        "r3_train_relation_target_text_support": context["target_text"].get((relation, direction), 0.0),
        "r3_train_relation_target_image_support": context["target_image"].get((relation, direction), 0.0),
    }


def resolve_run_pairs(summary: dict) -> dict[int, tuple[Path, Path]]:
    by_seed: dict[int, dict[str, Path]] = defaultdict(dict)
    for row in summary.get("endpoint_reproduction", []):
        by_seed[int(row["seed"])][str(row["expert"])] = Path(row["run_dir"])
    result = {}
    for seed, values in by_seed.items():
        if set(values) != {"A", "B"}:
            raise ValueError(f"Incomplete endpoint provenance for seed {seed}")
        result[seed] = (values["A"], values["B"])
    if set(result) != {1, 2, 3}:
        raise ValueError("Expected frozen run provenance for seeds 1/2/3")
    return result


def build(args: argparse.Namespace) -> None:
    import pandas as pd

    utility_path = Path(args.utility_table)
    summary_path = Path(args.full_ranking_summary)
    output_dir = Path(args.output_dir)
    for path in (utility_path, summary_path, output_dir):
        reject_test_path(path)
    utility = pd.read_csv(utility_path, compression="infer")
    if utility.empty or set(utility["split"].astype(str)) != {"dev"}:
        raise RuntimeError("Phase 3A builder accepts DEV utility rows only")
    if utility[["query_id", "original_triple_id", "alpha", "alpha0"]].isna().any().any():
        raise ValueError("Utility identity/action fields contain missing values")
    summary = load_json(summary_path)
    if summary.get("split") != "dev":
        raise RuntimeError("Full-ranking provenance must be DEV")
    if set(utility["pair_id"].astype(str)) != {str(summary["pair_name"])}:
        raise ValueError("Utility/full-ranking pair mismatch")
    expected_actions = sorted(utility["alpha"].astype(float).unique().tolist())
    expected_alpha0 = sorted(utility["alpha0"].astype(float).unique().tolist())
    if len(expected_alpha0) != 1:
        raise ValueError("Utility table must contain one frozen alpha0")
    if not np.isclose(expected_alpha0[0], float(summary["selection"]["global_alpha"]), atol=1e-12):
        raise ValueError("Utility alpha0 differs from the DEV-locked Global alpha")
    per_query_actions = utility.groupby("query_id", sort=False)["alpha"].apply(
        lambda values: tuple(sorted(float(value) for value in values))
    )
    if any(actions != tuple(expected_actions) for actions in per_query_actions):
        raise ValueError("Not every query has the same frozen local action grid")
    run_pairs = resolve_run_pairs(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "dev_action_response_features.csv.gz"
    source_path = output_dir / "candidate_score_source_manifest.json"
    feature_path = output_dir / "feature_manifest.json"
    for path in (table_path, source_path, feature_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    feature_path.write_text(json.dumps(feature_manifest(), indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        payload = {
            "schema_version": 1,
            "phase": "AACPI Phase 3A",
            "dry_run": True,
            "split": "dev",
            "pair_id": summary["pair_name"],
            "utility_table": portable_path(utility_path),
            "utility_sha256": sha256_file(utility_path),
            "full_ranking_summary": portable_path(summary_path),
            "full_ranking_summary_sha256": sha256_file(summary_path),
            "run_pairs": {str(seed): [portable_path(a), portable_path(b)] for seed, (a, b) in run_pairs.items()},
            "n_rows": len(utility),
            "n_queries": int(utility["query_id"].nunique()),
            "test_rows_accessed": 0,
            "test_evaluation_commands": 0,
        }
        source_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[DRY-RUN OK] {summary['pair_name']} rows={len(utility)}")
        return

    device = resolve_device(args.device)
    generated: dict[int, dict[str, float]] = {}
    utility_by_query = {str(key): rows for key, rows in utility.groupby("query_id", sort=False)}
    source_files: dict[str, str] = {}
    canonical_dataset_sources: set[Path] = set()
    for seed in sorted(run_pairs):
        run_a, run_b = run_pairs[seed]
        cfg_a, bundle_a, model_a, n_a, sources_a = load_frozen_model(run_a, device)
        cfg_b, bundle_b, model_b, n_b, sources_b = load_frozen_model(run_b, device)
        if int(cfg_a.get("system", {}).get("seed", -1)) != seed or int(cfg_b.get("system", {}).get("seed", -1)) != seed:
            raise RuntimeError("Frozen run seed differs from full-ranking provenance")
        if n_a != n_b or bundle_a.name != bundle_b.name:
            raise RuntimeError("Frozen experts are dataset-incompatible")
        if bundle_a.train_triples != bundle_b.train_triples or bundle_a.valid_triples != bundle_b.valid_triples:
            raise RuntimeError("Frozen expert TRAIN/DEV bundles differ")
        if bundle_a.entity2id != bundle_b.entity2id or bundle_a.relation2id != bundle_b.relation2id:
            raise RuntimeError("Frozen expert mappings differ")
        for path in set(sources_a + sources_b):
            if path.name in {"config_merged.json", "best.ckpt"}:
                source_files[portable_path(path)] = sha256_file(path)
            else:
                canonical_dataset_sources.add(path)
        context = train_context(bundle_a)
        seed_rows = utility[utility["seed"].astype(int) == seed]
        utility_triples = {
            (int(row.head), int(row.relation), int(row.tail))
            for row in seed_rows.drop_duplicates("original_triple_id").itertuples(index=False)
        }
        if utility_triples != set(bundle_a.valid_triples):
            raise RuntimeError("Utility original triples do not exactly match canonical DEV")
        for direction in ("head", "tail"):
            direction_rows = seed_rows[seed_rows["direction"].astype(str) == direction]
            queries = direction_rows.drop_duplicates("query_id", keep="first").reset_index()
            qbatch = int(args.query_batch_size or min(cfg_a["evaluation"].get("query_batch_size", 8), cfg_b["evaluation"].get("query_batch_size", 8)))
            chunk = int(args.chunk_size or min(cfg_a["evaluation"].get("chunk_size", 4096), cfg_b["evaluation"].get("chunk_size", 4096)))
            for start in range(0, len(queries), qbatch):
                block = queries.iloc[start : start + qbatch]
                triples = torch.tensor(block[["head", "relation", "tail"]].to_numpy(), dtype=torch.long)
                raw_a = score_unfiltered(model_a, triples, direction=direction, num_entities=n_a, chunk_size=chunk, device=device)
                raw_b = score_unfiltered(model_b, triples, direction=direction, num_entities=n_b, chunk_size=chunk, device=device)
                z_a = normalize_candidate_scores(raw_a, "query_zscore").numpy().astype(np.float64)
                z_b = normalize_candidate_scores(raw_b, "query_zscore").numpy().astype(np.float64)
                for local, query in enumerate(block.itertuples(index=False)):
                    query_rows = utility_by_query[str(query.query_id)]
                    alpha0 = float(query.alpha0)
                    anchor = alpha0 * z_a[local] + (1.0 - alpha0) * z_b[local]
                    static = cross_expert_features(z_a[local], z_b[local])
                    context_row = context_features(query, bundle_a, context)
                    for action_row in query_rows.itertuples(index=True):
                        alpha = float(action_row.alpha)
                        action = alpha * z_a[local] + (1.0 - alpha) * z_b[local]
                        response = action_response_features(anchor, action)
                        if np.isclose(alpha, alpha0, rtol=0.0, atol=1e-12):
                            validate_reference_response(response)
                        generated[int(action_row.Index)] = {**static, **response, **context_row}
                print(f"[FEATURES] pair={summary['pair_name']} seed={seed} direction={direction} queries={min(start+qbatch,len(queries))}/{len(queries)}", flush=True)
        del model_a, model_b
        if device == "cuda":
            torch.cuda.empty_cache()
    if set(generated) != set(utility.index.astype(int)):
        missing = sorted(set(utility.index.astype(int)) - set(generated))[:10]
        raise AssertionError(f"Feature generation incomplete; missing indices {missing}")
    additions = pd.DataFrame.from_dict(generated, orient="index").sort_index()
    if list(additions.columns) != [*R1_ADDITIONS, *R2_ADDITIONS, *R3_ADDITIONS]:
        raise AssertionError("Generated fields differ from frozen feature contract")
    if not np.isfinite(additions.to_numpy(dtype=np.float64)).all():
        raise ValueError("Generated Phase 3A features contain NaN/Inf")
    enriched = pd.concat([utility.reset_index(drop=True), additions.reset_index(drop=True)], axis=1)
    enriched.to_csv(table_path, index=False, compression="gzip")
    for path in sorted(canonical_dataset_sources):
        source_files[portable_path(path)] = sha256_file(path)
    payload = {
        "schema_version": 1,
        "phase": "AACPI Phase 3A",
        "split": "dev",
        "pair_id": summary["pair_name"],
        "dataset": summary["dataset"],
        "expert_a": summary["expert_a_name"],
        "expert_b": summary["expert_b_name"],
        "alpha0": expected_alpha0[0],
        "action_grid": expected_actions,
        "candidate_domain": "full_unfiltered",
        "target_identity_used_for_features": False,
        "score_normalization": "router.score_combination.normalize_candidate_scores(query_zscore)",
        "utility_table": {"path": portable_path(utility_path), "sha256": sha256_file(utility_path)},
        "full_ranking_summary": {"path": portable_path(summary_path), "sha256": sha256_file(summary_path)},
        "feature_contract": {"path": portable_path(feature_path), "sha256": sha256_file(feature_path)},
        "source_files": source_files,
        "output": {"path": portable_path(table_path), "sha256": sha256_file(table_path)},
        "n_rows": len(enriched),
        "n_queries": int(enriched["query_id"].nunique()),
        "n_original_triples": int(enriched["original_triple_id"].nunique()),
        "test_rows_accessed": 0,
        "test_evaluation_commands": 0,
        "expert_training_performed": False,
        "checkpoint_selection_performed": False,
    }
    source_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {table_path} ({len(enriched)} rows)")


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
