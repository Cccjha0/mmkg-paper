from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.feature_utils import load_cache_bundle


STRICT_CLEAN_FEATURES = [
    "direction",
    "relation_id",
    "relation_gain_prior",
    "relation_fusion_win_rate",
    "relation_support",
    "relation_is_visual_prior",
    "observed_has_img",
    "observed_text_img_cosine",
    "observed_img_missing_replaced",
    "candidate_has_img",
    "candidate_text_img_cosine",
    "candidate_img_missing_replaced",
    "candidate_text_norm",
    "candidate_img_norm",
    "candidate_is_observed_entity",
]


SCORE_AWARE_FEATURES = [
    "score_gate",
    "score_residual",
    "score_diff",
    "score_mean",
    "score_abs_diff",
    "score_max",
    "gate_rank_in_union",
    "residual_rank_in_union",
    "in_gate_topk",
    "in_residual_topk",
]


FORBIDDEN_FEATURES = [
    "is_target",
    "target_entity_id",
    "target_has_img",
    "target_regime",
    "rank_gate",
    "rank_residual",
    "rr_gate",
    "rr_residual",
    "rr_gain",
    "fusion_correct_score",
    "struct_correct_score",
    "correct_score",
]


METADATA_COLUMNS = [
    "query_id",
    "seed",
    "split",
    "head_id",
    "tail_id",
    "observed_entity_id",
    "candidate_entity_id",
    "target_entity_id",
    "is_target",
    "target_regime",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-aware router feature tables from exported candidate scores."
    )
    parser.add_argument("--dev-scores", nargs="+", default=["outputs/candidate_router/scores/dev_seed*_top100.parquet"])
    parser.add_argument("--test-scores", nargs="+", default=["outputs/candidate_router/scores/test_seed*_top100.parquet"])
    parser.add_argument("--relation-priors", default="outputs/router/raw/dev_relation_priors.csv")
    parser.add_argument("--cache-dir", default="data/cache/openbg_img")
    parser.add_argument("--out-dir", default="outputs/candidate_router/features")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200_000)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            files.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    unique = sorted(dict.fromkeys(files))
    if not unique:
        raise FileNotFoundError(f"No input files matched: {patterns}")
    return unique


def load_relation_priors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename_map = {
        "support": "relation_support",
        "mean_rr_gain": "relation_gain_prior",
        "is_visual_prior": "relation_is_visual_prior",
        "n_queries": "relation_support",
        "mean_delta_rr": "relation_gain_prior",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
    required = [
        "relation_id",
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_visual_prior",
    ]
    if "relation_fusion_win_rate" not in frame.columns and "fusion_win_rate" in frame.columns:
        frame["relation_fusion_win_rate"] = frame["fusion_win_rate"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Relation prior file is missing required columns: {missing}")
    return frame[required].copy()


def build_entity_feature_arrays(cache_dir: str | Path) -> dict[str, np.ndarray]:
    cache = load_cache_bundle(cache_dir)
    text_feat = cache["text_feat"].float()
    img_feat = cache["img_feat"].float()
    has_img = cache["has_img"].bool()

    text_norm = text_feat.norm(dim=1)
    img_norm = img_feat.norm(dim=1)
    cosine = F.cosine_similarity(text_feat, img_feat, dim=1)
    valid = has_img & text_norm.gt(0.0) & img_norm.gt(0.0)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))

    return {
        "has_img": has_img.to(dtype=torch.int64).numpy(),
        "text_img_cosine": cosine.numpy(),
        "img_missing_replaced": (~has_img).to(dtype=torch.int64).numpy(),
        "text_norm": text_norm.numpy(),
        "img_norm": img_norm.numpy(),
    }


def enrich_candidate_frame(
    frame: pd.DataFrame,
    relation_priors: pd.DataFrame,
    entity_features: dict[str, np.ndarray],
    top_k: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["score_abs_diff"] = frame["score_diff"].abs()
    frame["gate_rank_in_union"] = frame["candidate_rank_gate"].fillna(top_k + 1).astype("int64")
    frame["residual_rank_in_union"] = frame["candidate_rank_residual"].fillna(top_k + 1).astype("int64")
    frame["candidate_is_observed_entity"] = (
        frame["candidate_entity_id"].astype("int64") == frame["observed_entity_id"].astype("int64")
    ).astype("int64")

    frame = frame.merge(relation_priors, on="relation_id", how="left")
    prior_defaults = {
        "relation_gain_prior": 0.0,
        "relation_fusion_win_rate": 0.0,
        "relation_support": 0,
        "relation_is_visual_prior": 0,
    }
    for col, value in prior_defaults.items():
        frame[col] = frame[col].fillna(value)

    observed_ids = frame["observed_entity_id"].to_numpy(dtype=np.int64)
    candidate_ids = frame["candidate_entity_id"].to_numpy(dtype=np.int64)
    frame["observed_has_img"] = entity_features["has_img"][observed_ids]
    frame["observed_text_img_cosine"] = entity_features["text_img_cosine"][observed_ids]
    frame["observed_img_missing_replaced"] = entity_features["img_missing_replaced"][observed_ids]
    frame["candidate_has_img"] = entity_features["has_img"][candidate_ids]
    frame["candidate_text_img_cosine"] = entity_features["text_img_cosine"][candidate_ids]
    frame["candidate_img_missing_replaced"] = entity_features["img_missing_replaced"][candidate_ids]
    frame["candidate_text_norm"] = entity_features["text_norm"][candidate_ids]
    frame["candidate_img_norm"] = entity_features["img_norm"][candidate_ids]

    int_columns = [
        "seed",
        "relation_id",
        "head_id",
        "tail_id",
        "observed_entity_id",
        "candidate_entity_id",
        "target_entity_id",
        "is_target",
        "in_gate_topk",
        "in_residual_topk",
        "relation_support",
        "relation_is_visual_prior",
        "observed_has_img",
        "observed_img_missing_replaced",
        "candidate_has_img",
        "candidate_img_missing_replaced",
        "candidate_is_observed_entity",
        "gate_rank_in_union",
        "residual_rank_in_union",
    ]
    for col in int_columns:
        frame[col] = frame[col].fillna(0).astype("int64")

    output_columns = METADATA_COLUMNS + STRICT_CLEAN_FEATURES + SCORE_AWARE_FEATURES
    output_columns = list(dict.fromkeys(output_columns))
    missing = [col for col in output_columns if col not in frame.columns]
    if missing:
        raise RuntimeError(f"Internal error: missing output columns: {missing}")
    return frame[output_columns]


def empty_summary() -> dict:
    return {
        "rows": 0,
        "query_ids": set(),
        "seeds": set(),
        "target_rows": 0,
        "target_regime_counts": Counter(),
        "missing_values_in_deployable_features": Counter(),
    }


def update_summary(summary: dict, frame: pd.DataFrame) -> None:
    summary["rows"] += int(len(frame))
    summary["query_ids"].update(frame["query_id"].astype(str).unique())
    summary["seeds"].update(int(x) for x in frame["seed"].unique())
    summary["target_rows"] += int(frame["is_target"].sum())
    summary["target_regime_counts"].update(frame["target_regime"].astype(str).value_counts().to_dict())
    for col in STRICT_CLEAN_FEATURES + SCORE_AWARE_FEATURES:
        summary["missing_values_in_deployable_features"][col] += int(frame[col].isna().sum())


def finalize_summary(summary: dict) -> dict:
    queries = len(summary["query_ids"])
    return {
        "rows": int(summary["rows"]),
        "queries": int(queries),
        "seeds": sorted(summary["seeds"]),
        "target_rows": int(summary["target_rows"]),
        "avg_candidates_per_query": float(summary["rows"] / queries) if queries else 0.0,
        "target_regime_counts": {str(k): int(v) for k, v in summary["target_regime_counts"].items()},
        "missing_values_in_deployable_features": {
            str(k): int(v) for k, v in summary["missing_values_in_deployable_features"].items()
        },
    }


def build_split_features(
    *,
    files: list[Path],
    split: str,
    out_path: Path,
    relation_priors: pd.DataFrame,
    entity_features: dict[str, np.ndarray],
    top_k: int,
    batch_size: int,
) -> dict:
    writer: pq.ParquetWriter | None = None
    summary = empty_summary()
    try:
        for path in files:
            parquet_file = pq.ParquetFile(path)
            saw_rows = False
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                saw_rows = True
                raw_frame = batch.to_pandas()
                observed_split = set(raw_frame["split"].astype(str).unique())
                if observed_split != {split}:
                    raise ValueError(f"Unexpected split in {path}: {observed_split}, expected {split}")
                feature_frame = enrich_candidate_frame(raw_frame, relation_priors, entity_features, top_k)
                update_summary(summary, feature_frame)
                table = pa.Table.from_pandas(feature_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
            if not saw_rows:
                raise ValueError(f"Empty candidate score file: {path}")
    finally:
        if writer is not None:
            writer.close()
    return finalize_summary(summary)


def write_feature_contract(out_path: Path, args: argparse.Namespace, dev_files: list[Path], test_files: list[Path]) -> None:
    deployable = sorted(set(STRICT_CLEAN_FEATURES + SCORE_AWARE_FEATURES))
    leaked = sorted(set(deployable) & set(FORBIDDEN_FEATURES))
    if leaked:
        raise RuntimeError(f"Forbidden fields appear in deployable feature sets: {leaked}")

    payload = {
        "strict_clean_features": STRICT_CLEAN_FEATURES,
        "score_aware_features": SCORE_AWARE_FEATURES,
        "metadata_and_label_columns": METADATA_COLUMNS,
        "forbidden_features": FORBIDDEN_FEATURES,
        "feature_sets": {
            "CA-S1": STRICT_CLEAN_FEATURES,
            "CA-S2": [
                "direction",
                "relation_id",
                "relation_gain_prior",
                "relation_fusion_win_rate",
                "relation_support",
                "relation_is_visual_prior",
                *SCORE_AWARE_FEATURES,
            ],
            "CA-S3": STRICT_CLEAN_FEATURES + SCORE_AWARE_FEATURES,
        },
        "sources": {
            "dev_scores": [path.as_posix() for path in dev_files],
            "test_scores": [path.as_posix() for path in test_files],
            "relation_priors": args.relation_priors,
            "cache_dir": args.cache_dir,
            "top_k": args.top_k,
        },
        "leakage_check": {
            "forbidden_in_deployable_features": leaked,
            "note": "Forbidden columns may appear only as metadata, labels, evaluation fields, or post-hoc analysis fields.",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, dev_summary: dict, test_summary: dict) -> None:
    payload = {
        "dev": dev_summary,
        "test": test_summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dev_files = expand_inputs(args.dev_scores)
    test_files = expand_inputs(args.test_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relation_priors = load_relation_priors(Path(args.relation_priors))
    entity_features = build_entity_feature_arrays(args.cache_dir)

    dev_path = out_dir / "candidate_router_dev_top100.parquet"
    test_path = out_dir / "candidate_router_test_top100.parquet"
    contract_path = out_dir / "feature_contract.json"
    summary_path = out_dir / "candidate_router_feature_summary.json"

    dev_summary = build_split_features(
        files=dev_files,
        split="dev",
        out_path=dev_path,
        relation_priors=relation_priors,
        entity_features=entity_features,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    test_summary = build_split_features(
        files=test_files,
        split="test",
        out_path=test_path,
        relation_priors=relation_priors,
        entity_features=entity_features,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    write_feature_contract(contract_path, args, dev_files, test_files)
    write_summary(summary_path, dev_summary, test_summary)

    print(f"[OK] wrote dev features      -> {dev_path.as_posix()}")
    print(f"[OK] wrote test features     -> {test_path.as_posix()}")
    print(f"[OK] wrote feature contract  -> {contract_path.as_posix()}")
    print(f"[OK] wrote feature summary   -> {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
