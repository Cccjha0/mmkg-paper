import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.feature_utils import (
    build_posthoc_feature_rows,
    infer_cache_dir,
    load_cache_bundle,
)


POSTHOC_DEV_COLUMNS = [
    "query_id",
    "split",
    "direction",
    "target_has_img",
    "target_regime",
    "relation_id",
    "relation_gain_prior",
    "relation_fusion_win_rate",
    "relation_support",
    "relation_is_visual_prior",
    "text_img_cosine",
    "img_is_missing_replaced",
    "fusion_margin",
    "struct_margin",
    "fusion_correct_score",
    "struct_correct_score",
    "delta_margin",
    "rr_gate",
    "rr_residual",
    "rr_gain",
    "gain_label_d0",
    "gain_label_d001",
    "gain_label_d002",
]


POSTHOC_TEST_COLUMNS = [
    "query_id",
    "split",
    "direction",
    "target_has_img",
    "target_regime",
    "relation_id",
    "relation_gain_prior",
    "relation_fusion_win_rate",
    "relation_support",
    "relation_is_visual_prior",
    "text_img_cosine",
    "img_is_missing_replaced",
    "fusion_margin",
    "struct_margin",
    "fusion_correct_score",
    "struct_correct_score",
    "delta_margin",
    "rr_gate",
    "rr_residual",
]


EVAL_TARGET_COLUMNS = [
    "query_id",
    "rank_gate",
    "rr_gate",
    "rank_residual",
    "rr_residual",
    "rr_gain",
    "gain_label_d0",
    "gain_label_d001",
    "gain_label_d002",
    "target_regime",
    "direction",
    "relation_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared eval targets and posthoc contract feature tables.")
    parser.add_argument("--gate-dev", required=True)
    parser.add_argument("--residual-dev", required=True)
    parser.add_argument("--gate-test", required=True)
    parser.add_argument("--residual-test", required=True)
    parser.add_argument("--relation-priors", required=True)
    parser.add_argument("--out-dir", default="outputs/router/features")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-dir", default="ml/artifacts/outputs/openbg_img_gate_only/20260327_173820_seed1")
    return parser.parse_args()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def load_relation_prior_map_contract(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        relation_id = int(row["relation_id"])
        out[relation_id] = {
            "relation_gain_prior": float(row["mean_rr_gain"]),
            "relation_fusion_win_rate": float(row["fusion_win_rate"]),
            "relation_support": int(row["support"]),
            "relation_is_visual_prior": int(row["is_visual_prior"]),
        }
    return out


def normalize_query_rows(rows: list[dict], split: str) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "query_id": str(row["query_id"]),
                "split": split,
                "direction": str(row["direction"]),
                "relation_id": int(row["relation_id"]),
                "relation_name": row.get("relation_name", f"rel_{int(row['relation_id']):04d}"),
                "head_id": int(row["head_id"]),
                "tail_id": int(row["tail_id"]),
                "target_entity_id": int(row["target_entity_id"]),
                "target_position": str(row.get("target_position", row["direction"])),
                "target_has_img": int(row["target_has_img"]),
                "target_regime": str(row["target_regime"]),
                "rank": int(row["rank"]),
                "rr": float(row["rr"]),
                "score_margin": float(row.get("margin", row.get("score_margin", 0.0))),
                "correct_score": float(row["correct_score"]),
                "seed": int(row.get("source_seed", row.get("seed", 0))),
            }
        )
    return normalized


def enrich_rows(rows: list[dict], include_labels: bool) -> list[dict]:
    out = []
    for row in rows:
        rr_gate = float(row["rr_fusion"])
        rr_residual = float(row["rr_struct"])
        rr_gain = rr_gate - rr_residual
        base = {
            "query_id": row["query_id"],
            "split": row["split"],
            "direction": row["direction"],
            "target_has_img": int(row["target_has_img"]),
            "target_regime": row["target_regime"],
            "relation_id": int(row["relation_id"]),
            "relation_gain_prior": float(row["relation_gain_prior"]),
            "relation_fusion_win_rate": float(row["relation_fusion_win_rate"]),
            "relation_support": int(row["relation_support"]),
            "relation_is_visual_prior": int(row["relation_is_visual_prior"]),
            "text_img_cosine": float(row["text_img_cosine"]),
            "img_is_missing_replaced": int(row["img_is_missing_replaced"]),
            "fusion_margin": float(row["fusion_margin"]),
            "struct_margin": float(row["struct_margin"]),
            "fusion_correct_score": float(row["fusion_correct_score"]),
            "struct_correct_score": float(row["struct_correct_score"]),
            "delta_margin": float(row["delta_margin"]),
            "rr_gate": rr_gate,
            "rr_residual": rr_residual,
        }
        if include_labels:
            base["rr_gain"] = rr_gain
            base["gain_label_d0"] = int(rr_gain > 0.00)
            base["gain_label_d001"] = int(rr_gain > 0.01)
            base["gain_label_d002"] = int(rr_gain > 0.02)
        out.append(base)
    return out


def build_eval_targets(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        rr_gate = float(row["rr_fusion"])
        rr_residual = float(row["rr_struct"])
        rr_gain = rr_gate - rr_residual
        out.append(
            {
                "query_id": row["query_id"],
                "rank_gate": int(row["rank_fusion"]),
                "rr_gate": rr_gate,
                "rank_residual": int(row["rank_struct"]),
                "rr_residual": rr_residual,
                "rr_gain": rr_gain,
                "gain_label_d0": int(rr_gain > 0.00),
                "gain_label_d001": int(rr_gain > 0.01),
                "gain_label_d002": int(rr_gain > 0.02),
                "target_regime": row["target_regime"],
                "direction": row["direction"],
                "relation_id": int(row["relation_id"]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    cache_dir = infer_cache_dir(args.cache_dir, args.run_dir)
    cache_bundle = load_cache_bundle(cache_dir)
    prior_map = load_relation_prior_map_contract(read_table(Path(args.relation_priors)))

    gate_dev_rows = normalize_query_rows(read_table(Path(args.gate_dev)), "dev")
    residual_dev_rows = normalize_query_rows(read_table(Path(args.residual_dev)), "dev")
    gate_test_rows = normalize_query_rows(read_table(Path(args.gate_test)), "test")
    residual_test_rows = normalize_query_rows(read_table(Path(args.residual_test)), "test")

    dev_rows_raw = build_posthoc_feature_rows(
        gate_dev_rows,
        residual_dev_rows,
        prior_map,
        cache_bundle,
        label_by_query_id=None,
    )
    test_rows_raw = build_posthoc_feature_rows(
        gate_test_rows,
        residual_test_rows,
        prior_map,
        cache_bundle,
        label_by_query_id=None,
    )

    dev_rows = enrich_rows(dev_rows_raw, include_labels=True)
    test_rows = enrich_rows(test_rows_raw, include_labels=False)
    eval_target_rows = build_eval_targets(test_rows_raw)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path = out_dir / "router_features_posthoc_dev.parquet"
    test_path = out_dir / "router_features_posthoc_test.parquet"
    eval_path = out_dir / "router_eval_targets_shared_test.parquet"

    pd.DataFrame(dev_rows, columns=POSTHOC_DEV_COLUMNS).to_parquet(dev_path, index=False)
    pd.DataFrame(test_rows, columns=POSTHOC_TEST_COLUMNS).to_parquet(test_path, index=False)
    pd.DataFrame(eval_target_rows, columns=EVAL_TARGET_COLUMNS).to_parquet(eval_path, index=False)

    print(f"[OK] wrote dev features  -> {dev_path.as_posix()}")
    print(f"[OK] wrote test features -> {test_path.as_posix()}")
    print(f"[OK] wrote shared targets -> {eval_path.as_posix()}")


if __name__ == "__main__":
    main()
