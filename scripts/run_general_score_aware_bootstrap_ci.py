from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_candidate_router_paper_tables import markdown_table


POLICY_COLUMNS = {
    "gate": "rr_gate",
    "structural": "rr_residual",
    "global": "rr_global_interp",
    "direction": "rr_direction_interp",
    "relation": "rr_relation_interp",
    "query_soft": "rr_query_soft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired-bootstrap confidence intervals for general-protocol score-aware "
            "expert combination, without OpenBG-only E5/CA-S2 dependencies."
        )
    )
    parser.add_argument("--primary-query-rows", required=True)
    parser.add_argument("--comparison-query-rows", default=None)
    parser.add_argument("--primary-label", default="primary")
    parser.add_argument("--comparison-label", default="comparison")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-subgroups",
        action="store_true",
        help="Also bootstrap relation/query-soft gains separately by direction and modality regime.",
    )
    parser.add_argument(
        "--bootstrap-unit",
        choices=["query", "seed-query"],
        default="query",
        help="Average matching seeds per original query, or treat every seed-query as a unit.",
    )
    return parser.parse_args()


def original_query_id(query_id: str) -> str:
    parts = str(query_id).split("|")
    if len(parts) >= 3 and parts[1].isdigit():
        return "|".join([parts[0], *parts[2:]])
    return str(query_id)


def load_query_rows(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "query_id",
        "split",
        "seed",
        "direction",
        "relation_id",
        "target_regime",
        "dataset",
        "protocol_version",
        "target_score_semantics",
        "rr_gate",
        "rr_residual",
        "rr_global_interp",
        "rr_direction_interp",
        "rr_relation_interp",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame["query_id"].isna().any() or frame["query_id"].astype(str).str.len().eq(0).any():
        raise ValueError(f"{label} contains empty query_id values")
    if frame["query_id"].duplicated().any():
        duplicate = frame.loc[frame["query_id"].duplicated(), "query_id"].iloc[0]
        raise ValueError(f"{label} contains duplicate query_id: {duplicate}")
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError(f"{label} must contain TEST rows only")
    if set(frame["protocol_version"].astype(str)) != {"mmkg_general_v1"}:
        raise ValueError(f"{label} must use protocol_version=mmkg_general_v1")
    if set(frame["target_score_semantics"].astype(str)) != {"canonical_separate_target_score"}:
        raise ValueError(f"{label} must use canonical separate-target score semantics")

    available_policies = [column for column in POLICY_COLUMNS.values() if column in frame.columns]
    for column in available_policies:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{label} contains non-finite values in {column}")
    return frame[[*required, *[column for column in available_policies if column not in required]]].copy()


def prepare_units(frame: pd.DataFrame, bootstrap_unit: str) -> pd.DataFrame:
    frame = frame.copy()
    if bootstrap_unit == "seed-query":
        frame["unit_id"] = frame["query_id"].astype(str)
        return frame

    frame["unit_id"] = frame["query_id"].map(original_query_id)
    metadata = [
        "split",
        "direction",
        "relation_id",
        "target_regime",
        "dataset",
        "protocol_version",
        "target_score_semantics",
    ]
    for column in metadata:
        inconsistent = frame.groupby("unit_id")[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            unit_id = inconsistent[inconsistent].index[0]
            raise ValueError(f"Metadata column {column} varies across seeds for {unit_id}")
    policy_columns = [column for column in POLICY_COLUMNS.values() if column in frame.columns]
    prepared = (
        frame.groupby("unit_id", as_index=False)
        .agg(
            {
                **{column: "first" for column in metadata},
                **{column: "mean" for column in policy_columns},
                "seed": "nunique",
            }
        )
        .rename(columns={"seed": "n_seeds"})
    )
    if prepared["n_seeds"].nunique() != 1:
        raise ValueError("Every original query must contain the same number of seeds")
    return prepared


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("Paired metric vectors must be one-dimensional and aligned")
    if values_a.size == 0:
        raise ValueError("Cannot bootstrap an empty comparison")
    diff = values_a.astype(np.float64) - values_b.astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    batch_size = 64
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        sample = rng.integers(0, diff.size, size=(stop - start, diff.size))
        boot[start:stop] = diff[sample].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n": int(diff.size),
        "mrr_a": float(values_a.mean()),
        "mrr_b": float(values_b.mean()),
        "delta_mrr": float(diff.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
        "win_rate": float((diff > 0.0).mean()),
        "loss_rate": float((diff < 0.0).mean()),
        "tie_rate": float((diff == 0.0).mean()),
    }


def add_comparison(
    rows: list[dict],
    frame: pd.DataFrame,
    *,
    label_a: str,
    column_a: str,
    label_b: str,
    column_b: str,
    scope: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    metrics = paired_bootstrap_ci(
        frame[column_a].to_numpy(dtype=np.float64),
        frame[column_b].to_numpy(dtype=np.float64),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    rows.append(
        {
            "comparison": f"{label_a} vs {label_b}",
            "method_a": label_a,
            "method_b": label_b,
            "scope": scope,
            **metrics,
        }
    )


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

    primary = prepare_units(
        load_query_rows(Path(args.primary_query_rows), args.primary_label),
        args.bootstrap_unit,
    )
    rows: list[dict] = []
    comparisons = [
        ("global", "structural"),
        ("direction", "structural"),
        ("relation", "structural"),
        ("query_soft", "structural"),
        ("direction", "global"),
        ("relation", "global"),
        ("relation", "direction"),
        ("query_soft", "relation"),
    ]
    offset = 0
    for policy_a, policy_b in comparisons:
        column_a = POLICY_COLUMNS[policy_a]
        column_b = POLICY_COLUMNS[policy_b]
        if column_a not in primary.columns or column_b not in primary.columns:
            continue
        add_comparison(
            rows,
            primary,
            label_a=f"{args.primary_label}:{policy_a}",
            column_a=column_a,
            label_b=f"{args.primary_label}:{policy_b}",
            column_b=column_b,
            scope="all",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + offset,
        )
        offset += 1

    if args.include_subgroups:
        for scope_column in ("direction", "target_regime"):
            for scope_value, group in primary.groupby(scope_column, sort=True):
                for policy_a in ("relation", "query_soft"):
                    column_a = POLICY_COLUMNS[policy_a]
                    if column_a not in group.columns:
                        continue
                    add_comparison(
                        rows,
                        group,
                        label_a=f"{args.primary_label}:{policy_a}",
                        column_a=column_a,
                        label_b=f"{args.primary_label}:structural",
                        column_b=POLICY_COLUMNS["structural"],
                        scope=f"{scope_column}={scope_value}",
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + offset,
                    )
                    offset += 1

    if args.comparison_query_rows:
        comparison = prepare_units(
            load_query_rows(Path(args.comparison_query_rows), args.comparison_label),
            args.bootstrap_unit,
        )
        metadata = [
            "unit_id",
            "direction",
            "relation_id",
            "target_regime",
            "dataset",
            "protocol_version",
            "target_score_semantics",
        ]
        primary_ids = set(primary["unit_id"])
        comparison_ids = set(comparison["unit_id"])
        if primary_ids != comparison_ids:
            raise ValueError(
                "Primary/comparison query sets differ: "
                f"primary_only={len(primary_ids - comparison_ids)}, "
                f"comparison_only={len(comparison_ids - primary_ids)}"
            )
        merged = primary.merge(
            comparison,
            on="unit_id",
            how="inner",
            suffixes=("_primary", "_comparison"),
            validate="one_to_one",
        )
        for column in metadata[1:]:
            if not merged[f"{column}_primary"].equals(merged[f"{column}_comparison"]):
                raise ValueError(f"Primary/comparison metadata disagree for {column}")
        for policy in ("gate", "global", "direction", "relation", "query_soft"):
            column = POLICY_COLUMNS[policy]
            column_a = f"{column}_primary"
            column_b = f"{column}_comparison"
            if column_a not in merged.columns or column_b not in merged.columns:
                continue
            add_comparison(
                rows,
                merged,
                label_a=f"{args.primary_label}:{policy}",
                column_a=column_a,
                label_b=f"{args.comparison_label}:{policy}",
                column_b=column_b,
                scope="all",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + offset,
            )
            offset += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "general_score_aware_paired_bootstrap.csv"
    md_path = output_dir / "general_score_aware_paired_bootstrap.md"
    json_path = output_dir / "general_score_aware_paired_bootstrap.json"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "primary_label": args.primary_label,
                "comparison_label": args.comparison_label if args.comparison_query_rows else None,
                "bootstrap_unit": args.bootstrap_unit,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "include_subgroups": bool(args.include_subgroups),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {json_path}")


if __name__ == "__main__":
    main()
