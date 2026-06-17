from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_candidate_router_paper_tables import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired-bootstrap CIs for score-aware expert-combination comparisons."
    )
    parser.add_argument(
        "--score-ensemble-query-rows",
        default="outputs/score_ensemble/eval/score_ensemble_selected_query_rows.csv",
        help="Per-query selected score-interpolation rows from eval_score_ensemble_baselines.py.",
    )
    parser.add_argument(
        "--baseline-query-rows",
        default="outputs/router/eval/clean/baseline_locked_query_rows.csv",
        help="Locked clean-router per-query rows containing rr_regression_clean_router.",
    )
    parser.add_argument(
        "--ca-s2-query-glob",
        default="outputs/candidate_router/eval/ca_s2_full_ranking_seed*_query_rows.csv",
        help="Glob for CA-S2 full-ranking per-query rows.",
    )
    parser.add_argument("--output-dir", default="outputs/score_ensemble/eval")
    parser.add_argument("--paper-table-dir", default="docs/paper_tables")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-unit",
        choices=["query", "seed-query"],
        default="query",
        help=(
            "query: average the three seed-specific records for each original test query before bootstrapping; "
            "seed-query: bootstrap over all seed-query records directly."
        ),
    )
    return parser.parse_args()


def require_unique(frame: pd.DataFrame, name: str) -> None:
    if frame["query_id"].duplicated().any():
        dup = frame.loc[frame["query_id"].duplicated(), "query_id"].iloc[0]
        raise ValueError(f"{name} contains duplicate query_id: {dup}")


def load_score_ensemble(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "query_id",
        "split",
        "seed",
        "direction",
        "relation_id",
        "target_regime",
        "rr_global_interp",
        "rr_direction_interp",
        "rr_relation_interp",
        "rank_global_interp",
        "rank_direction_interp",
        "rank_relation_interp",
        "alpha_global",
        "alpha_direction",
        "alpha_relation",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    require_unique(frame, "score ensemble rows")
    return frame[required].copy()


def load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["query_id", "rr_regression_clean_router", "rr_residual", "rr_gate"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    require_unique(frame, "baseline rows")
    out = frame[required].copy()
    out = out.rename(
        columns={
            "rr_regression_clean_router": "rr_e5",
        }
    )
    return out


def load_ca_s2(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CA-S2 query rows matched: {pattern}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = ["query_id", "mixed_rr", "mixed_rank"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        frames.append(frame[required].copy())
    out = pd.concat(frames, ignore_index=True)
    require_unique(out, "CA-S2 rows")
    out = out.rename(columns={"mixed_rr": "rr_ca_s2", "mixed_rank": "rank_ca_s2"})
    return out


def paired_bootstrap_ci(diff: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = int(diff.size)
    observed = float(diff.mean())
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        boot[idx] = float(diff[sample].mean())
    low, high = np.percentile(boot, [2.5, 97.5])
    return observed, float(low), float(high)


def original_query_id(query_id: str) -> str:
    parts = str(query_id).split("|")
    if len(parts) >= 3 and parts[1].isdigit():
        return "|".join([parts[0], *parts[2:]])
    return str(query_id)


def average_over_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["original_query_id"] = frame["query_id"].map(original_query_id)
    rr_cols = [
        "rr_global_interp",
        "rr_direction_interp",
        "rr_relation_interp",
        "rr_e5",
        "rr_residual",
        "rr_gate",
        "rr_ca_s2",
    ]
    alpha_cols = [col for col in ["alpha_global", "alpha_direction", "alpha_relation"] if col in frame.columns]
    group_cols = ["original_query_id", "split", "direction", "relation_id", "target_regime"]
    counts = frame.groupby("original_query_id")["seed"].nunique()
    if counts.min() != counts.max():
        raise RuntimeError(
            "Inconsistent number of seeds per original query: "
            f"min={int(counts.min())}, max={int(counts.max())}"
        )
    averaged = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            {
                **{col: "mean" for col in rr_cols + alpha_cols},
                "query_id": "count",
            }
        )
        .rename(columns={"original_query_id": "query_id", "query_id": "n_seed_records"})
    )
    if averaged["query_id"].duplicated().any():
        raise RuntimeError("Averaged query IDs are not unique.")
    return averaged


def build_bootstrap_rows(frame: pd.DataFrame, n_bootstrap: int, seed: int, bootstrap_unit: str) -> list[dict]:
    comparisons = [
        (
            "Relation-specific interpolation vs. E5",
            "rr_relation_interp",
            "rr_e5",
            "score-aware > strict clean",
        ),
        (
            "Relation-specific interpolation vs. CA-S2",
            "rr_relation_interp",
            "rr_ca_s2",
            "interpolation > learned router",
        ),
        (
            "Global interpolation vs. CA-S2",
            "rr_global_interp",
            "rr_ca_s2",
            "simple score mix > learned router",
        ),
        (
            "Relation-specific interpolation vs. Global interpolation",
            "rr_relation_interp",
            "rr_global_interp",
            "relation refinement effect",
        ),
    ]
    rows = []
    for offset, (name, col_a, col_b, interpretation) in enumerate(comparisons):
        diff = (frame[col_a] - frame[col_b]).to_numpy(dtype=np.float64)
        delta, low, high = paired_bootstrap_ci(diff, n_bootstrap=n_bootstrap, seed=seed + offset)
        rows.append(
            {
                "comparison": name,
                "method_a": col_a,
                "method_b": col_b,
                "delta_mrr": delta,
                "ci_low": low,
                "ci_high": high,
                "n_queries": int(len(frame)),
                "n_bootstrap": int(n_bootstrap),
                "seed": int(seed + offset),
                "bootstrap_unit": bootstrap_unit,
                "interpretation": interpretation,
            }
        )
    return rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:+.4f}, {high:+.4f}]"


def write_latex(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Paired bootstrap confidence intervals for score-aware expert-combination comparisons.}",
        r"\label{tab:score_aware_bootstrap_ci}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.38\textwidth}ccp{0.24\textwidth}}",
        r"\toprule",
        r"Comparison & $\Delta$ MRR & 95\% CI & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {fmt(float(row['delta_mrr']))} & "
            f"{fmt_ci(float(row['ci_low']), float(row['ci_high']))} & {row['interpretation']} \\\\"
        )
    n_queries = int(rows[0]["n_queries"]) if rows else 0
    n_bootstrap = int(rows[0]["n_bootstrap"]) if rows else 0
    unit = str(rows[0].get("bootstrap_unit", "query")) if rows else "query"
    if unit == "query":
        note = (
            rf"\vspace{{0.4ex}}\caption*{{\footnotesize \textit{{Note:}} Intervals are computed over "
            rf"{n_queries:,} matched test-query reciprocal-rank differences after averaging seed-specific records, "
            rf"with {n_bootstrap:,} bootstrap resamples.}}"
        )
    else:
        note = (
            rf"\vspace{{0.4ex}}\caption*{{\footnotesize \textit{{Note:}} Intervals are computed over "
            rf"{n_queries:,} matched seed-query reciprocal-rank differences across three seeds, "
            rf"with {n_bootstrap:,} bootstrap resamples.}}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", note, r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paper_table_dir = Path(args.paper_table_dir)

    score = load_score_ensemble(Path(args.score_ensemble_query_rows))
    baseline = load_baseline(Path(args.baseline_query_rows))
    ca_s2 = load_ca_s2(args.ca_s2_query_glob)

    merged = score.merge(baseline, on="query_id", how="inner").merge(ca_s2, on="query_id", how="inner")
    if len(merged) != len(score) or len(merged) != len(baseline) or len(merged) != len(ca_s2):
        raise RuntimeError(
            "Query universe mismatch after merge: "
            f"merged={len(merged)} score={len(score)} baseline={len(baseline)} ca_s2={len(ca_s2)}"
        )
    require_unique(merged, "merged per-query RR")

    per_query_path = output_dir / "score_aware_per_query_rr.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(per_query_path, index=False)

    if args.bootstrap_unit == "query":
        bootstrap_frame = average_over_seeds(merged)
        averaged_path = output_dir / "score_aware_per_query_rr_seed_averaged.csv"
        bootstrap_frame.to_csv(averaged_path, index=False)
    else:
        bootstrap_frame = merged
        averaged_path = None

    rows = build_bootstrap_rows(
        bootstrap_frame,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        bootstrap_unit=args.bootstrap_unit,
    )
    ci_csv = output_dir / "score_aware_bootstrap_ci.csv"
    write_csv_rows(ci_csv, rows)

    md_frame = pd.DataFrame(rows)
    for col in ["delta_mrr", "ci_low", "ci_high"]:
        md_frame[col] = md_frame[col].map(lambda value: f"{float(value):+.4f}")
    (output_dir / "score_aware_bootstrap_ci.md").write_text(markdown_table(md_frame) + "\n", encoding="utf-8")

    tex_path = paper_table_dir / "table_score_aware_bootstrap_ci.tex"
    write_latex(tex_path, rows)

    print(f"[OK] wrote {per_query_path}")
    if averaged_path is not None:
        print(f"[OK] wrote {averaged_path}")
    print(f"[OK] wrote {ci_csv}")
    print(f"[OK] wrote {output_dir / 'score_aware_bootstrap_ci.md'}")
    print(f"[OK] wrote {tex_path}")


if __name__ == "__main__":
    main()
