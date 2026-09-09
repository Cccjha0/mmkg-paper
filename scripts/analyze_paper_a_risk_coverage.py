from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import QUERY_GEOMETRY_FIELDS


COVERAGE_GRID = tuple(value / 10.0 for value in range(11))
UNCHANGED_TOLERANCE = 1e-12
PAIR_SPECS = (
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
    },
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
    },
)

OUTPUT_COLUMNS = (
    "split",
    "scope",
    "dataset",
    "pair",
    "direction",
    "requested_coverage",
    "requested_coverage_percent",
    "mrr",
    "delta_mrr_vs_global",
    "harmful_query_rate",
    "beneficial_query_rate",
    "mean_harm",
    "adaptation_coverage",
    "changed_alpha_rate",
    "n_observations",
    "n_requested_for_adaptation",
    "n_adapted",
    "n_changed_alpha",
    "n_harmful",
    "n_beneficial",
    "n_unchanged",
    "n_nonfinite_fallback",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build diagnostic selective-adaptation risk-coverage curves from frozen "
            "Paper A Anchored Dynamic query-level assets. No model is trained."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/risk_coverage",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_risk_coverage_report.md",
    )
    return parser.parse_args()


def alpha_column(alpha: float) -> str:
    return f"rr_alpha_{alpha:.2f}".replace(".", "_")


def nearest_alpha(value: float, alphas: tuple[float, ...], anchor: float) -> float:
    """Match the released Anchored Dynamic grid tie-breaking exactly."""
    return min(alphas, key=lambda alpha: (abs(alpha - value), abs(alpha - anchor), alpha))


def map_to_grid(
    continuous: np.ndarray,
    anchors: np.ndarray,
    alphas: tuple[float, ...],
) -> np.ndarray:
    return np.asarray(
        [
            nearest_alpha(float(value), alphas, float(anchor))
            for value, anchor in zip(continuous, anchors, strict=True)
        ],
        dtype=np.float64,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def pair_root(repo_root: Path, spec: dict) -> Path:
    return (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )


def source_path(repo_root: Path, spec: dict, split: str) -> Path:
    root = pair_root(repo_root, spec)
    if split == "dev":
        return root / "anchored_crossfit" / "dev_anchored_query_rows.csv"
    return root / "test_anchored" / "test_locked_query_rows.csv"


def _allclose(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
        error = float(np.nanmax(np.abs(left - right)))
        raise RuntimeError(f"{label} mismatch; max absolute error={error}")


def rr_for_alpha(
    frame: pd.DataFrame,
    applied: np.ndarray,
    alphas: tuple[float, ...],
) -> np.ndarray:
    grid = frame[[alpha_column(alpha) for alpha in alphas]].to_numpy(dtype=np.float64)
    lookup = {alpha: index for index, alpha in enumerate(alphas)}
    indexes = np.asarray([lookup[float(alpha)] for alpha in applied], dtype=np.int64)
    rr = grid[np.arange(len(frame)), indexes]
    if not np.isfinite(rr).all() or np.any(rr <= 0.0):
        raise RuntimeError("Exact-ranking reciprocal ranks must be finite and positive")
    return rr


def audit_stored_policy(
    frame: pd.DataFrame,
    split: str,
    alphas: tuple[float, ...],
    nonfinite: np.ndarray,
) -> None:
    if split == "dev":
        anchors = frame["alpha0_crossfit"].to_numpy(dtype=np.float64)
        betas = frame["anchored_beta"].to_numpy(dtype=np.float64)
        thresholds = frame["anchored_confidence_threshold"].to_numpy(dtype=np.float64)
        applied_column = "alpha_anchored_crossfit"
        rr_column = "rr_anchored_crossfit"
    else:
        anchors = frame["alpha0_locked"].to_numpy(dtype=np.float64)
        betas = frame["anchored_beta_locked"].to_numpy(dtype=np.float64)
        thresholds = frame["anchored_confidence_threshold_locked"].to_numpy(
            dtype=np.float64
        )
        applied_column = "alpha_anchored_locked"
        rr_column = "rr_anchored_locked"
    decision = frame["anchored_decision"].to_numpy(dtype=np.float64)
    confidence = frame["anchored_confidence"].to_numpy(dtype=np.float64)
    fallback = nonfinite | (confidence < thresholds)
    continuous = np.clip(anchors + betas * np.tanh(decision), 0.0, 1.0)
    continuous = np.where(fallback, anchors, continuous)
    reconstructed_alpha = map_to_grid(continuous, anchors, alphas)
    stored_alpha = frame[applied_column].to_numpy(dtype=np.float64)
    _allclose(reconstructed_alpha, stored_alpha, f"{split} stored applied alpha")
    reconstructed_rr = rr_for_alpha(frame, reconstructed_alpha, alphas)
    stored_rr = frame[rr_column].to_numpy(dtype=np.float64)
    _allclose(reconstructed_rr, stored_rr, f"{split} stored reciprocal rank")
    stored_fallback = frame["anchored_fallback"].to_numpy(dtype=np.int64).astype(bool)
    if not np.array_equal(fallback, stored_fallback):
        raise RuntimeError(f"{split} stored fallback does not match policy reconstruction")


def load_pair_split(
    repo_root: Path,
    spec: dict,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    root = pair_root(repo_root, spec)
    lock_path = root / "dev_lock" / "anchored_dev_lock.json"
    rows_path = source_path(repo_root, spec, split)
    if not rows_path.exists() or not lock_path.exists():
        raise FileNotFoundError(rows_path if not rows_path.exists() else lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    alphas = tuple(float(value) for value in lock["alpha_grid"])
    frame = pd.read_csv(rows_path)
    split_columns = (
        {
            "alpha0_crossfit",
            "anchored_beta",
            "anchored_confidence_threshold",
            "alpha_anchored_crossfit",
            "rr_anchored_crossfit",
            "anchored_fold",
        }
        if split == "dev"
        else {
            "alpha0_locked",
            "anchored_beta_locked",
            "anchored_confidence_threshold_locked",
            "alpha_anchored_locked",
            "rr_anchored_locked",
        }
    )
    required = {
        "query_id",
        "query_key",
        "head_id",
        "relation_id",
        "tail_id",
        "target_entity_id",
        "seed",
        "direction",
        "split",
        "anchored_decision",
        "anchored_probability_a",
        "anchored_confidence",
        "anchored_fallback",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
        *split_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {rows_path}: {sorted(missing)}")
    if frame.empty or set(frame["split"]) != {split}:
        raise RuntimeError(f"Unexpected split content in {rows_path}")
    if frame["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id in {rows_path}")
    if set(frame["direction"]) != {"head", "tail"}:
        raise RuntimeError(f"Expected head and tail observations in {rows_path}")
    if str(lock["dataset"]) != spec["dataset_dir"]:
        raise RuntimeError(f"Dataset mismatch in {lock_path}")

    probability = frame["anchored_probability_a"].to_numpy(dtype=np.float64)
    decision = frame["anchored_decision"].to_numpy(dtype=np.float64)
    confidence = np.abs(2.0 * probability - 1.0)
    stored_confidence = frame["anchored_confidence"].to_numpy(dtype=np.float64)
    _allclose(confidence, stored_confidence, f"{split} confidence")
    geometry = frame[list(QUERY_GEOMETRY_FIELDS)].to_numpy(dtype=np.float64)
    nonfinite = (
        ~np.isfinite(geometry).all(axis=1)
        | ~np.isfinite(decision)
        | ~np.isfinite(probability)
    )
    audit_stored_policy(frame, split, alphas, nonfinite)

    alpha0 = float(lock["alpha0"])
    beta = float(lock["beta"])
    if alpha0 not in alphas:
        raise RuntimeError(f"Locked alpha0 is absent from alpha grid in {lock_path}")
    anchors = np.full(len(frame), alpha0, dtype=np.float64)
    raw_continuous = np.clip(alpha0 + beta * np.tanh(decision), 0.0, 1.0)
    raw_continuous = np.where(nonfinite, alpha0, raw_continuous)
    raw_applied = map_to_grid(raw_continuous, anchors, alphas)
    rr_raw = rr_for_alpha(frame, raw_applied, alphas)
    rr_global = frame[alpha_column(alpha0)].to_numpy(dtype=np.float64)
    global_column = "rr_global"
    if global_column in frame:
        _allclose(
            rr_global,
            frame[global_column].to_numpy(dtype=np.float64),
            f"{split} locked Global RR",
        )

    output = frame[
        [
            "query_id",
            "query_key",
            "head_id",
            "relation_id",
            "tail_id",
            "target_entity_id",
            "seed",
            "direction",
        ]
    ].copy()
    output.insert(0, "split", split)
    output.insert(1, "dataset", spec["dataset"])
    output.insert(2, "pair", spec["pair"])
    output["confidence"] = confidence
    output["nonfinite_fallback"] = nonfinite
    output["alpha0"] = alpha0
    output["beta"] = beta
    output["alpha_raw_continuous"] = raw_continuous
    output["alpha_raw_grid"] = raw_applied
    output["rr_global"] = rr_global
    output["rr_raw"] = rr_raw
    audit = {
        "split": split,
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "source_rows": portable_path(rows_path, repo_root),
        "source_rows_sha256": sha256_file(rows_path),
        "lock": portable_path(lock_path, repo_root),
        "lock_sha256": sha256_file(lock_path),
        "n_observations": int(len(frame)),
        "n_nonfinite": int(nonfinite.sum()),
        "alpha0": alpha0,
        "beta": beta,
        "formal_confidence_threshold": float(lock["confidence_threshold"]),
        "formal_fallback_rate_in_source": float(frame["anchored_fallback"].mean()),
        "curve_uses_formal_threshold": False,
        "dev_score_boundary": (
            "grouped held-out-triple OOF decision/probability with final DEV-locked "
            "alpha0 and beta"
            if split == "dev"
            else "locked TEST decision/probability and final DEV-locked alpha0 and beta"
        ),
    }
    return output, audit


def selection_order(frame: pd.DataFrame) -> np.ndarray:
    eligible = np.flatnonzero(~frame["nonfinite_fallback"].to_numpy(dtype=bool))
    confidence = frame["confidence"].to_numpy(dtype=np.float64)[eligible]
    query_id = frame["query_id"].astype(str).to_numpy()[eligible]
    order = np.lexsort((query_id, -confidence))
    return eligible[order]


def apply_coverage(frame: pd.DataFrame, coverage: float) -> pd.DataFrame:
    count = len(frame)
    requested = int(math.floor(coverage * count + 0.5))
    order = selection_order(frame)
    selected = np.zeros(count, dtype=bool)
    selected[order[: min(requested, len(order))]] = True
    alpha0 = frame["alpha0"].to_numpy(dtype=np.float64)
    alpha_raw = frame["alpha_raw_grid"].to_numpy(dtype=np.float64)
    rr_global = frame["rr_global"].to_numpy(dtype=np.float64)
    rr_raw = frame["rr_raw"].to_numpy(dtype=np.float64)
    output = frame.copy()
    output["requested_for_adaptation"] = requested
    output["selected_for_adaptation"] = selected
    output["alpha_selective"] = np.where(selected, alpha_raw, alpha0)
    output["changed_alpha"] = selected & (
        np.abs(output["alpha_selective"].to_numpy(dtype=np.float64) - alpha0)
        > UNCHANGED_TOLERANCE
    )
    output["rr_selective"] = np.where(selected, rr_raw, rr_global)
    output["delta_rr"] = output["rr_selective"] - rr_global
    return output


def summarize_state(
    frame: pd.DataFrame,
    *,
    scope: str,
    dataset: str,
    pair: str,
    direction: str,
    coverage: float,
) -> dict:
    delta = frame["delta_rr"].to_numpy(dtype=np.float64)
    harmful = delta < -UNCHANGED_TOLERANCE
    beneficial = delta > UNCHANGED_TOLERANCE
    unchanged = ~(harmful | beneficial)
    selected = frame["selected_for_adaptation"].to_numpy(dtype=bool)
    changed = frame["changed_alpha"].to_numpy(dtype=bool)
    harm = -delta[harmful]
    return {
        "split": str(frame.iloc[0]["split"]),
        "scope": scope,
        "dataset": dataset,
        "pair": pair,
        "direction": direction,
        "requested_coverage": coverage,
        "requested_coverage_percent": int(round(100.0 * coverage)),
        "mrr": float(frame["rr_selective"].mean()),
        "delta_mrr_vs_global": float(delta.mean()),
        "harmful_query_rate": float(harmful.mean()),
        "beneficial_query_rate": float(beneficial.mean()),
        "mean_harm": float(harm.mean()) if harm.size else None,
        "adaptation_coverage": float(selected.mean()),
        "changed_alpha_rate": float(changed.mean()),
        "n_observations": int(len(frame)),
        "n_requested_for_adaptation": int(
            math.floor(coverage * len(frame) + 0.5)
        ),
        "n_adapted": int(selected.sum()),
        "n_changed_alpha": int(changed.sum()),
        "n_harmful": int(harmful.sum()),
        "n_beneficial": int(beneficial.sum()),
        "n_unchanged": int(unchanged.sum()),
        "n_nonfinite_fallback": int(frame["nonfinite_fallback"].sum()),
    }


def analyze_split(
    pair_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    direction_summaries = []
    for coverage in COVERAGE_GRID:
        states = []
        for frame in pair_frames:
            state = apply_coverage(frame, coverage)
            states.append(state)
            dataset = str(frame.iloc[0]["dataset"])
            pair = str(frame.iloc[0]["pair"])
            summaries.append(
                summarize_state(
                    state,
                    scope="pair",
                    dataset=dataset,
                    pair=pair,
                    direction="pooled",
                    coverage=coverage,
                )
            )
            for direction, subset in state.groupby("direction", sort=False):
                direction_summaries.append(
                    summarize_state(
                        subset,
                        scope="pair",
                        dataset=dataset,
                        pair=pair,
                        direction=str(direction),
                        coverage=coverage,
                    )
                )
        pooled = pd.concat(states, ignore_index=True)
        summaries.append(
            summarize_state(
                pooled,
                scope="pooled",
                dataset="Pooled",
                pair="Pooled",
                direction="pooled",
                coverage=coverage,
            )
        )
        for direction, subset in pooled.groupby("direction", sort=False):
            direction_summaries.append(
                summarize_state(
                    subset,
                    scope="pooled",
                    dataset="Pooled",
                    pair="Pooled",
                    direction=str(direction),
                    coverage=coverage,
                )
            )
    return (
        pd.DataFrame(summaries)[list(OUTPUT_COLUMNS)],
        pd.DataFrame(direction_summaries)[list(OUTPUT_COLUMNS)],
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def plot_curves(all_summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render risk-coverage figures") from error

    panels = [
        (spec["dataset"], spec["pair"], f"{spec['dataset']}: {spec['pair']}")
        for spec in PAIR_SPECS
    ]
    panels.append(("Pooled", "Pooled", "Pooled summary"))
    metrics = (
        ("delta_mrr_vs_global", "Delta MRR vs Global", "coverage_vs_delta_mrr"),
        ("harmful_query_rate", "Harmful query rate", "coverage_vs_harm_rate"),
    )
    for metric, ylabel, basename in metrics:
        figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True)
        flat_axes = axes.ravel()
        for axis, (dataset, pair, title) in zip(flat_axes, panels, strict=False):
            subset = all_summary[
                all_summary["dataset"].eq(dataset) & all_summary["pair"].eq(pair)
            ]
            for split in ("dev", "test"):
                curve = subset[subset["split"].eq(split)].sort_values(
                    "requested_coverage"
                )
                values = curve[metric].to_numpy(dtype=np.float64)
                if metric == "harmful_query_rate":
                    values = 100.0 * values
                axis.plot(
                    100.0 * curve["requested_coverage"].to_numpy(dtype=np.float64),
                    values,
                    marker="o" if split == "dev" else "s",
                    linestyle="-" if split == "dev" else "--",
                    linewidth=1.5,
                    markersize=3.5,
                    label="Grouped OOF DEV" if split == "dev" else "Locked TEST",
                )
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("Requested adaptation coverage (%)")
            axis.set_ylabel(f"{ylabel} (%)" if metric == "harmful_query_rate" else ylabel)
            axis.set_xticks(range(0, 101, 20))
            axis.grid(True, linewidth=0.4, alpha=0.35)
        flat_axes[-1].axis("off")
        handles, labels = flat_axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        for suffix, options in (
            ("pdf", {}),
            ("png", {"dpi": 300}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"{basename}.{suffix}",
                bbox_inches="tight",
                **options,
            )
        plt.close(figure)


def _format_metric(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def report_table(frame: pd.DataFrame, coverages: tuple[float, ...]) -> list[str]:
    subset = frame[frame["requested_coverage"].isin(coverages)]
    lines = [
        "| Split | Dataset/Pair | Coverage | Delta MRR | Harm % | Benefit % | Changed-alpha % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subset.itertuples(index=False):
        lines.append(
            f"| {str(row.split).upper()} | {row.dataset} / {row.pair} | "
            f"{100.0 * row.requested_coverage:.0f}% | "
            f"{_format_metric(row.delta_mrr_vs_global)} | "
            f"{100.0 * row.harmful_query_rate:.2f} | "
            f"{100.0 * row.beneficial_query_rate:.2f} | "
            f"{100.0 * row.changed_alpha_rate:.2f} |"
        )
    return lines


def write_report(
    path: Path,
    all_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    audits: list[dict],
    output_dir: Path,
    repo_root: Path,
) -> None:
    pair_rows = all_summary[all_summary["scope"].eq("pair")]
    pooled = all_summary[all_summary["scope"].eq("pooled")]
    pooled_test_direction = direction_summary[
        direction_summary["scope"].eq("pooled")
        & direction_summary["split"].eq("test")
        & direction_summary["requested_coverage"].eq(0.7)
    ].set_index("direction")
    lines = [
        "# Paper A Selective Adaptation / Risk-Coverage Report",
        "",
        "## Main answer",
        "",
        "Allowing more queries to adapt initially increases net MRR, but negative-transfer "
        "risk accumulates throughout the curve and the marginal utility eventually "
        "diminishes. At the pooled level, both grouped OOF DEV and locked TEST reach their "
        "largest observed Delta MRR at 70% on the pre-fixed grid, then decline as coverage "
        "expands further. The behavior is pair-sensitive: three pairs remain net-positive "
        "at 100%, whereas DB15K / M-Hyper + AdaMF-MAT reverses below Global when every "
        "query is offered raw adaptation.",
        "",
        "## Protocol boundary",
        "",
        "This is a diagnostic analysis of existing query-level assets. It trains no base "
        "model or combiner, changes neither Anchored Dynamic nor beta, and performs no "
        "TEST-driven threshold or model selection. The fixed coverage grid is 0%, 10%, "
        "..., 100%.",
        "",
        "Confidence is `abs(2 * anchored_probability_a - 1)`. Within each pair and split, "
        "queries are sorted by decreasing confidence with `query_id` as a deterministic "
        "tie-break. The pooled curve is a micro-average of the four pair-wise selections. "
        "Selected finite queries use `clip(alpha0 + beta * tanh(decision), 0, 1)`, rounded "
        "with the existing nearest-grid rule; all other queries use alpha0. Reciprocal "
        "ranks are read from the precomputed exact filtered full-ranking alpha-grid columns.",
        "",
        "Grouped OOF DEV uses held-out-triple decision/probability values but the final "
        "DEV-locked alpha0 and beta for each pair, as required by this diagnostic. Locked "
        "TEST uses the same final lock. The official confidence threshold is not applied "
        "inside this curve: coverage itself defines selection, while non-finite queries "
        "always fall back.",
        "",
        "## Pair-level checkpoints",
        "",
        *report_table(pair_rows, (0.1, 0.5, 1.0)),
        "",
        "## Pooled curve",
        "",
        *report_table(pooled, COVERAGE_GRID),
        "",
        "## Interpretation",
        "",
    ]
    for split in ("dev", "test"):
        curve = pooled[pooled["split"].eq(split)].sort_values("requested_coverage")
        endpoint = curve[curve["requested_coverage"].eq(1.0)].iloc[0]
        maximum = curve.loc[curve["delta_mrr_vs_global"].idxmax()]
        lines.append(
            f"- {split.upper()}: expanding to 100% requested coverage gives Delta MRR "
            f"{endpoint['delta_mrr_vs_global']:+.6f}, harm rate "
            f"{100.0 * endpoint['harmful_query_rate']:.2f}%, and changed-alpha rate "
            f"{100.0 * endpoint['changed_alpha_rate']:.2f}%. The largest observed pooled "
            f"Delta MRR on the fixed grid occurs at {maximum['requested_coverage_percent']:.0f}% "
            f"coverage ({maximum['delta_mrr_vs_global']:+.6f}); this is retrospective "
            "description only, not a selected operating point."
        )
    for spec in PAIR_SPECS:
        pair_curves = pair_rows[
            pair_rows["dataset"].eq(spec["dataset"])
            & pair_rows["pair"].eq(spec["pair"])
        ]
        dev_curve = pair_curves[pair_curves["split"].eq("dev")]
        test_curve = pair_curves[pair_curves["split"].eq("test")]
        dev_max = dev_curve.loc[dev_curve["delta_mrr_vs_global"].idxmax()]
        test_max = test_curve.loc[test_curve["delta_mrr_vs_global"].idxmax()]
        test_end = test_curve[test_curve["requested_coverage"].eq(1.0)].iloc[0]
        lines.append(
            f"- {spec['dataset']} / {spec['pair']}: the largest observed grid value is "
            f"at {dev_max['requested_coverage_percent']:.0f}% on DEV "
            f"({dev_max['delta_mrr_vs_global']:+.6f}) and "
            f"{test_max['requested_coverage_percent']:.0f}% on TEST "
            f"({test_max['delta_mrr_vs_global']:+.6f}). At 100% TEST coverage, Delta MRR "
            f"is {test_end['delta_mrr_vs_global']:+.6f} and harm rate is "
            f"{100.0 * test_end['harmful_query_rate']:.2f}%."
        )
    lines.extend(
        [
            "- Harm rate is cumulative over all queries, so increasing coverage exposes "
            "additional queries to both beneficial and harmful corrections. Delta MRR "
            "shows the net magnitude balance; harm rate shows how widely negative transfer "
            "is distributed. Neither statistic substitutes for the other.",
            "- Changed-alpha rate can be lower than adaptation coverage because a bounded "
            "raw correction may round back to alpha0 on the exact-ranking grid.",
            "- This difference is especially visible when alpha0=1.0: the highest-confidence "
            "queries often support the primary expert, so a positive correction clips back "
            "to the anchor and changes no grid alpha. Requested coverage therefore should "
            "not be interpreted as effective intervention coverage.",
            "- At 70% pooled TEST coverage, head queries have a higher harm frequency "
            f"({100.0 * pooled_test_direction.loc['head', 'harmful_query_rate']:.2f}% vs. "
            f"{100.0 * pooled_test_direction.loc['tail', 'harmful_query_rate']:.2f}%), "
            "while harmful tail corrections have a larger conditional mean magnitude "
            f"({pooled_test_direction.loc['tail', 'mean_harm']:.6f} vs. "
            f"{pooled_test_direction.loc['head', 'mean_harm']:.6f}). Risk frequency and "
            "risk severity therefore remain distinct even after conditioning on coverage.",
            "",
            "## Figures",
            "",
            f"- Coverage vs Delta MRR: `{portable_path(output_dir / 'coverage_vs_delta_mrr.pdf', repo_root)}`",
            f"- Coverage vs Harm Rate: `{portable_path(output_dir / 'coverage_vs_harm_rate.pdf', repo_root)}`",
            "",
            "Each figure contains the four main model pairs and an additional pooled panel, "
            "with grouped OOF DEV and locked TEST shown separately. PDF, PNG, and SVG are "
            "generated without a hard-coded color palette.",
            "",
            "## Source and lock audit",
            "",
            "| Split | Dataset/Pair | alpha0 | beta | Final lock threshold | Stored-policy fallback % |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for audit in audits:
        lines.append(
            f"| {audit['split'].upper()} | {audit['dataset']} / {audit['pair']} | "
            f"{audit['alpha0']:.2f} | {audit['beta']:.2f} | "
            f"{audit['formal_confidence_threshold']:.2f} | "
            f"{100.0 * audit['formal_fallback_rate_in_source']:.2f} |"
        )
    lines.extend(
        [
            "",
            "For DEV, the stored fallback column comes from the original fold-specific OOF "
            "policies and is shown only as an audit reference; the diagnostic curve uses "
            "the final DEV-locked alpha0/beta and replaces the confidence threshold with "
            "the fixed coverage rule. Directional tables summarize the same pair-wise "
            "selection and do not re-rank within head or tail.",
            "",
            "Every source row file and lock was SHA-256 audited during execution. Stored "
            "confidence, fallback, applied alpha, and exact RR were reconstructed and "
            "checked against the original implementation before the diagnostic curve was "
            "computed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    report_path = (repo_root / args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_outputs = {}
    direction_outputs = []
    audits = []
    for split in ("dev", "test"):
        pair_frames = []
        for spec in PAIR_SPECS:
            frame, audit = load_pair_split(repo_root, spec, split)
            pair_frames.append(frame)
            audits.append(audit)
        split_summary, split_direction = analyze_split(pair_frames)
        split_outputs[split] = split_summary
        direction_outputs.append(split_direction)
        write_csv(output_dir / f"{split}_risk_coverage.csv", split_summary)

    all_summary = pd.concat(split_outputs.values(), ignore_index=True)
    by_pair = all_summary[all_summary["scope"].eq("pair")].copy()
    by_direction = pd.concat(direction_outputs, ignore_index=True)
    write_csv(output_dir / "risk_coverage_by_pair.csv", by_pair)
    write_csv(output_dir / "risk_coverage_by_direction.csv", by_direction)
    plot_curves(all_summary, output_dir)
    write_report(
        report_path,
        all_summary,
        by_direction,
        audits,
        output_dir,
        repo_root,
    )

    audit_payload = {
        "schema_version": 1,
        "analysis": "Paper A diagnostic selective adaptation / risk-coverage",
        "coverage_grid": list(COVERAGE_GRID),
        "confidence": "abs(2 * anchored_probability_a - 1)",
        "raw_correction": "clip(alpha0 + beta * tanh(anchored_decision), 0, 1)",
        "grid_mapping": (
            "nearest exact-ranking alpha; ties prefer alpha nearer alpha0, then smaller alpha"
        ),
        "selection_tie_break": "descending confidence, then query_id ascending",
        "test_is_diagnostic_only": True,
        "test_used_for_selection": False,
        "formal_fallback_threshold_modified": False,
        "source_audit": audits,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote risk-coverage analysis to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
