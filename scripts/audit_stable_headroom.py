from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_cross_seed_transfer import PAIR_LABELS, action_indices, load_pair
from scripts.exp2_information_common import (
    ALPHAS,
    PAIR_IDS,
    ZERO_TOLERANCE,
    portable_path,
    reject_test_path,
    sha256_file,
)


STABLE_CLASSIFICATIONS = (
    "E6_STABLE_HIGH",
    "E6_STABLE_COLLAPSE",
    "E6_STABLE_INTERMEDIATE",
)
GAIN_CLASSIFICATIONS = (
    "E6_GAIN_HIGHLY_CONCENTRATED",
    "E6_GAIN_DIFFUSE",
    "E6_GAIN_INTERMEDIATE",
)
FINAL_INTERPRETATIONS = (
    "FINAL_RESIDUAL_DOMINATED_LIMITS",
    "FINAL_STABLE_BUT_UNOBSERVABLE",
    "FINAL_MODELING_BOTTLENECK",
    "FINAL_SELECTIVE_RARE_OPPORTUNITY",
    "FINAL_MIXED_LIMITS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DEV-only Experiment 6 closure audit.")
    parser.add_argument("--contract", default="docs/protocols/EXP6_STABLE_HEADROOM_CONTRACT.json")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--exp1-root", default="outputs/complementarity_identifiability/exp1_landscape")
    parser.add_argument("--exp4-root", default="outputs/complementarity_identifiability/exp4_cross_seed_transfer")
    parser.add_argument("--exp5-root", default="outputs/complementarity_identifiability/exp5_local_identifiability")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp6_stable_headroom")
    parser.add_argument("--report", default="docs/reports/stable_headroom_gain_concentration_audit_2026-09-07.md")
    parser.add_argument("--decision-memo", default="docs/protocols/COMPLEMENTARITY_CLOSURE_DECISION_MEMO.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_contract(path: Path) -> dict:
    reject_test_path(path)
    contract = load_json(path)
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("Experiment 6 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("Experiment 6 pair inventory changed")
    if tuple(contract.get("seeds", [])) != (1, 2, 3):
        raise RuntimeError("Experiment 6 seed inventory changed")
    if tuple(contract.get("directions", [])) != ("head", "tail"):
        raise RuntimeError("Experiment 6 direction inventory changed")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("Experiment 6 alpha grid changed")
    prohibited = (
        "test_access", "test_commands_executed", "checkpoint_retraining", "checkpoint_reselection",
        "new_selector", "new_representation", "policy_tuning", "action_grid_modification",
        "experiment_4_5_result_modification",
    )
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("Experiment 6 prohibited-operation boundary changed")
    return contract


def choose_allowed_action(mean_rr: np.ndarray, allowed: np.ndarray, global_index: int) -> np.ndarray:
    """Select by max mean RR, nearest alpha0, smaller alpha; empty rows fall back to alpha0."""
    if mean_rr.ndim != 2 or allowed.shape != mean_rr.shape:
        raise ValueError("mean_rr and allowed must be aligned 2D arrays")
    candidate = allowed.copy()
    empty = ~candidate.any(axis=1)
    candidate[empty, global_index] = True
    masked = np.where(candidate, mean_rr, -np.inf)
    maxima = masked.max(axis=1, keepdims=True)
    tied = candidate & np.isclose(masked, maxima, rtol=0.0, atol=ZERO_TOLERANCE)
    preference = np.abs(ALPHAS - ALPHAS[global_index]) + ALPHAS * 1e-9
    return np.where(tied, preference, np.inf).argmin(axis=1).astype(np.int16)


def concentration_metrics(gains: np.ndarray, fractions: tuple[float, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
    gains = np.asarray(gains, dtype=np.float64)
    if gains.ndim != 1 or len(gains) == 0 or np.any(gains < -ZERO_TOLERANCE):
        raise ValueError("Concentration gains must be a non-empty non-negative vector")
    gains = np.maximum(gains, 0.0)
    order = np.argsort(-gains, kind="stable")
    descending = gains[order]
    total = float(descending.sum())
    n = len(descending)
    cumulative = np.cumsum(descending)
    curve = cumulative / total if total > 0 else np.zeros(n, dtype=np.float64)
    summary: dict[str, float | int] = {"total_gain": total}
    for fraction in fractions:
        count = max(1, int(math.ceil(fraction * n)))
        key = f"top{int(round(fraction * 100))}_gain_share"
        summary[key] = float(cumulative[count - 1] / total) if total > 0 else float("nan")
        summary[key.replace("_gain_share", "_triple_count")] = count
    if total > 0:
        q50_count = int(np.searchsorted(cumulative, 0.5 * total, side="left") + 1)
        ascending = np.sort(gains, kind="stable")
        indices = np.arange(1, n + 1, dtype=np.float64)
        gini = float(np.sum((2.0 * indices - n - 1.0) * ascending) / (n * total))
        denom = float(np.square(gains).sum())
        effective = float((total * total / denom) / n) if denom > 0 else float("nan")
    else:
        q50_count, gini, effective = 0, float("nan"), float("nan")
    summary.update({
        "q50_triple_count": q50_count,
        "q50": float(q50_count / n) if total > 0 else float("nan"),
        "gini": gini,
        "effective_support": effective,
    })
    return summary, order, curve


def bootstrap_part_a(
    cluster_arrays: dict[str, np.ndarray], samples: int, seed: int
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    lengths = {len(values) for values in cluster_arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Bootstrap cluster arrays are not aligned")
    n = lengths.pop()
    if n == 0:
        raise ValueError("No clusters to bootstrap")
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(samples, dtype=np.float64) for name in cluster_arrays}
    ratios = {name: np.empty(samples, dtype=np.float64) for name in ("consensus", "stable2", "stable3")}
    for start in range(0, samples, 64):
        stop = min(samples, start + 64)
        indices = rng.integers(0, n, size=(stop - start, n))
        for name, values in cluster_arrays.items():
            distributions[name][start:stop] = values[indices].mean(axis=1)
        raw = distributions["raw"][start:stop]
        for name in ratios:
            ratios[name][start:stop] = distributions[name][start:stop] / raw
    intervals = {
        name: tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
        for name, values in distributions.items()
    }
    ratio_intervals = {
        name: tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
        for name, values in ratios.items()
    }
    return intervals, ratio_intervals


def cluster_query_values(per_query: pd.DataFrame, column: str) -> np.ndarray:
    return (
        per_query.groupby("original_triple_id", sort=True)[column]
        .mean()
        .to_numpy(np.float64)
    )


def analyze_pair(
    pair_id: str,
    frame: pd.DataFrame,
    rr: np.ndarray,
    alpha0: float,
    exp1_row: pd.Series,
    exp4_row: pd.Series,
    contract: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    global_candidates = np.flatnonzero(np.isclose(ALPHAS, alpha0, rtol=0.0, atol=ZERO_TOLERANCE))
    if len(global_candidates) != 1:
        raise RuntimeError(f"alpha0 is not on the frozen grid for {pair_id}: {alpha0}")
    global_index = int(global_candidates[0])
    identity = frame.iloc[::3].reset_index(drop=True)
    rr3 = rr.reshape(-1, 3, len(ALPHAS))
    if len(identity) * 3 != len(frame):
        raise RuntimeError(f"Seed triplet reshape failed for {pair_id}")
    seeds = frame.seed.to_numpy(np.int16).reshape(-1, 3)
    if not np.array_equal(seeds, np.tile(np.asarray((1, 2, 3)), (len(identity), 1))):
        raise RuntimeError(f"Seed ordering failed for {pair_id}")

    rows = np.arange(len(identity))[:, None]
    seed_axis = np.arange(3)[None, :]
    global_rr = rr3[:, :, global_index]
    utility = rr3 - global_rr[:, :, None]
    oracle_index = action_indices(rr3, global_index)
    oracle_gain = rr3[rows, seed_axis, oracle_index] - global_rr
    raw_q = oracle_gain.mean(axis=1)

    mean_rr = rr3.mean(axis=1)
    consensus_index = action_indices(mean_rr, global_index)
    consensus_gain = mean_rr[np.arange(len(identity)), consensus_index] - mean_rr[:, global_index]

    positive_counts = (utility > 0.0).sum(axis=1)
    non_anchor = np.ones_like(positive_counts, dtype=bool)
    non_anchor[:, global_index] = False
    stable2_allowed = (positive_counts >= 2) & non_anchor
    stable3_allowed = (positive_counts == 3) & non_anchor
    stable2_index = choose_allowed_action(mean_rr, stable2_allowed, global_index)
    stable3_index = choose_allowed_action(mean_rr, stable3_allowed, global_index)
    stable2_gain = mean_rr[np.arange(len(identity)), stable2_index] - mean_rr[:, global_index]
    stable3_gain = mean_rr[np.arange(len(identity)), stable3_index] - mean_rr[:, global_index]
    stable2_nonempty = stable2_allowed.any(axis=1)
    stable3_nonempty = stable3_allowed.any(axis=1)
    consensus_changed = consensus_index != global_index

    per_query = pd.DataFrame({
        "dataset": identity.dataset.to_numpy(str),
        "pair_id": pair_id,
        "original_triple_id": identity.original_triple_id.to_numpy(str),
        "direction": identity.direction.to_numpy(str),
        "head_id": identity.head_id.to_numpy(),
        "relation_id": identity.relation_id.to_numpy(),
        "tail_id": identity.tail_id.to_numpy(),
        "alpha0": alpha0,
        "raw_oracle_gain": raw_q,
        "oracle_alpha_seed1": ALPHAS[oracle_index[:, 0]],
        "oracle_alpha_seed2": ALPHAS[oracle_index[:, 1]],
        "oracle_alpha_seed3": ALPHAS[oracle_index[:, 2]],
        "oracle_gain_seed1": oracle_gain[:, 0],
        "oracle_gain_seed2": oracle_gain[:, 1],
        "oracle_gain_seed3": oracle_gain[:, 2],
        "consensus_alpha": ALPHAS[consensus_index],
        "consensus_gain": consensus_gain,
        "consensus_changed": consensus_changed.astype(np.int8),
        "stable2_set_size": stable2_allowed.sum(axis=1),
        "stable2_nonempty": stable2_nonempty.astype(np.int8),
        "stable2_alpha": ALPHAS[stable2_index],
        "stable2_gain": stable2_gain,
        "stable3_set_size": stable3_allowed.sum(axis=1),
        "stable3_nonempty": stable3_nonempty.astype(np.int8),
        "stable3_alpha": ALPHAS[stable3_index],
        "stable3_gain": stable3_gain,
    })
    raw = float(raw_q.mean())
    if not np.isclose(raw, float(exp1_row.available_headroom), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Raw Oracle Headroom does not reproduce Experiment 1 for {pair_id}")
    if not np.isclose(raw, float(exp4_row.raw_oracle_headroom), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Raw Oracle Headroom does not reproduce Experiment 4 for {pair_id}")

    triple_meta = identity.drop_duplicates("original_triple_id")[
        ["dataset", "original_triple_id", "head_id", "relation_id", "tail_id"]
    ].set_index("original_triple_id")
    triple_gain = per_query.groupby("original_triple_id", sort=True).raw_oracle_gain.mean()
    if float(triple_gain.min()) < -ZERO_TOLERANCE:
        raise RuntimeError(f"Negative triple-level Oracle contribution for {pair_id}")
    triple_gain = triple_gain.clip(lower=0.0)
    concentration = triple_meta.loc[triple_gain.index].reset_index()
    concentration.insert(1, "pair_id", pair_id)
    concentration["oracle_gain"] = triple_gain.to_numpy(np.float64)
    fractions = tuple(float(value) for value in contract["concentration"]["top_fractions"])
    conc_summary, order, curve = concentration_metrics(concentration.oracle_gain.to_numpy(), fractions)
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    cumulative_by_row = np.empty(len(order), dtype=np.float64)
    cumulative_by_row[order] = curve
    concentration["descending_gain_rank"] = ranks
    concentration["cumulative_triple_fraction"] = ranks / len(ranks)
    concentration["cumulative_gain_fraction"] = cumulative_by_row
    if not np.isclose(float(concentration.oracle_gain.mean()), raw, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Triple gain concentration does not reproduce Raw Oracle for {pair_id}")

    cluster_arrays = {
        name: cluster_query_values(per_query, column)
        for name, column in {
            "raw": "raw_oracle_gain",
            "consensus": "consensus_gain",
            "stable2": "stable2_gain",
            "stable3": "stable3_gain",
            "consensus_changed_rate": "consensus_changed",
            "stable2_opportunity_rate": "stable2_nonempty",
            "stable3_opportunity_rate": "stable3_nonempty",
        }.items()
    }
    intervals, ratio_intervals = bootstrap_part_a(
        cluster_arrays, int(contract["bootstrap"]["samples"]), int(contract["bootstrap"]["seed"])
    )
    consensus = float(consensus_gain.mean())
    stable2 = float(stable2_gain.mean())
    stable3 = float(stable3_gain.mean())
    recovery_consensus = consensus / raw if raw != 0 else float("nan")
    recovery2 = stable2 / raw if raw != 0 else float("nan")
    recovery3 = stable3 / raw if raw != 0 else float("nan")
    summary = {
        "dataset": identity.dataset.iloc[0],
        "pair_id": pair_id,
        "pair_label": PAIR_LABELS[pair_id],
        "n_original_triples": int(concentration.original_triple_id.nunique()),
        "n_query_identities": int(len(identity)),
        "alpha0": alpha0,
        "raw_oracle_headroom": raw,
        "raw_ci95_low": intervals["raw"][0],
        "raw_ci95_high": intervals["raw"][1],
        "consensus_headroom": consensus,
        "consensus_ci95_low": intervals["consensus"][0],
        "consensus_ci95_high": intervals["consensus"][1],
        "consensus_recovery": recovery_consensus,
        "consensus_recovery_ci95_low": ratio_intervals["consensus"][0],
        "consensus_recovery_ci95_high": ratio_intervals["consensus"][1],
        "stable2_headroom": stable2,
        "stable2_ci95_low": intervals["stable2"][0],
        "stable2_ci95_high": intervals["stable2"][1],
        "stable2_recovery": recovery2,
        "stable2_recovery_ci95_low": ratio_intervals["stable2"][0],
        "stable2_recovery_ci95_high": ratio_intervals["stable2"][1],
        "stable3_headroom": stable3,
        "stable3_ci95_low": intervals["stable3"][0],
        "stable3_ci95_high": intervals["stable3"][1],
        "stable3_recovery": recovery3,
        "stable3_recovery_ci95_low": ratio_intervals["stable3"][0],
        "stable3_recovery_ci95_high": ratio_intervals["stable3"][1],
        "fragility2": 1.0 - recovery2,
        "fragility2_ci95_low": 1.0 - ratio_intervals["stable2"][1],
        "fragility2_ci95_high": 1.0 - ratio_intervals["stable2"][0],
        "fragility3": 1.0 - recovery3,
        "fragility3_ci95_low": 1.0 - ratio_intervals["stable3"][1],
        "fragility3_ci95_high": 1.0 - ratio_intervals["stable3"][0],
        "consensus_changed_rate": float(consensus_changed.mean()),
        "consensus_changed_ci95_low": intervals["consensus_changed_rate"][0],
        "consensus_changed_ci95_high": intervals["consensus_changed_rate"][1],
        "stable2_opportunity_rate": float(stable2_nonempty.mean()),
        "stable2_opportunity_ci95_low": intervals["stable2_opportunity_rate"][0],
        "stable2_opportunity_ci95_high": intervals["stable2_opportunity_rate"][1],
        "stable3_opportunity_rate": float(stable3_nonempty.mean()),
        "stable3_opportunity_ci95_low": intervals["stable3_opportunity_rate"][0],
        "stable3_opportunity_ci95_high": intervals["stable3_opportunity_rate"][1],
        "loso_stable_headroom": float(exp4_row.loso_stable_headroom),
        "loso_ci95_low": float(exp4_row.loso_ci95_low),
        "loso_ci95_high": float(exp4_row.loso_ci95_high),
        "loso_recovery": float(exp4_row.loso_recovery),
        "loso_recovery_ci95_low": float(exp4_row.loso_recovery_ci95_low),
        "loso_recovery_ci95_high": float(exp4_row.loso_recovery_ci95_high),
        "frozen_x4_oof_gain": float(exp4_row.frozen_x4_oof_gain),
        "frozen_x4_ci95_low": float(exp4_row.frozen_x4_ci95_low),
        "frozen_x4_ci95_high": float(exp4_row.frozen_x4_ci95_high),
        **conc_summary,
    }
    bootstrap_rows = []
    estimates = {
        "raw_oracle_headroom": (raw, intervals["raw"]),
        "consensus_headroom": (consensus, intervals["consensus"]),
        "consensus_recovery": (recovery_consensus, ratio_intervals["consensus"]),
        "stable2_headroom": (stable2, intervals["stable2"]),
        "stable2_recovery": (recovery2, ratio_intervals["stable2"]),
        "stable3_headroom": (stable3, intervals["stable3"]),
        "stable3_recovery": (recovery3, ratio_intervals["stable3"]),
        "fragility2": (1.0 - recovery2, (1.0 - ratio_intervals["stable2"][1], 1.0 - ratio_intervals["stable2"][0])),
        "fragility3": (1.0 - recovery3, (1.0 - ratio_intervals["stable3"][1], 1.0 - ratio_intervals["stable3"][0])),
        "consensus_changed_rate": (float(consensus_changed.mean()), intervals["consensus_changed_rate"]),
        "stable2_opportunity_rate": (float(stable2_nonempty.mean()), intervals["stable2_opportunity_rate"]),
        "stable3_opportunity_rate": (float(stable3_nonempty.mean()), intervals["stable3_opportunity_rate"]),
    }
    for metric, (estimate, interval) in estimates.items():
        bootstrap_rows.append({
            "dataset": identity.dataset.iloc[0], "pair_id": pair_id, "metric": metric,
            "estimate": estimate, "ci95_low": interval[0], "ci95_high": interval[1],
            "bootstrap_samples": int(contract["bootstrap"]["samples"]),
            "bootstrap_unit": "original_triple_id",
        })
    return per_query, concentration, summary, bootstrap_rows


def classify_stability(summary: pd.DataFrame, contract: dict) -> tuple[str, dict]:
    significant2 = summary.stable2_ci95_low > 0
    significant3 = summary.stable3_ci95_low > 0
    datasets3 = set(summary.loc[significant3, "dataset"])
    high_cfg = contract["stable_classification"]["stable_high"]
    high = (
        int(significant2.sum()) >= int(high_cfg["minimum_two_of_three_ci_lower_positive_pairs"])
        and float(summary.stable2_recovery.median()) >= float(high_cfg["median_two_of_three_recovery_gte"])
        and int(significant3.sum()) >= int(high_cfg["minimum_three_of_three_ci_lower_positive_pairs"])
        and datasets3 == {"mkg_w", "db15k"}
    )
    collapse_cfg = contract["stable_classification"]["stable_collapse"]
    collapse = (
        float(summary.stable2_recovery.median()) < float(collapse_cfg["median_two_of_three_recovery_lt"])
        and float(summary.stable3_recovery.median()) < float(collapse_cfg["median_three_of_three_recovery_lt"])
        and int((~significant3).sum()) >= int(collapse_cfg["minimum_three_of_three_not_significantly_positive_pairs"])
    )
    decision = "E6_STABLE_HIGH" if high else "E6_STABLE_COLLAPSE" if collapse else "E6_STABLE_INTERMEDIATE"
    evidence = {
        "stable2_ci_lower_positive_pairs": int(significant2.sum()),
        "median_stable2_recovery": float(summary.stable2_recovery.median()),
        "stable3_ci_lower_positive_pairs": int(significant3.sum()),
        "stable3_significant_datasets": sorted(datasets3),
        "median_stable3_recovery": float(summary.stable3_recovery.median()),
        "stable3_not_significantly_positive_pairs": int((~significant3).sum()),
        "stable_high_gate": bool(high),
        "stable_collapse_gate": bool(collapse),
    }
    return decision, evidence


def classify_concentration(summary: pd.DataFrame, contract: dict) -> tuple[str, dict]:
    high_cfg = contract["gain_classification"]["highly_concentrated"]
    diffuse_cfg = contract["gain_classification"]["diffuse"]
    top10_high = int((summary.top10_gain_share >= 0.50).sum())
    q50_low = int((summary.q50 <= 0.10).sum())
    top10_diffuse = int((summary.top10_gain_share <= 0.25).sum())
    q50_diffuse = int((summary.q50 >= 0.30).sum())
    high = (
        top10_high >= int(high_cfg["minimum_top10_share_gte_half_pairs"])
        and q50_low >= int(high_cfg["minimum_q50_lte_ten_percent_pairs"])
    )
    diffuse = (
        top10_diffuse >= int(diffuse_cfg["minimum_top10_share_lte_quarter_pairs"])
        and q50_diffuse >= int(diffuse_cfg["minimum_q50_gte_thirty_percent_pairs"])
    )
    decision = "E6_GAIN_HIGHLY_CONCENTRATED" if high else "E6_GAIN_DIFFUSE" if diffuse else "E6_GAIN_INTERMEDIATE"
    return decision, {
        "top10_share_gte_50pct_pairs": top10_high,
        "q50_lte_10pct_pairs": q50_low,
        "top10_share_lte_25pct_pairs": top10_diffuse,
        "q50_gte_30pct_pairs": q50_diffuse,
        "highly_concentrated_gate": bool(high),
        "diffuse_gate": bool(diffuse),
    }


def classify_joint(
    e4_payload: dict,
    e4_summary: pd.DataFrame,
    e5_payload: dict,
    stable_classification: str,
    gain_classification: str,
    contract: dict,
) -> tuple[str, dict]:
    e4 = str(e4_payload.get("classification", ""))
    e5 = str(e5_payload.get("final_classification", ""))
    e5_recorded = set(map(str, e5_payload.get("recorded_classifications", [])))
    x6_recovery = bool(e5_payload.get("evidence", {}).get("x6_recovery", False))
    e5_local_identifiable = e5 == "E5_LOCAL_IDENTIFIABLE" or "E5_LOCAL_IDENTIFIABLE" in e5_recorded
    cfg = contract["joint_interpretation"]["strong_e4_intermediate_definition"]
    significant_loso = e4_summary.loso_ci95_low > 0
    strong_e4 = (
        e4 == cfg["classification"]
        and int(significant_loso.sum()) >= int(cfg["minimum_loso_ci_lower_positive_pairs"])
        and set(e4_summary.loc[significant_loso, "dataset"]) == {"mkg_w", "db15k"}
        and float(e4_summary.loso_recovery.median()) >= float(cfg["median_loso_recovery_gte"])
    )
    gates = {
        "residual_dominated_limits": e4 == "E4_RESIDUAL_DOMINATED" and e5 == "E5_LOCAL_AMBIGUITY" and stable_classification == "E6_STABLE_COLLAPSE",
        "stable_but_unobservable": (e4 == "E4_SUBSTANTIALLY_TRANSFERABLE" or strong_e4) and stable_classification == "E6_STABLE_HIGH" and e5 == "E5_LOCAL_AMBIGUITY" and not x6_recovery,
        "modeling_bottleneck": (
            e4 == "E4_SUBSTANTIALLY_TRANSFERABLE"
            and stable_classification == "E6_STABLE_HIGH"
            and e5_local_identifiable
        ),
        "selective_rare_opportunity": e4 == "E4_INTERMEDIATE" and e5 == "E5_INTERMEDIATE" and gain_classification == "E6_GAIN_HIGHLY_CONCENTRATED" and stable_classification != "E6_STABLE_COLLAPSE",
    }
    if gates["residual_dominated_limits"]:
        decision = "FINAL_RESIDUAL_DOMINATED_LIMITS"
    elif gates["stable_but_unobservable"]:
        decision = "FINAL_STABLE_BUT_UNOBSERVABLE"
    elif gates["modeling_bottleneck"]:
        decision = "FINAL_MODELING_BOTTLENECK"
    elif gates["selective_rare_opportunity"]:
        decision = "FINAL_SELECTIVE_RARE_OPPORTUNITY"
    else:
        decision = "FINAL_MIXED_LIMITS"
    evidence = {
        "experiment4_classification": e4,
        "experiment5_classification": e5,
        "experiment6_stable_classification": stable_classification,
        "experiment6_gain_classification": gain_classification,
        "strong_e4_intermediate": bool(strong_e4),
        "e5_x6_recovery": x6_recovery,
        "e5_local_identifiable": e5_local_identifiable,
        "candidate_gates": {key: bool(value) for key, value in gates.items()},
    }
    return decision, evidence


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="40" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{html.escape(title)}</text>',
    ]


def svg_text(parts: list[str], x: float, y: float, value: str, size: int = 11, anchor: str = "start", fill: str = "#30343b") -> None:
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="{fill}">{html.escape(value)}</text>')


def write_headroom_figure(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("Raw", "raw_oracle_headroom", "raw_ci95_low", "raw_ci95_high", "#5b6573"),
        ("Consensus", "consensus_headroom", "consensus_ci95_low", "consensus_ci95_high", "#28778e"),
        ("2/3", "stable2_headroom", "stable2_ci95_low", "stable2_ci95_high", "#48a08a"),
        ("3/3", "stable3_headroom", "stable3_ci95_low", "stable3_ci95_high", "#87b75d"),
        ("LOSO", "loso_stable_headroom", "loso_ci95_low", "loso_ci95_high", "#d49a43"),
        ("X4 OOF*", "frozen_x4_oof_gain", "frozen_x4_ci95_low", "frozen_x4_ci95_high", "#c45d3c"),
    ]
    width, height = 1450, 690
    parts = svg_start(width, height, "Stable Headroom Decomposition")
    ymax = max(float(summary[high].max()) for _, _, _, high, _ in metrics) * 1.12
    ymin = min(0.0, min(float(summary[low].min()) for _, _, low, _, _ in metrics))
    left, right, top, bottom = 170, 1410, 80, 570
    yscale = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    zero = yscale(0.0)
    parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{right}" y2="{zero:.1f}" stroke="#888"/>')
    group_width = (right - left) / len(summary)
    bar_width = group_width / 8.0
    for pair_index, row in enumerate(summary.itertuples(index=False)):
        center = left + (pair_index + 0.5) * group_width
        for metric_index, (_, value_field, low_field, high_field, color) in enumerate(metrics):
            x = center + (metric_index - 2.5) * bar_width
            value = float(getattr(row, value_field)); low = float(getattr(row, low_field)); high = float(getattr(row, high_field))
            y = yscale(value)
            parts.append(f'<rect x="{x-bar_width*.38:.1f}" y="{min(y,zero):.1f}" width="{bar_width*.76:.1f}" height="{abs(zero-y):.1f}" fill="{color}"/>')
            parts.append(f'<line x1="{x:.1f}" y1="{yscale(high):.1f}" x2="{x:.1f}" y2="{yscale(low):.1f}" stroke="#222"/>')
        svg_text(parts, center, bottom + 24, row.pair_id.replace("mkgw_", "MW/").replace("db15k_", "DB/"), 10, "middle")
    for index, (label, _, _, _, color) in enumerate(metrics):
        x = 170 + index * 170
        parts.append(f'<rect x="{x}" y="620" width="14" height="14" fill="{color}"/>')
        svg_text(parts, x + 20, 632, label, 11)
    svg_text(parts, 40, 666, "MRR gain vs Global. Consensus/2-of-3/3-of-3 are ex-post diagnostics; LOSO is held-out-seed; *X4 uses fold-specific OOF Global.", 11, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_recovery_figure(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("2-of-3", "stable2_recovery", "stable2_recovery_ci95_low", "stable2_recovery_ci95_high", "#28778e", -8),
        ("3-of-3", "stable3_recovery", "stable3_recovery_ci95_low", "stable3_recovery_ci95_high", "#87b75d", 0),
        ("LOSO", "loso_recovery", "loso_recovery_ci95_low", "loso_recovery_ci95_high", "#c45d3c", 8),
    ]
    low = min(0.0, min(float(summary[field].min()) for _, field, _, _, _, _ in metrics))
    high = max(0.55, max(float(summary[field].max()) for _, field, _, _, _, _ in metrics) * 1.08)
    width, height = 1120, 560
    parts = svg_start(width, height, "Stable Recovery Forest Plot")
    left, right, top, bottom = 335, 1060, 75, 480
    xscale = lambda value: left + (value - low) / (high - low) * (right - left)
    for reference in (0.0, 0.10, 0.25, 0.50):
        x = xscale(reference)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#999" stroke-dasharray="4 4"/>')
        svg_text(parts, x, bottom + 22, f"{reference:.0%}", 10, "middle")
    for pair_index, row in enumerate(summary.itertuples(index=False)):
        y0 = top + 38 + pair_index * 62
        svg_text(parts, left - 18, y0 + 4, PAIR_LABELS[row.pair_id], 11, "end")
        for _, field, low_field, high_field, color, offset in metrics:
            value = float(getattr(row, field)); lo = float(getattr(row, low_field)); hi = float(getattr(row, high_field)); y = y0 + offset
            parts.append(f'<line x1="{xscale(lo):.1f}" y1="{y}" x2="{xscale(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{xscale(value):.1f}" cy="{y}" r="4" fill="{color}"/>')
    for index, (label, _, _, _, color, _) in enumerate(metrics):
        x = 520 + index * 150
        parts.append(f'<circle cx="{x}" cy="535" r="5" fill="{color}"/>'); svg_text(parts, x + 10, 539, label, 11)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_concentration_curves(concentration: pd.DataFrame, path: Path) -> None:
    width, height = 1220, 620
    parts = svg_start(width, height, "Cumulative Oracle-Gain Concentration")
    colors = ["#28778e", "#48a08a", "#7a6fac", "#c45d3c", "#d49a43", "#7d8f45"]
    for panel, dataset in enumerate(("mkg_w", "db15k")):
        left, right = 80 + panel * 590, 580 + panel * 590
        top, bottom = 90, 520
        parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#aaa" stroke-dasharray="5 5"/>')
        svg_text(parts, (left + right) / 2, 75, "MKG-W" if dataset == "mkg_w" else "DB15K", 14, "middle")
        subset_ids = [pair for pair in PAIR_IDS if (pair.startswith("mkgw") == (dataset == "mkg_w"))]
        for local_index, pair_id in enumerate(subset_ids):
            group = concentration.loc[concentration.pair_id == pair_id].sort_values("descending_gain_rank")
            xs = left + group.cumulative_triple_fraction.to_numpy() * (right - left)
            ys = bottom - group.cumulative_gain_fraction.to_numpy() * (bottom - top)
            points = f"{left},{bottom} " + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            color = colors[panel * 3 + local_index]
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            svg_text(parts, left + 12, top + 22 + local_index * 18, pair_id.split("_", 1)[1], 10, fill=color)
        svg_text(parts, (left + right) / 2, bottom + 34, "Top original-triple fraction", 11, "middle")
    svg_text(parts, 18, 305, "Cumulative gain fraction", 11)
    svg_text(parts, 65, 590, "Diagonal: uniform contribution reference (not a random baseline).", 11, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_concentration_summary(summary: pd.DataFrame, path: Path) -> None:
    metrics = [("Top10 gain share", "top10_gain_share", "#28778e"), ("Q50", "q50", "#c45d3c"), ("Effective support", "effective_support", "#7d8f45")]
    width, height = 1080, 540
    parts = svg_start(width, height, "Gain Concentration Summary")
    left, right, top = 350, 1015, 85
    for reference in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + reference * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="455" stroke="#ddd"/>')
        svg_text(parts, x, 478, f"{reference:.0%}", 10, "middle")
    for pair_index, row in enumerate(summary.itertuples(index=False)):
        y0 = top + 32 + pair_index * 56
        svg_text(parts, left - 18, y0 + 4, PAIR_LABELS[row.pair_id], 11, "end")
        for metric_index, (_, field, color) in enumerate(metrics):
            value = float(getattr(row, field)); y = y0 + (metric_index - 1) * 10
            parts.append(f'<circle cx="{left + value*(right-left):.1f}" cy="{y}" r="5" fill="{color}"/>')
    for index, (label, _, color) in enumerate(metrics):
        x = 410 + index * 210
        parts.append(f'<circle cx="{x}" cy="515" r="5" fill="{color}"/>'); svg_text(parts, x + 10, 519, label, 11)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def format_ci(value: float, low: float, high: float) -> str:
    return f"{value:+.6f} [{low:+.6f}, {high:+.6f}]"


def write_report(
    summary: pd.DataFrame,
    stable: str,
    stable_evidence: dict,
    gain: str,
    gain_evidence: dict,
    final: str,
    joint_evidence: dict,
    path: Path,
    contract: dict,
) -> None:
    lines = [
        "# Experiment 6 — Stable Headroom & Gain Concentration Audit",
        "", "Date: 2026-09-07", "Split: DEV only", "Experiment 4/5 status: frozen", "",
        "## Frozen classifications", "",
        f"- Experiment 4: `{joint_evidence['experiment4_classification']}`",
        f"- Experiment 5: `{joint_evidence['experiment5_classification']}`",
        f"- Experiment 6 stability: `{stable}`",
        f"- Experiment 6 concentration: `{gain}`",
        f"- Final joint interpretation: **`{final}`**", "",
        "## Stable headroom decomposition", "",
        "| Pair | Raw Oracle | Consensus | 2-of-3 | 3-of-3 | LOSO | Frozen X4 OOF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {format_ci(row.raw_oracle_headroom,row.raw_ci95_low,row.raw_ci95_high)} "
            f"| {format_ci(row.consensus_headroom,row.consensus_ci95_low,row.consensus_ci95_high)} "
            f"| {format_ci(row.stable2_headroom,row.stable2_ci95_low,row.stable2_ci95_high)} "
            f"| {format_ci(row.stable3_headroom,row.stable3_ci95_low,row.stable3_ci95_high)} "
            f"| {format_ci(row.loso_stable_headroom,row.loso_ci95_low,row.loso_ci95_high)} "
            f"| {format_ci(row.frozen_x4_oof_gain,row.frozen_x4_ci95_low,row.frozen_x4_ci95_high)} |"
        )
    lines.extend(["", "Consensus, two-of-three, and three-of-three are ex-post all-seed upper diagnostics. LOSO holds out an independently trained seed. Frozen X4 is the inference-time observable strict-OOF probe and retains its fold-specific OOF Global baseline. These quantities are deliberately not described as equivalent policies.", "", "## Recoveries and stable opportunities", "", "| Pair | Consensus recovery | 2-of-3 recovery | 3-of-3 recovery | LOSO recovery | P(A2 non-empty) | P(A3 non-empty) | P(consensus != alpha0) |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary.itertuples(index=False):
        lines.append(f"| {PAIR_LABELS[row.pair_id]} | {row.consensus_recovery:.1%} | {row.stable2_recovery:.1%} | {row.stable3_recovery:.1%} | {row.loso_recovery:.1%} | {row.stable2_opportunity_rate:.1%} | {row.stable3_opportunity_rate:.1%} | {row.consensus_changed_rate:.1%} |")
    lines.extend(["", "## Oracle gain concentration", "", "| Pair | Top1 | Top5 | Top10 | Top20 | Q50 | Gini | Effective support |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary.itertuples(index=False):
        lines.append(f"| {PAIR_LABELS[row.pair_id]} | {row.top1_gain_share:.1%} | {row.top5_gain_share:.1%} | {row.top10_gain_share:.1%} | {row.top20_gain_share:.1%} | {row.q50:.1%} | {row.gini:.3f} | {row.effective_support:.1%} |")
    lines.extend([
        "", "The concentration curve sorts non-negative original-triple Oracle gains from largest to smallest. Its diagonal denotes uniform contribution and is not a random baseline. The Gini implementation is the formula frozen in the Experiment 6 contract.",
        "", "## Frozen gates", "",
        f"- 2-of-3 CI lower > 0: {stable_evidence['stable2_ci_lower_positive_pairs']}/6; median recovery: {stable_evidence['median_stable2_recovery']:.1%}.",
        f"- 3-of-3 CI lower > 0: {stable_evidence['stable3_ci_lower_positive_pairs']}/6; median recovery: {stable_evidence['median_stable3_recovery']:.1%}; datasets: {', '.join(stable_evidence['stable3_significant_datasets']) or 'none'}.",
        f"- Top10 >=50%: {gain_evidence['top10_share_gte_50pct_pairs']}/6; Q50 <=10%: {gain_evidence['q50_lte_10pct_pairs']}/6.",
        f"- Strong E4 intermediate under the preregistered rule: {joint_evidence['strong_e4_intermediate']}.",
        "", "## Closure interpretation", "",
        "Raw Available → Seed-Stable / Transferable → Locally Observable → Empirically Deployable is treated as a sequence of increasingly restrictive evidence. Raw Oracle is not interpreted as headroom that merely awaits a better router.",
        "", "The final interpretation is selected mechanically from the frozen Experiment 4–6 classifications. No selector, representation, policy, gate, metric, or narrative was selected from TEST.",
        "", "## Figures", "",
        "1. `figure6_1_stable_headroom_decomposition.svg`",
        "2. `figure6_2_stable_recovery_forest.svg`",
        "3. `figure6_3_oracle_gain_concentration_curves.svg`",
        "4. `figure6_4_gain_concentration_summary.svg`",
        "", "## Integrity audit", "",
        "- TEST access = 0", "- TEST commands executed = 0", "- checkpoint retraining/reselection = 0",
        "- new selector / representation = 0", "- policy tuning = 0", "- action-grid modification = 0",
        "- Experiment 4/5 modification = 0", "- original-triple clustered bootstrap intact = yes",
        "- all direct source/output hashes recorded = yes (`audit_manifest.json`; self-hash excluded)",
        "- TEST remains locked until the generated decision memo is committed", "", final,
    ])
    if lines[-1] not in FINAL_INTERPRETATIONS:
        raise AssertionError("Report must end with one frozen final interpretation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_memo(
    summary: pd.DataFrame,
    e4: str,
    e5: str,
    stable: str,
    gain: str,
    final: str,
    path: Path,
) -> None:
    lines = [
        "# Complementarity Closure Decision Memo", "",
        "Status: `FROZEN_AFTER_DEV_CLOSURE_BEFORE_TEST`", "",
        "This memo freezes the complementarity study after Experiments 4–6. Its presence alone does not unlock TEST: the commit containing this memo, the Experiment 6 outputs, and the frozen TEST scripts must exist first.", "",
        "## 1–5. Frozen decisions", "",
        f"1. Experiment 4 classification: `{e4}`",
        f"2. Experiment 5 classification: `{e5}`",
        f"3. Experiment 6 stability classification: `{stable}`",
        f"4. Experiment 6 gain-concentration classification: `{gain}`",
        f"5. Final joint interpretation: **`{final}`**", "",
        "## 5. Final research questions", "",
        "- How much heterogeneous-expert Oracle headroom remains stable across independently trained seeds?",
        "- Is stable complementarity locally identifiable from the already frozen inference-time observable space?",
        "- Is Oracle gain diffuse enough to support general adaptation, or concentrated in rare high-value triples?",
        "- How does Raw Available headroom contract through Seed-Stable/Transferable, Locally Observable, and Empirically Deployable evidence?", "",
        "## 6. Frozen primary claims", "",
        "- Raw Oracle headroom is an answer-aware upper bound and is not evidence that a better router can recover it.",
        "- Consensus and two-/three-of-three quantities are ex-post stability upper diagnostics; LOSO is held-out-seed evidence; frozen X4 is the inference-time observable strict-OOF probe.",
        "- The DEV conclusion and paper framing are fixed by the joint interpretation above and cannot be reversed in response to TEST.",
        f"- Across the six DEV pairs, median two-of-three recovery is `{summary.stable2_recovery.median():.1%}`, median three-of-three recovery is `{summary.stable3_recovery.median():.1%}`, and `{int((summary.top10_gain_share >= .5).sum())}/6` pairs place at least half of Raw Oracle gain in the top 10% of triples.", "",
        "## 7. Main figures", "",
        "1. Experiment 6 Figure 6.1 — stable headroom decomposition.",
        "2. Experiment 6 Figure 6.3 — cumulative Oracle-gain concentration.",
        "3. Experiment 4 Figure 1 — cross-seed/LOSO headroom funnel.",
        "4. Experiment 5 Figure 4 — local purity lift versus realized utility.", "",
        "## 8. Frozen TEST metrics", "",
        "For every dataset/pair, report fixed-expert MRR, DEV-locked Global-alpha MRR, Raw Oracle headroom, Consensus/2-of-3/3-of-3 headroom and recoveries, stable-opportunity rates, LOSO held-out-seed headroom/recovery, Top1/5/10/20 gain shares, Q50, Gini, and Effective Support. Use original-triple clustered 95% CIs with the frozen 10,000-replicate procedure. TEST is confirmatory and cannot redefine any DEV classification gate.", "",
        "## 9. Frozen TEST commands", "",
        "After this memo is committed, run exactly:", "",
        "```powershell",
        "Set-Location G:\\mmkg-project-research",
        ".\\scripts\\run_complementarity_closure_test.ps1 -Mode Preflight -Python python",
        ".\\scripts\\run_complementarity_closure_test.ps1 -Mode Evaluate -Python python -Device cuda",
        ".\\scripts\\run_complementarity_closure_test.ps1 -Mode Analyze -Python python",
        "```", "",
        "Do not execute these commands before the commit containing this memo and the frozen TEST scripts.", "",
        "## 10. Forbidden post-TEST changes", "",
        "- new feature or representation", "- new selector", "- action-grid change", "- gate change",
        "- metric redefinition", "- title or narrative rewritten in response to TEST", "- checkpoint reselection",
        "- hyperparameter or policy tuning", "- repeated TEST runs after inspecting results", "",
        "TEST only validates extrapolation of the frozen DEV conclusion. Any operational failure may be repaired only without changing estimands, inputs, gates, or model/checkpoint selection, and the repair must be documented before rerunning an affected incomplete command.", "",
        f"Frozen joint interpretation: `{final}`", "", "TEST_STATUS_LOCKED_PENDING_MEMO_COMMIT",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inventory(paths: list[Path], role: str) -> list[dict]:
    records = []
    for path in sorted(set(paths), key=portable_path):
        if not path.exists():
            raise FileNotFoundError(path)
        records.append({"role": role, "path": portable_path(path), "sha256": sha256_file(path)})
    return records


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_contract(contract_path)
    exp1_root, exp4_root, exp5_root = Path(args.exp1_root), Path(args.exp4_root), Path(args.exp5_root)
    exp1_audit_path, exp1_stats_path = exp1_root / "audit_manifest.json", exp1_root / "pair_statistics.csv"
    exp4_audit_path, exp4_class_path, exp4_summary_path = exp4_root / "audit_manifest.json", exp4_root / "classification.json", exp4_root / "pair_summary.csv"
    exp5_audit_path, exp5_class_path = exp5_root / "audit_manifest.json", exp5_root / "classification.json"
    required = [exp1_audit_path, exp1_stats_path, exp4_audit_path, exp4_class_path, exp4_summary_path, exp5_audit_path, exp5_class_path]
    for path in required:
        reject_test_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    exp1_audit, exp4_audit, exp4_class, exp5_audit, exp5_class = map(load_json, [exp1_audit_path, exp4_audit_path, exp4_class_path, exp5_audit_path, exp5_class_path])
    if exp1_audit.get("gate", {}).get("decision") != "GO" or int(exp1_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 1 is not a clean frozen DEV input")
    if exp4_audit.get("split") != "dev" or int(exp4_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 4 is not a clean frozen DEV input")
    if exp5_audit.get("split") != "dev" or int(exp5_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 5 is not a clean frozen DEV input")
    if exp4_class.get("classification") not in ("E4_SUBSTANTIALLY_TRANSFERABLE", "E4_RESIDUAL_DOMINATED", "E4_INTERMEDIATE"):
        raise RuntimeError("Experiment 4 classification is invalid")
    if exp5_class.get("final_classification") not in ("E5_LOCAL_IDENTIFIABLE", "E5_LOCAL_AMBIGUITY", "E5_X6_RECOVERY", "E5_INTERMEDIATE"):
        raise RuntimeError("Experiment 5 classification is invalid")
    exp1_stats, exp4_summary = pd.read_csv(exp1_stats_path), pd.read_csv(exp4_summary_path)
    if set(exp1_stats.pair_id) != set(PAIR_IDS) or set(exp4_summary.pair_id) != set(PAIR_IDS):
        raise RuntimeError("Frozen pair inventory mismatch")

    sources = [
        contract_path, Path("docs/protocols/EXP6_STABLE_HEADROOM_PROTOCOL.md"), Path(__file__),
        Path("scripts/audit_cross_seed_transfer.py"), Path("scripts/exp2_information_common.py"),
        Path("scripts/run_exp6_stable_headroom.ps1"),
        Path("scripts/run_complementarity_closure_test.ps1"),
        Path("scripts/audit_complementarity_closure_test.py"),
        exp1_audit_path, exp1_stats_path,
        exp4_audit_path, exp4_class_path, exp4_summary_path, exp5_audit_path, exp5_class_path,
    ]
    loaded = []
    preflight = []
    for pair_id in PAIR_IDS:
        frame, rr, alpha0, manifest_path, source_path, _ = load_pair(pair_id, Path(args.utility_manifest_dir), exp1_stats)
        sources.extend([manifest_path, source_path])
        preflight.append({
            "pair_id": pair_id, "rows": int(len(frame)), "query_identities": int(len(frame) // 3),
            "original_triples": int(frame.original_triple_id.nunique()), "alpha0": alpha0,
            "canonical_source": portable_path(source_path), "raw_reproduction_checked": not args.dry_run,
        })
        loaded.append((pair_id, frame, rr, alpha0))
    if args.dry_run:
        print(json.dumps({
            "status": "preflight_ok", "split": "dev", "pairs": preflight,
            "experiment4_classification": exp4_class["classification"],
            "experiment5_classification": exp5_class["final_classification"],
            "test_access": 0, "checkpoint_execution": 0,
        }, indent=2))
        return

    output_dir, report_path, memo_path = Path(args.output_dir), Path(args.report), Path(args.decision_memo)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite: {output_dir}")
    for path in (report_path, memo_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query_frames, concentration_frames, summaries, bootstrap_rows = [], [], [], []
    for pair_id, frame, rr, alpha0 in loaded:
        per_query, concentration, summary, boot = analyze_pair(
            pair_id, frame, rr, alpha0,
            exp1_stats.loc[exp1_stats.pair_id == pair_id].iloc[0],
            exp4_summary.loc[exp4_summary.pair_id == pair_id].iloc[0], contract,
        )
        per_query_frames.append(per_query); concentration_frames.append(concentration); summaries.append(summary); bootstrap_rows.extend(boot)
        print(f"[OK] {pair_id}: raw={summary['raw_oracle_headroom']:.6f}, stable2={summary['stable2_headroom']:.6f}, stable3={summary['stable3_headroom']:.6f}, top10={summary['top10_gain_share']:.1%}")
    per_query = pd.concat(per_query_frames, ignore_index=True)
    concentration = pd.concat(concentration_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    bootstrap = pd.DataFrame(bootstrap_rows)
    stable, stable_evidence = classify_stability(summary, contract)
    gain, gain_evidence = classify_concentration(summary, contract)
    final, joint_evidence = classify_joint(exp4_class, exp4_summary, exp5_class, stable, gain, contract)
    summary["stable2_ci_lower_gt_zero"] = summary.stable2_ci95_low > 0
    summary["stable3_ci_lower_gt_zero"] = summary.stable3_ci95_low > 0

    per_query_path = output_dir / "stable_headroom_per_query.csv.gz"
    stable_summary_path = output_dir / "stable_headroom_pair_summary.csv"
    concentration_path = output_dir / "concentration_per_triple.csv.gz"
    concentration_summary_path = output_dir / "concentration_pair_summary.csv"
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    final_path = output_dir / "final_joint_interpretation.json"
    figure_paths = [
        output_dir / "figure6_1_stable_headroom_decomposition.svg",
        output_dir / "figure6_2_stable_recovery_forest.svg",
        output_dir / "figure6_3_oracle_gain_concentration_curves.svg",
        output_dir / "figure6_4_gain_concentration_summary.svg",
    ]
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    per_query.to_csv(per_query_path, index=False, compression=compression, lineterminator="\n")
    concentration.to_csv(concentration_path, index=False, compression=compression, lineterminator="\n")
    stable_columns = [column for column in summary.columns if not column.startswith(("top1_", "top5_", "top10_", "top20_", "q50", "gini", "effective_support", "total_gain"))]
    concentration_columns = ["dataset", "pair_id", "pair_label", "n_original_triples", "raw_oracle_headroom", "total_gain", "top1_gain_share", "top1_triple_count", "top5_gain_share", "top5_triple_count", "top10_gain_share", "top10_triple_count", "top20_gain_share", "top20_triple_count", "q50", "q50_triple_count", "gini", "effective_support"]
    summary[stable_columns].to_csv(stable_summary_path, index=False, lineterminator="\n")
    summary[concentration_columns].to_csv(concentration_summary_path, index=False, lineterminator="\n")
    bootstrap.to_csv(bootstrap_path, index=False, lineterminator="\n")
    final_payload = {
        "experiment4_classification": exp4_class["classification"],
        "experiment5_classification": exp5_class["final_classification"],
        "stable_classification": stable, "stable_evidence": stable_evidence,
        "gain_concentration_classification": gain, "gain_evidence": gain_evidence,
        "final_joint_interpretation": final, "joint_evidence": joint_evidence,
        "test_status": "LOCKED_PENDING_DECISION_MEMO_COMMIT",
    }
    final_path.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
    write_headroom_figure(summary, figure_paths[0]); write_recovery_figure(summary, figure_paths[1])
    write_concentration_curves(concentration, figure_paths[2]); write_concentration_summary(summary, figure_paths[3])
    write_report(summary, stable, stable_evidence, gain, gain_evidence, final, joint_evidence, report_path, contract)
    write_decision_memo(summary, exp4_class["classification"], exp5_class["final_classification"], stable, gain, final, memo_path)

    output_paths = [per_query_path, stable_summary_path, concentration_path, concentration_summary_path, bootstrap_path, final_path, *figure_paths, report_path, memo_path]
    audit = {
        "schema_version": 1, "experiment": contract["experiment"], "split": "dev",
        "stable_classification": stable, "gain_concentration_classification": gain,
        "final_joint_interpretation": final, "preflight": preflight,
        "sources_and_outputs": [*inventory(sources, "source"), *inventory(output_paths, "output")],
        "hash_inventory_note": "audit_manifest.json self-hash excluded to avoid recursive content",
        "operational_audit": {
            "test_access": 0, "test_commands_executed": 0, "checkpoint_retraining": 0,
            "checkpoint_reselection": 0, "new_selector": 0, "new_representation": 0,
            "policy_tuning": 0, "action_grid_modified": False, "experiment_4_5_modified": False,
            "original_triple_bootstrap_intact": True,
        },
        "decision_memo_generated": True,
        "test_status": "LOCKED_PENDING_DECISION_MEMO_COMMIT",
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {stable}; {gain}; {final}")
    print(f"[LOCKED] TEST remains prohibited until {memo_path} is committed")


if __name__ == "__main__":
    main()
