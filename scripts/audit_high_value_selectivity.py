from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_cross_seed_transfer import PAIR_LABELS
from scripts.exp2_information_common import (
    ALPHAS,
    PAIR_IDS,
    RR_COLUMNS,
    portable_path,
    reject_test_path,
    select_probe_actions,
    sha256_file,
)


METRICS = (
    "selected_mean_realized_utility",
    "selective_population_gain",
    "positive_transfer_rate",
    "negative_transfer_rate",
    "changed_rate",
    "stable2_opportunity_rate",
    "stable2_opportunity_enrichment",
    "stable2_gain_capture",
    "stable3_opportunity_rate",
    "stable3_opportunity_enrichment",
    "stable3_gain_capture",
    "consensus_gain_capture",
    "high_gain_precision",
    "high_gain_recall",
    "high_gain_enrichment",
    "raw_oracle_gain_capture",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen DEV-only high-value selectivity audit.")
    parser.add_argument("--contract", default="docs/protocols/FROZEN_HIGH_VALUE_SELECTIVITY_AUDIT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/part2_selectivity_audit")
    parser.add_argument("--report", default="docs/reports/high_value_opportunity_selectivity_audit_2026-09-07.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def load_contract(path: Path) -> dict:
    reject_test_path(path)
    contract = load_json(path)
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("Selectivity contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("Pair inventory changed")
    if not np.allclose(contract["input_contract"]["alpha_grid"], ALPHAS):
        raise RuntimeError("Alpha grid changed")
    if tuple(contract["ranking_rule"]["coverage_fractions"]) != (0.01, 0.05, 0.10, 0.20, 0.30):
        raise RuntimeError("Coverage ladder changed")
    if tuple(contract.get("metrics", [])) != METRICS:
        raise RuntimeError("Metric inventory changed")
    if contract["ranking_rule"]["primary_scalar"] != "predicted_u[row, chosen_action_index[row]]":
        raise RuntimeError("Primary ranking scalar changed")
    if contract["analysis_unit"]["cluster"] != "original_triple_id":
        raise RuntimeError("Bootstrap cluster changed")
    if any(int(value) != 0 for value in contract["prohibited"].values()):
        raise RuntimeError("A prohibited operation was enabled")
    return contract


def verify_frozen_sources(contract: dict) -> list[dict]:
    records = []
    for raw_path, expected in sorted(contract["source_hashes"].items()):
        path = Path(raw_path)
        reject_test_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
        records.append({"path": portable_path(path), "sha256": actual, "bytes": path.stat().st_size, "role": "frozen_source"})
    return records


def derived_seed(base: int, pair_id: str, coverage: float, purpose: str) -> int:
    token = f"{base}|{pair_id}|{coverage:.6f}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan), where=denominator != 0)


def top_selection(score: np.ndarray, query_id: np.ndarray, fraction: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    query_id = np.asarray(query_id, dtype=str)
    if score.ndim != 1 or len(score) != len(query_id) or not np.isfinite(score).all():
        raise ValueError("Invalid selectivity score")
    count = max(1, int(math.ceil(fraction * len(score))))
    order = np.lexsort((query_id, -score))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:count]] = True
    return selected


def metric_components(frame: pd.DataFrame, selected: np.ndarray, tolerance: float) -> dict[str, np.ndarray]:
    selected = np.asarray(selected, dtype=bool)
    gain = frame.realized_utility.to_numpy(np.float64)
    changed = frame.changed.to_numpy(np.float64)
    stable2 = frame.stable2_nonempty.to_numpy(np.float64)
    stable3 = frame.stable3_nonempty.to_numpy(np.float64)
    stable2_gain = frame.stable2_gain.to_numpy(np.float64)
    stable3_gain = frame.stable3_gain.to_numpy(np.float64)
    consensus_gain = frame.consensus_gain.to_numpy(np.float64)
    high = frame.high_gain_top10.to_numpy(np.float64)
    raw = frame.raw_oracle_gain.to_numpy(np.float64)
    sel = selected.astype(np.float64)
    return {
        "n": np.ones(len(frame), dtype=np.float64),
        "sel_n": sel,
        "sel_u": sel * gain,
        "sel_pos": sel * (gain > tolerance),
        "sel_neg": sel * (gain < -tolerance),
        "sel_changed": sel * changed,
        "all_s2": stable2,
        "sel_s2": sel * stable2,
        "all_s2_gain": stable2_gain,
        "sel_s2_gain": sel * stable2_gain,
        "all_s3": stable3,
        "sel_s3": sel * stable3,
        "all_s3_gain": stable3_gain,
        "sel_s3_gain": sel * stable3_gain,
        "all_consensus_gain": consensus_gain,
        "sel_consensus_gain": sel * consensus_gain,
        "all_high": high,
        "sel_high": sel * high,
        "all_raw": raw,
        "sel_raw": sel * raw,
    }


def metrics_from_sums(sums: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sel_n, n = sums["sel_n"], sums["n"]
    s2_rate = safe_divide(sums["sel_s2"], sel_n)
    s3_rate = safe_divide(sums["sel_s3"], sel_n)
    high_precision = safe_divide(sums["sel_high"], sel_n)
    return {
        "selected_mean_realized_utility": safe_divide(sums["sel_u"], sel_n),
        "selective_population_gain": safe_divide(sums["sel_u"], n),
        "positive_transfer_rate": safe_divide(sums["sel_pos"], sel_n),
        "negative_transfer_rate": safe_divide(sums["sel_neg"], sel_n),
        "changed_rate": safe_divide(sums["sel_changed"], sel_n),
        "stable2_opportunity_rate": s2_rate,
        "stable2_opportunity_enrichment": safe_divide(s2_rate, safe_divide(sums["all_s2"], n)),
        "stable2_gain_capture": safe_divide(sums["sel_s2_gain"], sums["all_s2_gain"]),
        "stable3_opportunity_rate": s3_rate,
        "stable3_opportunity_enrichment": safe_divide(s3_rate, safe_divide(sums["all_s3"], n)),
        "stable3_gain_capture": safe_divide(sums["sel_s3_gain"], sums["all_s3_gain"]),
        "consensus_gain_capture": safe_divide(sums["sel_consensus_gain"], sums["all_consensus_gain"]),
        "high_gain_precision": high_precision,
        "high_gain_recall": safe_divide(sums["sel_high"], sums["all_high"]),
        "high_gain_enrichment": safe_divide(high_precision, safe_divide(sums["all_high"], n)),
        "raw_oracle_gain_capture": safe_divide(sums["sel_raw"], sums["all_raw"]),
    }


def point_metrics(frame: pd.DataFrame, selected: np.ndarray, tolerance: float) -> dict[str, float]:
    components = metric_components(frame, selected, tolerance)
    sums = {name: np.asarray(value.sum(), dtype=np.float64) for name, value in components.items()}
    return {name: float(value) for name, value in metrics_from_sums(sums).items()}


def clustered_bootstrap(
    frame: pd.DataFrame,
    selections: dict[float, np.ndarray],
    samples: int,
    seed: int,
    tolerance: float,
) -> dict[float, dict[str, tuple[float, float]]]:
    groups, unique = pd.factorize(frame.original_triple_id, sort=True)
    n_groups = len(unique)
    first = next(iter(selections.values()))
    base = metric_components(frame, first, tolerance)
    base_names = ("n", "all_s2", "all_s2_gain", "all_s3", "all_s3_gain", "all_consensus_gain", "all_high", "all_raw")
    selected_names = (
        "sel_n", "sel_u", "sel_pos", "sel_neg", "sel_changed", "sel_s2", "sel_s2_gain",
        "sel_s3", "sel_s3_gain", "sel_consensus_gain", "sel_high", "sel_raw",
    )
    columns = [np.bincount(groups, weights=base[name], minlength=n_groups) for name in base_names]
    base_offsets = {name: index for index, name in enumerate(base_names)}
    offsets: dict[float, dict[str, int]] = {}
    for coverage, selected in selections.items():
        components = metric_components(frame, selected, tolerance)
        offsets[coverage] = dict(base_offsets)
        for name in selected_names:
            offsets[coverage][name] = len(columns)
            columns.append(np.bincount(groups, weights=components[name], minlength=n_groups))
    cluster_matrix = np.column_stack(columns).astype(np.float32)
    distributions = {coverage: {metric: np.empty(samples, dtype=np.float64) for metric in METRICS} for coverage in selections}
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_groups, 1.0 / n_groups, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        weights = rng.multinomial(n_groups, probabilities, size=stop - start).astype(np.float32)
        totals = (weights @ cluster_matrix).astype(np.float64)
        for coverage in selections:
            sums = {name: totals[:, index] for name, index in offsets[coverage].items()}
            values = metrics_from_sums(sums)
            for metric, result in values.items():
                distributions[coverage][metric][start:stop] = result
    return {
        coverage: {
            metric: tuple(float(value) for value in np.nanpercentile(values, [2.5, 97.5]))
            for metric, values in metrics.items()
        }
        for coverage, metrics in distributions.items()
    }


def matched_random_baselines(
    frame: pd.DataFrame,
    selections: dict[float, np.ndarray],
    samples: int,
    seed: int,
    tolerance: float,
) -> dict[float, dict[str, tuple[float, float, float]]]:
    work = frame[["seed", "direction", "relation_id"]].copy()
    work["row_index"] = np.arange(len(work))
    selection_columns = []
    for index, (coverage, selected) in enumerate(selections.items()):
        name = f"selected_{index}"
        work[name] = np.asarray(selected, dtype=np.int8)
        selection_columns.append((coverage, name))
    strata = []
    for _, group in work.groupby(["seed", "direction", "relation_id"], sort=True):
        indices = group.row_index.to_numpy(np.int64)
        counts = {coverage: int(group[name].sum()) for coverage, name in selection_columns}
        if any(counts.values()):
            strata.append((indices, counts))
    rng = np.random.default_rng(seed)
    distributions = {
        coverage: {metric: np.empty(samples, dtype=np.float64) for metric in METRICS}
        for coverage in selections
    }
    gain = frame.realized_utility.to_numpy(np.float64)
    selected_values = np.column_stack((
        gain,
        gain > tolerance,
        gain < -tolerance,
        frame.changed.to_numpy(np.float64),
        frame.stable2_nonempty.to_numpy(np.float64),
        frame.stable2_gain.to_numpy(np.float64),
        frame.stable3_nonempty.to_numpy(np.float64),
        frame.stable3_gain.to_numpy(np.float64),
        frame.consensus_gain.to_numpy(np.float64),
        frame.high_gain_top10.to_numpy(np.float64),
        frame.raw_oracle_gain.to_numpy(np.float64),
    ))
    fixed = {
        "n": float(len(frame)),
        "all_s2": float(frame.stable2_nonempty.sum()),
        "all_s2_gain": float(frame.stable2_gain.sum()),
        "all_s3": float(frame.stable3_nonempty.sum()),
        "all_s3_gain": float(frame.stable3_gain.sum()),
        "all_consensus_gain": float(frame.consensus_gain.sum()),
        "all_high": float(frame.high_gain_top10.sum()),
        "all_raw": float(frame.raw_oracle_gain.sum()),
    }
    for repetition in range(samples):
        chosen = {coverage: [] for coverage in selections}
        for indices, counts in strata:
            permutation = rng.permutation(indices)
            for coverage, count in counts.items():
                if count:
                    chosen[coverage].append(permutation[:count])
        for coverage in selections:
            chosen_indices = np.concatenate(chosen[coverage])
            totals = selected_values[chosen_indices].sum(axis=0)
            sums = dict(fixed)
            sums.update({
                "sel_n": float(len(chosen_indices)), "sel_u": totals[0], "sel_pos": totals[1],
                "sel_neg": totals[2], "sel_changed": totals[3], "sel_s2": totals[4],
                "sel_s2_gain": totals[5], "sel_s3": totals[6], "sel_s3_gain": totals[7],
                "sel_consensus_gain": totals[8], "sel_high": totals[9], "sel_raw": totals[10],
            })
            values = {name: float(value) for name, value in metrics_from_sums(sums).items()}
            for metric, value in values.items():
                distributions[coverage][metric][repetition] = value
    return {
        coverage: {
            metric: (
                float(np.nanmean(values)),
                float(np.nanpercentile(values, 2.5)),
                float(np.nanpercentile(values, 97.5)),
            )
            for metric, values in metrics.items()
        }
        for coverage, metrics in distributions.items()
    }


def load_pair(pair_id: str, contract: dict) -> tuple[pd.DataFrame, list[dict]]:
    inputs = contract["input_contract"]
    prediction_path = Path(inputs["probe_path_template"].format(pair_id=pair_id))
    asset_path = Path(inputs["query_asset_path_template"].format(pair_id=pair_id))
    utility_manifest_path = Path(inputs["utility_manifest_path_template"].format(pair_id=pair_id))
    for path in (prediction_path, asset_path, utility_manifest_path):
        reject_test_path(path)
    with np.load(prediction_path, allow_pickle=False) as prediction, np.load(asset_path, allow_pickle=False) as asset:
        for field in inputs["required_prediction_fields"]:
            if field not in prediction.files:
                raise RuntimeError(f"Missing prediction field {field}: {prediction_path}")
        for field in inputs["required_query_fields"]:
            if field not in asset.files:
                raise RuntimeError(f"Missing query field {field}: {asset_path}")
        query_id = asset["query_id"].astype(str)
        if not np.array_equal(query_id, prediction["query_id"].astype(str)) or len(np.unique(query_id)) != len(query_id):
            raise RuntimeError(f"X4/query asset alignment failed: {pair_id}")
        predicted_u = prediction["predicted_u"].astype(np.float64)
        global_index = prediction["fold_global_index"].astype(np.int16)
        chosen = prediction["chosen_action_index"].astype(np.int16)
        if predicted_u.shape != (len(query_id), len(ALPHAS)):
            raise RuntimeError(f"Prediction action grid failed: {pair_id}")
        if not np.isfinite(predicted_u).all() or np.max(np.abs(predicted_u[np.arange(len(query_id)), global_index])) != 0.0:
            raise RuntimeError(f"Global prediction is not exactly zero: {pair_id}")
        reproduced = np.empty_like(chosen)
        for index in np.unique(global_index):
            mask = global_index == index
            reproduced[mask] = select_probe_actions(predicted_u[mask], int(index))
        if not np.array_equal(reproduced, chosen):
            raise RuntimeError(f"Frozen chosen action cannot be reproduced: {pair_id}")
        frame = pd.DataFrame({
            "query_id": query_id,
            "outer_fold": prediction["outer_fold"].astype(np.int16),
            "fold_global_index": global_index,
            "chosen_action_index": chosen,
            "seed": asset["seed"].astype(np.int16),
            "direction": asset["direction"].astype(str),
            "head_id": asset["head_id"].astype(np.int64),
            "relation_id": asset["relation_id"].astype(np.int64),
            "tail_id": asset["tail_id"].astype(np.int64),
            "selectivity_score": predicted_u[np.arange(len(query_id)), chosen],
        })
    frame["original_triple_id"] = "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    frame["changed"] = (frame.chosen_action_index != frame.fold_global_index).astype(np.int8)
    frame["chosen_alpha"] = ALPHAS[frame.chosen_action_index]
    frame["fold_global_alpha"] = ALPHAS[frame.fold_global_index]
    if set(frame.seed.unique()) != set(inputs["required_seeds"]) or set(frame.direction.unique()) != set(inputs["required_directions"]):
        raise RuntimeError(f"Seed/direction inventory failed: {pair_id}")
    if not frame.groupby("original_triple_id").size().eq(6).all():
        raise RuntimeError(f"Every triple must retain both directions and all three seeds: {pair_id}")

    utility_manifest = load_json(utility_manifest_path)
    query_source = utility_manifest.get("source_query_rows", {})
    query_path = Path(query_source.get("path", ""))
    reject_test_path(query_path)
    if not query_path.is_file() or sha256_file(query_path) != query_source.get("sha256"):
        raise RuntimeError(f"Exact DEV query-row source failed hash verification: {pair_id}")
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS])
    if exact.query_id.duplicated().any():
        raise RuntimeError(f"Duplicate exact DEV query IDs: {pair_id}")
    exact = exact.set_index("query_id").reindex(frame.query_id)
    if exact.isna().any().any():
        raise RuntimeError(f"Missing exact DEV RR rows: {pair_id}")
    rr = exact.to_numpy(np.float64)
    rows = np.arange(len(frame))
    frame["realized_utility"] = rr[rows, frame.chosen_action_index] - rr[rows, frame.fold_global_index]

    stable_path = Path(inputs["stable_query_path"])
    concentration_path = Path(inputs["concentration_path"])
    stable = pd.read_csv(stable_path)
    stable = stable.loc[stable.pair_id == pair_id, [
        "original_triple_id", "direction", "stable2_nonempty", "stable2_gain",
        "stable3_nonempty", "stable3_gain", "consensus_gain", "raw_oracle_gain",
    ]]
    if stable.duplicated(["original_triple_id", "direction"]).any():
        raise RuntimeError(f"Duplicate Experiment 6 stable identities: {pair_id}")
    frame = frame.merge(stable, on=["original_triple_id", "direction"], how="left", validate="many_to_one")
    concentration = pd.read_csv(concentration_path)
    concentration = concentration.loc[concentration.pair_id == pair_id, [
        "original_triple_id", "oracle_gain", "descending_gain_rank"
    ]].rename(columns={"oracle_gain": "triple_oracle_gain"})
    if concentration.original_triple_id.duplicated().any():
        raise RuntimeError(f"Duplicate Experiment 6 concentration triples: {pair_id}")
    n_triples = len(concentration)
    top_count = max(1, int(math.ceil(0.10 * n_triples)))
    concentration["high_gain_top10"] = (concentration.descending_gain_rank <= top_count).astype(np.int8)
    frame = frame.merge(concentration, on="original_triple_id", how="left", validate="many_to_one")
    required = ["stable2_gain", "stable3_gain", "consensus_gain", "raw_oracle_gain", "triple_oracle_gain"]
    if frame[required].isna().any().any():
        raise RuntimeError(f"Experiment 6 join failed: {pair_id}")
    reproduced_triple_gain = frame.groupby("original_triple_id", sort=True).raw_oracle_gain.mean()
    declared_triple_gain = (
        frame.drop_duplicates("original_triple_id")
        .set_index("original_triple_id")
        .loc[reproduced_triple_gain.index, "triple_oracle_gain"]
    )
    if not np.allclose(reproduced_triple_gain, declared_triple_gain, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Experiment 6 triple-gain identity failed: {pair_id}")
    source_records = [{
        "path": portable_path(query_path), "sha256": query_source["sha256"],
        "bytes": query_path.stat().st_size, "role": "exact_dev_query_rows",
    }]
    frame.insert(0, "dataset", "mkg_w" if pair_id.startswith("mkgw_") else "db15k")
    frame.insert(1, "pair_id", pair_id)
    return frame, source_records


def analyze_pair(pair_id: str, contract: dict) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    frame, extra_sources = load_pair(pair_id, contract)
    tolerance = float(contract["targets"]["zero_tolerance"])
    coverages = tuple(float(value) for value in contract["ranking_rule"]["coverage_fractions"])
    selections = {coverage: top_selection(frame.selectivity_score.to_numpy(), frame.query_id.to_numpy(), coverage) for coverage in coverages}
    for coverage, selected in selections.items():
        frame[f"selected_top{int(round(coverage * 100))}"] = selected.astype(np.int8)
    bootstrap_cfg = contract["clustered_bootstrap"]
    print(f"[bootstrap] {pair_id}", flush=True)
    intervals = clustered_bootstrap(
        frame, selections, int(bootstrap_cfg["samples"]),
        derived_seed(int(bootstrap_cfg["seed"]), pair_id, 0.0, "bootstrap"), tolerance,
    )
    summary_rows, bootstrap_rows, random_rows = [], [], []
    random_cfg = contract["matched_random_baseline"]
    print(f"[matched-random] {pair_id}", flush=True)
    random_by_coverage = matched_random_baselines(
        frame, selections, int(random_cfg["samples"]),
        derived_seed(int(random_cfg["seed"]), pair_id, 0.0, "matched_random"), tolerance,
    )
    for coverage, selected in selections.items():
        observed = point_metrics(frame, selected, tolerance)
        random = random_by_coverage[coverage]
        row = {
            "dataset": frame.dataset.iloc[0], "pair_id": pair_id, "pair_label": PAIR_LABELS[pair_id],
            "coverage_target": coverage, "selected_count": int(selected.sum()),
            "total_count": int(len(frame)), "actual_coverage": float(selected.mean()),
            "score_threshold": float(frame.loc[selected, "selectivity_score"].min()),
        }
        for metric in METRICS:
            row[metric] = observed[metric]
            row[f"{metric}_ci95_low"] = intervals[coverage][metric][0]
            row[f"{metric}_ci95_high"] = intervals[coverage][metric][1]
            row[f"{metric}_random_mean"] = random[metric][0]
            row[f"{metric}_minus_random"] = observed[metric] - random[metric][0]
            bootstrap_rows.append({
                "dataset": frame.dataset.iloc[0], "pair_id": pair_id, "coverage_target": coverage,
                "metric": metric, "estimate": observed[metric],
                "ci95_low": intervals[coverage][metric][0], "ci95_high": intervals[coverage][metric][1],
                "bootstrap_samples": int(bootstrap_cfg["samples"]), "bootstrap_unit": "original_triple_id",
            })
            random_rows.append({
                "dataset": frame.dataset.iloc[0], "pair_id": pair_id, "coverage_target": coverage,
                "metric": metric, "observed": observed[metric], "random_mean": random[metric][0],
                "random_ci95_low": random[metric][1], "random_ci95_high": random[metric][2],
                "observed_minus_random_mean": observed[metric] - random[metric][0],
                "random_samples": int(random_cfg["samples"]),
                "matching_strata": "seed|direction|relation_id",
            })
        summary_rows.append(row)
    return frame, summary_rows, bootstrap_rows, random_rows + extra_sources


def svg_start(title: str, width: int = 1320, height: int = 740) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="38" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{html.escape(title)}</text>',
    ]


def write_coverage_figure(summary: pd.DataFrame, metric: str, ylabel: str, path: Path, reference: float | None = None) -> None:
    width, height = 1320, 740
    parts = svg_start(ylabel, width, height)
    pairs = list(PAIR_IDS)
    colors = ("#246a73", "#d17a22", "#7b5ea7", "#3a86a8", "#c8553d", "#6a994e")
    values = list(summary[metric].astype(float)) + list(summary[f"{metric}_random_mean"].astype(float))
    if reference is not None:
        values.append(reference)
    ymin, ymax = min(values + [0.0]), max(values + [0.0])
    pad = max((ymax - ymin) * 0.12, 1e-6)
    ymin, ymax = ymin - pad, ymax + pad
    left, right, top, bottom = 95, 1280, 82, 650
    x = lambda value: left + (value - 0.01) / 0.29 * (right - left)
    y = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#555"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#555"/>')
    if reference is not None:
        parts.append(f'<line x1="{left}" y1="{y(reference):.1f}" x2="{right}" y2="{y(reference):.1f}" stroke="#777" stroke-dasharray="5 5"/>')
    for tick in (0.01, 0.05, 0.10, 0.20, 0.30):
        parts.append(f'<text x="{x(tick):.1f}" y="{bottom + 22}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.0%}</text>')
    for index, pair_id in enumerate(pairs):
        block = summary.loc[summary.pair_id == pair_id].sort_values("coverage_target")
        points = " ".join(f'{x(float(row.coverage_target)):.1f},{y(float(row[metric])):.1f}' for _, row in block.iterrows())
        random_points = " ".join(f'{x(float(row.coverage_target)):.1f},{y(float(row[f"{metric}_random_mean"])):.1f}' for _, row in block.iterrows())
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="2.4"/>')
        parts.append(f'<polyline points="{random_points}" fill="none" stroke="{colors[index]}" stroke-width="1.2" stroke-dasharray="5 4" opacity="0.65"/>')
        for _, row in block.iterrows():
            parts.append(f'<circle cx="{x(float(row.coverage_target)):.1f}" cy="{y(float(row[metric])):.1f}" r="3.2" fill="{colors[index]}"/>')
        legend_y = 92 + index * 22
        parts.append(f'<line x1="930" y1="{legend_y}" x2="955" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>')
        parts.append(f'<text x="962" y="{legend_y + 4}" font-family="Arial" font-size="10">{html.escape(PAIR_LABELS[pair_id])}</text>')
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 52}" text-anchor="middle" font-family="Arial" font-size="12">Selected coverage (solid=frozen X4 ranking; dashed=matched random mean)</text>')
    for tick in np.linspace(ymin, ymax, 6):
        parts.append(f'<text x="{left - 10}" y="{y(float(tick)) + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{tick:.3f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(summary: pd.DataFrame, report: Path, output_dir: Path) -> None:
    primary = summary.loc[np.isclose(summary.coverage_target, 0.10)].copy()
    lines = [
        "# Frozen High-Value Opportunity Selectivity Audit",
        "",
        "Date: 2026-09-07",
        "",
        "## Scope and frozen interpretation boundary",
        "",
        "This DEV-only audit asks whether the already-frozen Experiment 2 X4 strict-OOF score ranks the seed-stable and high-Oracle-gain opportunities diagnosed by Experiment 6. It trains no model, introduces no feature or representation, and cannot change the frozen closure route.",
        "",
        "The primary ranking scalar is the predicted advantage of the action already chosen by the X4 probe. Unselected rows fall back to their fold-specific Global action. Coverage is selected within each dataset/expert pair, with lexicographic query-ID tie-breaking.",
        "",
        "## Primary 10% coverage results",
        "",
        "| Pair | Realized gain (population) | Selected utility | Stable-2 enrich X4/random | Stable-2 gain capture | High-gain enrich X4/random | High-gain recall | Negative transfer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {row.pair_label} | {row.selective_population_gain:+.6f} "
            f"[{row.selective_population_gain_ci95_low:+.6f}, {row.selective_population_gain_ci95_high:+.6f}] | "
            f"{row.selected_mean_realized_utility:+.6f} | {row.stable2_opportunity_enrichment:.2f}×/{row.stable2_opportunity_enrichment_random_mean:.2f}× | "
            f"{row.stable2_gain_capture:.1%} | {row.high_gain_enrichment:.2f}×/{row.high_gain_enrichment_random_mean:.2f}× | "
            f"{row.high_gain_recall:.1%} | {row.negative_transfer_rate:.1%} |"
        )
    stable_enriched = int((primary.stable2_opportunity_enrichment_minus_random > 0).sum())
    high_enriched = int((primary.high_gain_enrichment_minus_random > 0).sum())
    positive_gain = int((primary.selective_population_gain > 0).sum())
    ci_positive = int((primary.selective_population_gain_ci95_low > 0).sum())
    lines += [
        "",
        "## Descriptive synthesis",
        "",
        f"At 10% coverage, frozen X4 has positive selective population gain in {positive_gain}/6 pairs, with clustered CI lower above zero in {ci_positive}/6. Stable-2 enrichment exceeds its matched random mean in {stable_enriched}/6 pairs, and high-gain enrichment exceeds matched random in {high_enriched}/6 pairs.",
        "",
        "| Coverage | Positive gain | CI lower > 0 | Stable-2 enrich > random mean | High-gain enrich > random mean | Median negative transfer |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for coverage, block in summary.groupby("coverage_target", sort=True):
        lines.append(
            f"| {coverage:.0%} | {int((block.selective_population_gain > 0).sum())}/6 | "
            f"{int((block.selective_population_gain_ci95_low > 0).sum())}/6 | "
            f"{int((block.stable2_opportunity_enrichment_minus_random > 0).sum())}/6 | "
            f"{int((block.high_gain_enrichment_minus_random > 0).sum())}/6 | "
            f"{block.negative_transfer_rate.median():.1%} |"
        )
    lines += [
        "",
        "The matched-random comparison preserves the selected count within every seed × direction × relation stratum. Statements of `> random` compare with its Monte Carlo mean; they are descriptive and are not a newly introduced significance gate.",
        "",
        "The score therefore carries useful ranking information, but it is not a clean opportunity detector: median negative transfer remains about one quarter across the coverage ladder, and high-gain enrichment over matched random weakens as coverage expands.",
        "",
        "These comparisons are diagnostics of a previously frozen score, not a newly developed selective policy. They do not reopen selector development or alter `FINAL_SELECTIVE_RARE_OPPORTUNITY`.",
        "",
        "## Figures",
        "",
        f"1. [Selective realized gain]({os.path.relpath(output_dir / 'figure_p2_1_selective_gain.svg', report.parent).replace(os.sep, '/')})",
        f"2. [Stable-opportunity enrichment]({os.path.relpath(output_dir / 'figure_p2_2_stable2_enrichment.svg', report.parent).replace(os.sep, '/')})",
        f"3. [High-gain enrichment]({os.path.relpath(output_dir / 'figure_p2_3_high_gain_enrichment.svg', report.parent).replace(os.sep, '/')})",
        "",
        "## Machine-readable outputs",
        "",
        "- `selectivity_per_query.csv.gz`: frozen row scores, exact realized utilities, stable/high-gain targets, and coverage indicators.",
        "- `coverage_metrics.csv`: pair × coverage estimates, clustered intervals, and matched-random differences.",
        "- `bootstrap_ci.csv`: long-form original-triple clustered intervals.",
        "- `matched_random_baseline.csv`: long-form matched-random summaries.",
        "- `audit_manifest.json`: complete source/output hash inventory and operational audit.",
        "",
        "## Operational audit",
        "",
        "- TEST access = 0",
        "- new ranker / selector = 0",
        "- new representation / feature = 0",
        "- checkpoint retraining / reselection = 0",
        "- policy tuning = 0",
        "- closure route change = 0",
        "",
        "Frozen closure route retained: `FINAL_SELECTIVE_RARE_OPPORTUNITY`.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report = Path(args.contract), Path(args.output_dir), Path(args.report)
    contract = load_contract(contract_path)
    source_records = verify_frozen_sources(contract)
    if args.dry_run:
        for pair_id in PAIR_IDS:
            load_pair(pair_id, contract)
        print(json.dumps({
            "status": "preflight_pass", "pair_count": len(PAIR_IDS), "split": "dev",
            "test_access": 0, "systematic_statistics_run": 0,
        }, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames, summary_rows, bootstrap_rows, random_rows = [], [], [], []
    extra_sources = []
    for pair_id in PAIR_IDS:
        frame, summary, bootstrap, mixed = analyze_pair(pair_id, contract)
        all_frames.append(frame); summary_rows.extend(summary); bootstrap_rows.extend(bootstrap)
        for record in mixed:
            if "role" in record:
                extra_sources.append(record)
            else:
                random_rows.append(record)
        print(f"[done] {pair_id}")
    per_query = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    random = pd.DataFrame(random_rows)
    per_query.to_csv(output_dir / "selectivity_per_query.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "coverage_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    random.to_csv(output_dir / "matched_random_baseline.csv", index=False)
    write_coverage_figure(summary, "selective_population_gain", "Frozen X4 Selective Realized Gain", output_dir / "figure_p2_1_selective_gain.svg", 0.0)
    write_coverage_figure(summary, "stable2_opportunity_enrichment", "Stable-2 Opportunity Enrichment", output_dir / "figure_p2_2_stable2_enrichment.svg", 1.0)
    write_coverage_figure(summary, "high_gain_enrichment", "Top-10% Oracle-Gain Enrichment", output_dir / "figure_p2_3_high_gain_enrichment.svg", 1.0)
    write_report(summary, report, output_dir)
    outputs = [
        output_dir / "selectivity_per_query.csv.gz", output_dir / "coverage_metrics.csv",
        output_dir / "bootstrap_ci.csv", output_dir / "matched_random_baseline.csv",
        output_dir / "figure_p2_1_selective_gain.svg", output_dir / "figure_p2_2_stable2_enrichment.svg",
        output_dir / "figure_p2_3_high_gain_enrichment.svg", report,
    ]
    unique_sources = {record["path"]: record for record in source_records + extra_sources}
    manifest = {
        "schema_version": 1,
        "study": contract["study"],
        "split": "dev",
        "contract": {"path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
        "sources": list(unique_sources.values()),
        "outputs": [
            {"path": portable_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "frozen_score": contract["ranking_rule"]["primary_scalar"],
        "coverage_fractions": contract["ranking_rule"]["coverage_fractions"],
        "operational_audit": {
            "test_access": 0, "new_ranker": 0, "new_selector": 0,
            "new_representation": 0, "new_feature": 0,
            "checkpoint_retraining": 0, "checkpoint_reselection": 0,
            "policy_tuning": 0, "closure_route_change": 0,
        },
        "closure_route": "FINAL_SELECTIVE_RARE_OPPORTUNITY",
    }
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": portable_path(output_dir), "report": portable_path(report), "test_access": 0}, indent=2))


if __name__ == "__main__":
    main()
