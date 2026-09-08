from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_cross_seed_transfer import action_indices, original_triple_ids
from scripts.audit_stable_headroom import choose_allowed_action, concentration_metrics
from scripts.exp2_information_common import ALPHAS, RR_COLUMNS, ZERO_TOLERANCE, portable_path, sha256_file


PAIR_LABELS = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper + NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper + AdaMF-MAT",
    "mkgw_native_adamf": "MKG-W / NativE + AdaMF-MAT",
    "db15k_mhyper_native": "DB15K / M-Hyper + NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper + AdaMF-MAT",
    "db15k_native_adamf": "DB15K / NativE + AdaMF-MAT",
    "mkg_y_mhyper_native": "MKG-Y / M-Hyper + NativE",
    "mkg_y_mhyper_adamf": "MKG-Y / M-Hyper + AdaMF-MAT",
    "mkg_y_native_adamf": "MKG-Y / NativE + AdaMF-MAT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the one-time, memo-locked complementarity TEST export.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--x4-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def bootstrap(
    arrays: dict[str, np.ndarray], samples: int = 10000, seed: int = 20260907
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("TEST bootstrap arrays are not aligned")
    n = lengths.pop()
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(samples, dtype=np.float64) for name in arrays}
    ratios = {
        name: np.empty(samples, dtype=np.float64)
        for name in ("consensus", "stable2", "stable3", "direct_transfer", "loso", "x4")
    }
    for start in range(0, samples, 64):
        stop = min(samples, start + 64)
        indices = rng.integers(0, n, size=(stop - start, n))
        for name, values in arrays.items():
            distributions[name][start:stop] = values[indices].mean(axis=1)
        denominator = distributions["raw"][start:stop]
        for name in ratios:
            ratios[name][start:stop] = distributions[name][start:stop] / denominator
    intervals = {name: tuple(map(float, np.percentile(value, [2.5, 97.5]))) for name, value in distributions.items()}
    ratio_intervals = {name: tuple(map(float, np.percentile(value, [2.5, 97.5]))) for name, value in ratios.items()}
    return intervals, ratio_intervals


def cluster_mean(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame.groupby("original_triple_id", sort=True)[column].mean().to_numpy(np.float64)


def ranking_metrics(rr: np.ndarray) -> dict[str, float]:
    ranks = np.rint(1.0 / np.asarray(rr, dtype=np.float64)).astype(np.int64)
    return {
        "mrr": float(np.mean(rr)),
        "hits1": float(np.mean(ranks <= 1)),
        "hits3": float(np.mean(ranks <= 3)),
        "hits10": float(np.mean(ranks <= 10)),
    }


def analyze_pair(
    pair_id: str,
    dataset: str,
    query_path: Path,
    x4_path: Path,
    x4_summary_path: Path,
    selection: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    columns = [
        "dataset", "pair_name", "split", "query_id", "seed", "direction", "relation_id",
        "head_id", "tail_id", "rr_a", "rr_b", "alpha_global", *RR_COLUMNS,
    ]
    frame = pd.read_csv(query_path, usecols=columns)
    expected_dataset = dataset
    if len(frame) == 0 or set(frame.split.astype(str)) != {"test"}:
        raise RuntimeError(f"Missing/non-TEST rows for {pair_id}")
    if set(frame.pair_name.astype(str)) != {pair_id} or set(frame.dataset.astype(str)) != {expected_dataset}:
        raise RuntimeError(f"TEST pair identity mismatch for {pair_id}")
    if set(frame.seed.astype(int)) != {1, 2, 3} or set(frame.direction.astype(str)) != {"head", "tail"}:
        raise RuntimeError(f"TEST seed/direction inventory mismatch for {pair_id}")
    if frame.query_id.duplicated().any() or frame[RR_COLUMNS].isna().any().any():
        raise RuntimeError(f"Duplicate/incomplete TEST action rows for {pair_id}")
    frame["original_triple_id"] = original_triple_ids(frame)
    frame = frame.sort_values(["original_triple_id", "direction", "seed"], kind="stable").reset_index(drop=True)
    x4 = pd.read_csv(x4_path)
    if x4.query_id.duplicated().any() or set(x4.query_id.astype(str)) != set(frame.query_id.astype(str)):
        raise RuntimeError(f"Frozen X4 TEST query inventory mismatch for {pair_id}")
    x4 = frame[["query_id"]].merge(
        x4[["query_id", "selected_alpha", "selected_rr", "global_rr", "gain"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    x4_summary = load_json(x4_summary_path)
    if x4_summary.get("pair_id") != pair_id or x4_summary.get("split") != "test":
        raise RuntimeError(f"Frozen X4 TEST summary mismatch for {pair_id}")
    counts = frame.groupby("original_triple_id").size()
    if not (counts == 6).all():
        raise RuntimeError(f"Each TEST triple must contain 3 seeds × 2 directions: {pair_id}")
    alpha0 = float(selection["global_alpha"])
    global_candidates = np.flatnonzero(np.isclose(ALPHAS, alpha0, rtol=0.0, atol=ZERO_TOLERANCE))
    if len(global_candidates) != 1 or not np.allclose(frame.alpha_global.to_numpy(float), alpha0):
        raise RuntimeError(f"DEV-locked alpha0 mismatch for {pair_id}")
    global_index = int(global_candidates[0])
    rr = frame[RR_COLUMNS].to_numpy(np.float64)
    identity = frame.iloc[::3].reset_index(drop=True)
    rr3 = rr.reshape(-1, 3, len(ALPHAS))
    rows = np.arange(len(identity))[:, None]
    seed_axis = np.arange(3)[None, :]
    global_rr = rr3[:, :, global_index]
    mean_rr = rr3.mean(axis=1)
    oracle_index = action_indices(rr3, global_index)
    oracle_gain = rr3[rows, seed_axis, oracle_index] - global_rr
    raw_q = oracle_gain.mean(axis=1)
    consensus_index = action_indices(mean_rr, global_index)
    consensus_gain = mean_rr[np.arange(len(identity)), consensus_index] - mean_rr[:, global_index]
    positive_counts = ((rr3 - global_rr[:, :, None]) > 0.0).sum(axis=1)
    non_anchor = np.ones_like(positive_counts, dtype=bool)
    non_anchor[:, global_index] = False
    allowed2, allowed3 = (positive_counts >= 2) & non_anchor, (positive_counts == 3) & non_anchor
    index2 = choose_allowed_action(mean_rr, allowed2, global_index)
    index3 = choose_allowed_action(mean_rr, allowed3, global_index)
    gain2 = mean_rr[np.arange(len(identity)), index2] - mean_rr[:, global_index]
    gain3 = mean_rr[np.arange(len(identity)), index3] - mean_rr[:, global_index]

    loso = np.empty((len(identity), 3), dtype=np.float64)
    for held in range(3):
        train = [index for index in range(3) if index != held]
        chosen = action_indices(rr3[:, train, :].mean(axis=1), global_index)
        loso[:, held] = rr3[np.arange(len(identity)), held, chosen] - global_rr[:, held]
    loso_q = loso.mean(axis=1)
    direct = []
    for source in range(3):
        chosen = action_indices(rr3[:, source, :], global_index)
        for target in range(3):
            if source != target:
                direct.append(rr3[np.arange(len(identity)), target, chosen] - global_rr[:, target])
    direct_q = np.column_stack(direct).mean(axis=1)
    x4_q = x4.gain.to_numpy(np.float64).reshape(-1, 3).mean(axis=1)

    per_query = pd.DataFrame({
        "dataset": expected_dataset, "pair_id": pair_id,
        "original_triple_id": identity.original_triple_id.to_numpy(str),
        "direction": identity.direction.to_numpy(str), "head_id": identity.head_id.to_numpy(),
        "relation_id": identity.relation_id.to_numpy(), "tail_id": identity.tail_id.to_numpy(),
        "alpha0": alpha0, "raw_oracle_gain": raw_q,
        "consensus_alpha": ALPHAS[consensus_index], "consensus_gain": consensus_gain,
        "stable2_nonempty": allowed2.any(axis=1).astype(np.int8), "stable2_alpha": ALPHAS[index2], "stable2_gain": gain2,
        "stable3_nonempty": allowed3.any(axis=1).astype(np.int8), "stable3_alpha": ALPHAS[index3], "stable3_gain": gain3,
        "direct_transfer_gain": direct_q, "loso_gain": loso_q, "x4_gain": x4_q,
    })
    triple_meta = identity.drop_duplicates("original_triple_id")[["dataset", "original_triple_id", "head_id", "relation_id", "tail_id"]].set_index("original_triple_id")
    triple_gain = per_query.groupby("original_triple_id", sort=True).raw_oracle_gain.mean()
    concentration = triple_meta.loc[triple_gain.index].reset_index()
    concentration.insert(1, "pair_id", pair_id)
    concentration["oracle_gain"] = triple_gain.to_numpy(np.float64)
    conc, order, curve = concentration_metrics(concentration.oracle_gain.to_numpy(), (0.01, 0.05, 0.10, 0.20))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    cumulative = np.empty(len(order), dtype=np.float64)
    cumulative[order] = curve
    concentration["descending_gain_rank"] = ranks
    concentration["cumulative_triple_fraction"] = ranks / len(ranks)
    concentration["cumulative_gain_fraction"] = cumulative

    arrays = {name: cluster_mean(per_query, column) for name, column in {
        "raw": "raw_oracle_gain", "consensus": "consensus_gain", "stable2": "stable2_gain",
        "stable3": "stable3_gain", "direct_transfer": "direct_transfer_gain", "loso": "loso_gain",
        "x4": "x4_gain", "stable2_opportunity": "stable2_nonempty",
        "stable3_opportunity": "stable3_nonempty",
    }.items()}
    intervals, ratios = bootstrap(arrays)
    estimates = {name: float(values.mean()) for name, values in arrays.items()}
    raw = estimates["raw"]
    if not np.isclose(float(concentration.oracle_gain.mean()), raw, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"TEST concentration identity failed for {pair_id}")
    summary = {
        "dataset": expected_dataset, "pair_id": pair_id, "pair_label": PAIR_LABELS[pair_id],
        "n_original_triples": len(concentration), "n_query_identities": len(identity), "alpha0": alpha0,
        "expert_a_mrr": float(frame.rr_a.mean()), "expert_b_mrr": float(frame.rr_b.mean()),
        "global_mrr": float(rr[:, global_index].mean()),
        "x4_mrr": float(x4.selected_rr.mean()),
        "x4_changed_rate": float((x4.selected_alpha.to_numpy(float) != alpha0).mean()),
        "x4_negative_transfer_rate": float((x4.gain.to_numpy(float) < -ZERO_TOLERANCE).mean()),
        "x4_positive_transfer_rate": float((x4.gain.to_numpy(float) > ZERO_TOLERANCE).mean()),
        **conc,
    }
    for prefix, values in {
        "expert_a": frame.rr_a.to_numpy(float),
        "expert_b": frame.rr_b.to_numpy(float),
        "global": rr[:, global_index],
        "x4": x4.selected_rr.to_numpy(float),
    }.items():
        for metric_name, value in ranking_metrics(values).items():
            summary[f"{prefix}_{metric_name}"] = value
    if not np.isclose(float(x4_summary["test_gain"]), estimates["x4"], atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Frozen X4 TEST summary/gain mismatch for {pair_id}")
    metric_rows = []
    for name in ("raw", "consensus", "stable2", "stable3", "direct_transfer", "loso", "x4"):
        prefix = {"raw": "raw_oracle", "stable2": "stable2", "stable3": "stable3"}.get(name, name)
        summary[f"{prefix}_headroom"] = estimates[name]
        summary[f"{prefix}_ci95_low"] = intervals[name][0]
        summary[f"{prefix}_ci95_high"] = intervals[name][1]
        metric_rows.append({"dataset": expected_dataset, "pair_id": pair_id, "metric": f"{prefix}_headroom", "estimate": estimates[name], "ci95_low": intervals[name][0], "ci95_high": intervals[name][1], "bootstrap_samples": 10000, "bootstrap_unit": "original_triple_id"})
        if name != "raw":
            summary[f"{prefix}_recovery"] = estimates[name] / raw
            summary[f"{prefix}_recovery_ci95_low"] = ratios[name][0]
            summary[f"{prefix}_recovery_ci95_high"] = ratios[name][1]
            metric_rows.append({"dataset": expected_dataset, "pair_id": pair_id, "metric": f"{prefix}_recovery", "estimate": estimates[name] / raw, "ci95_low": ratios[name][0], "ci95_high": ratios[name][1], "bootstrap_samples": 10000, "bootstrap_unit": "original_triple_id"})
    for name in ("stable2_opportunity", "stable3_opportunity"):
        summary[f"{name}_rate"] = estimates[name]
        summary[f"{name}_ci95_low"] = intervals[name][0]
        summary[f"{name}_ci95_high"] = intervals[name][1]
    return per_query, concentration, summary, metric_rows


def main() -> None:
    args = parse_args()
    output_dir, report_path = Path(args.output_dir), Path(args.report)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Repeated TEST analysis is prohibited: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Repeated TEST report generation is prohibited: {report_path}")
    contract_path = Path(args.contract)
    contract = load_json(contract_path)
    if contract.get("status") != "frozen_before_one_time_test" or len(contract.get("pairs", [])) != 9:
        raise RuntimeError("Invalid frozen three-dataset TEST contract")
    per_query_frames, concentration_frames, summaries, bootstrap_rows = [], [], [], []
    source_paths = [
        Path(__file__),
        contract_path,
        Path("docs/protocols/COMPLEMENTARITY_CLOSURE_DECISION_MEMO.md"),
        Path("scripts/apply_frozen_x4_closure_test.py"),
        Path("scripts/eval_heterogeneous_complementarity.py"),
    ]
    source_paths.extend(Path(record["path"]) for record in contract["global_dev_sources"])
    for runs in contract["experts"].values():
        for run in runs:
            source_paths.extend((Path(run["run_dir"]) / "config_merged.json", Path(run["run_dir"]) / "best.ckpt"))
    for pair in contract["pairs"]:
        pair_id = pair["pair_id"]
        manifest_path = Path(pair["dev_utility_manifest"]["path"])
        selection_path = Path(pair["dev_selection"]["path"])
        query_path = Path(args.raw_root) / pair_id / "test_query_rows.csv"
        raw_summary_path = Path(args.raw_root) / pair_id / "test_summary.json"
        x4_path = Path(args.x4_root) / pair_id / "test_x4_policy_rows.csv.gz"
        x4_summary_path = Path(args.x4_root) / pair_id / "test_x4_summary.json"
        if not all(path.exists() for path in (query_path, raw_summary_path, x4_path, x4_summary_path)):
            raise FileNotFoundError(f"Incomplete one-time TEST export for {pair_id}")
        per_query, concentration, summary, rows = analyze_pair(
            pair_id, pair["dataset"], query_path, x4_path, x4_summary_path, load_json(selection_path)
        )
        per_query_frames.append(per_query)
        concentration_frames.append(concentration)
        summaries.append(summary)
        bootstrap_rows.extend(rows)
        source_paths.extend([
            manifest_path, selection_path, Path(pair["dev_x4_asset_manifest"]["path"]),
            Path(pair["dev_nested_selection"]["path"]), Path(pair["run_manifest"]["path"]),
            query_path, raw_summary_path, x4_path, x4_summary_path,
        ])
        print(
            f"[OK] {pair_id}: TEST raw={summary['raw_oracle_headroom']:.6f}, "
            f"stable2={summary['stable2_headroom']:.6f}, X4={summary['x4_headroom']:+.6f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query = pd.concat(per_query_frames, ignore_index=True)
    concentration = pd.concat(concentration_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    paths = [output_dir / "stable_headroom_per_query.csv.gz", output_dir / "concentration_per_triple.csv.gz", output_dir / "pair_summary.csv", output_dir / "bootstrap_ci.csv"]
    per_query.to_csv(paths[0], index=False, compression=compression, lineterminator="\n")
    concentration.to_csv(paths[1], index=False, compression=compression, lineterminator="\n")
    summary.to_csv(paths[2], index=False, lineterminator="\n")
    bootstrap_frame.to_csv(paths[3], index=False, lineterminator="\n")
    lines = [
        "# Complementarity Closure — One-Time Frozen TEST Audit",
        "",
        "This report applies only the metrics and commands frozen in the committed three-dataset DEV closure decision memo. It does not alter any DEV classification, gate, title, or narrative.",
        "",
        "The TEST baseline is the single full-DEV-selected alpha0 for each pair. Experiment 2 DEV OOF gains used fold-specific outer-train alpha0 values, so the DEV and TEST gain baselines are related but not numerically identical estimands.",
        "",
        "| Pair | Global MRR | Raw | Consensus | 2-of-3 | 3-of-3 | Direct transfer | LOSO | Frozen X4 | Top10 | Q50 | Effective support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.global_mrr:.6f} | {row.raw_oracle_headroom:.6f} | "
            f"{row.consensus_headroom:.6f} | {row.stable2_headroom:.6f} | {row.stable3_headroom:.6f} | "
            f"{row.direct_transfer_headroom:.6f} | {row.loso_headroom:.6f} | {row.x4_headroom:+.6f} | "
            f"{row.top10_gain_share:.1%} | {row.q50:.1%} | {row.effective_support:.1%} |"
        )
    lines.extend([
        "",
        "| Pair | Expert A MRR/H@1/H@3/H@10 | Expert B MRR/H@1/H@3/H@10 | Global MRR/H@1/H@3/H@10 | X4 MRR/H@1/H@3/H@10 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        def cell(prefix: str) -> str:
            return "/".join(f"{getattr(row, f'{prefix}_{metric}'):.6f}" for metric in ("mrr", "hits1", "hits3", "hits10"))
        lines.append(f"| {PAIR_LABELS[row.pair_id]} | {cell('expert_a')} | {cell('expert_b')} | {cell('global')} | {cell('x4')} |")
    lines.extend([
        "",
        "Raw Oracle and all-seed consensus/stable quantities remain ex-post upper diagnostics. Direct transfer and LOSO are seed-stability diagnostics. Frozen X4 is the only inference-time observable probe in this table.",
        "",
        "- TEST access: one-time, after committed decision memo",
        "- checkpoint reselection/retraining: 0",
        "- feature/selector/grid/gate changes: 0",
        "- DEV classifications, route, claims, title, and narrative modified: 0",
        "",
        "Frozen DEV interpretation retained: `FINAL_SELECTIVE_RARE_OPPORTUNITY`",
        "",
        "FROZEN_TEST_VALIDATION_COMPLETE",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(report_path)
    audit = {
        "schema_version": 1,
        "split": "test",
        "role": "one-time validation of frozen three-dataset DEV closure",
        "datasets": ["mkg_w", "db15k", "mkg_y"],
        "pair_count": 9,
        "frozen_dev_interpretation": "FINAL_SELECTIVE_RARE_OPPORTUNITY",
        "sources_and_outputs":
            [{"role": "source", "path": portable_path(path), "sha256": sha256_file(path)} for path in sorted(set(source_paths), key=portable_path)]
            + [{"role": "output", "path": portable_path(path), "sha256": sha256_file(path)} for path in paths],
        "operational_audit": {
            "test_access": 1,
            "checkpoint_retraining": 0,
            "checkpoint_reselection": 0,
            "new_selector": 0,
            "new_representation": 0,
            "action_grid_modified": False,
            "dev_gate_modified": False,
            "dev_narrative_modified": False,
            "test_tuning": 0,
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("[DONE] Frozen one-time TEST validation analyzed; DEV interpretation unchanged")


if __name__ == "__main__":
    main()
