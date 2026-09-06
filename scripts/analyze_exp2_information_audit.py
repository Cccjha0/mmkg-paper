from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    ALPHAS,
    PAIR_IDS,
    RR_COLUMNS,
    load_contract,
    policy_metrics,
    portable_path,
    reject_test_path,
    sha256_file,
    sign_metrics,
)


PAIR_LABELS = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper+NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper+AdaMF",
    "mkgw_native_adamf": "MKG-W / NativE+AdaMF",
    "db15k_mhyper_native": "DB15K / M-Hyper+NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper+AdaMF",
    "db15k_native_adamf": "DB15K / NativE+AdaMF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed Experiment 2 nested OOF probes and apply the frozen gate.")
    parser.add_argument("--run-root", default="outputs/complementarity_identifiability/exp2_information/runs")
    parser.add_argument("--asset-dir", default="outputs/complementarity_identifiability/exp2_information/assets")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--exp1-stats", default="outputs/complementarity_identifiability/exp1_landscape/pair_statistics.csv")
    parser.add_argument("--contract", default="docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp2_information")
    parser.add_argument("--report", default="docs/reports/information_identifiability_audit_2026-09-05.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative_link(target: Path, document: Path) -> str:
    return os.path.relpath(target.resolve(), document.resolve().parent).replace("\\", "/")


def load_pair_data(pair_id: str, asset_dir: Path, utility_dir: Path):
    asset_manifest = json.loads((asset_dir / f"{pair_id}_query_information_manifest.json").read_text(encoding="utf-8"))
    with np.load(asset_manifest["output"]["path"], allow_pickle=False) as asset:
        query_ids = asset["query_id"].astype(str)
        frame = pd.DataFrame(
            {
                "query_id": query_ids, "seed": asset["seed"].astype(int), "direction": asset["direction"].astype(str),
                "head_id": asset["head_id"].astype(int), "relation_id": asset["relation_id"].astype(int), "tail_id": asset["tail_id"].astype(int),
            }
        )
    frame["original_triple_id"] = "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    utility_manifest = json.loads((utility_dir / f"{pair_id}_dev_source_manifest.json").read_text(encoding="utf-8"))
    exact = pd.read_csv(utility_manifest["source_query_rows"]["path"], usecols=["query_id", *RR_COLUMNS]).set_index("query_id")
    rr = exact.loc[query_ids, RR_COLUMNS].to_numpy(np.float64)
    return frame, rr


def nested_select(pair_id: str, representation: str, learners: list[str], args, contract, available: float) -> tuple[dict, list[dict]]:
    run_root = Path(args.run_root)
    frame, rr = load_pair_data(pair_id, Path(args.asset_dir), Path(args.utility_manifest_dir))
    learner_data = {}
    for learner in learners:
        root = run_root / pair_id / representation.lower() / learner
        payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        with np.load(root / "oof_action_predictions.npz", allow_pickle=False) as prediction:
            if list(prediction["query_id"].astype(str)) != list(frame.query_id.astype(str)):
                raise RuntimeError(f"Prediction alignment mismatch: {root}")
            learner_data[learner] = {
                "payload": payload,
                "predicted": prediction["predicted_u"].astype(np.float64),
                "fold": prediction["outer_fold"].astype(int),
                "global": prediction["fold_global_index"].astype(int),
                "selection": pd.read_csv(root / "outer_fold_selections.csv"),
                "training": pd.read_csv(root / "training_fold_metrics.csv"),
            }
    fold_reference = next(iter(learner_data.values()))["fold"]
    global_reference = next(iter(learner_data.values()))["global"]
    predicted = np.empty_like(next(iter(learner_data.values()))["predicted"])
    selection_rows, train_gains, train_counts = [], [], []
    learner_order = {learner: index for index, learner in enumerate(learners)}
    for fold in range(1, 6):
        candidates = []
        for learner in learners:
            row = learner_data[learner]["selection"].loc[lambda value: value.outer_fold == fold].iloc[0]
            candidates.append((float(row.inner_probe_delta_mrr), float(row.inner_spearman), -learner_order[learner], learner, row))
        winner = max(candidates)
        learner, row = winner[3], winner[4]
        mask = fold_reference == fold
        if not np.array_equal(learner_data[learner]["fold"], fold_reference) or not np.array_equal(learner_data[learner]["global"], global_reference):
            raise RuntimeError("Learner OOF fold/global assignments differ")
        predicted[mask] = learner_data[learner]["predicted"][mask]
        train_row = learner_data[learner]["training"].loc[lambda value: value.outer_fold == fold].iloc[0]
        train_gains.append(float(train_row.training_gain)); train_counts.append(int(train_row["count"]))
        selection_rows.append({"pair_id": pair_id, "representation": representation, "outer_fold": fold, "selected_learner": learner, "selected_config": row.selected_config, "inner_probe_delta_mrr": row.inner_probe_delta_mrr, "inner_spearman": row.inner_spearman, "fold_global_alpha": row.fold_global_alpha})
    bootstrap = contract["bootstrap"]
    metrics, chosen = policy_metrics(rr, predicted, global_reference, frame.original_triple_id.to_numpy(str), available, int(bootstrap["samples"]), int(bootstrap["seed"]))
    actual_u = rr - rr[np.arange(len(rr)), global_reference][:, None]
    non_global = np.arange(len(ALPHAS))[None, :] != global_reference[:, None]
    metrics.update(sign_metrics(actual_u[non_global], predicted[non_global]))
    training_gain = float(np.average(train_gains, weights=train_counts))
    metrics.update({"training_gain": training_gain, "oof_gain": metrics["delta_mrr"], "train_oof_generalization_gap": training_gain - metrics["delta_mrr"]})
    root = run_root / pair_id / representation.lower() / "nested_selected"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "oof_action_predictions.npz", query_id=frame.query_id.to_numpy(str), outer_fold=fold_reference, fold_global_index=global_reference, predicted_u=predicted.astype(np.float32), chosen_action_index=chosen)
    pd.DataFrame(selection_rows).to_csv(root / "outer_fold_learner_selections.csv", index=False)
    slice_rows = []
    scopes = [("direction", value, frame.direction.astype(str) == value) for value in ("head", "tail")]
    scopes += [("seed", str(seed), frame.seed.astype(int) == seed) for seed in (1, 2, 3)]
    scopes += [
        (
            "seed_x_direction",
            f"{seed}_{direction}",
            (frame.seed.astype(int) == seed) & (frame.direction.astype(str) == direction),
        )
        for seed in (1, 2, 3)
        for direction in ("head", "tail")
    ]
    for offset, (scope, value, mask) in enumerate(scopes):
        result, _ = policy_metrics(
            rr[mask], predicted[mask], global_reference[mask],
            frame.loc[mask, "original_triple_id"].to_numpy(str), available,
            int(contract["bootstrap"]["samples"]), int(contract["bootstrap"]["seed"]) + offset + 101,
        )
        slice_actual = actual_u[mask]
        slice_non_global = np.arange(len(ALPHAS))[None, :] != global_reference[mask, None]
        result.update(sign_metrics(slice_actual[slice_non_global], predicted[mask][slice_non_global]))
        dataset = "mkg_w" if pair_id.startswith("mkgw_") else "db15k"
        slice_rows.append({"dataset": dataset, "pair_id": pair_id, "representation": representation, "learner": "nested_selected", "scope": scope, "value": value, **result})
    pd.DataFrame(slice_rows).to_csv(root / "seed_direction_metrics.csv", index=False)
    payload = {"schema_version": 1, "experiment": "Experiment 2 — Information–Identifiability Audit", "probe": "nested_selected", "pair_id": pair_id, "representation": representation, "metrics": metrics, "test_access": 0, "final_policy_development": 0}
    (root / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dataset = "mkg_w" if pair_id.startswith("mkgw_") else "db15k"
    return {"dataset": dataset, "pair_id": pair_id, "representation": representation, "learner": "nested_selected", **metrics}, selection_rows


def svg_header(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:12px;font-weight:700}.axis{font-size:9px}.note{font-size:10px;fill:#5f6368}</style>', f'<text x="20" y="28" class="title">{html.escape(title)}</text>']


def write_svg(path, parts):
    parts.append("</svg>"); path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def declared_hash_records(value):
    """Yield path/hash declarations nested in audited JSON manifests."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        source_files = value.get("source_files")
        if isinstance(source_files, dict):
            for path, expected in source_files.items():
                if isinstance(path, str) and isinstance(expected, str):
                    yield path, expected
        for key, nested in value.items():
            if key != "source_files":
                yield from declared_hash_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from declared_hash_records(nested)


def complete_hash_inventory(seed_paths: list[Path], excluded: set[Path]) -> list[dict]:
    """Resolve and verify the transitive source/output hash inventory."""
    records: dict[str, str] = {}
    queue = list(seed_paths)
    parsed_json: set[str] = set()
    excluded_resolved = {path.resolve() for path in excluded}
    while queue:
        path = Path(queue.pop()).resolve()
        if path in excluded_resolved or "preflight" in [part.lower() for part in path.parts]:
            continue
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Hash inventory source is missing: {path}")
        portable = portable_path(path)
        actual = sha256_file(path)
        previous = records.get(portable)
        if previous is not None and previous != actual:
            raise RuntimeError(f"Inconsistent hash inventory for {portable}")
        records[portable] = actual
        if path.suffix.lower() != ".json" or portable in parsed_json:
            continue
        parsed_json.add(portable)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Experiment 1 is an upstream, already-audited boundary. Experiment 2
        # consumes its gate/summary, not every large artifact listed inside it.
        # Record the Exp1 audit hash itself, but do not expand its transitive
        # output inventory (for example per_query_action_geometry.csv.gz).
        if str(payload.get("experiment", "")).startswith("Experiment 1"):
            continue
        for declared_path, expected in declared_hash_records(payload):
            source = Path(declared_path).resolve()
            if source in excluded_resolved:
                continue
            if not source.exists() or sha256_file(source) != expected:
                raise RuntimeError(f"Declared source hash mismatch: {declared_path}")
            queue.append(source)
    return [{"path": path, "sha256": records[path]} for path in sorted(records)]


def color(value, limit):
    normalized = max(-1.0, min(1.0, value / max(limit, 1e-12)))
    white, target = (247, 247, 247), ((33, 102, 172) if normalized < 0 else (178, 24, 43))
    rgb = [round(white[i] + abs(normalized) * (target[i] - white[i])) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def figures(all_metrics: pd.DataFrame, primary: pd.DataFrame, exp1: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    # Figure 1: six small information curves.
    parts = svg_header(1160, 700, "Information–Identifiability Curve")
    for panel, pair_id in enumerate(PAIR_IDS):
        col, row = panel % 3, panel // 3
        ox, oy, w, h = 65 + col * 375, 70 + row * 305, 305, 205
        data = primary[primary.pair_id == pair_id].set_index("representation").loc[[f"X{i}" for i in range(1, 7)]]
        values = np.r_[data.oof_gain.to_numpy(), data.training_gain.to_numpy(), exp1.loc[exp1.pair_id == pair_id, "available_headroom"].iloc[0]]
        low, high = min(-0.002, float(values.min())), max(0.002, float(values.max()))
        sx = lambda index: ox + index * w / 5
        sy = lambda value: oy + h - (value - low) / (high - low) * h
        parts.append(f'<text x="{ox}" y="{oy-12}" class="panel">{html.escape(PAIR_LABELS[pair_id])}</text>')
        parts.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+w}" y2="{sy(0):.1f}" stroke="#9aa0a6"/>')
        headroom = float(exp1.loc[exp1.pair_id == pair_id, "available_headroom"].iloc[0])
        parts.append(f'<line x1="{ox}" y1="{sy(headroom):.1f}" x2="{ox+w}" y2="{sy(headroom):.1f}" stroke="#888" stroke-dasharray="2 3"/>')
        for field, stroke, dash in (("oof_gain", "#376795", ""), ("training_gain", "#d1495b", "5 3")):
            points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(data[field]))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2" stroke-dasharray="{dash}"/>')
        for i in range(6): parts.append(f'<text x="{sx(i):.1f}" y="{oy+h+15}" text-anchor="middle" class="axis">X{i+1}</text>')
    parts += ['<line x1="420" y1="675" x2="445" y2="675" stroke="#376795" stroke-width="2"/><text x="452" y="679" class="note">OOF gain</text>', '<line x1="535" y1="675" x2="560" y2="675" stroke="#d1495b" stroke-width="2" stroke-dasharray="5 3"/><text x="567" y="679" class="note">train gain</text>', '<line x1="670" y1="675" x2="695" y2="675" stroke="#888" stroke-dasharray="2 3"/><text x="702" y="679" class="note">Available headroom</text>']
    path = output_dir / "figure1_information_identifiability_curve.svg"; write_svg(path, parts); paths.append(path)
    # Figure 2: Available versus empirical.
    parts = svg_header(780, 600, "Available vs Empirically Identifiable Headroom")
    merged = primary.merge(exp1[["pair_id", "available_headroom"]], on="pair_id")
    maximum = max(float(merged.available_headroom.max()), float(merged.oof_gain.max()), 0.001)
    minimum_y = min(-0.002, float(merged.oof_gain.min()))
    sx = lambda value: 80 + float(value) / maximum * 620
    sy = lambda value: 520 - (float(value) - minimum_y) / (maximum - minimum_y) * 430
    parts.append(f'<line x1="80" y1="{sy(0):.1f}" x2="700" y2="{sy(maximum):.1f}" stroke="#aaa" stroke-dasharray="4 3"/>')
    parts.append(f'<line x1="80" y1="{sy(0):.1f}" x2="700" y2="{sy(0):.1f}" stroke="#999"/>')
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for index, row in enumerate(merged.itertuples()):
        parts.append(f'<circle cx="{sx(row.available_headroom):.1f}" cy="{sy(row.oof_gain):.1f}" r="5" fill="{palette[int(row.representation[1:])-1]}"/>')
    for index, shade in enumerate(palette):
        x = 120 + index * 85
        parts.append(f'<circle cx="{x}" cy="555" r="5" fill="{shade}"/><text x="{x+10}" y="559" class="note">X{index+1}</text>')
    parts += ['<text x="390" y="590" text-anchor="middle" class="panel">Experiment 1 Available Headroom</text>', '<text x="20" y="300" transform="rotate(-90 20 300)" text-anchor="middle" class="panel">OOF Empirical Identifiable Headroom</text>']
    path = output_dir / "figure2_available_vs_empirical.svg"; write_svg(path, parts); paths.append(path)
    # Figure 3: pair panels, X x learner heatmaps.
    parts = svg_header(1260, 690, "X × learner OOF gain")
    learners = ["linear_huber", "hist_gbdt", "mlp_low", "mlp_high", "set_encoder"]
    limit = float(np.quantile(np.abs(all_metrics.delta_mrr), 0.98))
    for panel, pair_id in enumerate(PAIR_IDS):
        col, row = panel % 3, panel // 3; ox, oy = 55 + col * 410, 75 + row * 285
        parts.append(f'<text x="{ox}" y="{oy-12}" class="panel">{html.escape(PAIR_LABELS[pair_id])}</text>')
        for yi, learner in enumerate(learners):
            parts.append(f'<text x="{ox+95}" y="{oy+yi*35+22}" text-anchor="end" class="axis">{learner}</text>')
            for xi in range(1, 7):
                rows = all_metrics[(all_metrics.pair_id == pair_id) & (all_metrics.representation == f"X{xi}") & (all_metrics.learner == learner)]
                fill = "#d9d9d9" if rows.empty else color(float(rows.delta_mrr.iloc[0]), limit)
                parts.append(f'<rect x="{ox+105+(xi-1)*45}" y="{oy+yi*35}" width="43" height="33" fill="{fill}"/>')
        for xi in range(1, 7): parts.append(f'<text x="{ox+105+(xi-0.5)*45:.1f}" y="{oy+190}" text-anchor="middle" class="axis">X{xi}</text>')
    path = output_dir / "figure3_x_learner_heatmap.svg"; write_svg(path, parts); paths.append(path)
    # Figure 4: train-OOF gaps.
    parts = svg_header(900, 560, "Train–OOF generalization gap by information richness")
    maximum = max(float(primary.train_oof_generalization_gap.abs().max()), 1e-6)
    sx = lambda index: 100 + index * 125; sy = lambda value: 285 - float(value) / maximum * 210
    parts.append(f'<line x1="100" y1="285" x2="725" y2="285" stroke="#999"/>')
    for pair_index, pair_id in enumerate(PAIR_IDS):
        data = primary[primary.pair_id == pair_id].set_index("representation").loc[[f"X{i}" for i in range(1, 7)]]
        points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(data.train_oof_generalization_gap))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{palette[pair_index]}" stroke-width="2"/>')
        parts.append(f'<text x="750" y="{110+pair_index*30}" class="note" fill="{palette[pair_index]}">{html.escape(PAIR_LABELS[pair_id])}</text>')
    for i in range(6): parts.append(f'<text x="{sx(i):.1f}" y="515" text-anchor="middle" class="axis">X{i+1}</text>')
    path = output_dir / "figure4_train_oof_gap.svg"; write_svg(path, parts); paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report_path = Path(args.contract), Path(args.output_dir), Path(args.report)
    for path in (contract_path, output_dir, report_path): reject_test_path(path)
    contract = load_contract(contract_path); output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and ((output_dir / "audit_manifest.json").exists() or report_path.exists()):
        raise FileExistsError("Refusing to overwrite an existing Experiment 2 aggregate audit")
    exp1 = pd.read_csv(args.exp1_stats)
    all_rows, primary_rows, selection_rows = [], [], []
    for pair_id in PAIR_IDS:
        available = float(exp1.loc[exp1.pair_id == pair_id, "available_headroom"].iloc[0])
        for representation in [f"X{i}" for i in range(1, 7)]:
            learners = list(contract["learner_compatibility"][representation])
            for learner in learners:
                path = Path(args.run_root) / pair_id / representation.lower() / learner / "metrics.json"
                if not path.exists(): raise FileNotFoundError(f"Incomplete systematic run: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                dataset = "mkg_w" if pair_id.startswith("mkgw_") else "db15k"
                all_rows.append({"dataset": dataset, "pair_id": pair_id, "representation": representation, "learner": learner, **payload["metrics"]})
            if representation == "X6":
                primary_rows.append(all_rows[-1].copy())
            else:
                nested, selections = nested_select(pair_id, representation, learners, args, contract, available)
                primary_rows.append(nested); selection_rows.extend(selections)
    all_metrics, primary = pd.DataFrame(all_rows), pd.DataFrame(primary_rows)
    all_metrics.to_csv(output_dir / "metrics_by_x_learner.csv", index=False)
    primary.to_csv(output_dir / "primary_nested_probe_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "nested_learner_selections.csv", index=False)
    gate_rows = []
    for representation in [f"X{i}" for i in range(1, 7)]:
        rows = primary[primary.representation == representation]
        robust = (rows.delta_mrr > 0) & (rows.clustered_ci95_low > 0)
        native_positive = rows[rows.pair_id.isin(["mkgw_native_adamf", "db15k_native_adamf"])].delta_mrr.gt(0).all()
        recovery = rows.headroom_recovery.ge(0.10)
        mkgw_native_ci = float(rows.loc[rows.pair_id == "mkgw_native_adamf", "clustered_ci95_low"].iloc[0]) > 0
        passed = int(robust.sum()) >= 4 and native_positive and int(recovery.sum()) >= 3 and mkgw_native_ci
        gate_rows.append({"representation": representation, "robust_positive_pairs": int(robust.sum()), "both_native_adamf_positive": bool(native_positive), "recovery_ge_10pct_pairs": int(recovery.sum()), "mkgw_native_adamf_ci_lower_gt_zero": bool(mkgw_native_ci), "pass": bool(passed)})
    gate = pd.DataFrame(gate_rows); gate.to_csv(output_dir / "preliminary_query_level_gate.csv", index=False)
    decision = "PRELIMINARY GO" if gate["pass"].any() else "QUERY-LEVEL PRELIMINARY NO-GO"
    figure_paths = figures(all_metrics, primary, exp1, output_dir)
    report_lines = ["# Information–Identifiability Audit — Experiment 2", "", "Date: 2026-09-05", "", f"## Outcome", "", f"Frozen preliminary query-level gate: **{decision}**.", "", "These are finite-model strict-OOF probes and are reported as Empirical Identifiable Headroom, not theoretical `C_identifiable`.", "", "## Primary nested OOF probes", "", "| Pair | X | Fold-specific Global MRR | OOF MRR | OOF gain | 95% clustered CI | Recovery | Positive gain | Negative transfer | Changed | Train gain | Gap |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in primary.itertuples():
        report_lines.append(f"| {PAIR_LABELS[row.pair_id]} | {row.representation} | {row.fold_specific_global_mrr:.6f} | {row.oof_mrr:.6f} | {row.delta_mrr:+.6f} | [{row.clustered_ci95_low:+.6f}, {row.clustered_ci95_high:+.6f}] | {100*row.headroom_recovery:.1f}% | {100*row.positive_gain_rate:.1f}% | {100*row.negative_transfer_rate:.1f}% | {100*row.changed_rate:.1f}% | {row.training_gain:+.6f} | {row.train_oof_generalization_gap:+.6f} |")
    report_lines += ["", "## Utility-identifiability diagnostics", "", "| Pair | X | Spearman(pred U, U) | Positive AP lift | Harmful AP lift | Positive-vs-harmful AUROC |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in primary.itertuples():
        report_lines.append(f"| {PAIR_LABELS[row.pair_id]} | {row.representation} | {row.spearman_pred_u_actual_u:+.4f} | {row.positive_ap_lift:+.4f} | {row.harmful_ap_lift:+.4f} | {row.positive_vs_harmful_auroc:.4f} |")
    report_lines += ["", "## Preliminary gate by information representation", "", "| X | Robust-positive pairs | Both NativE+AdaMF positive | >=10% recovery pairs | MKG-W NativE+AdaMF CI lower > 0 | Pass |", "| --- | ---: | --- | ---: | --- | --- |"]
    for row in gate.to_dict(orient="records"):
        report_lines.append(
            f"| {row['representation']} | {row['robust_positive_pairs']}/6 | "
            f"{row['both_native_adamf_positive']} | {row['recovery_ge_10pct_pairs']}/6 | "
            f"{row['mkgw_native_adamf_ci_lower_gt_zero']} | {row['pass']} |"
        )
    report_lines += ["", "Experiment 3 remains the required next comparison regardless of this preliminary result. No query selector or final policy was developed.", "", "## Machine-readable results", "", "- `metrics_by_x_learner.csv`: every dataset/pair/X/learner result", "- `primary_nested_probe_metrics.csv`: fold-wise inner-selected primary probes", "- `nested_learner_selections.csv`: fold-specific learner choices", "- `runs/<pair>/<x>/<learner>/seed_direction_metrics.csv`: seed, direction, and seed × direction results", "", "## Figures", ""]
    for index, path in enumerate(figure_paths, 1): report_lines.append(f"{index}. [{path.stem}]({relative_link(path, report_path)})")
    report_lines += ["", "## Operational audit", "", "- TEST access = 0", "- full-DEV Global used for held-out folds = 0", "- checkpoint training/reselection = 0", "- AACPI resurrection = 0", "- final policy development = 0", "- candidate embeddings = 0", "", "All input and output hashes are recorded in the machine-readable `audit_manifest.json`.", ""]
    audit_manifest_path = output_dir / "audit_manifest.json"
    output_paths = [
        path for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != audit_manifest_path.resolve()
    ]
    utility_manifests = [Path(args.utility_manifest_dir) / f"{pair_id}_dev_source_manifest.json" for pair_id in PAIR_IDS]
    source_paths = [contract_path, Path(args.exp1_stats), *utility_manifests, *output_paths]
    inventory = complete_hash_inventory(source_paths, {audit_manifest_path})
    # Publish the human-readable report only after every declared source hash
    # has passed, so a failed audit cannot create a new partial final report.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    inventory.append({"path": portable_path(report_path), "sha256": sha256_file(report_path)})
    inventory.sort(key=lambda row: row["path"])
    manifest = {"schema_version": 1, "experiment": "Experiment 2 — Information–Identifiability Audit", "split": "dev", "decision": decision, "gate": gate_rows, "source_and_output_hash_count": len(inventory), "sources_and_outputs": inventory, "operational_audit": {"test_access": 0, "full_dev_global_for_heldout": 0, "checkpoint_training": 0, "checkpoint_reselection": 0, "aacpi_resurrection": 0, "final_policy_development": 0, "candidate_embeddings": 0}, "next_step_started": 0}
    audit_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gate": gate_rows}, indent=2))


if __name__ == "__main__":
    main()
