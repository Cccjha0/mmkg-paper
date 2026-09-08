from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ORDER = ("mkg_w", "db15k", "mkg_y")
DATASET_LABELS = {"mkg_w": "MKG-W", "db15k": "DB15K", "mkg_y": "MKG-Y"}
PAIR_SUFFIXES = ("mhyper_native", "mhyper_adamf", "native_adamf")
PAIR_NAMES = {
    "mhyper_native": "M-Hyper+NativE",
    "mhyper_adamf": "M-Hyper+AdaMF",
    "native_adamf": "NativE+AdaMF",
}
COLORS = {"mkg_w": "#28778e", "db15k": "#c45d3c", "mkg_y": "#48a08a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize frozen DEV complementarity closure evidence.")
    parser.add_argument("--contract", default="docs/protocols/CROSS_DATASET_COMPLEMENTARITY_CLOSURE_SYNTHESIS_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/cross_dataset_closure_synthesis")
    parser.add_argument("--report", default="docs/reports/cross_dataset_complementarity_closure_synthesis_2026-09-08.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def portable(path: Path) -> str:
    return path.as_posix()


def reject_test_path(path: Path) -> None:
    if "test" in {part.lower() for part in path.parts}:
        raise RuntimeError(f"TEST path prohibited: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_contract(path: Path) -> tuple[dict, list[dict]]:
    reject_test_path(path)
    contract = load_json(path)
    if contract.get("status") != "frozen_before_synthesis" or contract.get("split") != "dev":
        raise RuntimeError("Synthesis contract must be frozen and DEV-only")
    if tuple(contract.get("datasets", [])) != DATASET_ORDER:
        raise RuntimeError("Dataset inventory changed")
    if tuple(contract.get("pair_suffixes", [])) != PAIR_SUFFIXES:
        raise RuntimeError("Pair inventory changed")
    if contract.get("scope", {}).get("new_gate") is not False:
        raise RuntimeError("Cross-dataset synthesis must not introduce a gate")
    if contract.get("scope", {}).get("route_reselection") is not False:
        raise RuntimeError("Cross-dataset synthesis must not reselect the frozen route")
    if any(int(value) != 0 for value in contract.get("operational_boundaries", {}).values()):
        raise RuntimeError("Operational boundary changed")
    inventory = []
    for source in contract.get("sources", []):
        path = Path(source["path"])
        reject_test_path(path)
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {path}")
        inventory.append({"path": portable(path), "sha256": actual, "role": source["role"]})
    return contract, inventory


def role_path(contract: dict, role: str) -> Path:
    matches = [Path(item["path"]) for item in contract["sources"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source for role {role}, got {len(matches)}")
    return matches[0]


def validate_frozen_status(contract: dict) -> None:
    expected = contract["frozen_interpretations"]
    core_route = load_json(role_path(contract, "core_e3_decision"))
    core_e4 = load_json(role_path(contract, "core_e4_classification"))
    core_e5 = load_json(role_path(contract, "core_e5_classification"))
    core_e6 = load_json(role_path(contract, "core_e6_interpretation"))
    observed = {
        "core_exp3_route": core_route.get("decision"),
        "core_exp4_classification": core_e4.get("classification"),
        "core_exp5_classification": core_e5.get("final_classification"),
        "core_exp6_stable_classification": core_e6.get("stable_classification"),
        "core_exp6_concentration_classification": core_e6.get("gain_concentration_classification"),
        "core_exp6_joint_interpretation": core_e6.get("final_joint_interpretation"),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(f"Frozen interpretation mismatch for {key}: {value}")
    core_e2_gate = pd.read_csv(role_path(contract, "core_e2_gate"))
    if core_e2_gate["pass"].map(bool_value).any():
        raise RuntimeError("Frozen Experiment 2 query-level gate unexpectedly passes")
    y_expected = {
        "mkg_y_e2_assessment": ("outcome", "Y_E2_X4_REPLICATION_REPORTED"),
        "mkg_y_e3_assessment": ("outcome", "Y_E3_RESOLUTION_REPLICATION_REPORTED"),
        "mkg_y_e4_assessment": ("outcome", "Y_E4_CROSS_SEED_REPLICATION_REPORTED"),
        "mkg_y_e5_classification": ("final_classification", "Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED"),
        "mkg_y_e6_assessment": ("outcome", "Y_E6_CLOSURE_REPLICATION_REPORTED"),
    }
    for role, (field, value) in y_expected.items():
        if load_json(role_path(contract, role)).get(field) != value:
            raise RuntimeError(f"Frozen MKG-Y assessment mismatch: {role}")


def bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def suffix_from_pair(pair_id: str) -> str:
    for suffix in PAIR_SUFFIXES:
        if pair_id.endswith(suffix):
            return suffix
    raise RuntimeError(f"Unknown pair id: {pair_id}")


def normalize_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "dataset" not in result.columns:
        result["dataset"] = result["pair_id"].map(
            lambda value: "mkg_y" if str(value).startswith("mkg_y_")
            else ("mkg_w" if str(value).startswith("mkgw_") else "db15k")
        )
    result["pair_suffix"] = result["pair_id"].map(suffix_from_pair)
    return result


def load_family(contract: dict, prefix: str) -> dict[str, object]:
    if prefix == "core":
        return {
            "e1": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e1"))),
            "e2": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e2")).query("representation == 'X4'")),
            "e3": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e3"))),
            "e3_comparison": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e3_comparison"))),
            "e4": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e4"))),
            "e5": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e5"))),
            "e5_signal": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e5_signal"))),
            "e6": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e6_stable"))),
            "concentration": normalize_pair_frame(pd.read_csv(role_path(contract, "core_e6_concentration"))),
        }
    return {
        "e1": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e1"))),
        "e2": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e2"))),
        "e3": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e3"))),
        "e3_comparison": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e3_comparison"))),
        "e4": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e4"))),
        "e5": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e5"))),
        "e5_signal": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e5_signal"))),
        "e6": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e6_stable"))),
        "concentration": normalize_pair_frame(pd.read_csv(role_path(contract, "mkg_y_e6_concentration"))),
    }


def validate_family(family: dict[str, object], datasets: tuple[str, ...], contract: dict) -> None:
    expected = {(dataset, suffix) for dataset in datasets for suffix in PAIR_SUFFIXES}
    for key in ("e1", "e2", "e4", "e5_signal", "e6", "concentration", "e3_comparison"):
        frame = family[key]
        found = set(zip(frame.dataset, frame.pair_suffix))
        if found != expected:
            raise RuntimeError(f"{key} pair inventory mismatch: expected {expected}, got {found}")
    e3 = family["e3"]
    if set(e3.level) != {"L0", "L1", "L2", "L3", "L4", "L5"}:
        raise RuntimeError("E3 granularity inventory changed")
    if set(pd.to_numeric(family["e5"].k)) != set(contract["required_invariants"]["local_k_values"]):
        raise RuntimeError("E5 k inventory changed")
    for dataset, suffix in expected:
        if len(family["e5"].query("dataset == @dataset and pair_suffix == @suffix")) != 4:
            raise RuntimeError(f"E5 must contain four k rows for {dataset}/{suffix}")


def build_pair_evidence(contract: dict) -> pd.DataFrame:
    core = load_family(contract, "core")
    mkg_y = load_family(contract, "mkg_y")
    validate_family(core, ("mkg_w", "db15k"), contract)
    validate_family(mkg_y, ("mkg_y",), contract)
    families = [core, mkg_y]
    rows = []
    tol = float(contract["required_invariants"]["raw_headroom_tolerance"])
    for family in families:
        for dataset in sorted(set(family["e1"].dataset), key=DATASET_ORDER.index):
            for suffix in PAIR_SUFFIXES:
                def select(key: str) -> pd.Series:
                    frame = family[key]
                    return frame.loc[(frame.dataset == dataset) & (frame.pair_suffix == suffix)].iloc[0]
                e1, e2, e4, e6, concentration = map(select, ("e1", "e2", "e4", "e6", "concentration"))
                e5_primary = family["e5"].query("dataset == @dataset and pair_suffix == @suffix and k == 10").iloc[0]
                e5_signal = select("e5_signal")
                comparison = select("e3_comparison")
                best_level = comparison.best_group_level
                e3_best = family["e3"].query(
                    "dataset == @dataset and pair_suffix == @suffix and level == @best_level"
                ).iloc[0]
                raw_values = [float(e1.available_headroom), float(e4.raw_oracle_headroom), float(e6.raw_oracle_headroom)]
                if max(raw_values) - min(raw_values) > tol:
                    raise RuntimeError(f"Raw-headroom identity failed for {dataset}/{suffix}: {raw_values}")
                x4_values = [float(e2.delta_mrr), float(e4.frozen_x4_oof_gain), float(e6.frozen_x4_oof_gain)]
                if max(x4_values) - min(x4_values) > float(contract["required_invariants"]["x4_gain_tolerance"]):
                    raise RuntimeError(f"X4 identity failed for {dataset}/{suffix}: {x4_values}")
                loso_values = [float(e4.loso_stable_headroom), float(e6.loso_stable_headroom)]
                if max(loso_values) - min(loso_values) > float(contract["required_invariants"]["loso_tolerance"]):
                    raise RuntimeError(f"LOSO identity failed for {dataset}/{suffix}: {loso_values}")
                raw = raw_values[0]
                rows.append({
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "pair_id": e1.pair_id,
                    "pair_suffix": suffix,
                    "pair_name": PAIR_NAMES[suffix],
                    "pair_label": f"{DATASET_LABELS[dataset]} / {PAIR_NAMES[suffix]}",
                    "n_original_triples": int(e4.n_original_triples),
                    "global_alpha": float(e1.global_alpha),
                    "global_mrr": float(e1.global_mrr),
                    "available_headroom": raw,
                    "available_ci95_low": float(e1.headroom_ci95_low),
                    "available_ci95_high": float(e1.headroom_ci95_high),
                    "positive_opportunity_rate": float(e1.positive_opportunity_rate),
                    "best_group_level": best_level,
                    "best_group_gain": float(e3_best.delta_mrr),
                    "best_group_ci95_low": float(e3_best.clustered_ci95_low),
                    "best_group_ci95_high": float(e3_best.clustered_ci95_high),
                    "best_group_recovery": float(e3_best.delta_mrr) / raw,
                    "best_group_robust_positive": bool_value(e3_best.robust_positive),
                    "x4_oof_gain": x4_values[0],
                    "x4_ci95_low": float(e2.clustered_ci95_low),
                    "x4_ci95_high": float(e2.clustered_ci95_high),
                    "x4_recovery": float(e2.headroom_recovery),
                    "x4_robust_positive": float(e2.delta_mrr) > 0 and float(e2.clustered_ci95_low) > 0,
                    "cross_seed_transfer_headroom": float(e4.cross_seed_transfer_headroom),
                    "cross_seed_transfer_ci95_low": float(e4.transfer_ci95_low),
                    "cross_seed_transfer_ci95_high": float(e4.transfer_ci95_high),
                    "cross_seed_transfer_recovery": float(e4.transfer_recovery),
                    "loso_headroom": loso_values[0],
                    "loso_ci95_low": float(e4.loso_ci95_low),
                    "loso_ci95_high": float(e4.loso_ci95_high),
                    "loso_recovery": float(e4.loso_recovery),
                    "consensus_headroom": float(e6.consensus_headroom),
                    "consensus_recovery": float(e6.consensus_recovery),
                    "stable2_headroom": float(e6.stable2_headroom),
                    "stable2_ci95_low": float(e6.stable2_ci95_low),
                    "stable2_ci95_high": float(e6.stable2_ci95_high),
                    "stable2_recovery": float(e6.stable2_recovery),
                    "stable3_headroom": float(e6.stable3_headroom),
                    "stable3_ci95_low": float(e6.stable3_ci95_low),
                    "stable3_ci95_high": float(e6.stable3_ci95_high),
                    "stable3_recovery": float(e6.stable3_recovery),
                    "local_primary_k": 10,
                    "local_direction_agreement_lift": float(e5_primary.direction_agreement_lift),
                    "local_direction_lift_ci95_low": float(e5_primary.direction_agreement_lift_ci95_low),
                    "local_consensus_utility": float(e5_primary.neighbor_consensus_utility),
                    "local_consensus_ci95_low": float(e5_primary.neighbor_consensus_utility_ci95_low),
                    "local_consensus_utility_lift": float(e5_primary.consensus_utility_lift),
                    "local_consensus_lift_ci95_low": float(e5_primary.consensus_utility_lift_ci95_low),
                    "local_signal_pair": bool_value(e5_signal.local_signal_pair),
                    "local_satisfying_k_count": int(e5_signal.satisfying_k_count),
                    "local_satisfying_k": "" if pd.isna(e5_signal.satisfying_k) else str(e5_signal.satisfying_k),
                    "top10_gain_share": float(concentration.top10_gain_share),
                    "q50": float(concentration.q50),
                    "gini": float(concentration.gini),
                    "effective_support": float(concentration.effective_support),
                })
    result = pd.DataFrame(rows)
    order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    suffix_order = {suffix: index for index, suffix in enumerate(PAIR_SUFFIXES)}
    result["_dataset_order"] = result.dataset.map(order)
    result["_pair_order"] = result.pair_suffix.map(suffix_order)
    result = result.sort_values(["_dataset_order", "_pair_order"]).drop(columns=["_dataset_order", "_pair_order"])
    if len(result) != int(contract["required_invariants"]["pair_count"]):
        raise RuntimeError("Final synthesis pair count mismatch")
    return result.reset_index(drop=True)


def build_dataset_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        group = pairs.loc[pairs.dataset == dataset]
        rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "pair_count": len(group),
            "median_available_headroom": group.available_headroom.median(),
            "median_stable2_recovery": group.stable2_recovery.median(),
            "median_stable3_recovery": group.stable3_recovery.median(),
            "median_loso_recovery": group.loso_recovery.median(),
            "median_x4_recovery": group.x4_recovery.median(),
            "available_ci_lower_positive_pairs": int((group.available_ci95_low > 0).sum()),
            "stable2_ci_lower_positive_pairs": int((group.stable2_ci95_low > 0).sum()),
            "stable3_ci_lower_positive_pairs": int((group.stable3_ci95_low > 0).sum()),
            "loso_ci_lower_positive_pairs": int((group.loso_ci95_low > 0).sum()),
            "x4_ci_lower_positive_pairs": int((group.x4_ci95_low > 0).sum()),
            "best_group_robust_positive_pairs": int(group.best_group_robust_positive.sum()),
            "local_signal_pairs": int(group.local_signal_pair.sum()),
            "top10_share_ge_50pct_pairs": int((group.top10_gain_share >= 0.5).sum()),
            "q50_le_10pct_pairs": int((group.q50 <= 0.1).sum()),
            "median_effective_support": group.effective_support.median(),
        })
    return pd.DataFrame(rows)


def build_stage_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Raw Available", "available_headroom", None, "available_ci95_low"),
        ("All-seed Consensus", "consensus_headroom", "consensus_recovery", None),
        ("2-of-3 Stable", "stable2_headroom", "stable2_recovery", "stable2_ci95_low"),
        ("3-of-3 Stable", "stable3_headroom", "stable3_recovery", "stable3_ci95_low"),
        ("LOSO Stable", "loso_headroom", "loso_recovery", "loso_ci95_low"),
        ("Best Group OOF", "best_group_gain", "best_group_recovery", "best_group_ci95_low"),
        ("Frozen X4 OOF", "x4_oof_gain", "x4_recovery", "x4_ci95_low"),
    ]
    rows = []
    for stage_index, (stage, gain, recovery, ci_low) in enumerate(specs):
        rows.append({
            "stage_index": stage_index,
            "stage": stage,
            "pair_count": len(pairs),
            "median_gain": pairs[gain].median(),
            "median_recovery": 1.0 if recovery is None else pairs[recovery].median(),
            "positive_gain_pairs": int((pairs[gain] > 0).sum()),
            "ci_lower_positive_pairs": "n.a." if ci_low is None else int((pairs[ci_low] > 0).sum()),
        })
    return pd.DataFrame(rows)


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" font-family="Arial" font-size="18" font-weight="bold" fill="#222">{html.escape(title)}</text>',
    ]


def svg_text(parts: list[str], x: float, y: float, text: str, size: int = 11, anchor: str = "start", fill: str = "#333", weight: str = "normal") -> None:
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{html.escape(str(text))}</text>')


def write_evidence_chain_figure(pairs: pd.DataFrame, path: Path) -> None:
    width, height = 1040, 650
    left, right, top, bottom = 255, 1000, 82, 585
    xmin = min(-0.002, pairs.x4_ci95_low.min())
    xmax = max(0.05, pairs.available_ci95_high.max())
    xmap = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    parts = svg_start(width, height, "Cross-Dataset Complementarity Evidence Chain")
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
    for value in np.arange(0, xmax + 0.0001, 0.01):
        x = xmap(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ececec"/>')
        svg_text(parts, x, bottom + 22, f"{value:.2f}", 10, "middle")
    xzero = xmap(0)
    parts.append(f'<line x1="{xzero:.1f}" y1="{top}" x2="{xzero:.1f}" y2="{bottom}" stroke="#777" stroke-width="1.2"/>')
    stages = [("available_headroom", "#213547", 5.0), ("stable2_headroom", "#28778e", 4.4), ("loso_headroom", "#d49a2a", 4.0), ("x4_oof_gain", "#c45d3c", 4.0)]
    for index, row in pairs.iterrows():
        y = top + 28 + index * 53
        svg_text(parts, left - 12, y + 4, row.pair_label, 10, "end")
        values = [row[column] for column, _, _ in stages]
        parts.append(f'<polyline points="{" ".join(f"{xmap(v):.1f},{y:.1f}" for v in values)}" fill="none" stroke="#b9b9b9" stroke-width="1.5"/>')
        for (column, color, radius), value in zip(stages, values):
            parts.append(f'<circle cx="{xmap(value):.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="1"/>')
        if index in (2, 5):
            parts.append(f'<line x1="30" y1="{y+27:.1f}" x2="{right}" y2="{y+27:.1f}" stroke="#d6d6d6"/>')
    legends = [("Raw", "#213547"), ("2-of-3 stable", "#28778e"), ("LOSO", "#d49a2a"), ("Frozen X4 OOF", "#c45d3c")]
    for index, (label, color) in enumerate(legends):
        x = 275 + index * 155
        parts.append(f'<circle cx="{x}" cy="620" r="4" fill="{color}"/>')
        svg_text(parts, x + 10, 624, label, 10)
    svg_text(parts, (left + right) / 2, 610, "MRR gain vs Global", 11, "middle")
    parts.append("</svg>")
    write_text_lf(path, "\n".join(parts) + "\n")


def heat_color(value: float) -> str:
    value = max(-0.05, min(1.0, value))
    if value < 0:
        t = min(1.0, abs(value) / 0.05)
        return f"rgb({int(245)}, {int(235-80*t)}, {int(230-70*t)})"
    t = value
    return f"rgb({int(245-150*t)}, {int(247-70*t)}, {int(245-45*t)})"


def write_recovery_heatmap(pairs: pd.DataFrame, path: Path) -> None:
    columns = [
        ("Consensus", "consensus_recovery"), ("2-of-3", "stable2_recovery"),
        ("3-of-3", "stable3_recovery"), ("LOSO", "loso_recovery"),
        ("Best group", "best_group_recovery"), ("X4 OOF", "x4_recovery"),
    ]
    width, height = 980, 610
    left, top, cell_w, cell_h = 270, 88, 108, 48
    parts = svg_start(width, height, "Available-Headroom Recovery by Evidence Stage")
    for column, (label, _) in enumerate(columns):
        svg_text(parts, left + column * cell_w + cell_w / 2, 70, label, 10, "middle", weight="bold")
    for row_index, row in pairs.iterrows():
        y = top + row_index * cell_h
        svg_text(parts, left - 12, y + 30, row.pair_label, 10, "end")
        for column, (_, field) in enumerate(columns):
            value = float(row[field])
            x = left + column * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" fill="{heat_color(value)}" stroke="#fff"/>')
            svg_text(parts, x + (cell_w - 3) / 2, y + 29, f"{value:.1%}", 10, "middle")
        if row_index in (2, 5):
            parts.append(f'<line x1="30" y1="{y+cell_h-1}" x2="{left+len(columns)*cell_w}" y2="{y+cell_h-1}" stroke="#aaa"/>')
    svg_text(parts, left, 555, "Darker green indicates more Oracle headroom retained; red denotes negative recovery.", 10, fill="#666")
    parts.append("</svg>")
    write_text_lf(path, "\n".join(parts) + "\n")


def write_stability_observability_scatter(pairs: pd.DataFrame, path: Path) -> None:
    width, height = 850, 610
    left, right, top, bottom = 85, 800, 70, 525
    xmin, xmax, ymin, ymax = 0.25, 0.78, -0.04, 0.22
    xmap = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    ymap = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts = svg_start(width, height, "Stable Headroom and Frozen X4 Recovery")
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
    for value in (0.3, 0.4, 0.5, 0.6, 0.7):
        x = xmap(value); parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#eee"/>'); svg_text(parts, x, bottom + 22, f"{value:.0%}", 10, "middle")
    for value in (0.0, 0.05, 0.10, 0.15, 0.20):
        y = ymap(value); parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#eee"/>'); svg_text(parts, left - 10, y + 4, f"{value:.0%}", 10, "end")
    parts.append(f'<line x1="{left}" y1="{ymap(0)}" x2="{right}" y2="{ymap(0)}" stroke="#777"/>')
    offsets = {"mhyper_native": (-74, -10), "mhyper_adamf": (8, -8), "native_adamf": (8, 16)}
    for row in pairs.itertuples(index=False):
        x, y = xmap(row.stable2_recovery), ymap(row.x4_recovery)
        radius = 5 + 65 * row.available_headroom
        stroke_width = 3 if row.local_signal_pair else 1
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{COLORS[row.dataset]}" fill-opacity="0.82" stroke="#222" stroke-width="{stroke_width}"/>')
        dx, dy = offsets[row.pair_suffix]
        svg_text(parts, x + dx, y + dy, f"{row.dataset_label} {row.pair_name}", 9)
    svg_text(parts, (left + right) / 2, 575, "2-of-3 stable recovery", 11, "middle")
    svg_text(parts, 18, 300, "Frozen X4 OOF recovery", 11)
    svg_text(parts, 520, 550, "Thick outline: frozen LOCAL_SIGNAL_PAIR", 9, fill="#666")
    parts.append("</svg>")
    write_text_lf(path, "\n".join(parts) + "\n")


def write_concentration_deployability_scatter(pairs: pd.DataFrame, path: Path) -> None:
    width, height = 850, 610
    left, right, top, bottom = 85, 800, 70, 525
    xmin, xmax, ymin, ymax = 0.40, 0.72, -0.04, 0.22
    xmap = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    ymap = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts = svg_start(width, height, "Oracle-Gain Concentration and Frozen X4 Recovery")
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
    for value in (0.4, 0.5, 0.6, 0.7):
        x=xmap(value); parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#eee"/>'); svg_text(parts,x,bottom+22,f"{value:.0%}",10,"middle")
    for value in (0.0, 0.05, 0.10, 0.15, 0.20):
        y=ymap(value); parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#eee"/>'); svg_text(parts,left-10,y+4,f"{value:.0%}",10,"end")
    parts.append(f'<line x1="{left}" y1="{ymap(0)}" x2="{right}" y2="{ymap(0)}" stroke="#777"/>')
    for row in pairs.itertuples(index=False):
        x, y = xmap(row.top10_gain_share), ymap(row.x4_recovery)
        shape = f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10"' if row.local_signal_pair else f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"'
        parts.append(shape + f' fill="{COLORS[row.dataset]}" stroke="#222"/>')
        svg_text(parts, x + 8, y - 7, f"{row.dataset_label} {row.pair_name}", 9)
    svg_text(parts, (left + right) / 2, 575, "Top-10% original-triple gain share", 11, "middle")
    svg_text(parts, 18, 300, "Frozen X4 OOF recovery", 11)
    svg_text(parts, 470, 550, "Square: frozen LOCAL_SIGNAL_PAIR", 9, fill="#666")
    parts.append("</svg>")
    write_text_lf(path, "\n".join(parts) + "\n")


def assessment_payload(pairs: pd.DataFrame, datasets: pd.DataFrame, contract: dict) -> dict:
    native = pairs.loc[pairs.pair_suffix == "native_adamf"]
    return {
        "outcome": "CROSS_DATASET_CLOSURE_SYNTHESIS_COMPLETE",
        "scope": "descriptive synthesis; no new gate and no route reselection",
        "pair_count": len(pairs),
        "dataset_count": len(datasets),
        "evidence": {
            "available_ci_lower_positive_pairs": int((pairs.available_ci95_low > 0).sum()),
            "stable2_ci_lower_positive_pairs": int((pairs.stable2_ci95_low > 0).sum()),
            "stable3_ci_lower_positive_pairs": int((pairs.stable3_ci95_low > 0).sum()),
            "loso_ci_lower_positive_pairs": int((pairs.loso_ci95_low > 0).sum()),
            "x4_ci_lower_positive_pairs": int((pairs.x4_ci95_low > 0).sum()),
            "best_group_robust_positive_pairs": int(pairs.best_group_robust_positive.sum()),
            "local_signal_pairs": int(pairs.local_signal_pair.sum()),
            "top10_share_ge_50pct_pairs": int((pairs.top10_gain_share >= 0.5).sum()),
            "q50_le_10pct_pairs": int((pairs.q50 <= 0.1).sum()),
            "median_stable2_recovery": float(pairs.stable2_recovery.median()),
            "median_stable3_recovery": float(pairs.stable3_recovery.median()),
            "median_loso_recovery": float(pairs.loso_recovery.median()),
            "median_x4_recovery": float(pairs.x4_recovery.median()),
            "native_adamf_local_signal_datasets": native.loc[native.local_signal_pair, "dataset"].tolist(),
            "native_adamf_x4_ci_lower_positive_datasets": native.loc[native.x4_ci95_low > 0, "dataset"].tolist(),
        },
        "frozen_interpretations_preserved": contract["frozen_interpretations"],
        "synthesis_interpretation": "Oracle complementarity is broad and seed-stable, but only partially observable and deployable; effective gain is concentrated and the boundary is pair-dependent.",
        "paper_claim_boundary": [
            "Large Oracle headroom is not evidence that a better router can recover all complementarity.",
            "Seed-stable opportunities exist across all three datasets, but frozen inference-time X4 evidence recovers only a minority of available headroom.",
            "Local X4 neighborhoods rarely satisfy the frozen action-signal criterion, so action consistency cannot be claimed as generally identifiable.",
            "Oracle gains are often concentrated in a small original-triple subset, supporting a limits and selective-opportunity framing rather than universal adaptation."
        ],
        "test_status": "LOCKED",
        "next_stage_started": 0,
    }


def write_report(pairs: pd.DataFrame, datasets: pd.DataFrame, stages: pd.DataFrame, assessment: dict, path: Path, contract: dict) -> None:
    lines = [
        "# Cross-Dataset Complementarity Closure Synthesis",
        "", f"Date: {contract['effective_date']}", "Split: DEV only", "Datasets: MKG-W, DB15K, MKG-Y", "",
        "## Frozen status", "",
        "This report synthesizes the already frozen Experiment 1-6 and MKG-Y Y-E1-Y-E6 evidence. It does not define a new gate, rescale the six-pair gates, reselect a route, or develop a policy.", "",
        f"- Core route remains `{contract['frozen_interpretations']['core_exp3_route']}`.",
        f"- Core closure interpretation remains `{contract['frozen_interpretations']['core_exp6_joint_interpretation']}`.",
        "- MKG-Y remains a descriptive external replication.", "",
        "## Main evidence chain", "",
        "| Dataset / pair | Raw available | 2-of-3 stable | LOSO stable | Best group OOF | Frozen X4 OOF | Local signal | Top10 share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {row.pair_label} | {row.available_headroom:+.5f} | {row.stable2_headroom:+.5f} ({row.stable2_recovery:.1%}) "
            f"| {row.loso_headroom:+.5f} ({row.loso_recovery:.1%}) | {row.best_group_gain:+.5f} ({row.best_group_level}) "
            f"| {row.x4_oof_gain:+.5f} ({row.x4_recovery:.1%}) | {'yes' if row.local_signal_pair else 'no'} | {row.top10_gain_share:.1%} |"
        )
    lines += [
        "", "Raw Oracle, all-seed consensus and two-/three-of-three quantities are upper diagnostics. LOSO is a held-out-seed diagnostic. Best-group and X4 values are strict grouped OOF results. These quantities are not interchangeable.", "",
        "## Cross-dataset summary", "",
        "| Dataset | Median raw | Median Stable-2 recovery | Median Stable-3 recovery | Median LOSO recovery | Median X4 recovery | X4 CI>0 | Local signal | Top10>=50% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in datasets.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | {row.median_available_headroom:.5f} | {row.median_stable2_recovery:.1%} | {row.median_stable3_recovery:.1%} "
            f"| {row.median_loso_recovery:.1%} | {row.median_x4_recovery:.1%} | {row.x4_ci_lower_positive_pairs}/3 "
            f"| {row.local_signal_pairs}/3 | {row.top10_share_ge_50pct_pairs}/3 |"
        )
    evidence = assessment["evidence"]
    lines += [
        "", "Across all nine pairs:", "",
        f"- Raw available headroom has CI lower > 0 for {evidence['available_ci_lower_positive_pairs']}/9 pairs.",
        f"- 2-of-3 and 3-of-3 stable headroom have CI lower > 0 for {evidence['stable2_ci_lower_positive_pairs']}/9 and {evidence['stable3_ci_lower_positive_pairs']}/9 pairs.",
        f"- LOSO headroom has CI lower > 0 for {evidence['loso_ci_lower_positive_pairs']}/9 pairs.",
        f"- Frozen X4 OOF gain has CI lower > 0 for {evidence['x4_ci_lower_positive_pairs']}/9 pairs; median available-headroom recovery is {evidence['median_x4_recovery']:.1%}.",
        f"- The frozen local-identifiability criterion holds for only {evidence['local_signal_pairs']}/9 pairs.",
        f"- Top 10% of triples contribute at least half the Oracle gain for {evidence['top10_share_ge_50pct_pairs']}/9 pairs; Q50 is at most 10% for {evidence['q50_le_10pct_pairs']}/9.",
        "", "## Interpretation", "",
        "The stable layer does not collapse: every pair retains significant two-of-three, three-of-three and LOSO headroom. The sharpest loss occurs between stable/transferable opportunity and observable action structure. X4 produces significant OOF gains in most pairs, but its median recovery is small and local neighborhoods rarely satisfy the frozen action-consistency criterion.", "",
        "MKG-Y strengthens the external-validity side of this conclusion. Its M-Hyper pairs reproduce positive X4 gains, while MKG-Y NativE+AdaMF has negative X4 gain despite positive stable and LOSO headroom. Across all three datasets, NativE+AdaMF never passes the frozen local-signal criterion. This is evidence for a pair-dependent identifiability boundary, not for a universally recoverable routing target.", "",
        "The concentration results further limit the claim: effective Oracle gain often comes from a small fraction of original triples. This supports the frozen Limits / selective-rare-opportunity framing. It does not justify reopening selector development or treating the Oracle bound as deployable headroom.", "",
        "## Paper claim boundary", "",
    ]
    lines += [f"- {claim}" for claim in assessment["paper_claim_boundary"]]
    lines += [
        "", "## Figures", "",
        "1. `figure1_cross_dataset_evidence_chain.svg`",
        "2. `figure2_recovery_heatmap.svg`",
        "3. `figure3_stability_observability.svg`",
        "4. `figure4_concentration_deployability.svg`", "",
        "## Integrity audit", "",
        "- TEST access = 0", "- checkpoint execution/retraining/reselection = 0",
        "- new selector / representation / policy tuning = 0", "- gate or route change = 0",
        "- historical result modification = 0", "- all direct source/output hashes recorded = yes",
        "- TEST remains locked", "- subsequent stage started = 0", "",
        "CROSS_DATASET_CLOSURE_SYNTHESIS_COMPLETE",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report_path = Path(args.contract), Path(args.output_dir), Path(args.report)
    for path in (contract_path, output_dir, report_path):
        reject_test_path(path)
    contract, source_inventory = load_contract(contract_path)
    validate_frozen_status(contract)
    pairs = build_pair_evidence(contract)
    datasets = build_dataset_summary(pairs)
    stages = build_stage_summary(pairs)
    preflight = {
        "status": "PRECHECK_OK", "split": "dev", "pair_count": len(pairs),
        "dataset_count": len(datasets), "verified_source_count": len(source_inventory),
        "test_access": 0, "checkpoint_execution": 0,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return
    if not args.overwrite and ((output_dir.exists() and any(output_dir.iterdir())) or report_path.exists()):
        raise FileExistsError("Refusing to overwrite existing cross-dataset synthesis")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "pair_evidence_chain.csv"
    dataset_path = output_dir / "dataset_summary.csv"
    stage_path = output_dir / "stage_summary.csv"
    assessment_path = output_dir / "synthesis_assessment.json"
    figure_paths = [
        output_dir / "figure1_cross_dataset_evidence_chain.svg",
        output_dir / "figure2_recovery_heatmap.svg",
        output_dir / "figure3_stability_observability.svg",
        output_dir / "figure4_concentration_deployability.svg",
    ]
    pairs.to_csv(pair_path, index=False, lineterminator="\n")
    datasets.to_csv(dataset_path, index=False, lineterminator="\n")
    stages.to_csv(stage_path, index=False, lineterminator="\n")
    assessment = assessment_payload(pairs, datasets, contract)
    write_text_lf(assessment_path, json.dumps(assessment, indent=2) + "\n")
    write_evidence_chain_figure(pairs, figure_paths[0])
    write_recovery_heatmap(pairs, figure_paths[1])
    write_stability_observability_scatter(pairs, figure_paths[2])
    write_concentration_deployability_scatter(pairs, figure_paths[3])
    write_report(pairs, datasets, stages, assessment, report_path, contract)
    audit_path = output_dir / "audit_manifest.json"
    inventory = {item["path"]: item for item in source_inventory}
    code_paths = [contract_path, Path(__file__), Path("scripts/run_cross_dataset_closure_synthesis.ps1")]
    for path in code_paths:
        inventory[portable(path)] = {"path": portable(path), "sha256": sha256_file(path), "role": "synthesis_code_or_contract"}
    for path in [pair_path, dataset_path, stage_path, assessment_path, *figure_paths, report_path]:
        inventory[portable(path)] = {"path": portable(path), "sha256": sha256_file(path), "role": "synthesis_output"}
    audit = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "split": "dev",
        "preflight": preflight,
        "assessment": assessment,
        "source_and_output_hash_count": len(inventory),
        "sources_and_outputs": [inventory[key] for key in sorted(inventory)],
        "hash_inventory_note": "audit_manifest.json self-hash excluded to avoid recursive content",
        "operational_audit": contract["operational_boundaries"],
        "test_status": "LOCKED",
        "next_stage_started": 0,
    }
    write_text_lf(audit_path, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
