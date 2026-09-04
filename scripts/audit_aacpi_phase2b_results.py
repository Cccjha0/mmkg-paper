from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_aacpi_advantage_nested_cv import (
    ModelConfig,
    evaluate_predictions,
    inner_selection_key,
)


FLOAT_TOLERANCE = 1e-12
PAIR_IDS = (
    "mkgw_mhyper_native",
    "mkgw_mhyper_adamf",
    "mkgw_native_adamf",
    "db15k_mhyper_native",
    "db15k_mhyper_adamf",
    "db15k_native_adamf",
)
NATIVE_ADAMF_PAIRS = {"mkgw_native_adamf", "db15k_native_adamf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and aggregate AACPI V2 Phase 2B DEV OOF results."
    )
    parser.add_argument("--phase2b-dir", default="outputs/aacpi/phase2b")
    parser.add_argument(
        "--search-space",
        default="docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml",
    )
    parser.add_argument(
        "--report",
        default="docs/reports/aacpi_v2_phase2b_oof_audit_2026-09-04.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right) -> bool:
    return bool(
        np.allclose(
            np.asarray(left),
            np.asarray(right),
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
            equal_nan=False,
        )
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_selected_configs(pair_dir: Path) -> Counter:
    search = pd.read_csv(pair_dir / "inner_search_results.csv")
    selected = pd.read_csv(pair_dir / "outer_fold_selections.csv")
    if len(search) != 5 * 36 or len(selected) != 5:
        raise AssertionError(f"Unexpected nested-CV result count in {pair_dir}")

    for fold in range(1, 6):
        candidates = []
        for row in search.loc[search.outer_fold == fold].to_dict("records"):
            config = ModelConfig(
                hidden_width=int(row["hidden_width"]),
                learning_rate=float(row["learning_rate"]),
                negative_weight=float(row["negative_weight"]),
            )
            if config.config_id != row["config_id"]:
                raise AssertionError(f"Config ID mismatch in {pair_dir}, fold {fold}")
            candidates.append((inner_selection_key(row, config), config.config_id))
        expected = max(candidates, key=lambda item: item[0])[1]
        actual = selected.loc[
            selected.outer_fold == fold, "selected_config_id"
        ].iloc[0]
        if expected != actual:
            raise AssertionError(
                f"Inner selection mismatch in {pair_dir}, fold {fold}: "
                f"expected {expected}, found {actual}"
            )
    return Counter(selected["selected_config_id"].tolist())


def audit_pair(pair_dir: Path, search_space_path: Path) -> tuple[dict, dict]:
    pair_id = pair_dir.name
    metrics_path = pair_dir / "dev_oof_metrics.json"
    audit_path = pair_dir / "phase2b_input_audit.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    utility_path = Path(audit["utility_table"])
    if audit["split"] != "dev" or metrics["split"] != "dev":
        raise AssertionError(f"Non-DEV input found for {pair_id}")
    if audit["test_accessed"] is not False:
        raise AssertionError(f"TEST access flag is not false for {pair_id}")
    if sha256_file(utility_path) != audit["utility_sha256"]:
        raise AssertionError(f"Utility SHA mismatch for {pair_id}")
    if sha256_file(search_space_path) != audit["search_space_sha256"]:
        raise AssertionError(f"Search-space SHA mismatch for {pair_id}")

    utility = pd.read_csv(utility_path)
    prediction = pd.read_csv(pair_dir / "dev_oof_predictions.csv.gz")
    if len(utility) != len(prediction) or len(prediction) != metrics["n_rows"]:
        raise AssertionError(f"Row count mismatch for {pair_id}")

    row_key = ["original_triple_id", "query_id", "seed", "direction", "alpha"]
    if utility.duplicated(row_key).any() or prediction.duplicated(row_key).any():
        raise AssertionError(f"Duplicate query-action row for {pair_id}")
    if not utility[row_key].equals(prediction[row_key]):
        raise AssertionError(f"Utility/prediction row-key mismatch for {pair_id}")
    for field in ("alpha0", "delta_alpha", "rr_anchor", "rr_action", "advantage"):
        if not close(utility[field], prediction[field]):
            raise AssertionError(f"Utility/prediction {field} mismatch for {pair_id}")

    predicted = prediction["predicted_advantage_oof"].to_numpy()
    if not np.isfinite(predicted).all():
        raise AssertionError(f"Non-finite OOF prediction for {pair_id}")
    if set(prediction["outer_fold"].unique()) != {1, 2, 3, 4, 5}:
        raise AssertionError(f"Unexpected outer folds for {pair_id}")
    if prediction.groupby("original_triple_id")["outer_fold"].nunique().max() != 1:
        raise AssertionError(f"Original-triple leakage for {pair_id}")
    if prediction.groupby("query_id")["outer_fold"].nunique().max() != 1:
        raise AssertionError(f"Query leakage for {pair_id}")
    if prediction.groupby("query_id")["selected_config_id"].nunique().max() != 1:
        raise AssertionError(f"Mixed OOF models within a query for {pair_id}")

    reference = np.isclose(
        prediction["alpha"], prediction["alpha0"], rtol=0.0, atol=FLOAT_TOLERANCE
    )
    if not close(prediction.loc[reference, "advantage"], 0.0):
        raise AssertionError(f"Nonzero reference advantage for {pair_id}")
    if not close(
        prediction.loc[reference, "rr_action"],
        prediction.loc[reference, "rr_anchor"],
    ):
        raise AssertionError(f"Reference RR mismatch for {pair_id}")

    nonreference = ~reference
    recomputed = evaluate_predictions(
        prediction.loc[nonreference, "advantage"].to_numpy(),
        prediction.loc[nonreference, "predicted_advantage_oof"].to_numpy(),
        beta=0.02,
    )
    for name, expected in metrics["primary_metrics"].items():
        if not np.isclose(recomputed[name], expected, rtol=0.0, atol=FLOAT_TOLERANCE):
            raise AssertionError(f"Metric mismatch for {pair_id}: {name}")

    selected_counts = validate_selected_configs(pair_dir)
    per_fold = []
    for fold in range(1, 6):
        mask = nonreference & (prediction["outer_fold"] == fold)
        per_fold.append(
            evaluate_predictions(
                prediction.loc[mask, "advantage"].to_numpy(),
                prediction.loc[mask, "predicted_advantage_oof"].to_numpy(),
                beta=0.02,
            )
        )

    primary = metrics["primary_metrics"]
    checks = metrics["go_signal_components"]
    row = {
        "dataset": metrics["dataset"],
        "pair_id": pair_id,
        "n_rows": len(prediction),
        "n_original_triples": prediction["original_triple_id"].nunique(),
        "spearman": primary["spearman"],
        "positive_auprc": primary["positive_auprc"],
        "positive_prevalence": primary["positive_prevalence"],
        "positive_auprc_lift": primary["positive_auprc_lift"],
        "harmful_auprc": primary["harmful_auprc"],
        "harmful_prevalence": primary["harmful_prevalence"],
        "harmful_auprc_lift": primary["harmful_auprc_lift"],
        "highest_10pct_actual_mean_u": metrics["calibration"][-1]["actual_mean_u"],
        "h1_pair_pass": all(checks.values()),
        "fold_spearman_min": min(item["spearman"] for item in per_fold),
        "fold_spearman_max": max(item["spearman"] for item in per_fold),
        "fold_positive_lift_min": min(item["positive_auprc_lift"] for item in per_fold),
        "fold_positive_lift_max": max(item["positive_auprc_lift"] for item in per_fold),
        "fold_harmful_lift_min": min(item["harmful_auprc_lift"] for item in per_fold),
        "fold_harmful_lift_max": max(item["harmful_auprc_lift"] for item in per_fold),
        "selected_configs": ";".join(
            f"{config}:{count}" for config, count in sorted(selected_counts.items())
        ),
    }
    detail = {
        "pair_id": pair_id,
        "go_signal_components": checks,
        "outer_fold_metrics": per_fold,
        "selected_config_counts": dict(sorted(selected_counts.items())),
    }
    return row, detail


def render_report(rows: list[dict], decision: dict) -> str:
    lines = [
        "# AACPI V2 Phase 2B OOF Audit",
        "",
        "All results are DEV-only. No TEST evaluation or policy evaluation was performed.",
        "",
        "## Frozen H1 result",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        f"Pairs passing all four signals: {decision['pair_pass_count']}/6 "
        f"(required: {decision['required_pair_pass_count']}/6).",
        "",
        f"Both NativE + AdaMF-MAT pairs pass: "
        f"{str(decision['both_native_adamf_pass']).lower()} (required: true).",
        "",
        "| Dataset | Pair | Spearman | Positive AP lift | Harmful AP lift | "
        "Highest 10% actual mean U | Pair pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pair_id']} | {row['spearman']:.6f} | "
            f"{row['positive_auprc_lift']:+.6f} | {row['harmful_auprc_lift']:+.6f} | "
            f"{row['highest_10pct_actual_mean_u']:+.6f} | "
            f"{'PASS' if row['h1_pair_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            "- Utility and frozen search-space SHA-256 values match every run audit.",
            "- Every prediction row matches its frozen utility row and actual advantage.",
            "- Every original triple and query belongs to exactly one outer fold.",
            "- Every query-action row has one finite outer-fold OOF prediction.",
            "- Each selected outer-fold configuration reproduces the frozen inner-CV selection rule.",
            "- Reference-action advantages and RR values satisfy the anchor identities.",
            "",
            "## Interpretation",
            "",
            "DB15K NativE + AdaMF-MAT passes all four signals, but MKG-W NativE + "
            "AdaMF-MAT is near random for sign discrimination: harmful AP is below "
            "prevalence and its highest predicted-advantage bucket has negative actual "
            "mean advantage. The frozen H1 gate therefore fails.",
            "",
            "Under the preregistered response, Phase 2C and conservative policy work do "
            "not proceed from this run. The next research question is identifiability "
            "with the frozen 13 score-geometry features; MLP capacity and the action grid "
            "must not be expanded in response to this result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    phase2b_dir = Path(args.phase2b_dir)
    search_space_path = Path(args.search_space)
    missing = [pair_id for pair_id in PAIR_IDS if not (phase2b_dir / pair_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 2B pair directories: {missing}")

    rows = []
    details = []
    for pair_id in PAIR_IDS:
        row, detail = audit_pair(phase2b_dir / pair_id, search_space_path)
        rows.append(row)
        details.append(detail)

    pair_pass_count = sum(bool(row["h1_pair_pass"]) for row in rows)
    native_pass = {
        row["pair_id"]: bool(row["h1_pair_pass"])
        for row in rows
        if row["pair_id"] in NATIVE_ADAMF_PAIRS
    }
    decision = {
        "decision": "NO-GO",
        "pair_pass_count": pair_pass_count,
        "required_pair_pass_count": 4,
        "native_adamf_pass": native_pass,
        "both_native_adamf_pass": all(native_pass.values()),
        "h1_go": pair_pass_count >= 4 and all(native_pass.values()),
        "next_step": "investigate_identifiability_without_expanding_mlp_capacity",
        "phase2c_authorized_by_gate": False,
        "test_accessed": False,
    }
    if decision["h1_go"]:
        decision["decision"] = "H1-GO"
        decision["phase2c_authorized_by_gate"] = True
        decision["next_step"] = "run_phase2c_advantage_greedy"

    summary_csv = phase2b_dir / "phase2b_aggregate_summary.csv"
    summary_json = phase2b_dir / "phase2b_aggregate_audit.json"
    report_path = Path(args.report)
    write_csv(summary_csv, rows)
    summary_json.write_text(
        json.dumps(
            {"schema_version": 1, "decision": decision, "pairs": details},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows, decision), encoding="utf-8")
    print(
        f"[OK] audited={len(rows)} decision={decision['decision']} "
        f"pair_passes={pair_pass_count}/6 report={report_path}"
    )


if __name__ == "__main__":
    main()
