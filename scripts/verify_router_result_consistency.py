import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_json


EXPECTED_MAIN_MODELS = {
    "Gate-only",
    "Residual-only",
    "Full Model",
    "Oracle",
    "Rule-based",
    "logistic",
    "xgb",
}

EXPECTED_SUBGROUP_MODELS = {
    "Gate-only",
    "Residual-only",
    "Full Model",
    "Oracle",
    "Rule-based",
    "logistic",
    "xgb",
}

EXPECTED_REGIMES = {"head_has_img", "head_no_img", "tail_no_img"}


def require(path: str) -> dict:
    p = Path(path)
    return {
        "path": p.as_posix(),
        "exists": p.exists(),
        "size": p.stat().st_size if p.exists() else None,
    }


def render_markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def main() -> None:
    files = {
        "main_csv": require("outputs/router/eval/main_results_table.csv"),
        "main_md": require("outputs/router/eval/main_results_table.md"),
        "subgroup_csv": require("outputs/router/eval/subgroup_results_table.csv"),
        "subgroup_md": require("outputs/router/eval/subgroup_results_table.md"),
        "feature_ablation_csv": require("outputs/router/eval/feature_ablation.csv"),
        "feature_ablation_md": require("outputs/router/eval/feature_ablation.md"),
        "takeaways_md": require("outputs/router/eval/first_round_takeaways.md"),
        "threshold_logistic_csv": require("outputs/router/eval/threshold_scan_logistic_delta_0.01.csv"),
        "threshold_xgb_csv": require("outputs/router/eval/threshold_scan_xgb_delta_0.01.csv"),
        "fig_threshold": require("outputs/router/figures/threshold_coverage_mrr.png"),
        "fig_importance": require("outputs/router/figures/router_feature_importance.png"),
        "fig_subgroup_ratio": require("outputs/router/figures/selected_fusion_ratio_by_subgroup.png"),
    }

    main_rows = read_csv("outputs/router/eval/main_results_table.csv")
    subgroup_rows = read_csv("outputs/router/eval/subgroup_results_table.csv")
    feature_rows = read_csv("outputs/router/eval/feature_ablation.csv")

    main_models = {row["model"] for row in main_rows}
    subgroup_models = {row["model"] for row in subgroup_rows}

    learned_rows = [row for row in main_rows if row["category"] == "learned_router"]
    best_learned = max(learned_rows, key=lambda row: float(row["mrr"]))
    oracle_row = next(row for row in main_rows if row["model"] == "Oracle")
    residual_row = next(row for row in main_rows if row["model"] == "Residual-only")
    full_row = next(row for row in main_rows if row["model"] == "Full Model")
    rule_row = next(row for row in main_rows if row["model"] == "Rule-based")

    subgroup_coverage = {}
    for model in sorted(EXPECTED_SUBGROUP_MODELS):
        model_rows = [row for row in subgroup_rows if row["model"] == model]
        subgroup_coverage[model] = sorted({row["target_regime"] for row in model_rows})

    checks = {
        "all_required_files_exist": all(item["exists"] for item in files.values()),
        "main_table_has_expected_models": EXPECTED_MAIN_MODELS.issubset(main_models),
        "subgroup_table_has_expected_models": EXPECTED_SUBGROUP_MODELS.issubset(subgroup_models),
        "all_models_have_three_regimes_in_subgroup": all(set(regimes) == EXPECTED_REGIMES for regimes in subgroup_coverage.values()),
        "best_learned_is_xgb_delta_0.01_tau_0.7": (
            best_learned["model"] == "xgb"
            and str(best_learned["delta"]) == "0.01"
            and str(best_learned["tau"]) == "0.7"
        ),
        "oracle_beats_best_learned": float(oracle_row["mrr"]) > float(best_learned["mrr"]),
        "best_learned_beats_residual_only": float(best_learned["mrr"]) > float(residual_row["mrr"]),
        "best_learned_beats_full_model": float(best_learned["mrr"]) > float(full_row["mrr"]),
        "learned_beats_rule_based": float(best_learned["mrr"]) > float(rule_row["mrr"]),
        "feature_ablation_has_f1_to_f4_for_logistic_and_xgb": {(r["model"], r["feature_set"]) for r in feature_rows}
        == {
            ("logistic", "F1"),
            ("logistic", "F2"),
            ("logistic", "F3"),
            ("logistic", "F4"),
            ("xgb", "F1"),
            ("xgb", "F2"),
            ("xgb", "F3"),
            ("xgb", "F4"),
        },
    }

    canonical_outputs = [
        {"group": "final_table", "name": "main_results_table.csv", "path": files["main_csv"]["path"], "status": "final"},
        {"group": "final_table", "name": "main_results_table.md", "path": files["main_md"]["path"], "status": "final"},
        {"group": "final_table", "name": "subgroup_results_table.csv", "path": files["subgroup_csv"]["path"], "status": "final"},
        {"group": "final_table", "name": "subgroup_results_table.md", "path": files["subgroup_md"]["path"], "status": "final"},
        {"group": "final_table", "name": "feature_ablation.csv", "path": files["feature_ablation_csv"]["path"], "status": "final"},
        {"group": "final_table", "name": "feature_ablation.md", "path": files["feature_ablation_md"]["path"], "status": "final"},
        {"group": "final_summary", "name": "first_round_takeaways.md", "path": files["takeaways_md"]["path"], "status": "final"},
        {"group": "final_figure", "name": "threshold_coverage_mrr.png", "path": files["fig_threshold"]["path"], "status": "final"},
        {"group": "final_figure", "name": "router_feature_importance.png", "path": files["fig_importance"]["path"], "status": "final"},
        {"group": "final_figure", "name": "selected_fusion_ratio_by_subgroup.png", "path": files["fig_subgroup_ratio"]["path"], "status": "final"},
        {"group": "supporting_input", "name": "threshold_scan_logistic_delta_0.01.csv", "path": files["threshold_logistic_csv"]["path"], "status": "supporting"},
        {"group": "supporting_input", "name": "threshold_scan_xgb_delta_0.01.csv", "path": files["threshold_xgb_csv"]["path"], "status": "supporting"},
    ]

    payload = {
        "final_status": "PASS" if all(checks.values()) else "FAIL",
        "canonical_final_outputs": canonical_outputs,
        "checks": checks,
        "key_numbers": {
            "best_learned_model": best_learned["model"],
            "best_learned_delta": best_learned["delta"],
            "best_learned_tau": best_learned["tau"],
            "best_learned_mrr": float(best_learned["mrr"]),
            "oracle_mrr": float(oracle_row["mrr"]),
            "residual_only_mrr": float(residual_row["mrr"]),
            "full_model_mrr": float(full_row["mrr"]),
            "rule_based_mrr": float(rule_row["mrr"]),
        },
        "subgroup_coverage": subgroup_coverage,
        "notes": [
            "Canonical final tables are the markdown/csv files under outputs/router/eval refreshed by scripts/make_router_tables_figures.py.",
            "Threshold scan csv files are treated as supporting inputs to the final tables/figures, not as standalone final presentation artifacts.",
            "Raw query_eval, gain_labels, priors, feature tables, trained models, and per-tau routing predictions are intermediate or supporting artifacts rather than final paper-facing outputs.",
        ],
    }
    write_json("outputs/router/eval/final_results_manifest.json", payload)

    md_lines = [
        "# Final Results Manifest",
        "",
        f"- Final status: `{payload['final_status']}`",
        "",
        "## Canonical Final Outputs",
        "",
        render_markdown_table(canonical_outputs, ["group", "name", "status", "path"]),
        "",
        "## Consistency Checks",
        "",
        render_markdown_table(
            [{"check": key, "passed": value} for key, value in checks.items()],
            ["check", "passed"],
        ),
        "",
        "## Key Numbers",
        "",
        render_markdown_table(
            [{"metric": key, "value": value} for key, value in payload["key_numbers"].items()],
            ["metric", "value"],
        ),
        "",
        "## Subgroup Coverage",
        "",
        render_markdown_table(
            [{"model": model, "regimes": ", ".join(regimes)} for model, regimes in subgroup_coverage.items()],
            ["model", "regimes"],
        ),
        "",
        "## Notes",
        "",
    ]
    md_lines.extend([f"- {note}" for note in payload["notes"]])
    Path("outputs/router/eval/final_results_manifest.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[OK] wrote manifest json -> outputs/router/eval/final_results_manifest.json")
    print("[OK] wrote manifest md   -> outputs/router/eval/final_results_manifest.md")


if __name__ == "__main__":
    main()
