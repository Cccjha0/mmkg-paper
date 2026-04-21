import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import ROUTER_MODE_CLEAN, ROUTER_MODE_POSTHOC
from router.io_utils import read_csv, write_json


EXPECTED_REGIMES = {"head_has_img", "head_no_img", "tail_no_img"}
EXPECTED_CLEAN_MAIN_MODELS = {"Gate-only", "Residual-only", "Full Model", "Oracle", "rule", "logistic", "xgb"}
ILLEGAL_CLEAN_FEATURES = {
    "target_has_img",
    "target_regime",
    "fusion_correct_score",
    "struct_correct_score",
    "fusion_margin",
    "struct_margin",
    "delta_margin",
}


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


def clean_feature_columns_are_legal(path: Path) -> bool:
    if not path.exists():
        return False
    cols = json.loads(path.read_text(encoding="utf-8"))
    return all(col not in ILLEGAL_CLEAN_FEATURES for col in cols)


def best_row(rows: list[dict], router_mode: str) -> dict | None:
    candidates = [row for row in rows if row["router_mode"] == router_mode and row["model"] in {"logistic", "xgb"}]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["mrr"]))


def feature_ablation_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = read_csv(path)
    return {(row["model"], row["feature_set"]) for row in rows}


def scan_best_row(scan_dir: Path, model: str) -> dict | None:
    candidates = sorted(scan_dir.glob(f"threshold_scan_clean_{model}_delta_*_*.csv"))
    all_rows: list[dict] = []
    for path in candidates:
        all_rows.extend(read_csv(path))
    if not all_rows:
        return None
    return max(all_rows, key=lambda row: float(row["overall_mrr"]))


def consistent_best_with_main(clean_main_rows: list[dict], clean_scan_dir: Path) -> bool:
    clean_learned = [row for row in clean_main_rows if row["model"] in {"logistic", "xgb"}]
    if not clean_learned:
        return False
    main_best = max(clean_learned, key=lambda row: float(row["mrr"]))
    scan_rows = [row for row in [scan_best_row(clean_scan_dir, "logistic"), scan_best_row(clean_scan_dir, "xgb")] if row is not None]
    if not scan_rows:
        return False
    scan_best = max(scan_rows, key=lambda row: float(row["overall_mrr"]))
    return (
        str(main_best["model"]) == str(scan_best["model"])
        and str(main_best["delta"]) == str(scan_best["delta"])
        and float(main_best["tau"]) == float(scan_best["tau"])
        and abs(float(main_best["mrr"]) - float(scan_best["overall_mrr"])) < 1e-12
    )


def main() -> None:
    files = {
        "clean_main_csv": require("outputs/router/eval/clean/main_results_table_clean.csv"),
        "clean_subgroup_csv": require("outputs/router/eval/clean/subgroup_results_table_clean.csv"),
        "clean_feature_ablation_csv": require("outputs/router/eval/clean/feature_ablation_clean.csv"),
        "posthoc_main_csv": require("outputs/router/eval/posthoc/main_results_table_posthoc.csv"),
        "posthoc_subgroup_csv": require("outputs/router/eval/posthoc/subgroup_results_table_posthoc.csv"),
        "posthoc_feature_ablation_csv": require("outputs/router/eval/posthoc/feature_ablation_posthoc.csv"),
        "clean_threshold_scan": require("outputs/router/eval/clean"),
        "posthoc_threshold_scan": require("outputs/router/eval/posthoc"),
    }

    main_table_path = Path("outputs/router/eval/main_results_table.csv")
    subgroup_table_path = Path("outputs/router/eval/subgroup_results_table.csv")
    main_rows = read_csv(main_table_path) if main_table_path.exists() else []
    subgroup_rows = read_csv(subgroup_table_path) if subgroup_table_path.exists() else []
    clean_mode_main_rows = read_csv(Path(files["clean_main_csv"]["path"])) if Path(files["clean_main_csv"]["path"]).exists() else []

    clean_best = best_row(main_rows, ROUTER_MODE_CLEAN)
    posthoc_best = best_row(main_rows, ROUTER_MODE_POSTHOC)

    clean_feature_models = [
        path
        for path in Path("outputs/router/models/clean").glob("**/feature_columns.json")
    ] if Path("outputs/router/models/clean").exists() else []
    clean_legal = all(clean_feature_columns_are_legal(path) for path in clean_feature_models) if clean_feature_models else False

    subgroup_coverage = {}
    for mode in [ROUTER_MODE_CLEAN, ROUTER_MODE_POSTHOC]:
        mode_rows = [row for row in subgroup_rows if row["router_mode"] == mode]
        subgroup_coverage[mode] = {}
        for model in sorted({row["model"] for row in mode_rows}):
            subgroup_coverage[mode][model] = sorted({row["target_regime"] for row in mode_rows if row["model"] == model})

    checks = {
        "all_clean_required_files_exist": files["clean_main_csv"]["exists"] and files["clean_subgroup_csv"]["exists"],
        "all_posthoc_required_files_exist": files["posthoc_main_csv"]["exists"] and files["posthoc_subgroup_csv"]["exists"],
        "clean_main_table_has_expected_models": EXPECTED_CLEAN_MAIN_MODELS.issubset({row["model"] for row in clean_mode_main_rows}),
        "clean_scan_best_model_consistent_with_clean_main_table": consistent_best_with_main(
            clean_mode_main_rows,
            Path(files["clean_threshold_scan"]["path"]),
        ),
        "clean_feature_ablation_has_C1_to_C4": feature_ablation_pairs(Path(files["clean_feature_ablation_csv"]["path"])) == {
            ("logistic", "C1"),
            ("logistic", "C2"),
            ("logistic", "C3"),
            ("logistic", "C4"),
            ("xgb", "C1"),
            ("xgb", "C2"),
            ("xgb", "C3"),
            ("xgb", "C4"),
        },
        "posthoc_feature_ablation_has_PH1_to_PH4_or_full": feature_ablation_pairs(Path(files["posthoc_feature_ablation_csv"]["path"])).issuperset(
            {
                ("logistic", "PH1"),
                ("logistic", "PH2"),
                ("logistic", "PH3"),
                ("logistic", "PH4"),
                ("xgb", "PH1"),
                ("xgb", "PH2"),
                ("xgb", "PH3"),
                ("xgb", "PH4"),
            }
        ),
        "no_illegal_features_in_clean_feature_columns": clean_legal,
        "clean_subgroup_has_expected_regimes": all(set(regimes) == EXPECTED_REGIMES for regimes in subgroup_coverage.get("clean", {}).values()) if subgroup_coverage.get("clean") else False,
        "posthoc_subgroup_has_expected_regimes": all(set(regimes) == EXPECTED_REGIMES for regimes in subgroup_coverage.get("posthoc", {}).values()) if subgroup_coverage.get("posthoc") else False,
        "posthoc_best_ge_clean_best": (
            clean_best is not None and posthoc_best is not None and float(posthoc_best["mrr"]) >= float(clean_best["mrr"])
        ),
    }

    payload = {
        "final_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "key_rows": {
            "clean_best": clean_best,
            "posthoc_best": posthoc_best,
        },
        "notes": [
            "Main paper should consume clean outputs only.",
            "Posthoc outputs are analysis-only and may legally use target-aware or confidence-aware fields.",
            "The legality regression check scans clean model feature_columns.json files for forbidden fields.",
            "The clean best-model consistency check compares the best learned row in the clean main table against clean threshold scan csvs.",
        ],
    }
    write_json("outputs/router/eval/final_results_manifest.json", payload)

    md_lines = [
        "# Final Results Manifest",
        "",
        f"- Final status: `{payload['final_status']}`",
        "",
        "## Checks",
        "",
        render_markdown_table([{"check": key, "passed": value} for key, value in checks.items()], ["check", "passed"]),
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
