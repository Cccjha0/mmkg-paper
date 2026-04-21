import argparse
import json
import pickle
import re
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import ROUTER_MODE_CLEAN, ROUTER_MODE_POSTHOC
from router.io_utils import read_csv, write_csv
from router.router_models import CleanRuleBasedRouter, PosthocRuleBasedRouter, compute_feature_importance_rows
from router.routing_utils import compute_eval_summary, hard_route, select_expert_row


MAIN_HEADER = [
    "category",
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source",
]

SUBGROUP_HEADER = [
    "category",
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "target_regime",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-mode", choices=[ROUTER_MODE_CLEAN, ROUTER_MODE_POSTHOC], required=True)
    ap.add_argument("--eval-dir", default="outputs/router/eval")
    ap.add_argument("--figures-dir", default="outputs/router/figures")
    ap.add_argument("--eval-targets", default="outputs/router/features/router_eval_targets_shared_test.parquet")
    return ap.parse_args()


def metrics_from_query_eval_rows(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_queries": n,
        "mrr": sum(float(r["rr"]) for r in rows) / n if n else 0.0,
        "hits1": sum(int(r["rank"]) <= 1 for r in rows) / n if n else 0.0,
        "hits3": sum(int(r["rank"]) <= 3 for r in rows) / n if n else 0.0,
        "hits10": sum(int(r["rank"]) <= 10 for r in rows) / n if n else 0.0,
    }


def load_query_eval_baseline(expert: str) -> tuple[dict, dict[str, dict]]:
    rows = []
    for seed in [1, 2, 3]:
        path = Path("outputs/router/test") / f"{expert}_query_eval_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing query eval file: {path.as_posix()}")
        rows.extend(read_csv(path))
    overall = metrics_from_query_eval_rows(rows)
    by_regime = {}
    for regime in sorted({row["target_regime"] for row in rows}):
        bucket = [row for row in rows if row["target_regime"] == regime]
        by_regime[regime] = metrics_from_query_eval_rows(bucket)
    return overall, by_regime


def maybe_load_query_eval_baseline(expert: str) -> tuple[dict, dict[str, dict]] | None:
    try:
        return load_query_eval_baseline(expert)
    except FileNotFoundError:
        return None


def parse_main_results_summary() -> dict[str, dict]:
    path = Path("docs/MAIN_RESULTS_SUMMARY.md")
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    pattern = re.compile(
        r"^\|\s*(?P<model>[^|]+?)\s*\|\s*(?P<seeds>\d+)\s*\|\s*(?P<mrr>[0-9.]+)\s*[^\d|]+\s*[0-9.]+\s*\|\s*(?P<h1>[0-9.]+)\s*[^\d|]+\s*[0-9.]+\s*\|\s*(?P<h3>[0-9.]+)\s*[^\d|]+\s*[0-9.]+\s*\|\s*(?P<h10>[0-9.]+)\s*[^\d|]+\s*[0-9.]+",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        model = match.group("model").strip()
        out[model] = {
            "n_queries": None,
            "mrr": float(match.group("mrr")),
            "hits1": float(match.group("h1")),
            "hits3": float(match.group("h3")),
            "hits10": float(match.group("h10")),
        }
    return out


def render_markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def write_markdown_table(rows: list[dict], columns: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown_table(rows, columns), encoding="utf-8")


def default_test_feature_path(router_mode: str) -> Path:
    if router_mode == ROUTER_MODE_CLEAN:
        return Path("outputs/router/features/router_test_clean_features.csv")
    return Path("outputs/router/features/router_test_posthoc_features.csv")


def load_eval_targets(path: str) -> dict[str, dict]:
    if str(path).lower().endswith(".csv"):
        rows = read_csv(path)
    else:
        rows = __import__("pandas").read_parquet(path).to_dict(orient="records")
    return {str(row["query_id"]): row for row in rows}


def load_feature_importance_rows(router_mode: str) -> list[dict]:
    model_root = Path("outputs/router/models") / router_mode
    if not model_root.exists():
        return []

    rows: list[dict] = []
    for model_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        model_path = model_dir / "model.pkl"
        summary_path = model_dir / "train_summary.json"
        if not model_path.exists() or not summary_path.exists():
            continue
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in compute_feature_importance_rows(artifact):
            enriched = dict(row)
            enriched["router_mode"] = router_mode
            enriched["delta"] = summary.get("delta_tag", f"{float(summary['delta']):.2f}")
            enriched["is_query_time_legal"] = bool(summary.get("is_query_time_legal", router_mode == ROUTER_MODE_CLEAN))
            rows.append(enriched)
    rows.sort(key=lambda row: (row["model"], row["feature_set"], row["feature_name"]))
    return rows


def route_from_features(rows: list[dict], probs: list[float], tau: float, selected_by: str, eval_targets: dict[str, dict]) -> list[dict]:
    routed_rows = []
    for row, prob in zip(rows, probs):
        merged = dict(row)
        merged["router_prob"] = float(prob)
        merged["threshold"] = float(tau)
        target = eval_targets[str(row["query_id"])]
        eval_meta = {
            "target_regime": target["target_regime"],
            "rank_gate": target.get("rank_gate", 0),
            "rr_gate": target["rr_gate"],
            "rank_residual": target.get("rank_residual", 0),
            "rr_residual": target["rr_residual"],
        }
        use_fusion = hard_route(prob, tau)
        routed_rows.append(select_expert_row(merged, eval_meta, use_fusion, selected_by=selected_by))
    return routed_rows


def load_rule_based_eval(router_mode: str, eval_targets: dict[str, dict], tau: float = 0.5) -> tuple[dict, dict[str, dict]]:
    rows = read_csv(default_test_feature_path(router_mode))
    if router_mode == ROUTER_MODE_CLEAN:
        router = CleanRuleBasedRouter(gamma=0.0)
    else:
        router = PosthocRuleBasedRouter(gamma=0.0)
    probs = router.predict_proba_from_rows(rows)
    payload = compute_eval_summary(route_from_features(rows, probs, tau, "rule", eval_targets))
    return payload["overall"], payload["by_regime"]


def load_oracle_eval(router_mode: str, eval_targets: dict[str, dict]) -> tuple[dict, dict[str, dict]]:
    rows = read_csv(default_test_feature_path(router_mode))
    routed_rows = []
    for row in rows:
        merged = dict(row)
        target = eval_targets[str(row["query_id"])]
        rr_gate = float(target["rr_gate"])
        rr_residual = float(target["rr_residual"])
        eval_meta = {
            "target_regime": target["target_regime"],
            "rank_gate": target.get("rank_gate", 0),
            "rr_gate": rr_gate,
            "rank_residual": target.get("rank_residual", 0),
            "rr_residual": rr_residual,
        }
        use_fusion = int(rr_gate > rr_residual)
        merged["router_prob"] = 1.0 if use_fusion else 0.0
        merged["threshold"] = 0.5
        routed_rows.append(select_expert_row(merged, eval_meta, use_fusion, selected_by="oracle"))
    payload = compute_eval_summary(routed_rows)
    return payload["overall"], payload["by_regime"]


def best_routing_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: float(row["mrr"]))


def latest_threshold_scan(eval_mode_dir: Path, model: str) -> list[dict]:
    candidates = sorted(eval_mode_dir.glob(f"threshold_scan_*_{model}_delta_*_*.csv"))
    if not candidates:
        return []
    return read_csv(candidates[-1])


def latest_feature_ablation(eval_mode_dir: Path, router_mode: str) -> list[dict]:
    path = eval_mode_dir / f"feature_ablation_{router_mode}.csv"
    if not path.exists():
        return []
    return read_csv(path)


def write_feature_importance_outputs(router_mode: str, rows: list[dict], eval_mode_dir: Path, figures_mode_dir: Path) -> None:
    if not rows:
        return

    csv_path = eval_mode_dir / f"router_feature_importance_{router_mode}.csv"
    md_path = eval_mode_dir / f"router_feature_importance_{router_mode}.md"
    png_path = figures_mode_dir / f"router_feature_importance_{router_mode}.png"
    columns = ["router_mode", "delta", "model", "feature_set", "feature_name", "importance", "is_query_time_legal"]
    write_csv(csv_path, rows, columns)

    top_rows = sorted(rows, key=lambda row: float(row["importance"]), reverse=True)[:20]
    write_markdown_table(top_rows, columns, md_path)

    if plt is None:
        return

    picked = [row for row in rows if row["model"] == "xgb"]
    if not picked:
        picked = rows
    picked = sorted(picked, key=lambda row: float(row["importance"]), reverse=True)[:12]
    labels = [row["feature_name"] for row in picked][::-1]
    values = [float(row["importance"]) for row in picked][::-1]

    figures_mode_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, values, color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title(f"Router Feature Importance ({router_mode})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def save_threshold_figure(logistic_rows: list[dict], xgb_rows: list[dict], out_path: Path, router_mode: str) -> None:
    if plt is None or not logistic_rows or not xgb_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for label, rows, color in [("Logistic MRR", logistic_rows, "#1f77b4"), ("XGB MRR", xgb_rows, "#d62728")]:
        taus = [float(r["tau"]) for r in rows]
        mrrs = [float(r["overall_mrr"]) for r in rows]
        covs = [float(r["fusion_coverage"]) for r in rows]
        ax1.plot(taus, mrrs, marker="o", label=label, color=color)
        ax2.plot(taus, covs, marker="s", linestyle="--", label=label.replace("MRR", "Coverage"), color=color, alpha=0.65)

    ax1.set_xlabel("Threshold tau")
    ax1.set_ylabel("Overall MRR")
    ax2.set_ylabel("Fusion Coverage")
    ax1.set_title(f"Threshold vs Coverage vs MRR ({router_mode})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_takeaways(out_path: Path, router_mode: str, best_xgb: dict | None, best_logistic: dict | None, best_ablation: dict | None) -> None:
    lines = [f"# Router Takeaways ({router_mode})", ""]
    if best_xgb is not None:
        lines.append(f"- Best XGB row: delta={best_xgb['delta']}, tau={best_xgb['tau']}, mrr={float(best_xgb['mrr']):.4f}")
    if best_logistic is not None:
        lines.append(f"- Best logistic row: delta={best_logistic['delta']}, tau={best_logistic['tau']}, mrr={float(best_logistic['mrr']):.4f}")
    if best_ablation is not None:
        lines.append(
            f"- Best ablation row: model={best_ablation['model']}, feature_set={best_ablation['feature_set']}, overall_mrr={float(best_ablation['overall_mrr']):.4f}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    router_mode = args.router_mode
    eval_targets = load_eval_targets(args.eval_targets)
    summary_rows = parse_main_results_summary()
    gate_overall, gate_by_regime = load_query_eval_baseline("gate_only")
    residual_overall, residual_by_regime = load_query_eval_baseline("residual_only")
    full_model_bundle = maybe_load_query_eval_baseline("full_model")
    rule_overall, rule_by_regime = load_rule_based_eval(router_mode, eval_targets)
    oracle_overall, oracle_by_regime = load_oracle_eval(router_mode, eval_targets)

    eval_mode_dir = Path(args.eval_dir) / router_mode
    figures_mode_dir = Path(args.figures_dir) / router_mode
    routing_main_existing = read_csv(Path(args.eval_dir) / "main_results_table.csv")
    routing_subgroup_existing = read_csv(Path(args.eval_dir) / "subgroup_results_table.csv")
    routing_main = [r for r in routing_main_existing if r.get("router_mode") == router_mode and r["model"] in {"logistic", "xgb", "rule"}]
    routing_subgroup = [r for r in routing_subgroup_existing if r.get("router_mode") == router_mode and r["model"] in {"logistic", "xgb", "rule"}]

    main_rows = [
        {
            "category": "fixed_expert",
            "router_mode": router_mode,
            "feature_set": "",
            "is_query_time_legal": router_mode == ROUTER_MODE_CLEAN,
            "model": "Gate-only",
            "delta": "",
            "tau": "",
            "n_queries": gate_overall["n_queries"],
            "mrr": gate_overall["mrr"],
            "hits1": gate_overall["hits1"],
            "hits3": gate_overall["hits3"],
            "hits10": gate_overall["hits10"],
            "fusion_coverage": "",
            "source": "query_eval",
        },
        {
            "category": "fixed_expert",
            "router_mode": router_mode,
            "feature_set": "",
            "is_query_time_legal": router_mode == ROUTER_MODE_CLEAN,
            "model": "Residual-only",
            "delta": "",
            "tau": "",
            "n_queries": residual_overall["n_queries"],
            "mrr": residual_overall["mrr"],
            "hits1": residual_overall["hits1"],
            "hits3": residual_overall["hits3"],
            "hits10": residual_overall["hits10"],
            "fusion_coverage": "",
            "source": "query_eval",
        },
    ]
    if router_mode == ROUTER_MODE_CLEAN and "Full Model" in summary_rows:
        row = summary_rows["Full Model"]
        main_rows.append(
            {
                "category": "paper_baseline",
                "router_mode": router_mode,
                "feature_set": "",
                "is_query_time_legal": True,
                "model": "Full Model",
                "delta": "",
                "tau": "",
                "n_queries": "",
                "mrr": row["mrr"],
                "hits1": row["hits1"],
                "hits3": row["hits3"],
                "hits10": row["hits10"],
                "fusion_coverage": "",
                "source": "docs/MAIN_RESULTS_SUMMARY.md",
            }
        )
    main_rows.extend(
        [
            {
                "category": "oracle",
                "router_mode": router_mode,
                "feature_set": "oracle",
                "is_query_time_legal": False,
                "model": "Oracle",
                "delta": "",
                "tau": "",
                "n_queries": oracle_overall["n_queries"],
                "mrr": oracle_overall["mrr"],
                "hits1": oracle_overall["hits1"],
                "hits3": oracle_overall["hits3"],
                "hits10": oracle_overall["hits10"],
                "fusion_coverage": oracle_overall["fusion_coverage"],
                "source": f"recomputed_from_router_test_{router_mode}_features",
            },
            {
                "category": "rule_based",
                "router_mode": router_mode,
                "feature_set": "rule",
                "is_query_time_legal": router_mode == ROUTER_MODE_CLEAN,
                "model": "rule",
                "delta": "0.01",
                "tau": 0.5,
                "n_queries": rule_overall["n_queries"],
                "mrr": rule_overall["mrr"],
                "hits1": rule_overall["hits1"],
                "hits3": rule_overall["hits3"],
                "hits10": rule_overall["hits10"],
                "fusion_coverage": rule_overall["fusion_coverage"],
                "source": f"recomputed_from_router_test_{router_mode}_features",
            },
        ]
    )
    for row in routing_main:
        main_rows.append(
            {
                "category": "learned_router",
                "router_mode": row["router_mode"],
                "feature_set": row["feature_set"],
                "is_query_time_legal": row["is_query_time_legal"],
                "model": row["model"],
                "delta": row["delta"],
                "tau": row["tau"],
                "n_queries": row["n_queries"],
                "mrr": row["mrr"],
                "hits1": row["hits1"],
                "hits3": row["hits3"],
                "hits10": row["hits10"],
                "fusion_coverage": row["fusion_coverage"],
                "source": "router_eval_json",
            }
        )

    main_csv = eval_mode_dir / f"main_results_table_{router_mode}.csv"
    main_md = eval_mode_dir / f"main_results_table_{router_mode}.md"
    write_csv(main_csv, main_rows, MAIN_HEADER)
    write_markdown_table(main_rows, MAIN_HEADER, main_md)

    subgroup_rows = []
    for model_name, bucket, category in [
        ("Gate-only", gate_by_regime, "fixed_or_rule"),
        ("Residual-only", residual_by_regime, "fixed_or_rule"),
        ("rule", rule_by_regime, "fixed_or_rule"),
        ("Oracle", oracle_by_regime, "oracle"),
    ]:
        for regime, stats in sorted(bucket.items()):
            subgroup_rows.append(
                {
                    "category": category,
                    "router_mode": router_mode,
                    "feature_set": "rule" if model_name == "rule" else "",
                    "is_query_time_legal": router_mode == ROUTER_MODE_CLEAN if model_name == "rule" else "",
                    "model": model_name,
                    "delta": "0.01" if model_name == "rule" else "",
                    "tau": 0.5 if model_name == "rule" else "",
                    "target_regime": regime,
                    "n_queries": stats["n_queries"],
                    "mrr": stats["mrr"],
                    "hits1": stats["hits1"],
                    "hits3": stats["hits3"],
                    "hits10": stats["hits10"],
                    "fusion_coverage": stats.get("fusion_coverage", ""),
                    "source": "query_eval_or_recomputed",
                }
            )
    if router_mode == ROUTER_MODE_CLEAN and full_model_bundle is not None:
        _, full_model_by_regime = full_model_bundle
        for regime, stats in sorted(full_model_by_regime.items()):
            subgroup_rows.append(
                {
                    "category": "paper_baseline",
                    "router_mode": router_mode,
                    "feature_set": "",
                    "is_query_time_legal": True,
                    "model": "Full Model",
                    "delta": "",
                    "tau": "",
                    "target_regime": regime,
                    "n_queries": stats["n_queries"],
                    "mrr": stats["mrr"],
                    "hits1": stats["hits1"],
                    "hits3": stats["hits3"],
                    "hits10": stats["hits10"],
                    "fusion_coverage": "",
                    "source": "query_eval",
                }
            )
    for row in routing_subgroup:
        subgroup_rows.append(
            {
                "category": "learned_router",
                "router_mode": row["router_mode"],
                "feature_set": row["feature_set"],
                "is_query_time_legal": row["is_query_time_legal"],
                "model": row["model"],
                "delta": row["delta"],
                "tau": row["tau"],
                "target_regime": row["target_regime"],
                "n_queries": row["n_queries"],
                "mrr": row["mrr"],
                "hits1": row["hits1"],
                "hits3": row["hits3"],
                "hits10": row["hits10"],
                "fusion_coverage": row["fusion_coverage"],
                "source": "router_eval_json",
            }
        )

    subgroup_csv = eval_mode_dir / f"subgroup_results_table_{router_mode}.csv"
    subgroup_md = eval_mode_dir / f"subgroup_results_table_{router_mode}.md"
    write_csv(subgroup_csv, subgroup_rows, SUBGROUP_HEADER)
    write_markdown_table(subgroup_rows, SUBGROUP_HEADER, subgroup_md)

    threshold_xgb = latest_threshold_scan(eval_mode_dir, "xgb")
    threshold_logistic = latest_threshold_scan(eval_mode_dir, "logistic")
    feature_ablation = latest_feature_ablation(eval_mode_dir, router_mode)
    feature_importance = load_feature_importance_rows(router_mode)

    save_threshold_figure(threshold_logistic, threshold_xgb, figures_mode_dir / f"threshold_coverage_mrr_{router_mode}.png", router_mode)
    if feature_ablation:
        write_markdown_table(feature_ablation, list(feature_ablation[0].keys()), eval_mode_dir / f"feature_ablation_{router_mode}.md")
    write_feature_importance_outputs(router_mode, feature_importance, eval_mode_dir, figures_mode_dir)

    best_xgb = best_routing_row([row for row in main_rows if row["model"] == "xgb"]) if any(row["model"] == "xgb" for row in main_rows) else None
    best_logistic = best_routing_row([row for row in main_rows if row["model"] == "logistic"]) if any(row["model"] == "logistic" for row in main_rows) else None
    best_ablation = max(feature_ablation, key=lambda row: float(row["overall_mrr"])) if feature_ablation else None
    write_takeaways(eval_mode_dir / f"takeaways_{router_mode}.md", router_mode, best_xgb, best_logistic, best_ablation)

    print(f"[OK] wrote main table     -> {main_csv.as_posix()}")
    print(f"[OK] wrote subgroup table -> {subgroup_csv.as_posix()}")
    if feature_importance:
        print(f"[OK] wrote importance     -> {(eval_mode_dir / f'router_feature_importance_{router_mode}.csv').as_posix()}")
    print(f"[OK] wrote takeaways      -> {(eval_mode_dir / f'takeaways_{router_mode}.md').as_posix()}")


if __name__ == "__main__":
    main()
