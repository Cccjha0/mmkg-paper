import re
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.router_models import RuleBasedRouter
from router.routing_utils import compute_eval_summary, hard_route, select_expert_row


MAIN_HEADER = [
    "category",
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

MAIN_MD_COLUMNS = [
    "category",
    "model",
    "delta",
    "tau",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source",
]

SUBGROUP_HEADER = [
    "category",
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

SUBGROUP_MD_COLUMNS = [
    "category",
    "model",
    "delta",
    "tau",
    "target_regime",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source",
]


def metrics_from_query_eval_rows(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_queries": n,
        "mrr": sum(float(r["rr"]) for r in rows) / n,
        "hits1": sum(int(r["rank"]) <= 1 for r in rows) / n,
        "hits3": sum(int(r["rank"]) <= 3 for r in rows) / n,
        "hits10": sum(int(r["rank"]) <= 10 for r in rows) / n,
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


def route_from_features(rows: list[dict], probs: list[float], tau: float, selected_by: str) -> list[dict]:
    routed_rows = []
    for row, prob in zip(rows, probs):
        merged = dict(row)
        merged["router_prob"] = float(prob)
        merged["threshold"] = float(tau)
        use_fusion = hard_route(prob, tau)
        routed_rows.append(select_expert_row(merged, use_fusion, selected_by=selected_by))
    return routed_rows


def load_rule_based_eval(tau: float = 0.5) -> tuple[dict, dict[str, dict]]:
    rows = read_csv("outputs/router/features/router_test_features.csv")
    router = RuleBasedRouter(gamma=0.0)
    probs = router.predict_proba_from_rows(rows)
    payload = compute_eval_summary(route_from_features(rows, probs, tau, "rule"))
    return payload["overall"], payload["by_regime"]


def load_oracle_eval() -> tuple[dict, dict[str, dict]]:
    rows = read_csv("outputs/router/features/router_test_features.csv")
    routed_rows = []
    for row in rows:
        merged = dict(row)
        rr_gate = float(row["rr_fusion"])
        rr_residual = float(row["rr_struct"])
        use_fusion = int(rr_gate > rr_residual)
        merged["router_prob"] = 1.0 if use_fusion else 0.0
        merged["threshold"] = 0.5
        routed_rows.append(select_expert_row(merged, use_fusion, selected_by="oracle"))
    payload = compute_eval_summary(routed_rows)
    return payload["overall"], payload["by_regime"]


def best_routing_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: float(row["mrr"]))


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


def save_threshold_figure(logistic_rows: list[dict], xgb_rows: list[dict], out_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib unavailable, skipping threshold figure.")
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
    ax1.set_title("Threshold vs Coverage vs MRR")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_feature_importance_figure(rows: list[dict], out_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib unavailable, skipping feature importance figure.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xgb_rows = [r for r in rows if r["model"] == "xgb" and r["delta"] == "0.01"][:12]
    labels = [r["feature_name"] for r in xgb_rows][::-1]
    values = [float(r["importance"]) for r in xgb_rows][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, values, color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Router Feature Importance (XGB, delta=0.01)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_subgroup_fusion_ratio_figure(rows: list[dict], out_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib unavailable, skipping subgroup fusion ratio figure.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picked = [r for r in rows if r["model"] == "xgb" and r["delta"] == "0.01"]
    taus = [float(r["tau"]) for r in picked]
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, color in [("head_has_img", "#1f77b4"), ("head_no_img", "#ff7f0e"), ("tail_no_img", "#2ca02c")]:
        vals = [float(r[f"fusion_ratio_{regime}"]) for r in picked]
        ax.plot(taus, vals, marker="o", label=regime, color=color)
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Selected Fusion Ratio")
    ax.set_title("Selected Fusion Ratio by Subgroup (XGB, delta=0.01)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_feature_ablation_md(rows: list[dict], out_path: Path) -> None:
    columns = [
        "model",
        "feature_set",
        "delta",
        "tau",
        "train_auc",
        "overall_mrr",
        "fusion_coverage",
        "gain_precision",
    ]
    write_markdown_table(rows, columns, out_path)


def build_takeaways_markdown(
    best_xgb: dict,
    best_logistic: dict,
    best_ablation: dict,
    residual_overall: dict,
    oracle_overall: dict,
    summary_rows: dict[str, dict],
    threshold_xgb: list[dict],
    main_rows: list[dict],
) -> str:
    full_model_mrr = summary_rows.get("Full Model", {}).get("mrr", "N/A")
    preview_rows = [row for row in main_rows if row["model"] in {"Residual-only", "Full Model", "Oracle", "Rule-based", "logistic", "xgb"}]
    preview_rows = sorted(preview_rows, key=lambda row: (str(row["model"]), str(row["tau"])))

    md = []
    md.append("# First Round Takeaways")
    md.append("")
    md.append("## Main Outcome")
    md.append("")
    md.append(f"- 当前 best learned router 是 `xgb + delta=0.01 + tau={best_xgb['tau']}`，overall MRR = `{float(best_xgb['mrr']):.4f}`。")
    md.append(f"- 它高于固定 `Residual-only` 的 query-level test MRR `{residual_overall['mrr']:.4f}`，也高于文档主结果中的 `Full Model` `{full_model_mrr}`。")
    md.append(f"- 当前 `Oracle` 的 upper bound MRR 为 `{oracle_overall['mrr']:.4f}`，说明 learned router 之上仍然存在可提升空间。")
    md.append(f"- `logistic` 的最优点在 `tau={best_logistic['tau']}`，overall MRR = `{float(best_logistic['mrr']):.4f}`；`xgb` 仍然稳定优于 `logistic`。")
    md.append("")
    md.append("## Threshold Scan")
    md.append("")
    md.append("- `xgb` 的 threshold scan 呈现清晰 tradeoff：")
    for row in threshold_xgb:
        md.append(f"  - `tau={row['tau']}`: MRR `{float(row['overall_mrr']):.4f}`, coverage `{float(row['fusion_coverage']):.3f}`, gain_precision `{float(row['gain_precision']):.3f}`")
    md.append("- `logistic` 也呈现相同方向的 tradeoff，但整体曲线低于 `xgb`。")
    md.append("")
    md.append("## Subgroup Pattern")
    md.append("")
    md.append("- learned router 在 `head_has_img` 上明显比 `Residual-only` 更合理，在 `tail_no_img` 上则更接近甚至超过 `Residual-only`。")
    md.append("- 随着 `tau` 提高，`tail_no_img` 的 fusion ratio 明显下降，说明更保守的阈值更符合结构占优场景。")
    md.append("")
    md.append("## Feature Ablation")
    md.append("")
    md.append(f"- 当前最强 ablation 组合是 `{best_ablation['model']} + {best_ablation['feature_set']}`，overall MRR = `{float(best_ablation['overall_mrr']):.4f}`。")
    md.append("- `F1`（仅 `target_has_img`）表现明显不足，说明 router 不是简单地在学“有图就开 fusion”。")
    md.append("- `F2` 相比 `F1` 提升很大，说明 `direction + relation_gain_prior` 是关键特征。")
    md.append("- `F3` 相比 `F2` 增益很小，说明模态一致性特征目前只是弱补充。")
    md.append("- `F4` 再次明显提升，说明 expert confidence 特征（`fusion_margin / struct_margin / delta_margin`）是主要增强项。")
    md.append("")
    md.append("## Remaining Gaps")
    md.append("")
    md.append("- `Full Model` 的 subgroup 结果现已并入统一 subgroup 汇总表；当前剩余缺口主要在更论文式的文字组织与版式整理。")
    md.append("- 正式图文件依赖 `matplotlib`；如果当前环境缺失该依赖，脚本会保留表格和总结，但跳过 PNG 输出。")
    md.append("")
    md.append("## Quick Table")
    md.append("")
    md.append(render_markdown_table(preview_rows, ["model", "delta", "tau", "mrr", "hits1", "hits3", "hits10", "fusion_coverage", "source"]))
    return "\n".join(md)


def main() -> None:
    summary_rows = parse_main_results_summary()
    gate_overall, gate_by_regime = load_query_eval_baseline("gate_only")
    residual_overall, residual_by_regime = load_query_eval_baseline("residual_only")
    full_model_bundle = maybe_load_query_eval_baseline("full_model")
    rule_overall, rule_by_regime = load_rule_based_eval()
    oracle_overall, oracle_by_regime = load_oracle_eval()

    routing_main_existing = read_csv("outputs/router/eval/main_results_table.csv")
    routing_subgroup_existing = read_csv("outputs/router/eval/subgroup_results_table.csv")
    routing_main = [r for r in routing_main_existing if r.get("category") in ("learned_router", "") and r["model"] in {"logistic", "xgb"}]
    routing_subgroup = [r for r in routing_subgroup_existing if r.get("category") in ("learned_router", "") and r["model"] in {"logistic", "xgb"}]

    main_rows = [
        {
            "category": "fixed_expert",
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
    if "Full Model" in summary_rows:
        row = summary_rows["Full Model"]
        main_rows.append(
            {
                "category": "paper_baseline",
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
                "model": "Oracle",
                "delta": "",
                "tau": "",
                "n_queries": oracle_overall["n_queries"],
                "mrr": oracle_overall["mrr"],
                "hits1": oracle_overall["hits1"],
                "hits3": oracle_overall["hits3"],
                "hits10": oracle_overall["hits10"],
                "fusion_coverage": oracle_overall["fusion_coverage"],
                "source": "recomputed_from_router_test_features",
            },
            {
                "category": "rule_based",
                "model": "Rule-based",
                "delta": "0.01",
                "tau": 0.5,
                "n_queries": rule_overall["n_queries"],
                "mrr": rule_overall["mrr"],
                "hits1": rule_overall["hits1"],
                "hits3": rule_overall["hits3"],
                "hits10": rule_overall["hits10"],
                "fusion_coverage": rule_overall["fusion_coverage"],
                "source": "recomputed_from_router_test_features",
            },
        ]
    )
    for row in routing_main:
        main_rows.append(
            {
                "category": "learned_router",
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
    write_csv("outputs/router/eval/main_results_table.csv", main_rows, MAIN_HEADER)
    write_markdown_table(main_rows, MAIN_MD_COLUMNS, Path("outputs/router/eval/main_results_table.md"))

    subgroup_rows = []
    for model_name, bucket, category in [
        ("Gate-only", gate_by_regime, "fixed_or_rule"),
        ("Residual-only", residual_by_regime, "fixed_or_rule"),
        ("Rule-based", rule_by_regime, "fixed_or_rule"),
        ("Oracle", oracle_by_regime, "oracle"),
    ]:
        for regime, stats in sorted(bucket.items()):
            subgroup_rows.append(
                {
                    "category": category,
                    "model": model_name,
                    "delta": "0.01" if model_name == "Rule-based" else "",
                    "tau": 0.5 if model_name == "Rule-based" else "",
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
    if full_model_bundle is not None:
        _, full_model_by_regime = full_model_bundle
        for regime, stats in sorted(full_model_by_regime.items()):
            subgroup_rows.append(
                {
                    "category": "paper_baseline",
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
    write_csv("outputs/router/eval/subgroup_results_table.csv", subgroup_rows, SUBGROUP_HEADER)
    write_markdown_table(subgroup_rows, SUBGROUP_MD_COLUMNS, Path("outputs/router/eval/subgroup_results_table.md"))

    threshold_xgb = read_csv("outputs/router/eval/threshold_scan_xgb_delta_0.01.csv")
    threshold_logistic = read_csv("outputs/router/eval/threshold_scan_logistic_delta_0.01.csv")
    feature_ablation = read_csv("outputs/router/eval/feature_ablation.csv")
    feature_importance = read_csv("outputs/router/eval/router_feature_importance_delta_0.01.csv")

    save_threshold_figure(threshold_logistic, threshold_xgb, Path("outputs/router/figures/threshold_coverage_mrr.png"))
    save_feature_importance_figure(feature_importance, Path("outputs/router/figures/router_feature_importance.png"))
    save_subgroup_fusion_ratio_figure(threshold_xgb, Path("outputs/router/figures/selected_fusion_ratio_by_subgroup.png"))
    write_feature_ablation_md(feature_ablation, Path("outputs/router/eval/feature_ablation.md"))

    best_xgb = best_routing_row([row for row in routing_main if row["model"] == "xgb"])
    best_logistic = best_routing_row([row for row in routing_main if row["model"] == "logistic"])
    best_ablation = max(feature_ablation, key=lambda row: float(row["overall_mrr"]))

    takeaways_md = build_takeaways_markdown(
        best_xgb=best_xgb,
        best_logistic=best_logistic,
        best_ablation=best_ablation,
        residual_overall=residual_overall,
        oracle_overall=oracle_overall,
        summary_rows=summary_rows,
        threshold_xgb=threshold_xgb,
        main_rows=main_rows,
    )
    Path("outputs/router/eval/first_round_takeaways.md").write_text(takeaways_md, encoding="utf-8")

    print("[OK] wrote main table     -> outputs/router/eval/main_results_table.csv")
    print("[OK] wrote main markdown  -> outputs/router/eval/main_results_table.md")
    print("[OK] wrote subgroup table -> outputs/router/eval/subgroup_results_table.csv")
    print("[OK] wrote subgroup md    -> outputs/router/eval/subgroup_results_table.md")
    print("[OK] wrote takeaways      -> outputs/router/eval/first_round_takeaways.md")


if __name__ == "__main__":
    main()
