import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODEL_LABEL_OVERRIDES = {
    "openbg_img_complex": "ComplEx",
    "openbg_img_tucker": "TuckER",
    "openbg_img_text_only": "Text-only",
    "openbg_img_early": "Early Fusion",
    "openbg_img_gate_only": "Gate-only",
    "openbg_img_residual_only": "Residual-only",
    "openbg_img_gated_vec_res_rel": "Full Model",
    "openbg_img_apkgc": "APKGC",
    "openbg_img_native": "NativE",
    "openbg_img_adamf_mat": "AdaMF-MAT",
    "openbg_img_mhyper": "M-Hyper",
}

METRIC_KEYS = ["mrr", "hits@1", "hits@3", "hits@10", "tail_mrr", "head_mrr"]
PRIMARY_ORDER = [
    "ComplEx", "TuckER", "Text-only", "Early Fusion", "Gate-only", "Residual-only",
    "APKGC", "NativE", "AdaMF-MAT", "M-Hyper", "Full Model",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def safe_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def collect_group_metrics(outputs_root: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for test_metrics_path in sorted(outputs_root.rglob("test_metrics.json")):
        run_dir = test_metrics_path.parent
        exp_name = run_dir.parent.name
        label = MODEL_LABEL_OVERRIDES.get(exp_name, exp_name)
        cfg_path = run_dir / "config_merged.json"
        cfg = load_json(cfg_path) if cfg_path.exists() else {}
        metrics = load_json(test_metrics_path)
        grouped[label].append(
            {
                "seed": cfg.get("system", {}).get("seed", "-"),
                "run_dir": run_dir.relative_to(outputs_root).as_posix(),
                "metrics": metrics,
            }
        )
    return grouped


def build_summary(grouped: dict[str, list[dict]]) -> dict[str, dict]:
    summary = {}
    for label, rows in grouped.items():
        item = {"num_seeds": len(rows), "runs": rows, "stats": {}}
        for key in METRIC_KEYS:
            values = [float(r["metrics"][key]) for r in rows if key in r["metrics"]]
            if not values:
                continue
            item["stats"][key] = {
                "mean": statistics.mean(values),
                "std": safe_stdev(values),
                "values": values,
            }
        summary[label] = item
    return summary


def render_markdown(summary: dict[str, dict], outputs_root: Path) -> str:
    ordered_labels = [label for label in PRIMARY_ORDER if label in summary]
    ordered_labels += [label for label in sorted(summary) if label not in ordered_labels]

    lines = [
        "# 主结果摘要",
        "",
        "## 1. 说明",
        "",
        f"本文档基于 `{outputs_root.as_posix()}` 下各 run 目录中的 `test_metrics.json` 自动汇总，",
        "用于完成论文计划中的 `3.2 结果汇总`：",
        "",
        "- 汇总主模型与结构强基线的 test 指标",
        "- 计算各模型的 mean ± std",
        "- 形成可直接写入论文的主结果表",
        "",
        "当前汇总的模型包括：",
        "",
        "- `ComplEx`",
        "- `TuckER`",
        "- `Text-only`",
        "- `Early Fusion`",
        "- `Gate-only`",
        "- `Residual-only`",
        "- `Full Model`",
        "",
        "## 2. 主结果表",
        "",
        "| 模型 | Seeds | Test MRR | Hits@1 | Hits@3 | Hits@10 | Tail MRR | Head MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for label in ordered_labels:
        stats = summary[label]["stats"]
        lines.append(
            "| "
            f"{label} | {summary[label]['num_seeds']} | "
            f"{fmt(stats['mrr']['mean'], stats['mrr']['std'])} | "
            f"{fmt(stats['hits@1']['mean'], stats['hits@1']['std'])} | "
            f"{fmt(stats['hits@3']['mean'], stats['hits@3']['std'])} | "
            f"{fmt(stats['hits@10']['mean'], stats['hits@10']['std'])} | "
            f"{fmt(stats['tail_mrr']['mean'], stats['tail_mrr']['std'])} | "
            f"{fmt(stats['head_mrr']['mean'], stats['head_mrr']['std'])} |"
        )

    lines.extend(["", "## 3. 当前排序", ""])
    ranked = sorted(
        ((label, summary[label]["stats"]["mrr"]["mean"]) for label in ordered_labels),
        key=lambda x: x[1],
        reverse=True,
    )
    for idx, (label, mean_mrr) in enumerate(ranked, start=1):
        lines.append(f"{idx}. `{label}`: {mean_mrr:.4f}")

    best_label, best_mrr = ranked[0]
    worst_label, worst_mrr = ranked[-1]
    gap_full_gate = None
    gap_full_res = None
    gap_full_complex = None
    gap_full_tucker = None
    if "Full Model" in summary and "Gate-only" in summary:
        gap_full_gate = summary["Full Model"]["stats"]["mrr"]["mean"] - summary["Gate-only"]["stats"]["mrr"]["mean"]
    if "Full Model" in summary and "Residual-only" in summary:
        gap_full_res = summary["Full Model"]["stats"]["mrr"]["mean"] - summary["Residual-only"]["stats"]["mrr"]["mean"]
    if "Full Model" in summary and "ComplEx" in summary:
        gap_full_complex = summary["Full Model"]["stats"]["mrr"]["mean"] - summary["ComplEx"]["stats"]["mrr"]["mean"]
    if "Full Model" in summary and "TuckER" in summary:
        gap_full_tucker = summary["Full Model"]["stats"]["mrr"]["mean"] - summary["TuckER"]["stats"]["mrr"]["mean"]

    lines.extend(
        [
            "",
            "## 4. 初步观察",
            "",
            f"- 当前 test MRR 最高的是 `{best_label}`，平均为 `{best_mrr:.4f}`。",
            f"- 当前 test MRR 最低的是 `{worst_label}`，平均为 `{worst_mrr:.4f}`。",
        ]
    )
    if gap_full_gate is not None:
        lines.append(f"- `Full Model` 相比 `Gate-only` 的平均 MRR 提升为 `{gap_full_gate:.4f}`。")
    if gap_full_res is not None:
        lines.append(f"- `Full Model` 相比 `Residual-only` 的平均 MRR 差距为 `{gap_full_res:.4f}`。")
    if gap_full_complex is not None:
        lines.append(f"- `Full Model` 相比 `ComplEx` 的平均 MRR 差距为 `{gap_full_complex:.4f}`。")
    if gap_full_tucker is not None:
        lines.append(f"- `Full Model` 相比 `TuckER` 的平均 MRR 差距为 `{gap_full_tucker:.4f}`。")
    lines.extend(
        [
            "- 当前所有模型的 `head_mrr` 明显低于 `tail_mrr`，说明正式 `both` 协议显著严于早期 `tail-only` 观察结果。",
            "",
            "## 5. 逐模型 test 指标明细",
            "",
        ]
    )

    for idx, label in enumerate(ordered_labels, start=1):
        lines.append(f"### 5.{idx} {label}")
        lines.append("")
        lines.append("| Seed | Run 目录 | MRR | Hits@1 | Hits@3 | Hits@10 | Tail MRR | Head MRR |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(summary[label]["runs"], key=lambda x: str(x["seed"])):
            m = row["metrics"]
            lines.append(
                "| "
                f"{row['seed']} | `{row['run_dir']}` | "
                f"{m['mrr']:.4f} | {m['hits@1']:.4f} | {m['hits@3']:.4f} | {m['hits@10']:.4f} | "
                f"{m['tail_mrr']:.4f} | {m['head_mrr']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. 当前未完成项",
            "",
            "- 当前主结果表虽已纳入 `ComplEx` / `TuckER`，但对应的论文式结论与讨论仍待补充。",
            "- 当前尚未完成 `Full Model` 相对 `ComplEx / TuckER` 的正式差距分析。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--output-md", default="docs/MAIN_RESULTS_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/main_results_summary.json")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    grouped = collect_group_metrics(outputs_root)
    summary = build_summary(grouped)

    Path(args.output_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.output_md).write_text(render_markdown(summary, outputs_root), encoding="utf-8")
    print(f"[OK] wrote {args.output_md}")
    print(f"[OK] wrote {args.output_json}")


if __name__ == "__main__":
    main()
