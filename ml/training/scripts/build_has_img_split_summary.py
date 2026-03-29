from __future__ import annotations

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
}

PRIMARY_ORDER = ["ComplEx", "TuckER", "Text-only", "Early Fusion", "Gate-only", "Full Model", "Residual-only"]
METRIC_KEYS = [
    "mrr",
    "head_has_img_mrr",
    "head_no_img_mrr",
    "tail_no_img_mrr",
    "has_img_mrr",
    "no_img_mrr",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def collect_group_metrics(outputs_root: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for test_metrics_path in sorted(outputs_root.rglob("test_metrics.json")):
        run_dir = test_metrics_path.parent
        exp_name = run_dir.parent.name
        label = MODEL_LABEL_OVERRIDES.get(exp_name)
        if label is None:
            continue
        cfg_path = run_dir / "config_merged.json"
        cfg = load_json(cfg_path) if cfg_path.exists() else {}
        metrics = load_json(test_metrics_path)
        if "head_has_img_mrr" not in metrics:
            continue
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
        if rows:
            first_metrics = rows[0]["metrics"]
            item["split_counts"] = {
                "head_has_img_count": int(first_metrics.get("head_has_img_count", 0)),
                "head_no_img_count": int(first_metrics.get("head_no_img_count", 0)),
                "tail_has_img_count": int(first_metrics.get("tail_has_img_count", 0)),
                "tail_no_img_count": int(first_metrics.get("tail_no_img_count", 0)),
            }
        summary[label] = item
    return summary


def render_markdown(summary: dict[str, dict], outputs_root: Path) -> str:
    ordered_labels = [label for label in PRIMARY_ORDER if label in summary]
    ordered_labels += [label for label in sorted(summary) if label not in ordered_labels]

    lines = [
        "# has_img / no_img 分组结果摘要",
        "",
        "## 1. 说明",
        "",
        f"本摘要基于 `{outputs_root.as_posix()}` 下 run 目录中的 `test_metrics.json` 自动汇总。",
        "",
        "当前采用的正式口径是：",
        "",
        "- 保留当前 `paper_split`，不重划分数据",
        "- `head` 方向按真实 head 实体是否有图分组",
        "- `tail` 方向按真实 tail 实体是否有图分组",
        "- 但当前 `paper_split` 中 tail 目标实体全部无图，因此 tail 方向只有 `tail_no_img` 子组",
        "",
        "## 2. 分组规模",
        "",
    ]

    if ordered_labels:
        counts = summary[ordered_labels[0]].get("split_counts", {})
        lines.extend(
            [
                f"- `head_has_img_count = {counts.get('head_has_img_count', 0)}`",
                f"- `head_no_img_count = {counts.get('head_no_img_count', 0)}`",
                f"- `tail_has_img_count = {counts.get('tail_has_img_count', 0)}`",
                f"- `tail_no_img_count = {counts.get('tail_no_img_count', 0)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 主表",
            "",
            "| 模型 | Seeds | Overall MRR | Head has_img MRR | Head no_img MRR | Tail no_img MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for label in ordered_labels:
        stats = summary[label]["stats"]
        lines.append(
            "| "
            f"{label} | {summary[label]['num_seeds']} | "
            f"{fmt(stats['mrr']['mean'], stats['mrr']['std'])} | "
            f"{fmt(stats['head_has_img_mrr']['mean'], stats['head_has_img_mrr']['std'])} | "
            f"{fmt(stats['head_no_img_mrr']['mean'], stats['head_no_img_mrr']['std'])} | "
            f"{fmt(stats['tail_no_img_mrr']['mean'], stats['tail_no_img_mrr']['std'])} |"
        )

    lines.extend(["", "## 4. 初步观察", ""])
    lines.extend(
        [
            "- 当前 split 下，tail 方向不存在 `has_img` 子组，因此图像可用性分析主要发生在 head 方向。",
            "- `tail_no_img_mrr` 基本等价于 tail 方向主结果，可直接与整体 `tail_mrr` 对应理解。",
            "- 更值得关注的是：不同模型在 `head_has_img_mrr` 与 `head_no_img_mrr` 上的相对排序是否变化。",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--output-md", default="docs/HAS_IMG_SPLIT_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/has_img_split_summary.json")
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
