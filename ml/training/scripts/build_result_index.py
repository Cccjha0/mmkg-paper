import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODEL_LABEL_OVERRIDES = {
    "openbg_img_complex": "ComplEx",
    "openbg_img_text_only": "Text-only",
    "openbg_img_tucker": "TuckER",
    "openbg_img_early": "Early Fusion",
    "openbg_img_gate_only": "Gate-only",
    "openbg_img_residual_only": "Residual-only",
    "openbg_img_gated_vec_res_rel": "Full Model",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_metrics_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("metrics*.csv"))


def load_best_metrics(metrics_path: Path) -> dict | None:
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8-sig")))
    if not rows or "mrr" not in rows[0]:
        return None
    best = max(rows, key=lambda r: float(r.get("mrr", "-inf") or "-inf"))
    return {
        "num_rows": len(rows),
        "best_epoch": best.get("epoch", "-"),
        "best_mrr": float(best.get("mrr", 0.0)),
        "best_hits1": best.get("hits@1", "-"),
        "best_hits3": best.get("hits@3", "-"),
        "best_hits10": best.get("hits@10", "-"),
    }


def infer_model_label(exp_name: str, model_cfg: dict) -> str:
    if exp_name in MODEL_LABEL_OVERRIDES:
        return MODEL_LABEL_OVERRIDES[exp_name]

    use_fusion = model_cfg.get("use_fusion")
    use_residual = model_cfg.get("use_residual")
    if use_fusion is True and use_residual is False:
        return "Gate-only"
    if use_fusion is False and use_residual is True:
        return "Residual-only"
    if use_fusion is True and use_residual is True:
        return "Full Model"
    return model_cfg.get("name", "unknown")


def build_config_summary(cfg: dict) -> str:
    model = cfg.get("model", {})
    tr = cfg.get("training", {})
    items = []
    if "use_fusion" in model:
        items.append(f"use_fusion={model.get('use_fusion')}")
    if "use_residual" in model:
        items.append(f"use_residual={model.get('use_residual')}")
    if "use_normalized_mix" in model:
        items.append(f"use_normalized_mix={model.get('use_normalized_mix')}")
    if "img_dropout" in tr:
        items.append(f"img_dropout={tr.get('img_dropout')}")
    return ", ".join(items) if items else "-"


def determine_row_status(has_config: bool, has_ckpt: bool, has_metrics: bool) -> str:
    if has_config and has_ckpt and has_metrics:
        return "可用"
    if has_config or has_ckpt or has_metrics:
        return "参考可用"
    return "不完整"


def build_row_note(metrics_files: list[Path], row_status: str) -> str:
    if not metrics_files:
        return "缺少 metrics 文件"
    names = [m.name for m in metrics_files]
    if len(names) == 1 and names[0] == "metrics.csv":
        return "配置与结果完整"
    if len(names) == 1:
        return f"metrics 文件为 `{names[0]}`"
    return "存在多个 metrics 文件：" + ", ".join(f"`{name}`" for name in names)


def collect_rows(outputs_root: Path) -> list[dict]:
    rows = []
    for exp_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            cfg_path = run_dir / "config_merged.json"
            has_config = cfg_path.exists()
            cfg = load_json(cfg_path) if has_config else {}

            metrics_files = find_metrics_files(run_dir)
            has_metrics = bool(metrics_files)
            has_ckpt = (run_dir / "best.ckpt").exists()
            best_metrics = load_best_metrics(metrics_files[0]) if has_metrics else None

            row = {
                "exp_group": exp_dir.name,
                "model_label": infer_model_label(exp_dir.name, cfg.get("model", {})),
                "run_dir": run_dir.relative_to(outputs_root).as_posix(),
                "seed": cfg.get("system", {}).get("seed", "-"),
                "config_summary": build_config_summary(cfg),
                "best_mrr": "-" if not best_metrics else f"{best_metrics['best_mrr']:.4f}",
                "best_epoch": "-" if not best_metrics else best_metrics["best_epoch"],
                "status": determine_row_status(has_config, has_ckpt, has_metrics),
                "note": build_row_note(metrics_files, determine_row_status(has_config, has_ckpt, has_metrics)),
                "has_metrics": has_metrics,
            }
            rows.append(row)
    return rows


def summarize_groups(rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_label"]].append(row)

    complete_groups = {}
    partial_groups = {}
    for label, items in grouped.items():
        usable = [r for r in items if r["status"] == "可用"]
        if len(usable) >= 3:
            complete_groups[label] = items
        else:
            partial_groups[label] = items
    return complete_groups, partial_groups


def render_markdown(rows: list[dict], outputs_root: Path) -> str:
    complete_groups, partial_groups = summarize_groups(rows)

    lines = [
        "# 统一结果索引表",
        "",
        "## 1. 说明",
        "",
        f"本表用于统一记录当前 `{outputs_root.as_posix()}` 下可见实验结果，便于后续：",
        "",
        "- 快速定位 run 目录",
        "- 查找对应 baseline 与 seed",
        "- 判断哪些结果可直接用于论文比较",
        "- 判断哪些结果仍不完整，只能作为参考",
        "",
        "该文档由 `ml/training/scripts/build_result_index.py` 自动生成。",
        "",
        "## 2. 字段说明",
        "",
        "- `实验组`: 输出目录名，对应一个实验配置族",
        "- `模型口径`: 按当前项目语义整理后的模型解释",
        "- `run目录`: 相对 `ml/artifacts/outputs` 的运行目录",
        "- `seed`: 当前运行的随机种子",
        "- `配置`: 关键开关组合",
        "- `最佳MRR`: 当前 metrics 文件中记录到的最佳 `mrr`",
        "- `最佳epoch`: 达到最佳 `mrr` 的 epoch",
        "- `状态`: 当前结果是否适合直接纳入后续正式比较",
        "- `备注`: 对命名异常或文件完整性的补充说明",
        "",
        "## 3. 统一结果索引",
        "",
        "| 实验组 | 模型口径 | run目录 | seed | 配置 | 最佳MRR | 最佳epoch | 状态 | 备注 |",
        "|---|---|---|---:|---|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| `{row['exp_group']}` | `{row['model_label']}` | `{row['run_dir']}` | {row['seed']} | "
            f"`{row['config_summary']}` | {row['best_mrr']} | {row['best_epoch']} | {row['status']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## 4. 当前可直接使用的结果分组",
            "",
            "### 4.1 当前相对完整的组",
            "",
        ]
    )

    if complete_groups:
        for label in sorted(complete_groups):
            usable = [r for r in complete_groups[label] if r["status"] == "可用"]
            seeds = ", ".join(str(r["seed"]) for r in sorted(usable, key=lambda x: str(x["seed"])))
            lines.append(f"- `{label}`")
            lines.append(f"  - 当前已有 {len(usable)} 个可用 run，seed: {seeds}")
            lines.append("  - 可作为后续主结果比较与汇总的基础")
    else:
        lines.append("- 当前暂无满足条件的完整结果组")

    lines.extend(["", "### 4.2 当前部分可用的组", ""])

    if partial_groups:
        for label in sorted(partial_groups):
            usable = [r for r in partial_groups[label] if r["status"] == "可用"]
            lines.append(f"- `{label}`")
            lines.append(f"  - 当前有 {len(usable)} 个可用 run，总计 {len(partial_groups[label])} 个目录")
            lines.append("  - 可用于观察趋势，但不足以直接形成稳定结论")
    else:
        lines.append("- 当前无部分可用组")

    lines.extend(["", "## 5. 当前主要缺口", ""])
    results_dir = outputs_root / "results"
    if results_dir.exists() and len(list(results_dir.iterdir())) <= 1:
        lines.append("- 当前 `results` 聚合目录为空，仅有 `.gitkeep`")
    lines.append("- 当前五组主模型与两组结构强基线均已形成 3-seed 可用结果")
    lines.append("- 当前主要工作重心应从“补齐主模型结果”转向“强基线对比、分组分析与原因诊断”")

    lines.extend(
        [
            "",
            "## 6. 当前建议标签",
            "",
            "- `可用`",
            "  - 配置明确",
            "  - metrics 完整",
            "  - 可直接进入后续比较或汇总",
            "",
            "- `参考可用`",
            "  - 存在部分配置、checkpoint 或结果文件",
            "  - 可用于观察趋势，但不足以形成正式结论",
            "",
            "- `不完整`",
            "  - 缺失关键文件",
            "  - 只能保留目录信息，不能直接进入论文汇报",
            "",
            "## 7. 下一步维护任务",
            "",
            "- [ ] 将后续新实验统一追加到本索引表",
            "- [ ] 汇总七组模型的 `mean ± std` 并形成最终主结果表",
            "- [ ] 将 `test_metrics.json` 纳入后续自动索引与汇总逻辑",
            "- [ ] 开始 `Residual-only > Full Model` 的原因排查实验",
            "- [ ] 开始 `Full Model` 与 `ComplEx / TuckER` 的正式对比分析",
            "- [ ] 将 `results` 目录真正作为聚合输出目录使用",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="ml/artifacts/outputs")
    parser.add_argument("--output-md", default="docs/RESULT_INDEX.md")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_md = Path(args.output_md)

    rows = collect_rows(outputs_root)
    markdown = render_markdown(rows, outputs_root)
    output_md.write_text(markdown, encoding="utf-8")
    print(f"[OK] wrote {output_md}")


if __name__ == "__main__":
    main()
