from __future__ import annotations

import argparse
import csv
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
DEFAULT_MODEL_SET = ["Gate-only", "Full Model", "Residual-only"]

GATE_KEYS = [
    "g_mean_all",
    "g_std_all",
    "g_mean_img",
    "g_std_img",
    "g_mean_noimg",
    "g_std_noimg",
    "g_frac_img_in_sample",
]
RESIDUAL_KEYS = [
    "grad_residual",
    "grad_fusion",
    "grad_projection",
    "residual_scale_value",
    "mix_w_fusion",
    "mix_w_residual",
]
ALL_TRACKED_KEYS = GATE_KEYS + RESIDUAL_KEYS + ["mrr", "avg_loss", "epoch"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def ordered_labels(labels: list[str]) -> list[str]:
    out = [label for label in PRIMARY_ORDER if label in labels]
    out.extend(sorted(label for label in labels if label not in out))
    return out


def normalize_requested_models(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_MODEL_SET)

    out = []
    for value in values:
        out.append(MODEL_LABEL_OVERRIDES.get(value, value))
    return out


def find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("metrics*.csv"))
    if not candidates:
        return None
    return candidates[0]


def select_latest_runs(
    outputs_root: Path,
    requested_labels: set[str],
) -> tuple[dict[str, list[Path]], dict[str, dict[str, list[str]]]]:
    by_label_seed: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    duplicates: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for cfg_path in sorted(outputs_root.rglob("config_merged.json")):
        run_dir = cfg_path.parent
        exp_name = run_dir.parent.name
        label = MODEL_LABEL_OVERRIDES.get(exp_name, exp_name)
        if label not in requested_labels:
            continue
        if not (run_dir / "best.ckpt").exists():
            continue
        if find_metrics_csv(run_dir) is None:
            continue

        cfg = load_json(cfg_path)
        seed = int(cfg.get("system", {}).get("seed", -1))
        if seed < 0:
            continue
        by_label_seed[label][seed].append(run_dir)

    selected: dict[str, list[Path]] = {}
    for label, seed_map in by_label_seed.items():
        selected[label] = []
        for seed, candidates in sorted(seed_map.items()):
            candidates = sorted(candidates, key=lambda p: p.name)
            chosen = candidates[-1]
            selected[label].append(chosen)
            if len(candidates) > 1:
                duplicates[label][str(seed)] = [p.relative_to(outputs_root).as_posix() for p in candidates]
    return selected, duplicates


def load_metrics_rows(path: Path) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {key: safe_float(raw.get(key)) for key in reader.fieldnames or []}
            rows.append(row)
    return rows


def pick_best_row(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    valid_rows = [row for row in rows if row.get("mrr") is not None]
    if not valid_rows:
        raise RuntimeError("metrics csv has no valid mrr rows")
    return max(valid_rows, key=lambda row: float(row["mrr"]))


def extract_metric_subset(row: dict[str, float | None], keys: list[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        out[key] = float(value)
    return out


def aggregate_metric_rows(rows: list[dict], keys: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[key] = {
            "mean": statistics.mean(values),
            "std": safe_stdev(values),
            "values": values,
        }
    return out


def summarize_run(run_dir: Path, outputs_root: Path) -> dict:
    cfg = load_json(run_dir / "config_merged.json")
    exp_name = run_dir.parent.name
    label = MODEL_LABEL_OVERRIDES.get(exp_name, exp_name)
    seed = int(cfg.get("system", {}).get("seed", -1))

    metrics_csv = find_metrics_csv(run_dir)
    if metrics_csv is None:
        raise RuntimeError(f"No metrics csv found in {run_dir}")

    rows = load_metrics_rows(metrics_csv)
    if not rows:
        raise RuntimeError(f"Empty metrics csv: {metrics_csv}")

    best_row = pick_best_row(rows)
    first_row = rows[0]
    last_row = rows[-1]

    best_metrics = extract_metric_subset(best_row, ALL_TRACKED_KEYS)
    first_metrics = extract_metric_subset(first_row, ALL_TRACKED_KEYS)
    last_metrics = extract_metric_subset(last_row, ALL_TRACKED_KEYS)

    deltas = {}
    for key in GATE_KEYS + RESIDUAL_KEYS:
        if key in first_metrics and key in best_metrics:
            deltas[f"{key}_delta_best_minus_first"] = best_metrics[key] - first_metrics[key]
        if key in first_metrics and key in last_metrics:
            deltas[f"{key}_delta_last_minus_first"] = last_metrics[key] - first_metrics[key]

    return {
        "label": label,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "relative_run_dir": run_dir.relative_to(outputs_root).as_posix(),
        "metrics_csv": metrics_csv.name,
        "num_rows": len(rows),
        "best_epoch": int(best_metrics.get("epoch", 0)),
        "last_epoch": int(last_metrics.get("epoch", 0)),
        "best_dev": best_metrics,
        "first_eval": first_metrics,
        "last_eval": last_metrics,
        "deltas": deltas,
    }


def build_summary(run_results: dict[str, list[dict]]) -> dict:
    summary = {"models": {}}
    for label, rows in run_results.items():
        best_rows = [row["best_dev"] for row in rows]
        delta_rows = [row["deltas"] for row in rows]
        model_summary = {
            "num_seeds": len(rows),
            "runs": rows,
            "best_epoch_stats": aggregate_metric_rows(best_rows, GATE_KEYS + RESIDUAL_KEYS + ["mrr", "avg_loss", "epoch"]),
            "delta_stats": aggregate_metric_rows(
                delta_rows,
                [f"{key}_delta_best_minus_first" for key in GATE_KEYS + RESIDUAL_KEYS],
            ),
        }
        summary["models"][label] = model_summary
    return summary


def render_markdown(
    summary: dict,
    selected_runs: dict[str, list[Path]],
    duplicates: dict[str, dict[str, list[str]]],
    outputs_root: Path,
) -> str:
    labels = ordered_labels(list(summary["models"].keys()))

    lines = [
        "# Behavior Summary",
        "",
        "## 1. Purpose",
        "",
        "This document summarizes the current first-round behavior analysis based on training-time diagnostics already saved in run directories.",
        "",
        "Current focus:",
        "",
        "- gate mean / std behavior",
        "- residual scale behavior",
        "- fusion vs residual mixture weights",
        "- gradient-group diagnostics",
        "",
        "The statistics below are summarized at each run's best-dev epoch, i.e. the epoch with the highest recorded dev MRR in the metrics CSV.",
        "",
        "## 2. Selected Runs",
        "",
        f"Outputs root: `{outputs_root.as_posix()}`",
        "",
    ]

    for label in labels:
        rel_paths = [run.relative_to(outputs_root).as_posix() for run in selected_runs.get(label, [])]
        lines.append(f"- `{label}`: {', '.join(f'`{path}`' for path in rel_paths)}")

    duplicate_lines = []
    for label in labels:
        for seed, candidates in sorted(duplicates.get(label, {}).items()):
            duplicate_lines.append(f"- `{label}` seed `{seed}` had multiple runs; selected latest: `{candidates[-1]}`")
    if duplicate_lines:
        lines.extend(["", "Duplicate handling:"] + duplicate_lines)

    lines.extend(["", "## 3. Gate Statistics At Best Epoch", ""])
    lines.append("| Model | Seeds | Gate Mean (All) | Gate Std (All) | Gate Mean (Img) | Gate Mean (NoImg) | Img-NoImg Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        stats = summary["models"][label]["best_epoch_stats"]
        if "g_mean_all" not in stats:
            continue
        gap_mean = stats["g_mean_img"]["mean"] - stats["g_mean_noimg"]["mean"]
        gap_std = 0.0
        lines.append(
            "| "
            f"{label} | {summary['models'][label]['num_seeds']} | "
            f"{fmt(stats['g_mean_all']['mean'], stats['g_mean_all']['std'])} | "
            f"{fmt(stats['g_std_all']['mean'], stats['g_std_all']['std'])} | "
            f"{fmt(stats['g_mean_img']['mean'], stats['g_mean_img']['std'])} | "
            f"{fmt(stats['g_mean_noimg']['mean'], stats['g_mean_noimg']['std'])} | "
            f"{fmt(gap_mean, gap_std)} |"
        )

    lines.extend(["", "## 4. Residual And Mix Statistics At Best Epoch", ""])
    lines.append("| Model | Seeds | Residual Scale | Mix Fusion | Mix Residual | Grad Residual | Grad Fusion | Grad Projection |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        stats = summary["models"][label]["best_epoch_stats"]
        if "residual_scale_value" not in stats:
            continue
        lines.append(
            "| "
            f"{label} | {summary['models'][label]['num_seeds']} | "
            f"{fmt(stats['residual_scale_value']['mean'], stats['residual_scale_value']['std'])} | "
            f"{fmt(stats.get('mix_w_fusion', {'mean': 0.0, 'std': 0.0})['mean'], stats.get('mix_w_fusion', {'mean': 0.0, 'std': 0.0})['std'])} | "
            f"{fmt(stats.get('mix_w_residual', {'mean': 0.0, 'std': 0.0})['mean'], stats.get('mix_w_residual', {'mean': 0.0, 'std': 0.0})['std'])} | "
            f"{fmt(stats.get('grad_residual', {'mean': 0.0, 'std': 0.0})['mean'], stats.get('grad_residual', {'mean': 0.0, 'std': 0.0})['std'])} | "
            f"{fmt(stats.get('grad_fusion', {'mean': 0.0, 'std': 0.0})['mean'], stats.get('grad_fusion', {'mean': 0.0, 'std': 0.0})['std'])} | "
            f"{fmt(stats.get('grad_projection', {'mean': 0.0, 'std': 0.0})['mean'], stats.get('grad_projection', {'mean': 0.0, 'std': 0.0})['std'])} |"
        )

    lines.extend(["", "## 5. Delta From First Eval To Best Epoch", ""])
    lines.append("| Model | Gate Mean Delta | Gate Img Delta | Gate NoImg Delta | Residual Scale Delta | Mix Fusion Delta | Mix Residual Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        stats = summary["models"][label]["delta_stats"]
        if not stats:
            continue
        def get_stat(key: str) -> tuple[float, float]:
            item = stats.get(key)
            if not item:
                return 0.0, 0.0
            return item["mean"], item["std"]

        lines.append(
            "| "
            f"{label} | {fmt(*get_stat('g_mean_all_delta_best_minus_first'))} | "
            f"{fmt(*get_stat('g_mean_img_delta_best_minus_first'))} | "
            f"{fmt(*get_stat('g_mean_noimg_delta_best_minus_first'))} | "
            f"{fmt(*get_stat('residual_scale_value_delta_best_minus_first'))} | "
            f"{fmt(*get_stat('mix_w_fusion_delta_best_minus_first'))} | "
            f"{fmt(*get_stat('mix_w_residual_delta_best_minus_first'))} |"
        )

    lines.extend(
        [
            "",
            "## 6. Per-Run Best-Epoch Detail",
            "",
            "| Model | Seed | Run | Best Epoch | Best Dev MRR | Gate Mean (All) | Gate Mean (Img) | Gate Mean (NoImg) | Residual Scale | Mix Fusion | Mix Residual |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        for row in sorted(summary["models"][label]["runs"], key=lambda item: item["seed"]):
            best = row["best_dev"]
            lines.append(
                "| "
                f"{label} | {row['seed']} | `{row['relative_run_dir']}` | "
                f"{int(best.get('epoch', 0))} | "
                f"{best.get('mrr', 0.0):.4f} | "
                f"{best.get('g_mean_all', 0.0):.4f} | "
                f"{best.get('g_mean_img', 0.0):.4f} | "
                f"{best.get('g_mean_noimg', 0.0):.4f} | "
                f"{best.get('residual_scale_value', 0.0):.4f} | "
                f"{best.get('mix_w_fusion', 0.0):.4f} | "
                f"{best.get('mix_w_residual', 0.0):.4f} |"
            )

    lines.extend(
        [
            "",
            "## 7. First-Round Takeaways",
            "",
            "- This summary is based only on run-level diagnostics already saved during training; it does not yet provide relation-aware behavior analysis.",
            "- Gate statistics can already support the first half of `7.1` (mean/std and image-availability comparison).",
            "- Residual-scale, mix-weight, and gradient-group statistics can already support the first round of `7.2` and `7.3`.",
            "- The next follow-up should be a relation-aware behavior script to connect these diagnostics back to the completed `6.2 relation type` analysis.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--output-md", default="docs/BEHAVIOR_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/behavior_summary.json")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="model labels or experiment names; default: Gate-only Full Model Residual-only",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    requested_labels = set(normalize_requested_models(args.models))
    selected_runs, duplicates = select_latest_runs(outputs_root, requested_labels)
    if not selected_runs:
        raise RuntimeError("No matching runs with metrics CSV found for requested models.")

    run_results: dict[str, list[dict]] = defaultdict(list)
    for label in ordered_labels(list(selected_runs.keys())):
        for run_dir in selected_runs[label]:
            run_results[label].append(summarize_run(run_dir, outputs_root))

    summary = build_summary(run_results)
    summary["meta"] = {
        "outputs_root": outputs_root.as_posix(),
        "selected_models": ordered_labels(list(run_results.keys())),
        "selected_runs": {
            label: [run.relative_to(outputs_root).as_posix() for run in runs]
            for label, runs in selected_runs.items()
        },
        "duplicate_candidates": duplicates,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        render_markdown(summary, selected_runs, duplicates, outputs_root),
        encoding="utf-8",
    )

    print(f"[OK] wrote {output_md.as_posix()}")
    print(f"[OK] wrote {output_json.as_posix()}")


if __name__ == "__main__":
    main()
