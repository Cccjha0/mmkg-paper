from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260909
BOOTSTRAP_SAMPLES = 10000

PAIR_SPECS = (
    ("mkg_w", "mhyper_native", "MKG-W", "M-Hyper + NativE"),
    ("mkg_w", "mhyper_adamf", "MKG-W", "M-Hyper + AdaMF-MAT"),
    ("db15k", "mhyper_native", "DB15K", "M-Hyper + NativE"),
    ("db15k", "mhyper_adamf", "DB15K", "M-Hyper + AdaMF-MAT"),
)

OUTPUT_FIELDS = (
    "dataset",
    "pair",
    "scope_type",
    "scope_value",
    "seed",
    "direction",
    "n_queries",
    "primary_mrr",
    "global_mrr",
    "query_soft_mrr",
    "dynasemble_mrr",
    "anchored_mrr",
    "delta_dynasemble_vs_global",
    "delta_anchored_vs_global",
)

RISK_FIELDS = (
    "dataset",
    "pair",
    "n_queries",
    "harm_rate",
    "improvement_rate",
    "tie_rate",
    "mean_delta",
    "mean_harm_given_harm",
    "mean_gain_given_gain",
    "delta_q01",
    "delta_q05",
    "delta_q10",
    "cvar10",
    "severe_harm_rate_delta_le_minus_0_1",
    "top1_loss_rate",
    "top1_gain_rate",
    "top10_loss_rate",
    "top10_gain_rate",
)

SELECTOR_FIELDS = (
    "dataset",
    "pair",
    "seed",
    "global_mrr",
    "dynasemble_mrr",
    "delta_dynasemble_vs_global",
    "delta_ci95_low",
    "delta_ci95_high",
    "weight_mean",
    "effective_alpha_mean",
    "weight_zero_fraction",
    "selector_behavior",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the locked Paper A DynaSemble four-pair table.")
    parser.add_argument(
        "--root",
        default="outputs/paper_a_safe_correction/dynasemble",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(rows: list[dict[str, str]], column: str) -> float:
    return sum(float(row[column]) for row in rows) / len(rows)


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def risk_row(rows: list[dict[str, str]], dataset: str, pair: str) -> dict:
    deltas = [float(row["rr_dynasemble"]) - float(row["rr_global"]) for row in rows]
    sorted_deltas = sorted(deltas)
    harmful = [value for value in deltas if value < 0.0]
    beneficial = [value for value in deltas if value > 0.0]
    tail_count = max(1, int(len(deltas) * 0.1 + 0.999999999))
    top1_losses = sum(
        int(row["rank_global"]) <= 1 < int(row["rank_dynasemble"]) for row in rows
    )
    top1_gains = sum(
        int(row["rank_dynasemble"]) <= 1 < int(row["rank_global"]) for row in rows
    )
    top10_losses = sum(
        int(row["rank_global"]) <= 10 < int(row["rank_dynasemble"]) for row in rows
    )
    top10_gains = sum(
        int(row["rank_dynasemble"]) <= 10 < int(row["rank_global"]) for row in rows
    )
    return {
        "dataset": dataset,
        "pair": pair,
        "n_queries": len(rows),
        "harm_rate": len(harmful) / len(rows),
        "improvement_rate": len(beneficial) / len(rows),
        "tie_rate": (len(rows) - len(harmful) - len(beneficial)) / len(rows),
        "mean_delta": sum(deltas) / len(deltas),
        "mean_harm_given_harm": sum(harmful) / len(harmful),
        "mean_gain_given_gain": sum(beneficial) / len(beneficial),
        "delta_q01": quantile(sorted_deltas, 0.01),
        "delta_q05": quantile(sorted_deltas, 0.05),
        "delta_q10": quantile(sorted_deltas, 0.10),
        "cvar10": sum(sorted_deltas[:tail_count]) / tail_count,
        "severe_harm_rate_delta_le_minus_0_1": sum(
            value <= -0.1 for value in deltas
        )
        / len(deltas),
        "top1_loss_rate": top1_losses / len(rows),
        "top1_gain_rate": top1_gains / len(rows),
        "top10_loss_rate": top10_losses / len(rows),
        "top10_gain_rate": top10_gains / len(rows),
    }


def clustered_bootstrap_delta(rows: list[dict[str, str]]) -> tuple[float, float]:
    clusters: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"]))
        clusters[key].append(
            float(row["rr_dynasemble"]) - float(row["rr_global"])
        )
    values = np.asarray(
        [sum(cluster) / len(cluster) for cluster in clusters.values()],
        dtype=np.float64,
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    chunk_size = 128
    for start in range(0, BOOTSTRAP_SAMPLES, chunk_size):
        width = min(chunk_size, BOOTSTRAP_SAMPLES - start)
        indexes = generator.integers(0, values.size, size=(width, values.size))
        samples[start : start + width] = values[indexes].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summary_row(
    rows: list[dict[str, str]],
    dataset: str,
    pair: str,
    scope_type: str,
    scope_value: str,
    seed: str = "",
    direction: str = "",
) -> dict:
    primary = mean(rows, "rr_a")
    global_mrr = mean(rows, "rr_global")
    query_soft = mean(rows, "rr_query_soft")
    dynasemble = mean(rows, "rr_dynasemble")
    anchored = mean(rows, "rr_anchored")
    return {
        "dataset": dataset,
        "pair": pair,
        "scope_type": scope_type,
        "scope_value": scope_value,
        "seed": seed,
        "direction": direction,
        "n_queries": len(rows),
        "primary_mrr": primary,
        "global_mrr": global_mrr,
        "query_soft_mrr": query_soft,
        "dynasemble_mrr": dynasemble,
        "anchored_mrr": anchored,
        "delta_dynasemble_vs_global": dynasemble - global_mrr,
        "delta_anchored_vs_global": anchored - global_mrr,
    }


def format_markdown(rows: list[dict]) -> str:
    overall = [row for row in rows if row["scope_type"] == "overall"]
    lines = [
        "| Dataset | Pair | Primary | Global | Query-soft | DynaSemble | Anchored | DynaSemble - Global | Anchored - Global |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            "| {dataset} | {pair} | {primary_mrr:.6f} | {global_mrr:.6f} | "
            "{query_soft_mrr:.6f} | {dynasemble_mrr:.6f} | {anchored_mrr:.6f} | "
            "{delta_dynasemble_vs_global:+.6f} | {delta_anchored_vs_global:+.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_rows = []
    risk_rows = []
    selector_rows = []
    audit = []
    for dataset_dir, pair_dir, dataset_label, pair_label in PAIR_SPECS:
        pair_root = root / dataset_dir / pair_dir
        required = (
            pair_root / "lock.json",
            pair_root / "test_summary.json",
            pair_root / "test_query_rows.csv",
            pair_root / "results_by_seed.csv",
            pair_root / "results_by_direction.csv",
            pair_root / "clustered_bootstrap_ci.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete DynaSemble pair {dataset_dir}/{pair_dir}: {missing}")
        summary = json.loads((pair_root / "test_summary.json").read_text(encoding="utf-8"))
        if summary.get("stage") != "test" or summary.get("expert_a_name") != "M-Hyper":
            raise RuntimeError(f"Unexpected TEST summary identity under {pair_root}")
        rows = read_rows(pair_root / "test_query_rows.csv")
        current_lock_hash = sha256_file(pair_root / "lock.json")
        if summary.get("dev_lock_sha256") != current_lock_hash:
            raise RuntimeError(f"TEST summary lock hash mismatch under {pair_root}")
        if int(summary.get("n_rows", -1)) != len(rows):
            raise RuntimeError(f"TEST summary row count mismatch under {pair_root}")
        if not rows or any(row.get("rr_query_soft", "") == "" for row in rows):
            raise RuntimeError(f"Missing matched Query-soft values under {pair_root}")
        if any(row.get("rr_anchored", "") == "" for row in rows):
            raise RuntimeError(f"Missing matched Anchored values under {pair_root}")
        output_rows.append(summary_row(rows, dataset_label, pair_label, "overall", "all"))
        risk_rows.append(risk_row(rows, dataset_label, pair_label))
        grouped_direction: dict[str, list[dict[str, str]]] = defaultdict(list)
        grouped_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
        grouped_seed_direction: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped_direction[row["direction"]].append(row)
            grouped_seed[row["seed"]].append(row)
            grouped_seed_direction[(row["seed"], row["direction"])].append(row)
        for direction in ("head", "tail"):
            output_rows.append(
                summary_row(
                    grouped_direction[direction],
                    dataset_label,
                    pair_label,
                    "direction",
                    direction,
                    direction=direction,
                )
            )
        for seed in ("1", "2", "3"):
            seed_summary = summary_row(
                grouped_seed[seed],
                dataset_label,
                pair_label,
                "seed",
                seed,
                seed=seed,
            )
            output_rows.append(
                seed_summary
            )
            weight_rows = read_rows(pair_root / "test_weight_diagnostics.csv")
            weight = next(
                row
                for row in weight_rows
                if row["seed"] == seed and row["direction"] == "all"
            )
            zero_fraction = float(weight["weight_zero_fraction"])
            ci95_low, ci95_high = clustered_bootstrap_delta(grouped_seed[seed])
            selector_rows.append(
                {
                    "dataset": dataset_label,
                    "pair": pair_label,
                    "seed": seed,
                    "global_mrr": seed_summary["global_mrr"],
                    "dynasemble_mrr": seed_summary["dynasemble_mrr"],
                    "delta_dynasemble_vs_global": seed_summary[
                        "delta_dynasemble_vs_global"
                    ],
                    "delta_ci95_low": ci95_low,
                    "delta_ci95_high": ci95_high,
                    "weight_mean": float(weight["weight_mean"]),
                    "effective_alpha_mean": float(weight["effective_alpha_mean"]),
                    "weight_zero_fraction": zero_fraction,
                    "selector_behavior": (
                        "collapsed_to_secondary"
                        if zero_fraction == 1.0
                        else "active"
                    ),
                }
            )
            for direction in ("head", "tail"):
                output_rows.append(
                    summary_row(
                        grouped_seed_direction[(seed, direction)],
                        dataset_label,
                        pair_label,
                        "seed_direction",
                        f"seed{seed}_{direction}",
                        seed=seed,
                        direction=direction,
                    )
                )
        audit.append(
            {
                "dataset": dataset_label,
                "pair": pair_label,
                "n_rows": len(rows),
                "lock_sha256": current_lock_hash,
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "four_pair_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    with (root / "four_pair_risk_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RISK_FIELDS)
        writer.writeheader()
        writer.writerows(risk_rows)
    with (root / "four_pair_selector_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTOR_FIELDS)
        writer.writeheader()
        writer.writerows(selector_rows)
    (root / "four_pair_summary.md").write_text(
        format_markdown(output_rows), encoding="utf-8"
    )
    (root / "four_pair_summary_audit.json").write_text(
        json.dumps({"schema_version": 1, "pairs": audit}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()
