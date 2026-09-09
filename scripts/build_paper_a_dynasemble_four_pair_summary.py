from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


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
            output_rows.append(
                summary_row(
                    grouped_seed[seed],
                    dataset_label,
                    pair_label,
                    "seed",
                    seed,
                    seed=seed,
                )
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
