from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight protocol sanity tables for OpenBG-IMG.")
    parser.add_argument("--split-dir", default="data/datasets/openbg_img/paper_split")
    parser.add_argument("--has-img", default="data/cache/openbg_img/has_img.pt")
    parser.add_argument("--output-dir", default="docs/protocol_audit")
    return parser.parse_args()


def entity_id(text: str) -> int:
    if text.startswith("ent_"):
        return int(text.split("_", 1)[1])
    return int(text)


def read_triples(path: Path) -> list[tuple[int, str, int]]:
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            triples.append((entity_id(parts[0]), parts[1], entity_id(parts[2])))
    return triples


def summarize(values: list[int]) -> dict:
    arr = np.array(values, dtype=np.float64)
    log_arr = np.log1p(arr)
    return {
        "count": int(arr.size),
        "mean_degree": float(arr.mean()) if arr.size else 0.0,
        "median_degree": float(np.median(arr)) if arr.size else 0.0,
        "q25_degree": float(np.quantile(arr, 0.25)) if arr.size else 0.0,
        "q75_degree": float(np.quantile(arr, 0.75)) if arr.size else 0.0,
        "mean_log_degree": float(log_arr.mean()) if arr.size else 0.0,
        "median_log_degree": float(np.median(log_arr)) if arr.size else 0.0,
    }


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_degree_table(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Training-graph degree by test-time target-side regime. Entity degree is computed only from training triples.}",
        r"\label{tab:degree_by_target_regime}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Regime & \#Queries & Mean degree & Median degree & Q25 & Q75 & Mean log-degree \\",
        r"\midrule",
    ]
    order = ["head_has_img", "head_no_img", "tail_no_img", "tail_has_img"]
    by_regime = {row["regime"]: row for row in rows}
    for regime in order:
        if regime not in by_regime:
            continue
        row = by_regime[regime]
        lines.append(
            f"\\texttt{{{regime.replace('_', r'\_')}}} & {row['count']:,} & {fmt(row['mean_degree'])} & "
            f"{fmt(row['median_degree'])} & {fmt(row['q25_degree'])} & {fmt(row['q75_degree'])} & "
            f"{fmt(row['mean_log_degree'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Protocol Audit Sanity Summary",
        "",
        "Entity degree is computed from training triples only. Test triples are used only to group target entities by prediction-side regime.",
        "",
        "| Regime | #Queries | Mean degree | Median degree | Q25 | Q75 | Mean log-degree |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['regime']}` | {row['count']} | {row['mean_degree']:.2f} | {row['median_degree']:.2f} | "
            f"{row['q25_degree']:.2f} | {row['q75_degree']:.2f} | {row['mean_log_degree']:.2f} |"
        )
    lines.extend(
        [
            "",
            "This sanity check does not eliminate structural confounding. It records the most direct degree alternative so the role--modality interpretation remains explicitly protocol-aware.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = read_triples(split_dir / "OpenBG-IMG_paper_train.tsv")
    test = read_triples(split_dir / "OpenBG-IMG_paper_test.tsv")
    has_img = torch.load(args.has_img, map_location="cpu").to(dtype=torch.bool)

    degree = Counter()
    for head, _, tail in train:
        degree[head] += 1
        degree[tail] += 1

    buckets: dict[str, list[int]] = defaultdict(list)
    for head, _, tail in test:
        head_regime = "head_has_img" if bool(has_img[head].item()) else "head_no_img"
        tail_regime = "tail_has_img" if bool(has_img[tail].item()) else "tail_no_img"
        buckets[head_regime].append(degree[head])
        buckets[tail_regime].append(degree[tail])

    rows = []
    for regime in ["head_has_img", "head_no_img", "tail_no_img", "tail_has_img"]:
        if regime in buckets:
            row = {"regime": regime}
            row.update(summarize(buckets[regime]))
            rows.append(row)

    payload = {
        "degree_definition": "undirected count of train triples in which the entity appears as head or tail",
        "split_dir": str(split_dir),
        "rows": rows,
    }
    (output_dir / "degree_by_target_regime.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_degree_table(output_dir / "table_degree_by_target_regime.tex", rows)
    write_summary(output_dir / "protocol_audit_summary.md", rows)
    print(f"[OK] wrote {output_dir / 'degree_by_target_regime.json'}")
    print(f"[OK] wrote {output_dir / 'table_degree_by_target_regime.tex'}")
    print(f"[OK] wrote {output_dir / 'protocol_audit_summary.md'}")


if __name__ == "__main__":
    main()
