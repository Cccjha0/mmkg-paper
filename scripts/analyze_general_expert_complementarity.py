from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize dataset-local fusion/structural query complementarity."
    )
    parser.add_argument("--fusion", required=True, help="Fusion query-eval CSV")
    parser.add_argument("--structural", required=True, help="Structural query-eval CSV")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_and_pair(fusion_rows: list[dict], structural_rows: list[dict], split: str) -> list[tuple[dict, dict]]:
    for expert, rows in (("fusion", fusion_rows), ("structural", structural_rows)):
        splits = {str(row.get("split", "")) for row in rows}
        if splits != {split}:
            raise RuntimeError(f"{expert} rows must contain only split={split!r}; found {sorted(splits)}")
        protocols = {str(row.get("protocol_version", "")) for row in rows}
        if protocols != {"mmkg_general_v1"}:
            raise RuntimeError(f"{expert} rows must use mmkg_general_v1; found {sorted(protocols)}")
    fusion_by_id = {row["query_id"]: row for row in fusion_rows}
    structural_by_id = {row["query_id"]: row for row in structural_rows}
    if len(fusion_by_id) != len(fusion_rows) or len(structural_by_id) != len(structural_rows):
        raise RuntimeError("Duplicate query_id found in expert exports.")
    if set(fusion_by_id) != set(structural_by_id):
        raise RuntimeError("Fusion and structural exports contain different query_id sets.")
    datasets = {
        str(row.get("dataset", ""))
        for row in [*fusion_rows, *structural_rows]
    }
    if len(datasets) != 1 or "" in datasets:
        raise RuntimeError(f"Complementarity analysis must be dataset-local; found {sorted(datasets)}")
    return [(fusion_by_id[key], structural_by_id[key]) for key in sorted(fusion_by_id)]


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    fusion_wins = sum(float(row["delta_rr"]) > 0.0 for row in rows)
    structural_wins = sum(float(row["delta_rr"]) < 0.0 for row in rows)
    ties = count - fusion_wins - structural_wins
    fusion_mrr = sum(float(row["rr_fusion"]) for row in rows) / count if count else 0.0
    structural_mrr = sum(float(row["rr_structural"]) for row in rows) / count if count else 0.0
    oracle_mrr = sum(max(float(row["rr_fusion"]), float(row["rr_structural"])) for row in rows) / count if count else 0.0
    best_fixed = max(fusion_mrr, structural_mrr)
    return {
        "count": count,
        "fusion_wins": fusion_wins,
        "fusion_win_pct": fusion_wins / count if count else 0.0,
        "structural_wins": structural_wins,
        "structural_win_pct": structural_wins / count if count else 0.0,
        "ties": ties,
        "tie_pct": ties / count if count else 0.0,
        "mean_delta_rr": sum(float(row["delta_rr"]) for row in rows) / count if count else 0.0,
        "fusion_mrr": fusion_mrr,
        "structural_mrr": structural_mrr,
        "best_fixed_mrr": best_fixed,
        "hard_oracle_mrr": oracle_mrr,
        "oracle_headroom": oracle_mrr - best_fixed,
    }


def support_bucket(support: int) -> str:
    if support < 5:
        return "lt5"
    if support < 20:
        return "5_19"
    if support < 50:
        return "20_49"
    return "ge50"


def build_analysis_rows(pairs: list[tuple[dict, dict]]) -> list[dict]:
    relation_support = Counter(int(fusion["relation_id"]) for fusion, _ in pairs)
    rows = []
    for fusion, structural in pairs:
        for field in ("dataset", "split", "direction", "relation_id", "head_id", "tail_id", "target_entity_id"):
            if str(fusion.get(field, "")) != str(structural.get(field, "")):
                raise RuntimeError(f"Expert metadata mismatch for query_id={fusion['query_id']}: {field}")
        relation_id = int(fusion["relation_id"])
        rr_fusion = float(fusion["rr"])
        rr_structural = float(structural["rr"])
        row = {
            "query_id": fusion["query_id"],
            "dataset": fusion["dataset"],
            "split": fusion["split"],
            "direction": fusion["direction"],
            "relation_id": relation_id,
            "target_regime": fusion["target_regime"],
            "target_has_text": fusion.get("target_has_text", ""),
            "target_has_img": fusion.get("target_has_img", ""),
            "rr_fusion": rr_fusion,
            "rr_structural": rr_structural,
            "delta_rr": rr_fusion - rr_structural,
            "relation_support": relation_support[relation_id],
            "relation_support_bucket": support_bucket(relation_support[relation_id]),
        }
        if fusion.get("degree_bucket", "") not in ("", None):
            row["degree_bucket"] = fusion["degree_bucket"]
        rows.append(row)
    return rows


def group_summaries(rows: list[dict]) -> list[dict]:
    output = [{"group_type": "overall", "group_value": "all", **summarize(rows)}]
    for field in ("direction", "target_regime", "relation_id", "relation_support_bucket", "degree_bucket"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get(field, "") not in ("", None):
                buckets[str(row[field])].append(row)
        for value, bucket in sorted(buckets.items()):
            output.append({"group_type": field, "group_value": value, **summarize(bucket)})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = validate_and_pair(read_rows(Path(args.fusion)), read_rows(Path(args.structural)), args.split)
    rows = build_analysis_rows(pairs)
    groups = group_summaries(rows)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "expert_complementarity_per_query.csv", rows)
    write_csv(out_dir / "expert_complementarity_groups.csv", groups)
    payload = {
        "dataset": rows[0]["dataset"] if rows else None,
        "protocol_version": "mmkg_general_v1",
        "split": args.split,
        "selection_role": "validation_analysis" if args.split == "dev" else "locked_posthoc_reporting",
        "overall": summarize(rows),
        "sources": {"fusion": args.fusion, "structural": args.structural},
    }
    (out_dir / "expert_complementarity_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote complementarity analysis -> {out_dir}")


if __name__ == "__main__":
    main()
