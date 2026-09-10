"""Read-only audit of old Paper A exports; writes evidence, never relabels them."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_a_safe_correction/information_boundary"


def inspect_export(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            observed = row["tail_id"] if row["direction"] == "head" else row["head_id"]
            key = (row["seed"], row["direction"], row["relation_id"], observed)
            groups[key].append(row)
    multi = [rows for rows in groups.values() if len({r["target_entity_id"] for r in rows}) > 1]
    fields = [name for name in next(iter(groups.values()))[0] if name.startswith("geometry_")]
    changed = [rows for rows in multi if len({tuple(r[f] for f in fields) for r in rows}) > 1]
    weight_fields = [f for f in ("alpha_anchored_continuous", "alpha_anchored_locked",
                                "alpha_query_soft_locked") if f in next(iter(groups.values()))[0]]
    result = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": sum(map(len, groups.values())),
        "multi_gold_observable_query_groups": len(multi),
        "groups_with_different_geometry": len(changed),
        "groups_with_different_weights": {
            field: sum(len({r[field] for r in rows}) > 1 for rows in multi) for field in weight_fields
        },
        "interpretation": "Artifact-level association; causal counterexample is covered by the toy code-path tests.",
    }
    if changed:
        sample = changed[0]
        varying = [f for f in fields if len({r[f] for r in sample}) > 1]
        result["example"] = [{f: r[f] for f in ["seed", "direction", "head_id", "relation_id",
                                              "tail_id", "target_entity_id", *varying, *weight_fields]}
                             for r in sample[:3]]
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    for dataset in ("mkg_w", "db15k"):
        for pair in ("mhyper_native_seed123", "mhyper_adamf_seed123", "native_adamf_seed123"):
            base = ROOT / "outputs" / dataset / "anchored_dynamic" / pair
            for relative in ("full_ranking/dev_query_rows.csv", "test_full_ranking/test_query_rows.csv",
                             "test_anchored/test_locked_query_rows.csv"):
                path = base / relative
                if path.exists():
                    assets.append(inspect_export(path))
    required = ["manifest.json", "train.tsv", "valid.tsv", "test.tsv", "entity2id.json", "relation2id.json",
                "text_feat.pt", "img_feat.pt", "has_text.pt", "has_img.pt"]
    missing = {dataset: [name for name in required if not (ROOT / "data/datasets" / dataset / "processed" / name).exists()]
               for dataset in ("mkg_w", "db15k")}
    report = {
        "audit": "B07_C04", "status": "BLOCKED_PENDING_CORRECTED_REEXPORT_AND_REEVALUATION",
        "historical_results_validated_for_submission": False,
        "missing_processed_assets": missing, "exports": assets,
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exports": len(assets), "missing_processed_assets": missing,
                      "evidence": str(OUT / "audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
