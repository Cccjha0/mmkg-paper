from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from router.constants import QUERY_GEOMETRY_FIELDS
from scripts.build_aacpi_utility_table import build_utility_table, local_action_grid
from scripts.crossfit_heterogeneous_dev_policies import alpha_column


def toy_inputs(root: Path, *, split: str = "dev") -> tuple[Path, Path]:
    alphas = tuple(round(index * 0.05, 2) for index in range(21))
    rows = []
    for direction in ("head", "tail"):
        row = {
            "pair_name": "toy_aacpi",
            "dataset": "toy",
            "protocol_version": "toy_v1",
            "expert_a_name": "Expert A",
            "expert_b_name": "Expert B",
            "query_key": f"{direction}|r=2|h=1|t=3|target={1 if direction == 'head' else 3}",
            "query_id": f"{split}|1|{direction}|r=2|h=1|t=3",
            "split": split,
            "seed": 1,
            "direction": direction,
            "relation_id": 2,
            "head_id": 1,
            "tail_id": 3,
            "target_entity_id": 1 if direction == "head" else 3,
            "rank_a": 11,
            "rank_b": 11,
            "rr_a": 1.0 / 11.0,
            "rr_b": 1.0 / 11.0,
            "rr_oracle": 1.0 / 11.0,
        }
        for field_index, field in enumerate(QUERY_GEOMETRY_FIELDS):
            row[field] = float(field_index + (1 if direction == "tail" else 0))
        for alpha in alphas:
            rank = 1 + int(round(abs(alpha - 0.5) * 20.0))
            row[alpha_column(alpha)] = 1.0 / rank
        row["rr_global"] = row[alpha_column(0.5)]
        rows.append(row)

    query_path = root / "dev_query_rows.csv"
    with query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "pair_name": "toy_aacpi",
        "dataset": "toy",
        "protocol_version": "toy_v1",
        "expert_a_name": "Expert A",
        "expert_b_name": "Expert B",
        "seeds": [1],
        "score_normalization": "query_zscore",
        "alpha_grid": list(alphas),
        "global_alpha": 0.5,
        "global_dev_mrr": 1.0,
    }
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return query_path, selection_path


class AACPIUtilityTableTest(unittest.TestCase):
    def test_local_action_grid_clips_deduplicates_and_keeps_anchor(self) -> None:
        self.assertEqual(
            local_action_grid(0.6),
            (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9),
        )
        self.assertEqual(local_action_grid(1.0), (0.7, 0.8, 0.9, 0.95, 1.0))

    def test_builder_writes_leakage_safe_dev_advantage_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path, selection_path = toy_inputs(root)
            result = build_utility_table(query_path, selection_path, root / "out")
            summary = result["summary"]

            self.assertEqual(summary["split"], "dev")
            self.assertEqual(summary["n_original_triples"], 1)
            self.assertEqual(summary["n_query_instances"], 2)
            self.assertEqual(summary["n_query_action_rows"], 18)
            self.assertEqual(summary["positive_opportunity_query_rate"], 0.0)
            self.assertTrue(summary["validation"]["group_coverage_passed"])

            with gzip.open(result["table_path"], "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["original_triple_id"] for row in rows}, {"h=1|r=2|t=3"})
            reference = [row for row in rows if float(row["alpha"]) == 0.5]
            self.assertEqual(len(reference), 2)
            self.assertTrue(all(float(row["advantage"]) == 0.0 for row in reference))
            self.assertTrue(all(row["rank_action"] == row["rank_anchor"] for row in reference))

    def test_builder_rejects_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path, selection_path = toy_inputs(root, split="test")
            with self.assertRaisesRegex(RuntimeError, "DEV-only"):
                build_utility_table(query_path, selection_path, root / "out")


if __name__ == "__main__":
    unittest.main()
