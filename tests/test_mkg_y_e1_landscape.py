from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/complementarity_identifiability/mkg_y_y_e1_landscape"


class MkgYE1LandscapeOutputTests(unittest.TestCase):
    def test_replication_assessment_and_pair_statistics(self) -> None:
        assessment = json.loads((OUTPUT / "replication_assessment.json").read_text(encoding="utf-8"))
        stats = pd.read_csv(OUTPUT / "pair_statistics.csv")
        self.assertEqual(assessment["outcome"], "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT")
        self.assertFalse(assessment["is_progression_gate"])
        self.assertEqual(assessment["significant_headroom_pairs"], 3)
        self.assertEqual(assessment["test_status"], "LOCKED")
        self.assertEqual(len(stats), 3)
        self.assertTrue((stats["available_headroom"] > 0).all())
        self.assertTrue((stats["headroom_ci95_low"] > 0).all())
        self.assertTrue((stats["positive_opportunity_rate"] >= 0.25).all())
        self.assertTrue(
            ((stats["oracle_mrr"] - stats["global_mrr"] - stats["available_headroom"]).abs() < 1e-12).all()
        )

    def test_complete_seed_direction_geometry_and_dev_boundary(self) -> None:
        frame = pd.read_csv(
            OUTPUT / "per_query_action_geometry.csv.gz",
            usecols=["pair_id", "original_triple_id", "seed", "direction"],
        )
        self.assertEqual(len(frame), 3 * 2665 * 3 * 2)
        self.assertEqual(set(frame["seed"]), {1, 2, 3})
        self.assertEqual(set(frame["direction"]), {"head", "tail"})
        sizes = frame.groupby(["pair_id", "original_triple_id"]).size()
        self.assertEqual(set(sizes), {6})

    def test_audit_declares_zero_test_and_no_policy(self) -> None:
        audit = json.loads((OUTPUT / "audit_manifest.json").read_text(encoding="utf-8"))
        operational = audit["operational_audit"]
        self.assertEqual(audit["split"], "dev")
        self.assertEqual(audit["filter_fact_scope"], "train_dev")
        self.assertEqual(operational["test_access"], 0)
        self.assertEqual(operational["checkpoint_evaluation"], 0)
        self.assertEqual(operational["selector_training"], 0)
        self.assertEqual(operational["policy_runs"], 0)

    def test_required_svg_figures_are_well_formed(self) -> None:
        for name in (
            "figure1_global_to_oracle.svg",
            "figure2_action_landscape_heatmaps.svg",
            "figure3_gwd_distributions.svg",
            "figure4_relation_direction_consistency.svg",
        ):
            path = OUTPUT / name
            root = ET.parse(path).getroot()
            self.assertTrue(root.tag.endswith("svg"))
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
