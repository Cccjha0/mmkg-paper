from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.training.tests.test_aacpi_utility_table import toy_inputs
from scripts.analyze_aacpi_phase2a import analyze_pair, best_action
from scripts.build_aacpi_utility_table import build_utility_table


class AACPIPhase2ADiagnosticsTest(unittest.TestCase):
    def test_best_action_prefers_anchor_on_an_rr_plateau(self) -> None:
        rows = [
            {"query_id": "q", "alpha": "0.4", "delta_alpha": "-0.1", "rr_action": "0.5", "rr_anchor": "0.5", "advantage": "0"},
            {"query_id": "q", "alpha": "0.5", "delta_alpha": "0", "rr_action": "0.5", "rr_anchor": "0.5", "advantage": "0"},
            {"query_id": "q", "alpha": "0.6", "delta_alpha": "0.1", "rr_action": "0.5", "rr_anchor": "0.5", "advantage": "0"},
        ]
        selected = best_action(rows)
        self.assertEqual(selected["alpha"], 0.5)
        self.assertEqual(selected["direction"], "stay")

    def test_pair_diagnostics_keep_winner_and_action_targets_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path, selection_path = toy_inputs(root)
            built = build_utility_table(query_path, selection_path, root / "utility")
            summary, landscape, confusion = analyze_pair(
                Path(built["manifest_path"]), root / "phase2a"
            )

            self.assertEqual(summary["n_queries"], 2)
            self.assertEqual(summary["beneficial_deviation_rate"], 0.0)
            self.assertEqual(summary["winner_direction_agreement_all_queries"], 0.0)
            self.assertEqual(summary["max_radius_boundary_best_rate"], 0.0)
            self.assertEqual(len(landscape), 9)
            self.assertTrue(any(row["matrix"] == "winner_label_vs_best_action_class" for row in confusion))
            self.assertTrue(Path(summary["query_diagnostics"]).exists())


if __name__ == "__main__":
    unittest.main()
