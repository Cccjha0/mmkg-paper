from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.audit_complementarity_landscape import (
    ALPHA_COLUMNS,
    ALPHA_GRID,
    clustered_bootstrap,
    compute_query_geometry,
    deterministic_best_indices,
)


class ComplementarityLandscapeTest(unittest.TestCase):
    def test_deterministic_best_prefers_global_then_lower_alpha(self) -> None:
        rr = np.zeros((2, len(ALPHA_GRID)), dtype=np.float64)
        rr[0, [8, 10, 12]] = 1.0
        rr[1, [8, 12]] = 1.0
        selected = deterministic_best_indices(rr, alpha0=0.5)
        self.assertEqual(ALPHA_GRID[selected[0]], 0.5)
        self.assertEqual(ALPHA_GRID[selected[1]], 0.4)

    def test_geometry_width_distance_plateau_and_fragmentation(self) -> None:
        values = np.full(len(ALPHA_GRID), 0.2, dtype=np.float64)
        values[[2, 3, 7]] = [0.3, 0.3, 0.4]
        row = {
            "dataset": "toy",
            "expert_a_name": "A",
            "expert_b_name": "B",
            "query_id": "dev|1|head|q",
            "seed": 1,
            "direction": "head",
            "relation_id": 2,
            "head_id": 3,
            "tail_id": 4,
        }
        row.update(dict(zip(ALPHA_COLUMNS, values)))
        geometry = compute_query_geometry(pd.DataFrame([row]), pair_id="toy_pair", alpha0=0.5)
        actual = geometry.iloc[0]
        self.assertAlmostEqual(actual["gain_amplitude"], 0.2)
        self.assertAlmostEqual(actual["beneficial_basin_width"], 3 / 21)
        self.assertAlmostEqual(actual["min_beneficial_distance"], 0.15)
        self.assertEqual(actual["best_alpha"], 0.35)
        self.assertEqual(actual["best_action_direction"], "toward_b")
        self.assertEqual(actual["beneficial_components"], 2)
        self.assertTrue(actual["beneficial_fragmented"])
        self.assertAlmostEqual(actual["plateau_ratio"], 18 / 20)

    def test_bootstrap_clusters_seeds_and_directions_by_triple(self) -> None:
        frame = pd.DataFrame(
            {
                "original_triple_id": ["t1"] * 6 + ["t2"] * 6,
                "gain_amplitude": [0.1] * 6 + [0.3] * 6,
            }
        )
        result = clustered_bootstrap(frame, n_bootstrap=200, seed=7)
        self.assertEqual(result["n_original_triple_clusters"], 2)
        self.assertLessEqual(result["headroom_ci95_low"], 0.2)
        self.assertGreaterEqual(result["headroom_ci95_high"], 0.2)


if __name__ == "__main__":
    unittest.main()
