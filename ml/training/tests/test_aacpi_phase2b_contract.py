from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.train_aacpi_advantage_nested_cv import (
    average_precision,
    calibration_rows,
    evaluate_predictions,
    load_yaml,
    model_configs,
    roc_auc,
    validate_search_space,
)


class AACPIPhase2BContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.search_path = Path("docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml")

    def test_frozen_search_space_has_exactly_36_low_capacity_configs(self) -> None:
        space = load_yaml(self.search_path)
        validate_search_space(space)
        configs = model_configs(space)
        self.assertEqual(len(configs), 36)
        self.assertEqual({config.hidden_width for config in configs}, {32, 64, 128})
        self.assertEqual({config.negative_weight for config in configs}, {1.0, 1.5, 2.0, 3.0})

    def test_detection_metrics_reward_perfect_advantage_ordering(self) -> None:
        actual = np.asarray([-0.5, -0.1, 0.0, 0.1, 0.5], dtype=np.float64)
        predicted = actual.copy()
        metrics = evaluate_predictions(actual, predicted, beta=0.02)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["positive_auprc"], 1.0)
        self.assertAlmostEqual(metrics["positive_auroc"], 1.0)
        self.assertAlmostEqual(metrics["harmful_auprc"], 1.0)
        self.assertAlmostEqual(metrics["harmful_auroc"], 1.0)
        self.assertAlmostEqual(average_precision(actual > 0, predicted), 1.0)
        self.assertAlmostEqual(roc_auc(actual < 0, -predicted), 1.0)

    def test_calibration_uses_six_locked_rank_buckets(self) -> None:
        actual = np.linspace(-0.5, 0.5, 100)
        rows = calibration_rows(actual, actual)
        self.assertEqual([row["bucket"] for row in rows], [
            "lowest_10pct",
            "10_to_30pct",
            "30_to_50pct",
            "50_to_70pct",
            "70_to_90pct",
            "highest_10pct",
        ])
        self.assertLess(rows[0]["actual_mean_u"], rows[-1]["actual_mean_u"])
        self.assertGreater(rows[0]["harmful_rate"], rows[-1]["harmful_rate"])


if __name__ == "__main__":
    unittest.main()
