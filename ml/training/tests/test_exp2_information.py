from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_exp2_union_top100 import candidate_block
from scripts.exp2_information_common import (
    ALPHAS,
    grouped_folds,
    load_contract,
    policy_metrics,
    representation_features,
    select_global_alpha,
    select_probe_actions,
)


class Experiment2InformationTest(unittest.TestCase):
    def test_feature_ladder_is_cumulative_and_x6_is_frozen_without_embeddings(self) -> None:
        path = Path("docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
        contract = load_contract(path)
        features = representation_features(contract)
        for lower, upper in zip(("X1", "X2", "X3", "X4"), ("X2", "X3", "X4", "X5")):
            self.assertTrue(set(features[lower]).issubset(features[upper]))
        self.assertEqual(contract["representations"]["X6_candidate"]["top_k_per_expert"], 100)
        self.assertEqual(contract["representations"]["X6_candidate"]["inherits"], "X5")
        self.assertEqual(contract["representations"]["X6_candidate"]["candidate_embeddings"], "excluded_before_first_systematic_run")
        self.assertFalse(set(contract["prohibited_existing_fields"]) & set(features["X5"]))

    def test_global_selection_uses_only_training_mask_and_frozen_ties(self) -> None:
        rr = np.zeros((3, len(ALPHAS)))
        rr[:2, 8] = 1.0
        rr[:2, 12] = 1.0
        rr[2, 20] = 100.0
        index, _ = select_global_alpha(rr, np.asarray([True, True, False]))
        self.assertEqual(ALPHAS[index], 0.4)

    def test_probe_ties_prefer_fold_global(self) -> None:
        predicted = np.zeros((2, len(ALPHAS)))
        predicted[1, 4] = 0.1
        chosen = select_probe_actions(predicted, global_index=10)
        self.assertEqual(chosen[0], 10)
        self.assertEqual(chosen[1], 4)

    def test_grouped_outer_fold_keeps_six_observations_together(self) -> None:
        rows = []
        for triple in range(10):
            for seed in (1, 2, 3):
                for direction in ("head", "tail"):
                    rows.append({"head_id": triple, "relation_id": triple % 2, "tail_id": triple + 20, "original_triple_id": f"h={triple}|r={triple%2}|t={triple+20}", "seed": seed, "direction": direction})
        frame = pd.DataFrame(rows)
        folds, _ = grouped_folds(frame, 5, 17)
        self.assertTrue(frame.assign(fold=folds).groupby("original_triple_id").fold.nunique().eq(1).all())

    def test_candidate_block_has_frozen_fields_and_union(self) -> None:
        a = np.linspace(-2, 2, 250)
        b = -a
        fields, ids = candidate_block(a, b, 100)
        self.assertEqual(fields.shape, (200, 10))
        self.assertEqual(len(ids), 200)
        self.assertTrue(np.isfinite(fields).all())

    def test_policy_metrics_use_clustered_query_gains(self) -> None:
        rr = np.zeros((12, len(ALPHAS)))
        rr[:, 10] = 0.2
        rr[:6, 12] = 0.3
        predicted = np.zeros_like(rr)
        predicted[:6, 12] = 1.0
        groups = np.asarray(["a"] * 6 + ["b"] * 6)
        metrics, chosen = policy_metrics(rr, predicted, np.full(12, 10), groups, 0.05, 200, 3)
        self.assertAlmostEqual(metrics["delta_mrr"], 0.05)
        self.assertAlmostEqual(metrics["headroom_recovery"], 1.0)
        self.assertTrue((chosen[:6] == 12).all())
        self.assertTrue((chosen[6:] == 10).all())


if __name__ == "__main__":
    unittest.main()
