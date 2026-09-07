from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_adaptive_resolution import (
    ALPHAS,
    degree_edges,
    fit_group_policy,
    load_contract,
    modality_states,
    select_action,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for triple in range(4):
        for seed in (1, 2, 3):
            for direction in ("head", "tail"):
                rows.append(
                    {
                        "query_id": f"q{triple}-{seed}-{direction}",
                        "seed": seed,
                        "direction": direction,
                        "head_id": triple,
                        "relation_id": 0,
                        "tail_id": triple + 10,
                        "original_triple_id": f"t{triple}",
                        "degree_value": float(triple),
                        "modality_state": "none",
                    }
                )
    return pd.DataFrame(rows)


class AdaptiveResolutionTest(unittest.TestCase):
    def test_contract_is_frozen_dev_only(self) -> None:
        contract = load_contract(Path("docs/protocols/EXP3_ADAPTIVE_RESOLUTION_CONTRACT.json"))
        self.assertEqual(contract["levels"]["L4"]["minimum_support_candidates"], [10, 25, 50, 100])
        self.assertFalse(contract["levels"]["L5"]["new_training"])
        self.assertEqual(contract["test_access"], 0)

    def test_support_counts_original_triples_not_seed_direction_rows(self) -> None:
        frame = synthetic_frame()
        rr = np.zeros((len(frame), len(ALPHAS)))
        rr[:, 12] = 1.0
        train = np.ones(len(frame), dtype=bool)
        policy, _, _ = fit_group_policy(frame, rr, train, "L2", global_index=10, minimum_support=5, edges=None)
        self.assertEqual(policy, {})
        policy, rows, _ = fit_group_policy(frame, rr, train, "L2", global_index=10, minimum_support=4, edges=None)
        self.assertEqual(next(iter(policy.values())), 12)
        self.assertEqual(rows[0]["support_original_triples"], 4)

    def test_degree_edges_ignore_outer_held_values(self) -> None:
        frame = synthetic_frame()
        held = frame.original_triple_id == "t3"
        frame.loc[held, "degree_value"] = 1000.0
        edges = degree_edges(frame, (~held).to_numpy(), [0.25, 0.5, 0.75])
        self.assertLess(float(edges.max()), 1000.0)

    def test_action_tie_prefers_fold_global(self) -> None:
        rr = np.ones((5, len(ALPHAS)))
        selected = select_action(rr, np.arange(5), global_index=17)
        self.assertEqual(selected, 17)

    def test_modality_state_uses_known_side_binary_fields(self) -> None:
        states = modality_states(np.asarray([0, 1, 0, 1]), np.asarray([0, 0, 1, 1]))
        self.assertEqual(states.tolist(), ["none", "text_only", "image_only", "text_and_image"])


if __name__ == "__main__":
    unittest.main()
