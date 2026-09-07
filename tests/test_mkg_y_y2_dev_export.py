from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_mkg_y_y2_dev_export import preflight


class MkgYY2DevExportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "docs/protocols/MKG_Y_Y2_DEV_FULL_RANKING_CONTRACT.json").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_dev_only_protocol(self) -> None:
        contract = self.contract
        self.assertEqual(contract["status"], "frozen_before_y2_export")
        self.assertEqual(contract["split"], "dev")
        self.assertEqual(contract["ranking_protocol"]["filter_fact_scope"], "train_dev")
        self.assertEqual(contract["test_policy"]["test_rows_opened"], 0)
        self.assertEqual(contract["prohibited"]["test_access"], 0)
        self.assertEqual(contract["ranking_protocol"]["alpha_grid"]["values"], [index / 20 for index in range(21)])

    def test_exact_three_pair_three_seed_inventory(self) -> None:
        self.assertEqual(len(self.contract["pairs"]), 3)
        self.assertEqual(len(self.contract["experts"]), 3)
        for expert in self.contract["experts"].values():
            self.assertEqual(sorted(expert["runs"]), ["1", "2", "3"])
            for run in expert["runs"].values():
                self.assertEqual(len(run["checkpoint_sha256"]), 64)

    def test_preflight_accepts_frozen_assets_without_test_rows(self) -> None:
        result = preflight(self.contract)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["acceptance_progression"], "Y2_DEV_EXPORT_ALLOWED")
        self.assertEqual(result["checkpoint_count"], 9)
        self.assertEqual(result["test_rows_opened"], 0)


if __name__ == "__main__":
    unittest.main()
