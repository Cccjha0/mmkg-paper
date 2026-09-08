import json
from pathlib import Path

from scripts.exp2_information_common import representation_features


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/protocols/COMPLEMENTARITY_CLOSURE_TEST_CONTRACT.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_closure_test_contract_has_exact_three_dataset_inventory():
    contract = load(CONTRACT_PATH)
    expected = {
        f"{prefix}_{suffix}"
        for prefix in ("mkgw", "db15k", "mkg_y")
        for suffix in ("mhyper_native", "mhyper_adamf", "native_adamf")
    }
    assert contract["status"] == "frozen_before_one_time_test"
    assert contract["pair_count"] == 9
    assert {row["pair_id"] for row in contract["pairs"]} == expected
    assert contract["alpha_grid"] == [round(index * 0.05, 2) for index in range(21)]
    assert contract["score_normalization"] == "query_zscore"


def test_closure_test_x4_schema_and_final_configs_are_preexisting():
    contract = load(CONTRACT_PATH)
    core = load(ROOT / "docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
    mkg_y = load(ROOT / "docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json")
    assert contract["x4_feature_fields"] == representation_features(core)["X4"]
    assert contract["x4_feature_fields"] == representation_features(mkg_y)["X4"]
    for pair in contract["pairs"]:
        source = mkg_y if pair["dataset"] == "mkg_y" else core
        assert pair["final_x4"]["learner"] in contract["learner_order"]
        assert pair["final_x4"]["config"] in source["hyperparameters"][pair["final_x4"]["learner"]]
        assert [run["seed"] for run in contract["experts"][pair["expert_a_key"]]] == [1, 2, 3]
        assert [run["seed"] for run in contract["experts"][pair["expert_b_key"]]] == [1, 2, 3]
