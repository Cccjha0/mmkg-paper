from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json"
OUTPUT_ROOT = ROOT / "outputs/complementarity_identifiability/mkg_y_y_e2_information"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_freezes_only_original_x4_replication() -> None:
    contract = load_contract()
    assert contract["status"] == "frozen_before_first_systematic_run"
    assert contract["split"] == "dev"
    assert contract["protocol_profile"] == "mkg_y_frozen_x4_replication"
    assert set(contract["representations"]) == {"X1", "X2_additions", "X3_additions", "X4_additions"}
    assert contract["learner_compatibility"] == {
        "X4": ["linear_huber", "hist_gbdt", "mlp_low", "mlp_high"]
    }
    assert contract["x6_status"] == "excluded_no_independent_justification"
    assert contract["test_access"] == 0
    assert contract["final_policy_development"] == 0


def test_contract_source_boundaries_match_disk() -> None:
    contract = load_contract()
    prerequisite = contract["prerequisite"]
    assert sha256(ROOT / prerequisite["manifest"]) == prerequisite["sha256"]
    headroom = contract["source_boundaries"]["available_headroom"]
    assert sha256(ROOT / headroom["path"]) == headroom["sha256"]
    for pair_id, source in contract["source_boundaries"]["full_ranking_manifests"].items():
        path = ROOT / source["path"]
        assert sha256(path) == source["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pair"]["pair_id"] == pair_id
        assert payload["pair"]["filter_fact_scope"] == "train_dev"
        assert payload["pair"]["test_rows_opened"] == 0


def test_utility_scaffolding_uses_exact_dev_sources_and_train_dev_filtering() -> None:
    contract = load_contract()
    expected_alphas = np.asarray(contract["alpha_grid"], dtype=float)
    for pair_id in contract["pair_ids"]:
        path = OUTPUT_ROOT / "utility_tables" / f"{pair_id}_dev_source_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["split"] == "dev"
        assert manifest["filtered_ranking_protocol"]["filter_fact_scope"] == "train_dev"
        query_path = ROOT / manifest["source_query_rows"]["path"]
        assert sha256(query_path) == manifest["source_query_rows"]["sha256"]
        query = pd.read_csv(query_path, usecols=["query_id", "seed", "direction", *[f"rr_alpha_{alpha:.2f}".replace(".", "_") for alpha in expected_alphas]])
        assert len(query) == 15990
        assert query.query_id.nunique() == 15990
        assert sorted(query.seed.unique().tolist()) == [1, 2, 3]
        assert set(query.direction) == {"head", "tail"}


def test_nested_protocol_keeps_fold_specific_global_and_grouping() -> None:
    nested = load_contract()["nested_cv"]
    assert nested["outer_folds"] == 5
    assert nested["inner_folds"] == 3
    assert nested["grouping_key"] == "original_triple_id"
    assert nested["global_alpha_scope"] == "outer_train_only"
    assert load_contract()["action_descriptors"] == ["alpha", "delta_alpha", "abs_delta_alpha"]


def test_preflight_records_zero_compute_and_test_access() -> None:
    payload = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["verified_source_count"] >= 58
    assert payload["test_access"] == 0
    assert payload["checkpoint_inference"] == 0
    assert payload["learner_training"] == 0


def test_server_runner_exposes_only_x4_systematic_runs() -> None:
    script = (ROOT / "scripts/run_mkg_y_y_e2_x4_oof.ps1").read_text(encoding="utf-8")
    assert '"--representation", "X4"' in script
    assert '"--representation", "X5"' not in script
    assert '"--representation", "X6"' not in script
    assert "build_exp2_union_top100.py" not in script
    assert "extract_aacpi_frozen_query_latents.py" not in script
