from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from scripts.audit_mkg_y_y0_y1_acceptance import (
    manifests_match_except_registry_hash,
    normalized_embedded_manifest,
    validate_metric_frame,
)


def test_manifest_comparison_ignores_only_whole_registry_hash() -> None:
    old = {"hashes": {"train": "a"}, "source_lock": {"lock_sha256": "old", "verified_files": {"train": "x"}}}
    new = copy.deepcopy(old)
    new["source_lock"]["lock_sha256"] = "new"
    assert manifests_match_except_registry_hash(old, new)
    new["source_lock"]["verified_files"]["train"] = "changed"
    assert not manifests_match_except_registry_hash(old, new)
    assert "lock_sha256" not in normalized_embedded_manifest(old)["source_lock"]


def test_metric_validation_accepts_finite_monotone_dev_trace() -> None:
    frame = pd.DataFrame({
        "epoch": [5, 10], "mrr": [0.2, 0.3], "hits@1": [0.1, 0.2],
        "hits@3": [0.3, 0.4], "hits@10": [0.5, 0.6],
    })
    assert validate_metric_frame(frame) == []


def test_metric_validation_rejects_nonfinite_and_hits_order() -> None:
    nonfinite = pd.DataFrame({
        "epoch": [5], "mrr": [np.nan], "hits@1": [0.1], "hits@3": [0.2], "hits@10": [0.3],
    })
    assert validate_metric_frame(nonfinite)
    wrong_hits = pd.DataFrame({
        "epoch": [5], "mrr": [0.2], "hits@1": [0.4], "hits@3": [0.3], "hits@10": [0.5],
    })
    assert "Hits@K is not monotone" in validate_metric_frame(wrong_hits)
