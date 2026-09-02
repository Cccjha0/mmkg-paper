from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from router.query_geometry import QUERY_GEOMETRY_FIELDS, query_geometry_rows, query_geometry_tensor
from scripts.crossfit_anchored_dynamic import apply_policy, main, nearest_alpha
from scripts.crossfit_heterogeneous_dev_policies import (
    alpha_column,
    assign_grouped_folds,
    best_alpha,
)
from scripts.eval_score_ensemble_baselines import query_features


def test_query_geometry_contract_is_finite_and_excludes_answer_fields() -> None:
    scores_a = torch.tensor([[4.0, 2.0, float("-inf"), 1.0], [3.0, 3.0, 3.0, 3.0]])
    scores_b = torch.tensor([[1.0, 3.0, float("-inf"), 2.0], [2.0, 2.0, 2.0, 2.0]])
    features = query_geometry_tensor(scores_a, scores_b, "tail")
    rows = query_geometry_rows(scores_a, scores_b, "tail")

    assert features.shape == (2, len(QUERY_GEOMETRY_FIELDS))
    assert torch.isfinite(features).all()
    assert all(row["geometry_direction_tail"] == 1.0 for row in rows)
    forbidden = ("target", "reference", "rank", "rr", "relation")
    assert not any(token in field for field in QUERY_GEOMETRY_FIELDS for token in forbidden)


def test_shared_query_geometry_preserves_legacy_and_general_feature_order() -> None:
    scores_a = torch.tensor([[4.0, 2.0, 1.0]])
    scores_b = torch.tensor([[1.0, 3.0, 2.0]])
    relations = torch.tensor([17])
    general = query_features(
        scores_a,
        scores_b,
        "head",
        relations,
        protocol_version=MMKG_GENERAL_V1,
    )
    legacy = query_features(
        scores_a,
        scores_b,
        "head",
        relations,
        protocol_version=OPENBG_LEGACY_V1,
    )

    assert general.shape == (1, len(QUERY_GEOMETRY_FIELDS))
    assert legacy.shape == (1, len(QUERY_GEOMETRY_FIELDS) + 1)
    assert legacy[0, 0] == general[0, 0]
    assert legacy[0, 1] == 17.0
    assert np.array_equal(legacy[0, 2:], general[0, 1:])


def test_anchored_policy_is_bounded_quantized_and_falls_back_to_anchor() -> None:
    alphas = tuple(round(index * 0.05, 2) for index in range(21))
    rows = []
    for _ in range(3):
        row = {}
        for alpha in alphas:
            row[f"rr_alpha_{alpha:.2f}".replace(".", "_")] = alpha
        rows.append(row)
    policy = apply_policy(
        rows,
        decision=np.asarray([-10.0, 10.0, 10.0]),
        probability_a=np.asarray([0.0, 1.0, 0.51]),
        nonfinite=np.asarray([False, False, False]),
        alpha0=0.6,
        beta=0.2,
        confidence_threshold=0.1,
        alphas=alphas,
    )

    assert policy["applied"].tolist() == [0.4, 0.8, 0.6]
    assert policy["fallback"].tolist() == [False, False, True]
    assert np.all(policy["continuous"] >= 0.4)
    assert np.all(policy["continuous"] <= 0.8)
    assert policy["rr"].tolist() == policy["applied"].tolist()
    assert nearest_alpha(0.625, alphas, anchor=0.6) == 0.6


def test_anchored_crossfit_cli_writes_dynamic_expert_labels() -> None:
    alphas = (0.0, 0.4, 0.5, 0.6, 0.8, 1.0)
    rows = []
    for triple_index in range(15):
        h, relation, tail = triple_index, triple_index % 3, triple_index + 100
        target_alpha = 0.8 if triple_index % 2 == 0 else 0.4
        for seed in (1, 2, 3):
            for direction in ("head", "tail"):
                alpha_rr = {
                    alpha: 1.0 / (1.0 + 5.0 * abs(alpha - target_alpha))
                    for alpha in alphas
                }
                signal = 1.0 if target_alpha > 0.6 else -1.0
                row = {
                    "pair_name": "toy_pair",
                    "split": "dev",
                    "query_key": f"{direction}|r={relation}|h={h}|t={tail}",
                    "query_id": f"dev|{seed}|{direction}|r={relation}|h={h}|t={tail}",
                    "seed": seed,
                    "direction": direction,
                    "relation_id": relation,
                    "head_id": h,
                    "tail_id": tail,
                    "rr_a": alpha_rr[1.0],
                    "rr_b": alpha_rr[0.0],
                    "rr_equal": alpha_rr[0.5],
                    "rr_oracle": max(alpha_rr[1.0], alpha_rr[0.0]),
                }
                row.update({alpha_column(alpha): value for alpha, value in alpha_rr.items()})
                for index, field in enumerate(QUERY_GEOMETRY_FIELDS):
                    row[field] = (
                        float(direction == "tail")
                        if field == "geometry_direction_tail"
                        else signal * (index + 1)
                    )
                rows.append(row)

    assignment, _ = assign_grouped_folds(rows, folds=5, fold_seed=20260901)
    for fold in range(5):
        train_rows = [row for row in rows if assignment[f"h={row['head_id']}|r={row['relation_id']}|t={row['tail_id']}"] != fold]
        alpha0, _ = best_alpha(train_rows, alphas)
        for row in rows:
            key = f"h={row['head_id']}|r={row['relation_id']}|t={row['tail_id']}"
            if assignment[key] == fold:
                row["crossfit_fold"] = fold + 1
                row["alpha_global_crossfit"] = alpha0
                row["rr_global_crossfit"] = row[alpha_column(alpha0)]
                row["rr_relation_crossfit"] = row[alpha_column(alpha0)]

    test_temp_root = Path.cwd() / ".codex_tmp"
    test_temp_root.mkdir(exist_ok=True)
    temp = test_temp_root / f"anchored_dynamic_test_{uuid.uuid4().hex}"
    temp.mkdir()
    try:
        query_path = temp / "query_rows.csv"
        selection_path = temp / "selection.json"
        output_dir = temp / "anchored"
        with query_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selection_path.write_text(
            json.dumps(
                {
                    "pair_name": "toy_pair",
                    "dataset": "toy",
                    "expert_a_name": "Expert Alpha",
                    "expert_b_name": "Expert Beta",
                    "alpha_grid": list(alphas),
                }
            ),
            encoding="utf-8",
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                "crossfit_anchored_dynamic.py",
                "--query-rows",
                str(query_path),
                "--selection-json",
                str(selection_path),
                "--output-dir",
                str(output_dir),
                "--folds",
                "5",
                "--betas",
                "0.10,0.20",
                "--confidence-thresholds",
                "0.00,0.10",
            ]
            main()
        finally:
            sys.argv = original_argv

        results = list(
            csv.DictReader((output_dir / "dev_anchored_results.csv").open(encoding="utf-8"))
        )
        assert results[0]["method"] == "Expert Alpha"
        assert results[1]["method"] == "Expert Beta"
        summary = json.loads((output_dir / "dev_anchored_summary.json").read_text(encoding="utf-8"))
        assert summary["leakage_guard"].startswith("all seeds and both directions")
        assert summary["diagnostics"]["saturation_rate"] == 0.0
    finally:
        shutil.rmtree(temp)
