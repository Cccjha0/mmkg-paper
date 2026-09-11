import numpy as np
import pandas as pd
import pytest

from scripts.audit_paper_a_data_checkpoint import (
    check_export, check_history, matching_split_encoding, split_bytes,
)
import hashlib


def example_export():
    triples = [(0, 1, 2), (2, 0, 3)]
    rows = []
    for seed in (1, 2, 3):
        for direction in ("head", "tail"):
            for i, (h, r, t) in enumerate(triples):
                rows.append(dict(seed=seed, direction=direction, query_id=f"{seed}-{direction}-{i}",
                                 head_id=h, relation_id=r, tail_id=t,
                                 target_entity_id=h if direction == "head" else t,
                                 rank_a=2, rr_a=0.5, rank_b=4, rr_b=0.25))
    return pd.DataFrame(rows), triples


def test_all_gold_rows_survive_missing_modality_independent_export_and_row_order():
    frame, triples = example_export()
    check_export(frame.sample(frac=1, random_state=3), triples)


@pytest.mark.parametrize("corruption", ["duplicate", "wrong_gold", "wrong_triple", "wrong_rr", "missing"])
def test_same_count_or_incomplete_export_is_rejected(corruption):
    frame, triples = example_export()
    if corruption == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif corruption == "wrong_gold":
        frame.loc[0, "target_entity_id"] = 99
    elif corruption == "wrong_triple":
        frame.loc[0, "relation_id"] = 99
    elif corruption == "wrong_rr":
        frame.loc[0, "rr_a"] = 0.25
    else:
        frame = frame.iloc[:-1]
    with pytest.raises(AssertionError):
        check_export(frame, triples)


def history():
    frame = pd.DataFrame({"epoch": [5, 10, 15, 20], "mrr": [0.2, 0.3, 0.3, 0.29],
                          "avg_loss": [2.0, 1.5, 1.2, 1.1]})
    cfg = {"training": {"eval_every": 5, "epochs": 20, "termination_policy": "fixed_budget"},
           "evaluation": {"run_test": False}}
    return frame, cfg


def test_first_dev_max_selected_and_non_best_export_rejected():
    frame, cfg = history()
    assert check_history(frame, cfg, 0.3)["best_epoch"] == 10
    with pytest.raises(AssertionError):
        check_history(frame, cfg, 0.29)


def test_truncated_budget_or_nonfinite_history_rejected():
    frame, cfg = history()
    with pytest.raises(AssertionError):
        check_history(frame.iloc[:-1], cfg, 0.3)
    frame.loc[1, "avg_loss"] = np.nan
    with pytest.raises(AssertionError):
        check_history(frame, cfg, 0.3)


def test_split_hash_requires_exact_membership_order_and_line_endings():
    rows = [(1, 0, 2), (3, 0, 2)]
    digest = hashlib.sha256(split_bytes(rows, "\r\n")).hexdigest()
    assert matching_split_encoding(rows, digest) == "CRLF"
    with pytest.raises(AssertionError):
        matching_split_encoding(rows[::-1], digest)
