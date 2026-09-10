import numpy as np
import pandas as pd
import pytest

from scripts.audit_paper_a_boundary_rerun import compare_rows, invariance


def rows(direction="tail"):
    return pd.DataFrame({"seed": [1, 1, 1], "direction": [direction] * 3,
                         "head_id": [0, 0, 3] if direction == "tail" else [1, 2, 2],
                         "tail_id": [1, 2, 2] if direction == "tail" else [0, 0, 3],
                         "relation_id": [0, 0, 0], "target_entity_id": [1, 2, 2],
                         "weight": [.5, .5, .9]})


@pytest.mark.parametrize("direction", ["head", "tail"])
def test_only_same_observable_query_is_compared_and_small_drift_is_not_hidden(direction):
    frame = rows(direction)
    result = invariance(frame, ["weight"])
    assert result["multi_gold_groups"] == 1
    assert result["fields"]["weight"]["nonidentical_groups"] == 0
    frame.loc[1, "weight"] += 1e-8
    assert invariance(frame, ["weight"])["fields"]["weight"]["nonidentical_groups"] == 1


def test_missing_observations_cannot_pass_by_inner_join():
    frame = rows()
    with pytest.raises(ValueError, match="complete one-to-one"):
        compare_rows(frame, frame.iloc[:2], ["weight"])


def test_duplicate_observations_cannot_pass():
    frame = rows()
    with pytest.raises(pd.errors.MergeError):
        compare_rows(frame, pd.concat([frame, frame.iloc[:1]]), ["weight"])


def test_nonfinite_values_cannot_pass_invariance():
    frame = rows()
    frame.loc[1, "weight"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        invariance(frame, ["weight"])


def test_row_order_does_not_change_alignment():
    frame = rows()
    assert compare_rows(frame, frame.iloc[::-1], ["weight"])["max_errors"] == {"weight": 0.0}
