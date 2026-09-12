"""Diagnostic information boundary and opportunity/decomposition regression checks."""
import numpy as np
import pandas as pd
import pytest

from scripts.analyze_paper_a_complementarity import observable_support, rank_ratio_bin, summarize, join_oof


def test_unknown_gold_and_its_modality_attributes_do_not_change_query_support():
    query = pd.DataFrame({"direction":["tail","head"], "head_id":[0,2], "tail_id":[1,3], "relation_id":[7,8]})
    img, txt = np.array([1,0,1,0]), np.array([0,1,1,0])
    known, before = observable_support(query,img,txt)
    np.testing.assert_array_equal(known,[0,3])
    np.testing.assert_array_equal(before,["image-only","neither"])
    # Unknown endpoints can even be out of the candidate universe: no lookup.
    query.loc[0,"tail_id"] = 99999
    query.loc[1,"head_id"] = -50
    img[[1,2]], txt[[1,2]] = 1-img[[1,2]], 1-txt[[1,2]]
    np.testing.assert_array_equal(observable_support(query,img,txt)[1],before)


def test_known_entity_support_is_not_a_constant_dummy():
    query = pd.DataFrame({"direction":["tail"],"head_id":[0],"tail_id":[1]})
    assert observable_support(query,[0,0],[0,0])[1][0] == "neither"
    assert observable_support(query,[1,0],[1,0])[1][0] == "both"


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError,match="direction"):
        observable_support(pd.DataFrame({"direction":["other"],"head_id":[0],"tail_id":[0]}),[1],[1])


def test_fixed_rank_ratio_boundaries():
    np.testing.assert_array_equal(rank_ratio_bin(np.array([1,2,1,4,1]),np.array([1,1,4,1,5])),
                                  ["equal","(1,2]","(2,4]","(2,4]",">4"])


def fixture():
    return pd.DataFrame(dict(head_id=[0,1,2],relation_id=[0,0,0],tail_id=[1,2,0],direction=["tail"]*3,
        rr_a=[1.,.1,.5],rr_b=[.1,1.,.5],rank_a=[1,10,2],rank_b=[10,1,2],
        reference_rr=[1.,.1,.5],adc_rr=[.5,.2,.5],applied_alpha=[.5,.5,1.],anchor_alpha=[1.]*3,
        explicit_fallback=[False,False,True]))


def test_oracle_is_mean_max_not_max_mean_and_is_not_ensemble_ceiling():
    frame = fixture()
    value = summarize(frame,len(frame))
    assert value["oracle_gap"] == pytest.approx(.3)
    assert value["a_win_rate"] == value["b_win_rate"] == value["tie_rate"] == pytest.approx(1/3)
    # A mixture may beat both endpoint gold ranks; do not clamp its diagnostic gain.
    frame.loc[2,"adc_rr"] = 1.
    assert summarize(frame.iloc[2:],len(frame))["adc_mrr"] > .5


def test_weighted_conditional_contributions_recover_total_and_empty_is_zero():
    frame = fixture()
    total = summarize(frame,3)
    pieces = [summarize(frame.iloc[ix],3) for ix in ([0],[1,2],[])]
    assert sum(p["delta_contribution"] for p in pieces) == pytest.approx(total["delta_mrr"])
    assert total["delta_mrr"] == pytest.approx(total["mean_gain"]-total["mean_loss"])
    assert pieces[-1]["mass"] == pieces[-1]["delta_contribution"] == 0
    assert "delta_mrr" not in pieces[-1]  # undefined conditional mean stays missing


def oof_fixture():
    raw = pd.DataFrame(dict(query_id=[f"q{i}" for i in range(5)],seed=[1]*5,direction=["tail"]*5,
                            head_id=list(range(5)),relation_id=[0]*5,tail_id=[5]*5,rr_a=[.5]*5))
    oof = raw.assign(fold=range(1,6),rr_method=.6,rr_global_crossfit=.5,alpha_applied=.8,alpha0=1.,fallback=0)
    return raw,oof


def test_shuffled_oof_rows_join_by_full_identity():
    raw,oof = oof_fixture()
    result = join_oof(raw,oof.iloc[::-1])
    assert result.query_id.tolist() == raw.query_id.tolist()
    assert result.adc_rr.eq(.6).all()


def test_oof_identity_or_endpoint_mutation_is_rejected():
    raw,oof = oof_fixture()
    oof.loc[0,"head_id"] = 42
    with pytest.raises(ValueError,match="metadata"):
        join_oof(raw,oof)
    raw,oof = oof_fixture()
    oof.loc[0,"rr_a"] = .9
    with pytest.raises(ValueError,match="endpoint"):
        join_oof(raw,oof)
