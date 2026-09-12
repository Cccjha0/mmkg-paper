import numpy as np
import pandas as pd
import pytest

from scripts.audit_paper_a_claims_cost import align, reverse_interval, time_ratio


def rows():
    return pd.DataFrame(dict(seed=[1,1,2],direction=['head','tail','head'],
                             head_id=[4,4,4],relation_id=[2,2,2],tail_id=[9,9,9],rr=[.5,.25,.125]))


def test_pairing_respects_seed_direction_and_triple_when_shuffled():
    frame=rows()
    np.testing.assert_array_equal(align(frame,frame.iloc[[2,0,1]],['rr']).rr,frame.rr)


@pytest.mark.parametrize('mutation',['missing','duplicate','target_changed','direction_changed'])
def test_pairing_rejects_observation_mismatch(mutation):
    frame=rows(); changed=frame.copy()
    if mutation=='missing': changed=changed.iloc[:2]
    elif mutation=='duplicate': changed=pd.concat([changed,changed.iloc[:1]])
    elif mutation=='target_changed': changed.loc[0,'tail_id']=10
    else: changed.loc[0,'direction']='tail'
    with pytest.raises(ValueError): align(frame,changed,['rr'])


def test_reversing_paired_effect_also_reverses_interval_endpoints():
    assert reverse_interval([.001,.003]) == [-.003,-.001]
    assert reverse_interval([-.001,.003]) == [-.003,.001]
    with pytest.raises(ValueError): reverse_interval([.003,.001])


def test_total_time_ratio_is_not_relative_overhead():
    assert time_ratio(3.,.5)==6.
    assert time_ratio(3.,.5)!=5.
    for a,b in ((1.,0.),(-1.,1.),(np.nan,1.)):
        with pytest.raises(ValueError): time_ratio(a,b)
