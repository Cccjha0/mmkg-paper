import itertools
import numpy as np
import pytest
from scipy.special import expit
from router.component_ablation import component_weights,match_interventions,outcome_summary,METHODS


def test_center_change_preserves_fallback_and_changes_only_accepted_origin():
    g=np.array([0.,.15,1.]);p=expit(g)
    full=component_weights(g,p,[False]*3,1.,.5,.1,'ADC')
    center=component_weights(g,p,[False]*3,1.,.5,.1,'Center-0.5')
    np.testing.assert_array_equal(full,[1,1,1])
    np.testing.assert_array_equal(center,[1,1,.9])
    # At an already neutral anchor, the maps are identical (including projection ties).
    np.testing.assert_array_equal(component_weights(g,p,[False]*3,.5,.5,.1,'ADC'),
                                  component_weights(g,p,[False]*3,.5,.5,.1,'Center-0.5'))


def test_mapping_range_and_threshold_are_distinct_controls():
    g=np.array([-.3,.3]);p=expit(g);invalid=np.zeros(2,bool)
    full=component_weights(g,p,invalid,.8,.5,.2,'ADC')
    no=component_weights(g,p,invalid,.8,.5,.2,'No-gate')
    np.testing.assert_array_equal(full,[.8,.8]);assert np.any(no!=full)
    np.testing.assert_array_equal(no,component_weights(g,p,invalid,.8,.5,0,'ADC'))
    wide=component_weights(g,p,invalid,.8,.5,0,'Radius-1')
    assert np.max(abs(wide-.8))>np.max(abs(no-.8))
    shrink=component_weights([1.],expit([1.]),[False],1.,.5,0,'Shrink-fixed')
    assert shrink[0]<1 and component_weights([1.],expit([1.]),[False],1.,.5,0,'ADC')[0]==1


def test_shared_guard_and_linear_clip_local_bound():
    g=np.array([np.nan,np.inf,-np.inf,0.,-5.,5.]);p=expit(g)
    for method in METHODS:
        w=component_weights(g,p,[False]*6,.8,.3,0,method)
        np.testing.assert_array_equal(w[:3],[.8]*3)
        if method in ('ADC','Linear-p','Clip-g','Shrink-fixed'):assert np.max(abs(w-.8))<=.3+1e-12


def test_exact_stratified_matching_and_uniform_expectation():
    repetitions=np.array([2,2,2,1,1]);strata=[0,0,0,1,1]
    a,b,ledger=match_interventions(strata,repetitions,[1,1,1,1,0],[1,0,0,1,1])
    assert repetitions@a==pytest.approx(3) and repetitions@b==pytest.approx(3)
    np.testing.assert_allclose(a,[1/3,1/3,1/3,1,0]);np.testing.assert_allclose(b,[1,0,0,.5,.5])
    values=np.array([.6,-.2,.1]);enumerated=np.mean([values[list(k)].sum() for k in itertools.combinations(range(3),1)])
    assert values@a[:3]==pytest.approx(enumerated)
    assert sum(r['retained_observations'] for r in ledger)==3


def test_zero_quota_unchanged_pairs_and_bad_strata():
    a,b,_=match_interventions([0,0],[1,1],[1,1],[0,0]);assert not a.any() and not b.any()
    a,b,_=match_interventions([0,0],[1,1],[1,0],[1,0]);np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):match_interventions([0,0],[1,2],[1,1],[1,1])


def test_expected_metrics_use_discrete_outcomes_and_matched_denominators():
    ref=np.array([1.,.5]);rr=np.array([.5,1.]);alpha=np.array([.6,.6]);anchor=np.array([.8,.8])
    r=outcome_summary(rr,ref,alpha,anchor,[.5,.5])
    assert r['utility']==0 and r['harm']==.25 and r['harm_active']==.5
    assert r['mean_loss']==r['mean_gain']==.125 and r['active_n']==1
    zero=outcome_summary(rr,ref,alpha,anchor,[0,0]);assert np.isnan(zero['harm_active']) and zero['utility']==0
