"""Denominators, exact RR events, decomposition and clustered ratio sampling."""
import numpy as np
import pytest
from router.outcome_risk_diagnostics import observations,summarize,ratios,cluster_ratio_intervals,FIELDS


def rows(base,rank,active=None):
    base=np.asarray(base);rank=np.asarray(rank)
    active=(base!=rank) if active is None else np.asarray(active,bool)
    return observations(1/rank,1/base,np.where(active,.5,1.),np.ones(len(rank)))[0]


def test_active_harm_denominator_differs_from_all_and_rank_unchanged_is_not_inactive():
    x=rows([1,2,2,1,1,1],[2,1,2,1,1,1],[1,1,1,0,0,0]);r=summarize(x)
    assert (r['n'],r['active_n'],r['harm_n'],r['benefit_n'],r['unchanged_n'])==(6,3,1,1,4)
    assert r['harm']==pytest.approx(1/6) and r['harm_active']==pytest.approx(1/3)
    assert r['active_unchanged_n']==1 and r['inactive_n']==3
    assert r['mean_gain']==pytest.approx(r['benefit']*r['conditional_gain'])
    assert r['mean_loss']==pytest.approx(r['harm']*r['conditional_loss'])
    assert r['mean_gain']-r['mean_loss']==pytest.approx(r['utility'])


def test_empty_conditional_events_are_undefined_while_unconditional_mass_is_zero():
    r=summarize(rows([1]*6,[1]*6))
    assert r['mean_loss']==r['mean_gain']==0 and r['top1_retention']==1
    for k in ('harm_active','conditional_loss','conditional_gain','loss_q95','top1_active_retention'):assert np.isnan(r[k])


def test_rank_threshold_events_use_exact_rational_boundaries():
    # The floating 1/6 - 1/15 is slightly below .1; the mathematical loss is .1.
    r=summarize(rows([6,1,1,1,10,10],[15,2,10,11,1,10]))
    assert r['severe_10_n']==4 and r['severe_50_n']==3
    assert r['top1_n']==3 and r['top1_lost_n']==3 and r['top1_outside10_n']==1
    assert r['new_top1_n']==1 and r['hits1_delta']==pytest.approx(-2/6)


def test_tail_quantiles_condition_on_harm_and_rare_large_losses_are_visible():
    x=rows([1,2,10,100,2,1],[100,3,20,101,1,1]);r=summarize(x)
    loss=np.array([.99,1/2-1/3,.05,1/100-1/101])
    assert r['loss_q50']==pytest.approx(np.quantile(loss,.5,method='linear'))
    assert r['loss_max']==pytest.approx(.99) and r['worst_tenth_harm_n']==1
    assert r['worst_tenth_loss_share']==pytest.approx(.99/loss.sum())


def test_ratio_bootstrap_matches_explicit_full_cluster_draws_before_subset_selection():
    x=np.vstack([rows([1]*6,[2,1,1,1,1,1]),rows([1]*6,[2,2,2,1,1,1]),rows([2]*6,[1,1,1,2,2,2])])
    ids=np.repeat(np.arange(3),6);r=cluster_ratio_intervals(x,ids,7,100)
    ix=np.random.Generator(np.random.PCG64(7)).integers(3,size=(100,3))
    explicit=ratios(x.reshape(3,6,-1)[ix].sum(axis=(1,2)))
    for name,v in explicit.items():
        good=v[np.isfinite(v)]
        assert r[name+'_valid_replicates']==len(good)
        if len(good):np.testing.assert_allclose([r[name+'_lo'],r[name+'_hi']],np.quantile(good,[.025,.975]),atol=1e-14)
    # Resampling only active observations would silently change this estimand.
    assert r['harm_active_hi']!=r['harm_hi']


def test_zero_denominator_draws_are_counted_per_metric_and_incomplete_clusters_rejected():
    x=np.vstack([rows([1]*6,[2,1,1,1,1,1]),rows([1]*6,[1]*6)])
    ids=np.repeat(np.arange(2),6);r=cluster_ratio_intervals(x,ids,9,100)
    assert 0<r['harm_active_valid_replicates']<100 and r['harm_valid_replicates']==100
    with pytest.raises(ValueError):cluster_ratio_intervals(x[:-1],ids[:-1],9,100)
    with pytest.raises(ValueError):observations([.5],[1.],[1.],[1.])
    with pytest.raises(ValueError):observations([.7],[1.],[.5],[1.])
