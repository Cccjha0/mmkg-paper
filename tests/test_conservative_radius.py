import numpy as np
import pandas as pd
import pytest

from scripts.analyze_paper_a_conservative_radius import ALPHAS, project, metrics, nondominated, choose
from scripts.crossfit_anchored_dynamic import nearest_alpha


def test_projection_preserves_released_ties_and_neighbors():
    mid=(ALPHAS[:-1]+ALPHAS[1:])/2
    values=np.concatenate([mid,np.nextafter(mid,0),np.nextafter(mid,1),np.linspace(0,1,1001)])
    for anchor in ALPHAS:
        expected=[nearest_alpha(float(x),tuple(ALPHAS),float(anchor)) for x in values]
        np.testing.assert_array_equal(ALPHAS[project(values,anchor)],expected)


def test_small_action_bound_does_not_bound_rr_damage():
    a=np.array([1e-9,0.,-1.]);b=np.array([0.,1.,-1.])
    a=(a-a.mean())/a.std();b=(b-b.mean())/b.std()
    beta=1e-6
    mixed=(1-beta)*a+beta*b
    assert np.all(np.abs(mixed-a)<=beta*np.abs(a-b)+1e-15)
    assert 1/(1+sum(a>a[0]))==1
    assert 1/(1+sum(mixed>mixed[0]))==.5


def test_global_has_zero_risk_and_undefined_conditional_loss():
    r=metrics(np.array([.5,1]),np.array([.5,1]),np.array([.55,.55]),.55)
    assert r['mean_loss']==r['harm_rate']==r['action_rate']==0
    assert r['conditional_loss'] is None


def test_realized_loss_precedes_selector_averaging():
    rr=np.array([.8,.2]);reference=np.array([.5,.5])
    r=metrics(rr,reference)
    assert r['mean_loss']==pytest.approx(.15)
    assert metrics(np.array([rr.mean()]),np.array([.5]))['mean_loss']==0
    assert r['delta_mrr']==pytest.approx(r['mean_gain']-r['mean_loss'])
    assert r['action_rate'] is None


def test_pareto_and_restricted_selection_do_not_imply_global_optimality():
    f=pd.DataFrame(dict(delta_mrr=[0,.1,.09],mean_loss=[0,.01,.02]))
    assert nondominated(f).tolist()==[True,True,False]
    points=[dict(beta=.5,tau=.1,mrr=.35),dict(beta=.75,tau=0,mrr=.36),dict(beta=.5,tau=.2,mrr=.35)]
    assert choose(points,.5)['tau']==.2
    assert choose(points,1.)['beta']==.75
