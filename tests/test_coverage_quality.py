import itertools
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from router.coverage_quality import NestedOrder, rates, aggregate
from router.rejection_diagnostics import observable_units
from router.ties_harm_diagnostics import RankedBinary
from scripts.analyze_paper_a_confidence_harm import average_precision


def test_mechanical_harm_does_not_imply_conditional_improvement():
    # Higher-confidence proposal harms; accepting the next safe proposal lowers
    # conditional risk even though cumulative all-denominator harm stays fixed.
    gate=NestedOrder([0,0],[.9,.1],[0,1])
    values=np.array([[1,1,1,-.5,.5,.1],[1,1,0,.5,0,.2]])
    totals,_=gate.totals(values); r=rates(totals,2)
    assert np.all(np.diff(totals[:,2])>=0)
    assert r[5,2]==r[10,2]==.5
    assert r[5,3]==1 and r[10,3]==.5
    assert np.isnan(r[0,3:5]).all()
    assert all(np.all(~gate.mask(j)|gate.mask(j+1)) for j in range(10))


def test_accepted_and_active_denominators_differ():
    x=np.array([10,2,1,-.1,.2,.3],float)
    r=rates(x,20)
    np.testing.assert_allclose(r[:5],[.5,.1,.05,.1,.5])


def test_random_expectation_matches_enumeration_and_nested_budgets():
    gate=NestedOrder([0]*4,[4,3,2,1],np.arange(4))
    values=np.array([[2,2,1,-.5,.5,.2],[2,0,0,0,0,0],
                     [2,2,0,.2,0,.4],[2,2,2,-.8,.8,.6]])
    point,expected=gate.totals(values)
    enumerated=np.mean([values[list(v)].sum(axis=0) for v in itertools.combinations(range(4),2)],axis=0)
    np.testing.assert_allclose(expected[5],enumerated)
    draws=gate.random_totals(values,27,64)
    np.testing.assert_allclose(draws[:,:,0],np.broadcast_to(point[:,0],(64,11)))
    assert np.all(np.diff(draws[:,:,2],axis=1)>=0)
    np.testing.assert_allclose(draws[:,-1],np.broadcast_to(point[-1],(64,6)))
    # Mean of conditional ratios is generally not ratio of expected counts.
    unequal=np.array([[4,1,1,-.5,.5,.1],[4,3,0,.5,0,.2]])
    assert not np.isclose(rates(unequal,8)[:,4].mean(),rates(unequal.mean(axis=0),8)[4])


def test_gold_free_units_and_stratum_repetitions():
    # Two possible golds for one observable query share a single decision.
    units,inv=observable_units([1,1,1],[0,0,0],['tail']*3,[2]*3,[7,7,8])
    assert inv[0]==inv[1] and sorted(units.repetitions)==[1,2]
    gate=NestedOrder([0,0],[.3,.8],[.1,.2])
    keep=gate.mask(5)[inv]
    assert keep[0]==keep[1]
    with pytest.raises(ValueError):NestedOrder([0],[np.inf],[1])


def test_macro_micro_and_conditional_weights():
    x=np.array([[4,2,1,0,1,1],[12,10,8,0,8,1]],float); sizes=[10,30]
    micro=aggregate(x,sizes,'micro'); macro=aggregate(x,sizes,'macro')
    assert micro[2]==9/40 and macro[2]==pytest.approx((1/10+8/30)/2)
    assert micro[4]==9/12 and macro[4]==pytest.approx((1/2+8/10)/2)
    assert micro[4]!=pytest.approx(np.average([.5,.8],weights=sizes))


def test_ap_is_step_weighted_with_ties_not_trapezoid_or_ratio():
    y=np.array([0,1,1,0,1]);s=np.array([.9,.8,.8,.6,.1])
    old=average_precision(y,s);new=RankedBinary(y,s,np.arange(5)).evaluate(np.ones(5))
    expected=average_precision_score(y,s)
    np.testing.assert_allclose([old,new[2]],[expected,expected])
    precision,recall,_=precision_recall_curve(y,s)
    assert not np.isclose(expected,auc(recall,precision))
    assert new[3]==pytest.approx(expected-y.mean())
    assert average_precision(y,np.ones(5))==pytest.approx(y.mean())
    # Pooled AP requires sorting across pairs; it is not a size-weighted AP mean.
    first=average_precision_score(y[:3],s[:3]);second=average_precision_score(y[3:],s[3:])
    assert not np.isclose(expected,(3*first+2*second)/5)
