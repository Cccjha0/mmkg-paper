"""B08/B10: executable fallback, deterministic selection, endpoint boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router.anchored_inference import guarded_outputs, apply_locked_features
from router.endpoint_audit import compare_endpoints, shared_endpoint_ranks
from scripts.crossfit_heterogeneous_dev_policies import best_alpha as fold_anchor
from scripts.eval_heterogeneous_complementarity import best_alpha as full_anchor, score_expert_block
from ml.training.src.eval import filtered_ranking as shared


def model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(solver='liblinear',random_state=7)).fit(
        [[0.,0.],[1.,.1],[2.,.2],[3.,.3]],[0,0,1,1])


def test_invalid_features_are_split_before_any_preprocessing(monkeypatch):
    fitted=model(); seen=[]; original=fitted[0].transform
    def transform(x):
        assert np.isfinite(x).all()
        seen.extend(x.tolist())
        return original(x)
    monkeypatch.setattr(fitted[0],'transform',transform)
    x=np.array([[0.,0.],[np.nan,1e308],[np.inf,0.],[-np.inf,np.nan],[2.,.2]])
    result=guarded_outputs(fitted,x)
    assert result['invalid'].tolist()==[False,True,True,True,False]
    assert seen==[[0.,0.],[2.,.2]]
    np.testing.assert_array_equal(result['decision'][result['invalid']],0.)
    np.testing.assert_array_equal(result['probability_a'][result['invalid']],.5)


def test_all_invalid_batch_never_calls_transform_or_predict(monkeypatch):
    fitted=model()
    def forbidden(*args,**kwargs):
        raise AssertionError('Invalid row reached an estimator')
    for obj,name in ((fitted[0],'transform'),(fitted[1],'transform'),(fitted[2],'decision_function'),(fitted[2],'predict_proba')):
        monkeypatch.setattr(obj,name,forbidden)
    out=apply_locked_features(fitted,[[np.nan,np.nan],[np.inf,-np.inf],[np.nan,1e308]],
                              alpha0=.55,beta=.45,threshold=0,alphas=np.arange(21)/20)
    assert out['invalid'].all() and out['fallback'].all() and not out['predicted'].any()
    np.testing.assert_array_equal(out['applied'],.55)
    np.testing.assert_array_equal(out['query_soft_applied'],.55)


def test_preprocessing_overflow_is_removed_before_classifier(monkeypatch):
    fitted=model(); original=fitted[-1].decision_function
    def decision(x):
        assert np.isfinite(x).all()
        return original(x)
    monkeypatch.setattr(fitted[-1],'decision_function',decision)
    out=guarded_outputs(fitted,[[0.,1e308],[1.,.1]])
    assert out['invalid_reason'].tolist()==['nonfinite_preprocessing','']


def test_nonfinite_logit_is_removed_before_probability_prediction(monkeypatch):
    fitted=model(); fitted[1].mean_[:]=0; fitted[1].scale_[:]=1
    fitted[-1].coef_[:]=2
    original=fitted[-1].predict_proba
    def probability(x):
        assert np.max(np.abs(x))<100
        return original(x)
    monkeypatch.setattr(fitted[-1],'predict_proba',probability)
    out=guarded_outputs(fitted,[[1e308,1e308],[0.,0.]])
    assert out['invalid_reason'].tolist()==['nonfinite_logit','']


def test_finite_batch_matches_locked_pipeline_and_keeps_row_order():
    fitted=model(); x=np.array([[3.,.3],[0.,0.],[1.,.1]])
    out=guarded_outputs(fitted,x)
    np.testing.assert_array_equal(out['decision'],fitted.decision_function(x))
    np.testing.assert_array_equal(out['probability_a'],fitted.predict_proba(x)[:,1])
    assert not out['invalid'].any()


def test_legacy_predict_before_fallback_can_fail_after_nan_imputation():
    fitted=model()
    with np.errstate(over='ignore'), pytest.raises(ValueError,match='infinity'):
        fitted.predict_proba([[np.nan,1e308]])


def test_invalid_probabilities_return_to_anchor(monkeypatch):
    fitted=model()
    monkeypatch.setattr(fitted[-1],'predict_proba',lambda x: np.array([[0.,np.nan],[0.,1.1],[.5,.5]]))
    out=apply_locked_features(fitted,[[0.,0.],[1.,.1],[2.,.2]],alpha0=.75,beta=.25,threshold=0,alphas=[0.,.75,1.])
    assert out['invalid_reason'].tolist()==['invalid_probability','invalid_probability','']
    np.testing.assert_array_equal(out['applied'][:2],.75)


def test_empty_batch_does_not_call_estimators(monkeypatch):
    fitted=model()
    def forbidden(*args,**kwargs): raise AssertionError('Empty batch reached estimator')
    monkeypatch.setattr(fitted[0],'transform',forbidden)
    assert not len(guarded_outputs(fitted,np.empty((0,2)))['decision'])


@pytest.mark.parametrize('choose',[full_anchor,fold_anchor])
def test_global_anchor_ties_prefer_half_then_lower_and_never_override_better_mrr(choose):
    rows=[{f'rr_alpha_{a:.2f}'.replace('.','_'):.5 for a in (0.,.25,.5,.75,1.)}]
    assert choose(rows,(1.,.75,.5,.25,0.))[0]==.5
    assert choose(rows,(.75,.25))[0]==.25
    rows[0]['rr_alpha_1_00']=np.nextafter(.5,1.)
    assert choose(rows,(0.,.25,.5,.75,1.))[0]==1.


class ToyModel(torch.nn.Module):
    def __init__(self,values):
        super().__init__(); self.values=torch.tensor(values,dtype=torch.float32)
    def score_head(self,triples):
        return self.values[triples[:,0]]
    def score_tail(self,triples):
        return self.values[triples[:,2]]


@pytest.mark.parametrize('direction',['head','tail'])
@pytest.mark.parametrize('values',[[2.,2.,1.,0.],[4.,4.,4.,4.],[float('inf'),2.,1.,float('nan')]])
def test_endpoints_match_actual_shared_evaluator_per_row_with_ties_and_nonfinite(monkeypatch,direction,values):
    model=ToyModel(values)
    q=torch.tensor([[1,0,0],[2,0,0]] if direction=='head' else [[0,0,1],[0,0,2]])
    facts={(0,0):torch.tensor([1,2])}
    expert=SimpleNamespace(model=model,num_entities=4,query_batch_size=2,chunk_size=3)
    _,gold,raw=score_expert_block(expert,q,direction,{},'cpu',retain_unfiltered=True)
    monkeypatch.setattr(shared,'_metrics_from_ranks',lambda ranks,ks: {'ranks':ranks.tolist()})
    fn=shared._filtered_head_ranking_eval if direction=='head' else shared._filtered_tail_ranking_eval
    expected=fn(model,q,facts,num_entities=4,chunk_size=3,query_batch_size=2,device='cpu')['ranks']
    counts,direct,_=compare_endpoints(raw,gold,q,direction,facts)
    assert direct.tolist()==expected
    assert counts['export_vs_shared_mismatches']==counts['alpha_zero_vs_shared_mismatches']==counts['alpha_one_vs_shared_mismatches']==0
    assert shared_endpoint_ranks(raw,gold,q,direction,facts,dense=False).tolist()==expected


def test_float32_normalization_can_merge_distinct_scores_at_the_endpoint():
    raw=torch.tensor([[0.,1.,1e8]])
    counts,direct,normalized=compare_endpoints(raw,raw[:,0],torch.tensor([[0,0,0]]),'tail',{})
    assert direct.item()==3 and normalized.item()==2
    assert counts['normalized_vs_raw_mismatches']==1


def test_nonfinite_active_scores_preclude_a_universal_affine_equivalence_claim():
    raw=torch.tensor([[1.,2.,float('inf')]])
    _,direct,normalized=compare_endpoints(raw,raw[:,0],torch.tensor([[0,0,0]]),'tail',{})
    assert direct.item()==3 and normalized.item()==2
