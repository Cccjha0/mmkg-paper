"""B01-B03: independent normalization arithmetic and representation boundaries."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router.feature_redundancy import expansion_matrix, collapsed_raw_coefficients, augmented_standardized_map
from router.query_geometry import query_geometry_tensor
from router.score_combination import normalize_candidate_scores
from scripts.eval_heterogeneous_complementarity import query_zscore_with_reference, ranks_against_reference, mixed_ranks, endpoint_safe_mixed_ranks
from scripts.analyze_paper_a_feature_dimension import DIMENSIONS, signals
from scripts.analyze_paper_a_matched_alternatives import Inputs


def scalar_definition(row, eps):
    finite=[v for v in row if math.isfinite(v)]
    n=max(1,len(finite)); mean=sum(finite)/n
    sigma=math.sqrt(sum((v-mean)**2 for v in finite)/n)
    return [(v-mean)/(sigma+eps) if math.isfinite(v) else -math.inf for v in row]


@pytest.mark.parametrize('row',[
    [1.,2.,6.,10.], [4.,4.,4.,4.], [2.,2.+1e-11,2.-1e-11,2.],
    [1.,3.,math.nan,math.inf], [math.nan,-math.inf,math.inf,math.nan],
    [math.nan,4.,math.inf,-math.inf],
])
def test_raw_row_matches_independent_population_definition(row):
    raw=torch.tensor([row],dtype=torch.float64)
    expected=torch.tensor([scalar_definition(row,1e-8)],dtype=torch.float64)
    normalized=normalize_candidate_scores(raw,'query_zscore')
    exported,reference=query_zscore_with_reference(raw,torch.tensor([2.],dtype=torch.float64))
    torch.testing.assert_close(normalized,expected,rtol=1e-12,atol=1e-15)
    torch.testing.assert_close(exported,expected,rtol=1e-12,atol=1e-15)
    assert torch.equal(torch.isneginf(exported),~torch.isfinite(raw))


def test_each_model_uses_own_support_and_normalization_is_batch_independent():
    a=torch.tensor([[1.,3.,math.nan,5.]],dtype=torch.float64)
    b=torch.tensor([[math.nan,10.,20.,30.]],dtype=torch.float64)
    for raw in (a,b):
        expected=torch.tensor([scalar_definition(raw[0].tolist(),1e-8)])
        actual=normalize_candidate_scores(raw,'query_zscore')
        torch.testing.assert_close(actual,expected.to(torch.float64),rtol=1e-6,atol=1e-8)
        batched=torch.cat([raw,torch.tensor([[1e3,2e3,3e3,4e3]],dtype=torch.float64)])
        torch.testing.assert_close(normalize_candidate_scores(batched,'query_zscore')[:1],actual,rtol=0,atol=0)
    za,ga=query_zscore_with_reference(a,torch.tensor([3.]))
    zb,gb=query_zscore_with_reference(b,torch.tensor([10.]))
    # In the paper export, interior mixtures exclude candidates missing in either
    # expert; router.combine_expert_scores has a different fill policy, not used here.
    assert mixed_ranks(za,zb,ga,gb,.5).item()==2
    assert endpoint_safe_mixed_ranks(za,zb,ga,gb,1.,torch.tensor([3]),torch.tensor([4])).item()==3


def test_geometry_std_is_sample_and_invalid_geometry_sanitization_is_batch_scoped():
    raw=torch.tensor([[1.,2.,3.,4.]])
    geom=query_geometry_tensor(raw,raw,'tail')
    assert geom[0,4].item()==pytest.approx(math.sqrt(5/3))
    abnormal=torch.tensor([[math.nan,0.,1.,2.]])
    extra=torch.tensor([[-100.,-99.,-98.,-97.]])
    single=query_geometry_tensor(abnormal,abnormal,'tail')[0]
    batch=query_geometry_tensor(torch.cat([abnormal,extra]),torch.cat([abnormal,extra]),'tail')[0]
    assert single[4]!=batch[4]  # Document, do not claim universal row-wise fallback.


def test_legacy_nonfinite_gold_comparison_is_not_a_valid_recovery_policy():
    raw=torch.full((1,4),math.nan)
    z,gold=query_zscore_with_reference(raw,torch.tensor([math.nan]))
    assert torch.isneginf(z).all() and torch.isnan(gold).all()
    assert ranks_against_reference(z,gold).item()==1  # IEEE comparisons false; not evidence of a correct answer.


def test_redundant_features_preserve_affine_predictions_and_change_l2_geometry():
    rng=np.random.default_rng(73); x9=rng.normal(size=(120,9)); x13=x9@expansion_matrix().T
    y=(x9[:,1]-x9[:,5]>.1).astype(int)
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                        LogisticRegression(solver='liblinear',class_weight='balanced',random_state=7)).fit(x13,y)
    coef,bias=collapsed_raw_coefficients(model)
    np.testing.assert_allclose(model.decision_function(x13),x9@coef+bias,rtol=1e-12,atol=1e-12)
    scaler9=StandardScaler().fit(x9)
    m=augmented_standardized_map(scaler9,model[-2])
    np.testing.assert_allclose(np.c_[scaler9.transform(x9),np.ones(len(x9))]@m.T,
                               np.c_[model[-2].transform(x13),np.ones(len(x9))],atol=1e-12)
    target=rng.normal(size=10)
    gram=m.T@m
    minimal=m@np.linalg.solve(gram,target)
    np.testing.assert_allclose(m.T@minimal,target,atol=1e-12)
    assert minimal@minimal==pytest.approx(target@np.linalg.solve(gram,target))
    assert not np.allclose(gram,np.eye(10))
    # Without scaling, a,b,a-b yield penalty (2a^2+2ab+2b^2)/3.
    d=np.array([[1,0],[0,1],[1,-1]])
    np.testing.assert_allclose(np.linalg.inv(d.T@d),np.array([[2,1],[1,2]])/3)


def test_independent_medians_can_break_difference_dependency():
    x=np.array([[0.,0.,0.],[100.,1.,99.],[2.,100.,-98.]])
    out=SimpleImputer(strategy='median').fit(x).transform([[np.nan,np.nan,np.nan]])[0]
    assert out[0]-out[1]!=out[2]


def test_sufficient_margin_condition_is_not_necessary_for_a_realized_action():
    a=np.array([1.,-1.]);b=-a;anchor=.75;beta=.5
    ref=anchor*a+(1-anchor)*b;d=a-b
    assert not ref[0]-ref[1]>beta*abs(d[0]-d[1])
    actual=.8*a+.2*b
    assert np.argmax(actual)==np.argmax(ref)==0


def test_dimension_review_uses_nested_feature_sets_and_blocks_test_in_dev():
    assert DIMENSIONS[9]==DIMENSIONS[13][:9]
    with pytest.raises(ValueError,match='DEV phase cannot open TEST'):
        Inputs('DEV').frame(Path(__file__).parent/'test_rows.csv')


def test_raw_audit_distinguishes_constants_from_nonfinite_and_overflow():
    from scripts.audit_paper_a_raw_score_contract import exception_counts
    raw=torch.tensor([[4.,4.],[math.nan,math.nan],[3e38,-3e38]])
    counts,bad=exception_counts(raw,torch.tensor([4.,math.nan,0.]))
    assert bad.tolist()==[False,True,True]
    assert counts['constant_finite_rows']==1
    assert counts['empty_finite_rows']==1
    assert counts['nonfinite_gold_rows']==1
    assert counts['nonfinite_moment_rows']==1
