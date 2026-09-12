"""Feature-only inference must not require or consume offline evaluation metadata."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router.information_boundary import SCORE_INFORMATION_CONTRACT
from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.apply_anchored_safe import apply_frame


@pytest.fixture
def sample():
    rng=np.random.default_rng(71)
    x=rng.normal(size=(30,13))
    fitted=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(solver='liblinear')).fit(x,np.arange(30)%2)
    lock=dict(score_information_contract=SCORE_INFORMATION_CONTRACT,query_geometry_fields=list(QUERY_GEOMETRY_FIELDS),
              alpha0=.55,beta=.45,confidence_threshold=.1,alpha_grid=(np.arange(21)/20).tolist())
    return pd.DataFrame(x[:5],columns=QUERY_GEOMETRY_FIELDS),fitted,lock


def test_feature_only_frame_produces_weights_without_any_answer_or_rank(sample):
    frame,model,lock=sample
    result=apply_frame(frame,model,lock)
    assert len(result)==len(frame) and result.predicted.all()
    assert np.isfinite(result.applied).all()
    assert not any(k.startswith(('rank','rr_')) for k in result)


@pytest.mark.parametrize('metadata',[
    {'seed':99,'direction':'unused','head_id':-1,'tail_id':-1,'target_entity_id':-1},
    {**{f'rr_alpha_{a:.2f}'.replace('.','_'):np.nan for a in np.arange(21)/20},'rr_a':np.inf,'rr_b':-np.inf,'rank_a':-1},
    {'seed':np.nan,'target_entity_id':np.inf,'gold':'unavailable','filter':'not supplied'},
])
def test_cache_and_seed_metadata_cannot_change_the_shared_policy(sample,metadata):
    frame,model,lock=sample
    expected=apply_frame(frame,model,lock)
    decorated=frame.assign(**metadata)
    actual=apply_frame(decorated,model,lock)
    pd.testing.assert_frame_equal(actual[expected.columns],expected)


def test_shared_selector_can_apply_each_observation_individually(sample):
    frame,model,lock=sample
    batched=apply_frame(frame,model,lock)
    singles=pd.concat([apply_frame(frame.iloc[[i]],model,lock) for i in range(len(frame))],ignore_index=True)
    np.testing.assert_allclose(singles.decision,batched.decision,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(singles.applied,batched.applied)


def test_rank_columns_cannot_substitute_for_missing_observable_features(sample):
    frame,model,lock=sample
    with pytest.raises(ValueError,match='Missing feature column'):
        apply_frame(frame.drop(columns=QUERY_GEOMETRY_FIELDS[0]).assign(rr_a=1.,rr_b=.5),model,lock)
