"""Selection boundaries, exact tie rules and seed/triple uncertainty denominators."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from scripts.analyze_paper_a_selection_seed import (FIELDS,GRID_COLUMNS,ACTIONS,folds_for,inner_oof,
    candidate_matrix,select_index,seed_summary,observable_overlap)
from router.rejection_diagnostics import paired_intervals


def toy():
    rows=[]
    for h in range(12):
        for seed in (1,2,3):
            for direction in ('head','tail'):
                r=dict(head_id=h,relation_id=h%2,tail_id=h+1,seed=seed,direction=direction,
                       rr_a=.5 if h%2 else .25,rr_b=.25 if h%2 else .5)
                r.update({f:float(h)/20 for f in FIELDS})
                r.update({f:.1+.05*j+(h%3)*.0001 for j,f in enumerate(GRID_COLUMNS)})
                rows.append(r)
    return pd.DataFrame(rows)


def test_original_triple_fold_assignments_ignore_gold_outcomes_and_retain_six_rows():
    f=toy();ids=folds_for(f,3,2026091209)
    changed=f.copy();changed['rr_a']=999;changed['rr_b']=-999
    np.testing.assert_array_equal(ids,folds_for(changed,3,2026091209))
    for h,part in f.groupby('head_id'):assert len(set(ids[part.index]))==1 and len(part)==6


def test_exact_policy_ties_and_explicit_global_are_separate_candidates():
    score=np.ones(41)
    assert select_index(score,True)==0
    assert ACTIONS[select_index(score,False)].strength==.05
    assert ACTIONS[select_index(score,False)].tau==.3
    score[10]+=1e-13  # Original exact tie rule, not matched-family 1e-12 tolerance.
    assert select_index(score,True)==10
    with pytest.raises(ValueError):select_index(np.full(41,np.nan),True)


def test_inner_models_never_see_their_heldout_triples_and_row_weighted_scores_match():
    f=toy();ids=folds_for(f,3,2026091209);calls=[]
    class Model:
        def __init__(self,seen):self.seen=seen
        def decision_function(self,x):
            assert self.seen.isdisjoint(set(x[:,0]))
            return x[:,0]*2-1
        def predict_proba(self,x):
            p=1/(1+np.exp(-self.decision_function(x)))
            return np.column_stack((1-p,p))
        def __getitem__(self,i):return SimpleNamespace(n_iter_=np.array([1]))
    def fit(rows,fields,random_state):
        seen={r[fields[0]] for r in rows};calls.append((seen,random_state))
        return Model(seen),None,None
    actual,scopes=inner_oof(f,2,fit_fn=fit)
    assert len(calls)==len(scopes)==3
    expected=[]
    for k,scope in enumerate(scopes):
        held=f.loc[ids==k];g=held[FIELDS[0]].to_numpy()*2-1;p=1/(1+np.exp(-g))
        expected.append(candidate_matrix(held,(g,p,np.zeros(len(g),bool)),scope['anchor']))
        assert scope['random_state']==20261103+k
        assert scope['fit_triples']+scope['heldout_triples']==12
    np.testing.assert_allclose(actual,np.concatenate(expected).mean(axis=0))


def test_outer_holdout_outcomes_cannot_change_the_supplied_inner_training_scope():
    f=toy();ids=folds_for(f,5,20260901);train=f.loc[ids!=0].copy()
    changed=f.copy();changed.loc[ids==0,GRID_COLUMNS+['rr_a','rr_b']]=123
    pd.testing.assert_frame_equal(train,changed.loc[ids!=0])


def test_seed_sd_is_sample_sd_of_paired_effects_not_subtracted_marginal_sds():
    a=np.array([.2,.4,.6]);b=a-.01
    mean,sd=seed_summary(a-b)
    assert mean==pytest.approx(.01) and sd<1e-15
    assert seed_summary(a)[1]==pytest.approx(.2)
    with pytest.raises(ValueError):seed_summary([1.,2.])


def test_triple_bootstrap_keeps_six_rows_and_is_not_a_seed_bootstrap():
    delta=np.tile(np.array([.1,-.1,.2,-.2,.3,-.3]),3)[:,None]
    point,lo,hi=paired_intervals(delta,np.repeat(np.arange(3),6),17,replicates=50)
    np.testing.assert_allclose([point,lo,hi],0,atol=1e-15)
    with pytest.raises(ValueError):paired_intervals(delta[:-1],np.repeat(np.arange(3),6)[:-1],17,50)


def test_observable_overlap_is_distinct_from_triple_overlap():
    train=pd.DataFrame([dict(seed=1,direction='tail',head_id=0,relation_id=2,tail_id=3)])
    held=pd.DataFrame([dict(seed=1,direction='tail',head_id=0,relation_id=2,tail_id=4),
                       dict(seed=1,direction='head',head_id=0,relation_id=2,tail_id=4)])
    assert observable_overlap(train,held)==(1,2)
