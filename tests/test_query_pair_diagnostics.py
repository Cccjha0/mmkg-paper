"""Meaningful synthetic checks of grouping, purging, pairing and CI units."""
import numpy as np
import pandas as pd
import pytest
from scripts.analyze_paper_a_query_pair import (query_keys,purge_training,size_control,
    inventory,cluster_intervals,TRIPLE)


def frame(triples):
    return pd.DataFrame([dict(head_id=h,relation_id=r,tail_id=t,seed=s,direction=d,
        split='dev',score_information_contract='unfiltered_features_and_normalization_v2',rr_a=.5,rr_b=.25)
        for h,r,t in triples for s in (1,2,3) for d in ('head','tail')])


def test_query_keys_ignore_gold_for_each_direction_but_not_the_known_entity():
    f=frame([(1,2,3)])
    changed=f.copy();changed.loc[changed.direction=='head','head_id']=99
    changed.loc[changed.direction=='tail','tail_id']=88
    assert query_keys(f)==query_keys(changed)
    changed.loc[changed.direction=='tail','head_id']=77
    assert query_keys(f)!=query_keys(changed)


def test_purge_drops_whole_triples_for_either_direction_and_keeps_six_observations():
    held=frame([(1,2,3)])
    train=frame([(1,2,4),(5,2,3),(6,2,7),(1,9,3)])
    kept=purge_training(train,held)
    assert set(map(tuple,kept[TRIPLE].to_numpy()))=={(6,2,7),(1,9,3)}
    assert kept.groupby(TRIPLE).size().eq(6).all()
    assert not set(query_keys(kept)) & set(query_keys(held))
    poisoned=held.copy();poisoned[['rr_a','rr_b']]=999
    pd.testing.assert_frame_equal(kept,purge_training(train,poisoned))


def test_relation_size_control_is_deterministic_and_preserves_quotas_without_outcomes():
    train=frame([(h,h%2,h+1) for h in range(20)])
    purged=train[train.head_id<9]
    a=size_control(train,purged,3)
    b=size_control(train.sample(frac=1,random_state=5),purged,3)
    assert set(map(tuple,a[TRIPLE].to_numpy()))==set(map(tuple,b[TRIPLE].to_numpy()))
    assert a.groupby('relation_id').size().to_dict()==purged.groupby('relation_id').size().to_dict()


def test_inventory_distinguishes_unique_keys_answer_rows_and_cross_fold_rows():
    f=frame([(1,2,3),(1,2,4),(8,2,4)])
    rows={r['direction']:r for r in inventory(f,np.repeat([1,2,2],6))}
    assert rows['tail']['unique_keys']==2 and rows['tail']['repeated_keys']==1
    assert rows['tail']['rows_on_repeated_keys']==2 and rows['tail']['cross_fold_keys']==1
    assert rows['tail']['rows_on_cross_fold_keys']==2
    assert rows['head']['repeated_keys']==1 and rows['head']['cross_fold_keys']==0
    assert rows['head']['answer_rows']==3  # Three seeds do not inflate semantic counts.


def test_cluster_percentiles_match_explicit_six_row_resampling_and_reversed_contrast():
    f=frame([(1,1,2),(2,1,3),(3,1,4)])
    x=np.arange(18)/100;d=np.column_stack([x,-x,np.zeros(18)])
    p,lo,hi,meta=cluster_intervals(f,d,19,100)
    rng=np.random.Generator(np.random.PCG64(19));ix=rng.integers(3,size=(100,3))
    explicit=d.reshape(3,6,3)[ix].mean(axis=(1,2))
    np.testing.assert_allclose(p,d.mean(axis=0))
    np.testing.assert_allclose([lo,hi],np.quantile(explicit,[.025,.975],axis=0,method='linear'))
    assert lo[0]==pytest.approx(-hi[1]) and hi[0]==pytest.approx(-lo[1])
    assert lo[2]==hi[2]==0 and meta['clusters']==3


def test_bootstrap_row_order_invariance_and_fixed_seed_dispersion_not_resampled():
    f=frame([(2,1,3),(1,1,2),(3,1,4)])
    d=np.tile([.1,-.1,.2,-.2,.3,-.3],3)[:,None]
    p,lo,hi,m=cluster_intervals(f,d,77,100)
    order=np.random.default_rng(1).permutation(len(f))
    q,l,h,n=cluster_intervals(f.iloc[order],d[order],77,100)
    np.testing.assert_allclose([p,lo,hi],[q,l,h],atol=1e-15)
    np.testing.assert_allclose([p,lo,hi],0,atol=1e-15)
    assert m==n
    with pytest.raises(ValueError):cluster_intervals(f.iloc[:-1],d[:-1],77,100)

