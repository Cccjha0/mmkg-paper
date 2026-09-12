import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from router.grid_sensitivity import METHODS, actions
from scripts.review_paper_a_grid_sensitivity import check_cell, equal_frame, numeric_summary, returned_path, REPLAY, TRIPLE


def fixture():
    frame = pd.DataFrame(dict(seed=[1,1],direction=['tail','tail'],head_id=[0,0],relation_id=[0,0],tail_id=[1,2],
                              g=[.4,.4],p=[1/(1+np.exp(-.4))]*2,nonfinite=[False,False],anchor=[.55,.55],beta=[.45,.45],tau=[.1,.1]))
    weights, fallback = actions(frame.g,frame.p,frame.nonfinite,frame.anchor,frame.beta,frame.tau)
    frame['fallback'] = fallback
    for method in METHODS:
        frame['alpha_'+method] = weights[method]
    for method in REPLAY:
        frame['historical_rank_'+method] = [1,2]
    prepared = frame.copy()
    for method in ('Primary','Secondary',*METHODS):
        frame['rank_'+method] = [1,2]
    for method in METHODS:
        frame['execution_alpha_'+method] = weights[method].astype(np.float32).astype(float)
    order = frame[TRIPLE].to_numpy()
    receipt = dict(rows=2,seed=1,direction='tail',num_entities=3,
                   triple_order_sha256=hashlib.sha256(json.dumps(order.tolist(),separators=(',',':')).encode()).hexdigest(),
                   historical_rank_mismatch_counts={m:0 for m in REPLAY})
    return frame,receipt,prepared,order


def test_returned_cell_and_zero_global_loss():
    frame,receipt,prepared,order=fixture()
    assert check_cell(frame,receipt,prepared,order)==0
    summary=pd.DataFrame(numeric_summary(frame)).set_index('method')
    assert summary.loc['Global','mean_loss']==0 and summary.loc['Global','delta_mrr']==0


@pytest.mark.parametrize('column,value', [('alpha_ADC_continuous',.9),('rank_ADC_grid_005',3),('rank_ADC_continuous',1.5),('seed',2)])
def test_rejects_changed_action_rank_or_role(column,value):
    frame,receipt,prepared,order=fixture()
    if isinstance(value,float):
        frame[column]=frame[column].astype(float)
    frame.loc[0,column]=value
    with pytest.raises((ValueError,AssertionError)):
        check_cell(frame,receipt,prepared,order)


def test_rejects_incomplete_or_reordered_cells():
    frame,receipt,prepared,order=fixture()
    for altered in (frame.iloc[:1],frame.iloc[::-1]):
        with pytest.raises((ValueError,AssertionError)):
            check_cell(altered,receipt,prepared,order)


def test_rejects_false_summary_even_when_other_columns_agree():
    frame,*_=fixture()
    a=pd.DataFrame(numeric_summary(frame)); b=a.copy(); b.loc[1,'mrr']+=.001
    with pytest.raises(AssertionError):
        equal_frame(a,b,['method'])


@pytest.mark.parametrize('path',['../outside.json','outputs/paper_a_safe_correction/grid_sensitivity_review_v1/../../outside.json','C:/outside.json'])
def test_returned_paths_cannot_escape_bundle(path):
    with pytest.raises(ValueError):
        returned_path(path)
