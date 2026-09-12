"""Reject false endpoint-equivalence receipts and test exact action substitution."""
import copy
import json

import numpy as np
import pytest

from scripts.review_paper_a_endpoint_contract import RETURN, verify, check_differences, endpoint_substitution, COUNTS


def test_complete_return_retains_real_inequivalence():
    result,cells,differences,_=verify()
    assert result['cells']==72 and result['totals']['rows']==474732
    assert len(differences)==1293 and len(cells)==72
    assert result['totals']['export_vs_shared_mismatches']==0
    assert not result['normalized_endpoint_equivalence']


@pytest.mark.parametrize('change',['duplicate','missing','count','false_pass','selection','receipt','order','checkpoint'])
def test_altered_return_is_rejected(change):
    report=json.loads((RETURN/'audit.json').read_text())
    if change=='duplicate':report['cells'][1]=copy.deepcopy(report['cells'][0])
    elif change=='missing':report['cells'].pop()
    elif change=='count':report['totals']['normalized_vs_raw_mismatches']=0
    elif change=='false_pass':report['status']='endpoint_equivalence_checks_passed'
    elif change=='selection':report['policy_selection']=True
    elif change=='receipt':report['cells'][0]['differences_file']='../outside.json'
    elif change=='order':report['cells'][0]['triple_order_sha256']='0'*64
    elif change=='checkpoint':report['cells'][0]['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):verify(report=report)


def fixture_rows():
    canonical=[[0,0,1],[1,0,2]]
    rows=[dict(split_index=0,triple=canonical[0],raw_rank=2,normalized_rank=1)]
    totals={k:0 for k in COUNTS};totals.update(rows=2,normalized_vs_raw_mismatches=1,raw_rr_sum=1.,normalized_rr_sum=1.5)
    return rows,dict(totals=totals,examples=copy.deepcopy(rows)),canonical


@pytest.mark.parametrize('change',['duplicate','triple','rank','conservation','example'])
def test_inconsistent_difference_rows_are_rejected(change):
    rows,cell,canonical=fixture_rows()
    check_differences(rows,cell,canonical,3)
    if change=='duplicate':rows+=copy.deepcopy(rows);cell['totals']['normalized_vs_raw_mismatches']=2;cell['examples']=rows
    elif change=='triple':rows[0]['triple']=[2,0,1]
    elif change=='rank':rows[0]['normalized_rank']=0
    elif change=='conservation':cell['totals']['normalized_rr_sum']=1.4
    elif change=='example':cell['examples']=[]
    with pytest.raises(ValueError):check_differences(rows,cell,canonical,3)


def test_substitution_changes_exact_endpoints_only():
    alpha=np.array([0.,np.nextafter(0.,1.),.5,np.nextafter(1.,0.),1.])
    result=endpoint_substitution(np.full(5,.25),alpha,np.full(5,.5),np.full(5,.125))
    np.testing.assert_array_equal(result,[.375,.25,.25,.25,.75])


def test_policy_effect_uses_the_matching_counterfactual_reference():
    raw_global=np.array([.5,.25]); raw_adc=np.array([.5,.3])
    d_a=np.array([.5,.25]);d_b=np.zeros(2)
    new_global=endpoint_substitution(raw_global,[1,1],d_a,d_b)
    new_adc=endpoint_substitution(raw_adc,[1,.5],d_a,d_b)
    assert new_adc[0]-new_global[0]==raw_adc[0]-raw_global[0]
    np.testing.assert_allclose(new_adc-new_global,[0.,-.2],rtol=0,atol=1e-15)
