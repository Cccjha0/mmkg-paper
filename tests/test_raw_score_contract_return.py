"""Returned-report checks must reject missing coverage, altered sources and false passes."""
import copy
import hashlib
import json

import pytest

from scripts.review_paper_a_raw_score_contract import ROOT, RETURNED, match_source_bytes, verify


def test_returned_report_matches_bound_splits_and_checkpoints():
    result = verify()
    assert result['score_rows'] == 474732
    assert result['checkpoints'] == 18 and result['cells'] == 72
    assert not result['historical_raw_bitwise_equality_established']


@pytest.mark.parametrize('change', [
    'duplicate_cell', 'missing_cell', 'wrong_rows', 'wrong_order', 'wrong_role',
    'hidden_nonfinite', 'negative_count', 'wrong_checkpoint', 'wrong_code', 'new_selection',
])
def test_false_pass_is_rejected(change):
    report = json.loads((ROOT / RETURNED).read_text())
    if change == 'duplicate_cell':
        report['cells'][1] = copy.deepcopy(report['cells'][0])
    elif change == 'missing_cell':
        report['cells'].pop()
    elif change == 'wrong_rows':
        report['cells'][0]['rows'] -= 1
    elif change == 'wrong_order':
        report['cells'][0]['triple_order_sha256'] = '0' * 64
    elif change == 'wrong_role':
        report['cells'][0]['seed'] = 2
    elif change == 'hidden_nonfinite':
        report['cells'][0]['nonfinite_gold_rows'] = 1
    elif change == 'negative_count':
        report['cells'][0]['exceptional_rows'] = -1
    elif change == 'wrong_checkpoint':
        key = next(k for k in report['sources'] if k.endswith('/best.ckpt'))
        report['sources'][key] = '0' * 64
    elif change == 'wrong_code':
        report['sources']['scripts/eval_heterogeneous_complementarity.py'] = '0' * 64
    elif change == 'new_selection':
        report['policy_selection'] = True
    with pytest.raises(ValueError):
        verify(report=report)


def test_newline_exception_does_not_accept_code_changes():
    local = b'x = 1\ny = 2\n'
    server = local.replace(b'\n', b'\r\n')
    digest = hashlib.sha256(server).hexdigest()
    assert match_source_bytes(local, digest, True) == 'newline_only_to_CRLF'
    with pytest.raises(ValueError):
        match_source_bytes(local, digest, False)
    with pytest.raises(ValueError):
        match_source_bytes(local.replace(b'x = 1', b'x = 3'), digest, True)
