"""The evidence reviewer must reject silent seed deletion and crossed DEV folds."""
import itertools
import json
import zipfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_paper_a_dynasemble_review import BASES, SEEDS, LRS, EPOCHS, selected_cv, check_split
from scripts.export_paper_a_dynasemble_cache_evidence import export, sha


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1]/'tmp'
    path = root/('dyna_review_test_'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


def records():
    return pd.DataFrame([dict(base_seed=b, selector_seed=s, fold=f, learning_rate=lr,
        epoch=e, variant='R3_softplus', mrr=.35, n=(f+1)*2)
        for b, s, f, lr, e in itertools.product(BASES, SEEDS, range(3), LRS, EPOCHS)])


def test_cv_requires_every_seed_even_when_coverage_equal_across_candidates():
    frame = records()
    with pytest.raises(ValueError, match='matrix'):
        selected_cv(frame[frame.selector_seed != 37])


def test_cv_rejects_duplicate_replacing_missing_observation():
    frame = records()
    frame.iloc[0] = frame.iloc[1]
    with pytest.raises(ValueError, match='matrix'):
        selected_cv(frame)


def test_cv_uses_observation_weighting_and_declared_tie_break():
    frame = records()
    winner, _ = selected_cv(frame)
    assert winner['epochs'] == 1 and winner['learning_rate'] == 1e-5
    mask = (frame.learning_rate == 5e-5) & (frame.epoch == 3)
    frame.loc[mask, 'mrr'] = frame.loc[mask, 'fold'].map({0: .1, 1: .2, 2: .6})
    winner, _ = selected_cv(frame)
    assert winner['epochs'] == 3 and winner['learning_rate'] == 5e-5
    assert winner['heldout_dev_mrr'] == pytest.approx((.1+2*.2+3*.6)/6)


@pytest.mark.parametrize('corruption', ['fit_overlap', 'head_tail_crossing'])
def test_split_rejects_leakage_and_direction_mismatch(corruption):
    folds = np.array([0, 1, 2, 0, 1, 2])
    signature = dict(fit_ids=[1, 2, 4, 5], validation_ids=[0, 3, 6, 9])
    check_split(signature, 6, folds, 0)
    if corruption == 'fit_overlap':
        signature['fit_ids'][0] = 0
    else:
        signature['validation_ids'][-1] = 10
    with pytest.raises(ValueError, match='DEV'):
        check_split(signature, 6, folds, 0)


@pytest.mark.parametrize('tamper', [False, True])
def test_small_cache_export_preserves_bound_bytes_and_rejects_tampering(workspace, tamper):
    tmp_path = workspace
    root = tmp_path/'run'
    for pair in ('mkgw_mhyper_native', 'mkgw_mhyper_adamf', 'db15k_mhyper_native', 'db15k_mhyper_adamf'):
        lock = {'cache_manifests': {}}
        (root/pair/'test').mkdir(parents=True)
        for split, bs in itertools.product(('dev', 'test'), BASES):
            folder = root/pair/'cache'/f'{split}_seed{bs}'
            folder.mkdir(parents=True)
            for name in ('queries', 'features'):
                np.save(folder/(name+'.npy'), np.arange(8).reshape(2, 4))
            manifest = {'array_sha256': {n: sha(folder/(n+'.npy')) for n in ('queries', 'features')}}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            bound = sha(folder/'manifest.json')
            if split == 'dev':
                lock['cache_manifests'][str(bs)] = bound
            else:
                (root/pair/'test'/f'R3_softplus_b{bs}_s11.json').write_text(json.dumps({'identity': {'cache_manifest_sha256': bound}}))
            (folder/'a.npy').write_bytes(b'large candidate placeholder: must not enter small bundle')
        (root/pair/'dev_lock.json').write_text(json.dumps(lock))
    if tamper:
        (folder/'features.npy').write_bytes(b'changed after manifest')
        with pytest.raises(ValueError, match='array changed'):
            export(root)
    else:
        export(root)
        with zipfile.ZipFile(tmp_path/'run_cache_evidence.zip') as z:
            assert len(z.namelist()) == 72
            assert all(name.endswith(('/manifest.json', '/queries.npy', '/features.npy')) for name in z.namelist())
            for name in z.namelist():
                assert z.read(name) == (root/name).read_bytes()
