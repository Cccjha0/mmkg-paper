"""B02: small, single-thread CPU cache review; lock DEV before opening TEST."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.feature_redundancy import FIELDS9, FIELDS13, dependency_diagnostics, augmented_standardized_map
from router.matched_actions import Action, ALPHAS
from scripts.analyze_paper_a_matched_alternatives import (
    Inputs, check_frame, apply, evaluate, summaries, check_repeat_weights, write_json,
)
from scripts.analyze_paper_a_conservative_radius import INPUT, PAIRS, GRID_COLUMNS, KEY, sha, close, require
from scripts.ablate_anchored_dynamic import feature_matrix, fit_geometry_model, model_outputs
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha

OUT = ROOT/'outputs/paper_a_safe_correction/feature_dimension_review_v1'
PROTOCOL = ROOT/'docs/protocols/paper_a_feature_dimension_review.md'
CODE = ('router/feature_redundancy.py', 'router/matched_actions.py',
        'scripts/analyze_paper_a_feature_dimension.py', 'scripts/analyze_paper_a_matched_alternatives.py',
        'scripts/analyze_paper_a_conservative_radius.py', 'scripts/ablate_anchored_dynamic.py',
        'scripts/crossfit_anchored_dynamic.py', 'scripts/crossfit_heterogeneous_dev_policies.py',
        'router/constants.py', 'router/query_geometry.py', 'router/score_combination.py',
        'scripts/eval_heterogeneous_complementarity.py')
DIMENSIONS = {9: FIELDS9, 13: FIELDS13}


def signals(frame, model, fields):
    x = frame[list(fields)].to_numpy(dtype=float, copy=True)
    invalid = ~np.isfinite(x).all(axis=1)
    x[~np.isfinite(x)] = np.nan
    return (*model_outputs(model, x), invalid)


def dimension_diagnostics(pair, scope, frame, models):
    m = augmented_standardized_map(models[9][-2], models[13][-2])
    eigen = np.linalg.eigvalsh(m.T @ m)
    require(eigen.min() > 0, 'Augmented map must have full column rank')
    return dict(pair=pair, scope=scope, **dependency_diagnostics(frame[list(FIELDS13)].to_numpy(), models[13]),
                min_augmented_gram_eigenvalue=float(eigen.min()), max_augmented_gram_eigenvalue=float(eigen.max()),
                augmented_gram_condition=float(eigen.max()/eigen.min()),
                n_iter_9=int(models[9][-1].n_iter_[0]), n_iter_13=int(models[13][-1].n_iter_[0]))


def check_original(frame, sig, grid, anchor, action):
    close(sig[0], frame.anchored_decision, '13D decision replay failed')
    close(sig[1], frame.anchored_probability_a, '13D probability replay failed')
    rr, alpha, _ = apply(grid, sig, anchor, action)
    close(rr, frame.rr_anchored_locked, '13D original ADC RR replay failed')
    close(alpha, frame.alpha_anchored_locked, '13D original ADC weights replay failed')


def write_frames(frames):
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT/name, index=False)
    return {name: sha(OUT/name) for name in frames}


def paired_rows(pair, scope, frame, result):
    rows = []
    for policy in ('ADC', 'Query-soft'):
        delta = result[f'{policy}-13'][0] - result[f'{policy}-9'][0]
        rows.append(dict(pair=pair, scope=scope, policy=policy, n=len(frame), delta_13_minus_9=float(delta.mean()),
                         **{f'seed_{s}':float(delta[frame.seed.to_numpy()==s].mean()) for s in (1,2,3)},
                         **{f'direction_{d}':float(delta[frame.direction.to_numpy()==d].mean()) for d in ('head','tail')}))
    return rows


def lock_dev():
    require(not (OUT/'dev_locks.json').exists(), 'Existing DEV lock; use a new review version, do not overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'query_rows').mkdir(exist_ok=True)
    inputs = Inputs('DEV')
    for rel in (*CODE, PROTOCOL.relative_to(ROOT).as_posix()):
        inputs.bind(ROOT/rel, False)
    choices, candidates, diagnostics, pooled, cells, paired, fits = [], [], [], [], [], [], []
    locks, artifacts = {}, []
    for pair in PAIRS:
        folder = INPUT/pair
        full = inputs.frame(folder/'dev_lock/dev_locked_query_rows.csv')
        check_frame(full, 'dev')
        old_lock = inputs.obj(folder/'dev_lock/anchored_dev_lock.json')
        old_path = folder/'dev_lock'/old_lock['model_file']
        require(sha(old_path)==old_lock['model_sha256'], 'Historical model hash mismatch')
        originals = inputs.frame(folder/'p3_ablation/dev_p3_selected_query_rows.csv').set_index('query_id')
        baseline = inputs.frame(folder/'baseline_crossfit/dev_crossfit_query_rows.csv')
        check_frame(baseline, 'dev')
        rows = baseline.to_dict('records')
        assignment, _ = assign_grouped_folds(rows, 5, 20260901)
        folds = np.array([assignment[triple_key(r)] for r in rows])
        oof = {name: (np.full(len(rows), np.nan), np.full(len(rows), np.nan))
               for name in ('ADC-9','ADC-13','Query-soft-9','Query-soft-13')}
        refs, anchors = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        pair_locks = {}
        for fold in (None, 0, 1, 2, 3, 4):
            scope = 'full_dev_selection' if fold is None else f'fold_{fold+1}'
            train = full if fold is None else baseline.loc[folds != fold].reset_index(drop=True)
            held = None if fold is None else baseline.loc[folds == fold].reset_index(drop=True)
            records = train.to_dict('records')
            anchor = old_lock['alpha0'] if fold is None else best_alpha(records, tuple(ALPHAS))[0]
            grid = train[GRID_COLUMNS].to_numpy()
            models = {}
            for dim, fields in DIMENSIONS.items():
                state = 20260902 if fold is None else 20260902+fold*10+2
                model = inputs.model(old_path) if fold is None and dim==13 else fit_geometry_model(records, fields=fields, random_state=state)[0]
                models[dim] = model
                sig = signals(train, model, fields)
                action, selected, all_candidates, unique = evaluate(grid, sig, anchor, 'ADC+Global')
                choices.append(dict(pair=pair, scope=scope, dimension=dim, anchor=anchor,
                                    **asdict(action), selection_mrr=selected['mrr'], distinct_action_vectors=unique))
                candidates.extend(dict(pair=pair, scope=scope, dimension=dim, **r) for r in all_candidates)
                fits.append(dict(pair=pair, scope=scope, dimension=dim, random_state=state,
                                 fit_rows=int((train.rr_a != train.rr_b).sum()), n_iter=int(model[-1].n_iter_[0]),
                                 reused=fold is None and dim==13))
                if fold is None:
                    if dim==13:
                        check_original(full, sig, grid, anchor, Action('ADC+Global',old_lock['beta'],old_lock['confidence_threshold']))
                        path = old_path
                    else:
                        path = OUT/'models'/f'{pair}_9.pkl'
                        with path.open('wb') as handle: pickle.dump(model, handle)
                        artifacts.append(path)
                    pair_locks[str(dim)] = dict(action=asdict(action), model_file=path.relative_to(ROOT).as_posix(), model_sha256=sha(path))
                else:
                    hs = signals(held, model, fields)
                    hg = held[GRID_COLUMNS].to_numpy()
                    for name, setting in [('ADC',action),('Query-soft',Action('Shrink',1.,0.))]:
                        rr, alpha, _ = apply(hg, hs, anchor, setting)
                        oof[f'{name}-{dim}'][0][folds==fold] = rr
                        oof[f'{name}-{dim}'][1][folds==fold] = alpha
                    prev = originals.loc[held.query_id]
                    if dim==13:
                        require((prev.fold==fold+1).all(), 'Historical fold identities mismatch')
                        close(oof['ADC-13'][0][folds==fold], prev.rr_method, '13D OOF ADC replay failed')
                    reference = hg[:,int(round(anchor*20))]
                    close(reference, prev.rr_global_crossfit, 'OOF anchor replay failed')
                    refs[folds==fold], anchors[folds==fold] = reference, anchor
            diagnostics.append(dimension_diagnostics(pair,scope,train,models))
        require(all(np.isfinite(x).all() for values in oof.values() for x in values), 'Incomplete OOF coverage')
        check_repeat_weights_by_fold(baseline, folds, oof)
        summary, by_cell = summaries(pair, baseline, oof, refs, anchors)
        pooled.extend(summary); cells.extend(by_cell)
        paired.extend(paired_rows(pair,'DEV_OOF',baseline,oof))
        output = baseline[KEY].copy(); output['fold'] = folds+1; output['rr_global']=refs; output['anchor']=anchors
        for name,(rr,alpha) in oof.items(): output['rr_'+name]=rr; output['alpha_'+name]=alpha
        path = OUT/'query_rows'/f'{pair}_dev.csv.gz'
        output.to_csv(path,index=False,compression={'method':'gzip','mtime':0}); artifacts.append(path)
        locks[pair] = dict(anchor=old_lock['alpha0'], dimensions=pair_locks)
        print('[DEV LOCKED] '+pair,flush=True)
    outputs = write_frames({'dev_choices.csv':choices,'dev_candidates.csv':candidates,'dev_oof_summary.csv':pooled,
                           'dev_seed_direction.csv':cells,'dev_paired.csv':paired,'fit_diagnostics.csv':fits,
                           'dev_dependency_diagnostics.csv':diagnostics})
    outputs.update({p.relative_to(OUT).as_posix():sha(p) for p in artifacts})
    write_json(OUT/'dev_locks.json',dict(version='feature_dimension_review_v1',status='all_dev_locks_complete',
               pairs=locks,sources=inputs.sources,outputs=outputs,test_used_for_selection=False,model_fits=44,full_models_reused=4))


def check_repeat_weights_by_fold(frame, folds, result):
    for fold in np.unique(folds):
        mask = folds==fold
        check_repeat_weights(frame.loc[mask], {k:(rr[mask],a[mask]) for k,(rr,a) in result.items()})


def apply_test():
    require(not (OUT/'test_audit.json').exists(), 'TEST results already exist; do not overwrite')
    lock_path = OUT/'dev_locks.json'
    lock = json.loads(lock_path.read_text())
    require(lock['status']=='all_dev_locks_complete' and set(lock['pairs'])==set(PAIRS),'Incomplete DEV locks')
    for rel,value in lock['sources'].items(): require(sha(ROOT/rel)==value,'DEV source changed: '+rel)
    for name,value in lock['outputs'].items(): require(sha(OUT/name)==value,'DEV output changed: '+name)
    inputs = Inputs('TEST'); inputs.bind(lock_path,False)
    pooled,cells,paired,diagnostics,checks,artifacts = [],[],[],[],[],[]
    for pair in PAIRS:
        setting = lock['pairs'][pair]; anchor = setting['anchor']
        frame = inputs.frame(INPUT/pair/'test_anchored/test_locked_query_rows.csv'); check_frame(frame,'test')
        grid,ref = frame[GRID_COLUMNS].to_numpy(),frame.rr_global.to_numpy()
        models,result = {},{}
        for dim,fields in DIMENSIONS.items():
            info = setting['dimensions'][str(dim)]; path=ROOT/info['model_file']
            require(sha(path)==info['model_sha256'],'Frozen model mismatch')
            inputs.bind(path,dim==13)
            with path.open('rb') as handle: model=pickle.load(handle)
            models[dim]=model; sig=signals(frame,model,fields)
            action=Action(**info['action'])
            if dim==13: check_original(frame,sig,grid,anchor,action)
            for name,act in [('ADC',action),('Query-soft',Action('Shrink',1.,0.))]:
                rr,alpha,_=apply(grid,sig,anchor,act); result[f'{name}-{dim}']=rr,alpha
        check_repeat_weights(frame,result)
        summary,by_cell=summaries(pair,frame,result,ref,anchor); pooled.extend(summary);cells.extend(by_cell)
        paired.extend(paired_rows(pair,'TEST',frame,result))
        diagnostics.append(dimension_diagnostics(pair,'TEST',frame,models))
        output=frame[KEY].copy();output['rr_global']=ref
        for name,(rr,alpha) in result.items():output['rr_'+name]=rr;output['alpha_'+name]=alpha
        path=OUT/'query_rows'/f'{pair}_test.csv.gz'
        output.to_csv(path,index=False,compression={'method':'gzip','mtime':0});artifacts.append(path)
        checks.append(dict(pair=pair,n=len(frame),original_13d_replayed=True,gold_repeat_grid_weights_invariant=True))
        print('[TEST APPLIED] '+pair,flush=True)
    outputs=write_frames({'test_summary.csv':pooled,'test_seed_direction.csv':cells,'test_paired.csv':paired,'test_dependency_diagnostics.csv':diagnostics})
    outputs.update({p.relative_to(OUT).as_posix():sha(p) for p in artifacts})
    write_json(OUT/'test_audit.json',dict(version='feature_dimension_review_v1',status='feature_dimension_checks_passed',failures=[],
                sources={**lock['sources'],**inputs.sources},outputs=outputs,dev_outputs=lock['outputs'],
                dev_lock_sha256=sha(lock_path),checks=checks,test_used_for_selection=False,
                limitations=['Retrospective, no new blind confirmation or confidence intervals.',
                             'C=1 matched; different representations imply different effective regularization.',
                             'Finite exported features do not certify raw candidate-score finiteness.']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('lock-dev','apply-test'))
    args=parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        (lock_dev if args.phase=='lock-dev' else apply_test)()
