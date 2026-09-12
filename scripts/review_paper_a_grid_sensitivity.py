"""Independently audit B09 returned weights/ranks; no model scoring or selection."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.grid_sensitivity import METHODS, RESOLUTIONS, actions
from scripts.analyze_paper_a_conservative_radius import PAIRS, KEY, sha, require
from scripts.paper_a_grid_sensitivity import CODE, PROTOCOL, TRIPLE, REPLAY, read_frame, weight_diagnostics
from scripts.review_paper_a_raw_score_contract import match_source_bytes

PREFIX = 'outputs/paper_a_safe_correction/grid_sensitivity_review_v1'
RETURN = ROOT / (PREFIX + '_return')
OUT = ROOT / 'outputs/paper_a_safe_correction/grid_sensitivity_return_review_v1'


def returned_path(rel, folder=RETURN):
    relative = PurePosixPath(rel)
    require('..' not in relative.parts and relative.is_relative_to(PREFIX), 'Invalid returned path')
    return folder / relative.relative_to(PREFIX).as_posix()


def equal_frame(actual, expected, keys):
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), 'Duplicate comparison key')
    a, b = [f.set_index(keys).sort_index() for f in (actual, expected)]
    require(set(a.columns) == set(b.columns), 'Different result columns')
    pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False, rtol=0, atol=1e-12)


def audit_actions(frame):
    weights, fallback = actions(frame.g, frame.p, frame.nonfinite, frame.anchor, frame.beta, frame.tau)
    np.testing.assert_array_equal(frame.fallback, fallback)
    for method, alpha in weights.items():
        np.testing.assert_allclose(frame['alpha_' + method], alpha, rtol=0, atol=1e-12)
        if 'execution_alpha_' + method in frame:
            np.testing.assert_array_equal(frame['execution_alpha_' + method], np.asarray(alpha, dtype=np.float32).astype(float))
    return weights


def numeric_summary(frame):
    """Recompute aggregates directly from integer ranks, independently of report()."""
    ref = 1 / frame.rank_Global.to_numpy()
    result = []
    for method in METHODS:
        rr = 1 / frame['rank_' + method].to_numpy()
        delta = rr - ref
        bad, good = delta < -1e-12, delta > 1e-12
        alpha = frame['alpha_' + method].to_numpy()
        result.append(dict(method=method, n=len(frame), mrr=rr.mean(), global_mrr=ref.mean(), delta_mrr=delta.mean(),
                           harm_rate=bad.mean(), benefit_rate=good.mean(), rank_change_rate=(bad | good).mean(),
                           conditional_loss=-delta[bad].mean() if bad.any() else np.nan,
                           mean_loss=np.maximum(-delta, 0).mean(), mean_gain=np.maximum(delta, 0).mean(),
                           action_rate=(np.abs(alpha-frame.anchor)>1e-12).mean(),
                           mean_abs_alpha_deviation=np.abs(alpha-frame.anchor).mean()))
    return result


def check_cell(frame, receipt, prepared, canonical):
    require(len(frame) == receipt['rows'] == len(canonical), 'Cell row count mismatch')
    require(set(frame.seed) == {receipt['seed']} and set(frame.direction) == {receipt['direction']}, 'Wrong cell role')
    require(not frame.duplicated(KEY).any(), 'Duplicate query in cell')
    np.testing.assert_array_equal(frame[TRIPLE], canonical)
    ordered = frame[TRIPLE].to_numpy().tolist()
    require(hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).hexdigest() == receipt['triple_order_sha256'], 'Canonical order mismatch')
    select = prepared[(prepared.seed == receipt['seed']) & (prepared.direction == receipt['direction'])]
    equal_frame(frame[prepared.columns], select, KEY)
    audit_actions(frame)
    ranks = frame[['rank_' + m for m in ('Primary', 'Secondary', *METHODS)]].to_numpy()
    require(np.isfinite(ranks).all() and (ranks == np.rint(ranks)).all()
            and (ranks >= 1).all() and (ranks <= receipt['num_entities']).all(), 'Invalid returned rank')
    mismatch = {m: int((frame['rank_' + m] != frame['historical_rank_' + m]).sum()) for m in REPLAY}
    require(mismatch == receipt['historical_rank_mismatch_counts'], 'False historical replay counter')
    require(not any(mismatch.values()), 'Historical rank replay requires review')
    return sum(mismatch.values())


def verify(folder=RETURN):
    sources, code_checks, checks = {}, [], []
    def bind(path, expected=None):
        path = Path(path)
        value = sha(path)
        if expected is not None:
            require(value == expected, 'Hash mismatch: ' + str(path))
        sources[path.relative_to(ROOT).as_posix()] = value
        return path
    def obj(path, expected=None):
        return json.loads(bind(path, expected).read_text(encoding='utf-8'))
    def csv(path, expected=None):
        return read_frame(bind(path, expected))
    plan = obj(folder/'plan.json')
    plan_sha = sha(folder/'plan.json')
    local = obj(ROOT/PREFIX/'local_preparation_audit.json')
    require(set(plan['sources']) == set(local['sources']), 'Source inventory changed from frozen preparation')
    require(plan['current_git_commit'] == 'a2c1728e3c4b20f4a4daf52d6be1a4e63a528663', 'Unexpected server revision')
    require(plan['methods'] == list(METHODS) and plan['resolutions'] == RESOLUTIONS, 'Changed fixed configuration')
    require(plan['status'] == 'prepared_before_new_scoring' and plan['selector_fits'] == 20, 'Wrong plan stage')
    require(plan['test_rows_opened'] is False and plan['new_grid_selection'] is False and plan['original_grid_retained'] is True, 'Unexpected selection')
    require(len(plan['fold_fits']) == 20 and all(f['oof_replayed'] for f in plan['fold_fits']), 'OOF fit replay failed')
    for rel, value in plan['sources'].items():
        path = bind(ROOT/rel)
        mode = match_source_bytes(path.read_bytes(), value, allow_newlines=rel.endswith('.py'))
        code_checks.append(dict(path=rel, match=mode, local_sha256=sha(path), server_sha256=value))
    baseline = obj(ROOT/'paper_a_draft/rerun_source_manifest.json')['sources']
    cp_manifest = obj(ROOT/'paper_a_draft/data_checkpoint_source_manifest.json')['sources']
    ledger = csv(ROOT/'outputs/paper_a_safe_correction/data_checkpoint_review_v1/checkpoint_diagnostics.csv')
    require(set(plan['checkpoints']) == set(ledger.run) and len(ledger) == 18, 'Missing checkpoint')
    for _, entry in ledger.iterrows():
        actual = plan['checkpoints'][entry.run]
        expected = dict(dataset=entry.dataset, model=entry.model, seed=int(entry.seed),
                        checkpoint_sha256=entry.checkpoint_sha256, config_sha256=cp_manifest[entry.run+'/config_merged.json'])
        require(actual == expected and actual['checkpoint_sha256'] == cp_manifest[entry.run+'/best.ckpt'], 'Checkpoint identity mismatch')
    canonical = {}
    for dataset in ('mkg_w', 'db15k'):
        rel = f'outputs/paper_a_safe_correction/data_checkpoint_review_v1/{dataset}_split_rows.csv.gz'
        rows = csv(ROOT/rel, cp_manifest[rel])
        for phase, name in [('dev','valid'), ('test','test')]:
            sub = rows[rows.canonical_split == name].sort_values('canonical_row_index')
            canonical[dataset, phase] = sub[['head','relation','tail']].to_numpy(dtype=int)
    summary, by_cell, effects, diagnostics, receipt_sources, times = [], [], [], [], {}, {'dev':[], 'test':[]}
    review = obj(folder/'review.json')
    require(review['plan_sha256'] == plan_sha and review['historical_grid_replay_exact'] is True, 'Unverified server report')
    require(review['status'] == 'grid_scoring_review_ready' and review['selector_fits'] == 20, 'Unexpected server status')
    require(review['new_grid_selection'] is False and review['current_paper_results_replaced'] is False and review['new_confidence_intervals'] is False, 'Unplanned selection or inference')
    for phase in ('dev', 'test'):
        complete = obj(folder/(phase+'_complete.json'))
        require(complete == review['phase_checks'][0 if phase=='dev' else 1], 'Changed completion record')
        expected_receipts = {f'{PREFIX}/cells/{phase}_{pair}_s{s}_{d}.json' for pair in PAIRS for s in (1,2,3) for d in ('head','tail')}
        require(set(complete['receipts']) == expected_receipts and complete['cells'] == 24, 'Incomplete cell inventory')
        require(complete['phase'] == phase and complete['plan_sha256'] == plan_sha and complete['new_grid_selection'] is False
                and complete['historical_rank_mismatches'] == 0 and complete['status'] == 'all_cells_scored', 'Invalid phase completion')
        for pair in PAIRS:
            lock_rel = f'outputs/paper_a_safe_correction/information_boundary_v2/{pair}/dev_lock/anchored_dev_lock.json'
            lock = obj(ROOT/lock_rel, baseline[lock_rel])
            require(plan['full_locks'][pair] == {k:lock[k] for k in plan['full_locks'][pair]}, 'Changed full-DEV lock')
            prepared = csv(folder/'prepared'/f'{pair}_{phase}.csv.gz')
            require(not prepared.duplicated(KEY).any() and (prepared.groupby(TRIPLE).size()==6).all(), 'Incomplete action plan')
            if phase == 'dev':
                info = plan['dev_plans'][pair]
                bind(returned_path(info['path'], folder), info['sha256'])
                require(len(prepared) == info['rows'], 'Incomplete DEV plan')
                old_rel = f'outputs/paper_a_safe_correction/information_boundary_v2/{pair}/p3_ablation/dev_p3_selected_query_rows.csv'
                old = csv(ROOT/old_rel, baseline[old_rel]).set_index(KEY).sort_index()
                b_rel = f'outputs/paper_a_safe_correction/information_boundary_v2/{pair}/baseline_crossfit/dev_crossfit_query_rows.csv'
                base = csv(ROOT/b_rel, baseline[b_rel]).set_index(KEY).sort_index()
                expected_rr = {'Primary':base.rr_a, 'Secondary':base.rr_b, 'Global':old.rr_global_crossfit,
                               'ADC_grid_005':old.rr_method, 'Query-soft_grid_005':old.rr_query_soft}
                aligned = prepared.set_index(KEY).sort_index()
                np.testing.assert_array_equal(aligned.fold, old.fold)
                np.testing.assert_allclose(aligned.anchor, old.alpha0, rtol=0, atol=1e-12)
                np.testing.assert_allclose(aligned.alpha_ADC_grid_005, old.alpha_applied, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(aligned.fallback, old.fallback.astype(bool))
                settings_rel = f'outputs/paper_a_safe_correction/information_boundary_v2/{pair}/p3_ablation/dev_p3_selected_by_fold.csv'
                settings = csv(ROOT/settings_rel, baseline[settings_rel]).set_index('fold')
                for fold, part in aligned.groupby('fold'):
                    require(set(part.beta)=={settings.loc[fold].selected_beta} and set(part.tau)=={settings.loc[fold].selected_confidence_threshold}, 'Changed fold parameters')
            else:
                old_rel = f'outputs/paper_a_safe_correction/information_boundary_v2/{pair}/test_anchored/test_locked_query_rows.csv'
                old = csv(ROOT/old_rel, baseline[old_rel]).set_index(KEY).sort_index()
                aligned = prepared.set_index(KEY).sort_index()
                np.testing.assert_allclose(aligned.g, old.anchored_decision, rtol=0, atol=1e-12)
                np.testing.assert_allclose(aligned.p, old.anchored_probability_a, rtol=0, atol=1e-12)
                require(set(aligned.anchor)=={lock['alpha0']} and set(aligned.beta)=={lock['beta']} and set(aligned.tau)=={lock['confidence_threshold']}, 'Changed TEST parameters')
                expected_rr = {'Primary':old.rr_a, 'Secondary':old.rr_b, 'Global':old.rr_global,
                               'ADC_grid_005':old.rr_anchored_locked, 'Query-soft_grid_005':old.rr_query_soft_locked}
                info = dict(path=old_rel, sha256=baseline[old_rel], prepared_path=f'{PREFIX}/prepared/{pair}_test.csv.gz',
                            prepared_sha256=sha(folder/'prepared'/f'{pair}_test.csv.gz'))
            require(aligned.index.equals(old.index), 'Original query coverage mismatch')
            for method, rr in expected_rr.items():
                np.testing.assert_allclose(1/aligned['historical_rank_'+method], rr, rtol=0, atol=1e-12)
            parts = []
            for seed in (1,2,3):
                for direction in ('head','tail'):
                    rel = f'{PREFIX}/cells/{phase}_{pair}_s{seed}_{direction}.json'
                    receipt = obj(returned_path(rel, folder), complete['receipts'][rel])
                    receipt_sources[rel] = complete['receipts'][rel]
                    require((receipt['pair'], receipt['phase'], receipt['seed'], receipt['direction']) == (pair,phase,seed,direction), 'Wrong receipt role')
                    require(receipt['input'] == info and receipt['plan_sha256'] == plan_sha and receipt['methods'] == list(METHODS), 'Changed cell plan/input')
                    require(receipt['rows_path'] == rel[:-5]+'.csv.gz', 'Wrong output path')
                    require(receipt['candidates_exported'] is False and receipt['new_grid_selection'] is False, 'Unexpected cell behavior')
                    expected_batches = []
                    for name in (lock['expert_a_name'], lock['expert_b_name']):
                        run = next(r for r,c in plan['checkpoints'].items() if (c['dataset'],c['model'],c['seed'])==(lock['dataset'],name,seed))
                        cfg = obj(ROOT/run/'config_merged.json')
                        expected_batches.append(cfg['evaluation']['query_batch_size'])
                        require(receipt['num_entities'] == cfg['_dataset_manifest']['counts']['entities'], 'Wrong candidate count')
                    require(receipt['scorer_query_batches']==expected_batches and receipt['outer_batch']==max(expected_batches), 'Batch contract changed')
                    require(receipt['runtime']['execution_weight_dtype']=='float32', 'Execution dtype changed')
                    frame = csv(returned_path(receipt['rows_path'], folder), receipt['rows_sha256'])
                    receipt_sources[receipt['rows_path']] = receipt['rows_sha256']
                    check_cell(frame, receipt, prepared, canonical[lock['dataset'],phase])
                    by_cell.extend(dict(pair=pair, split=phase, seed=seed, direction=direction, **r) for r in numeric_summary(frame))
                    times[phase].append(datetime.fromisoformat(receipt['completed_at_utc']))
                    parts.append(frame)
            frame = pd.concat(parts, ignore_index=True)
            summary.extend(dict(pair=pair, split=phase, **r) for r in numeric_summary(frame))
            diagnostics.extend(dict(pair=pair, split=phase, **r) for r in weight_diagnostics(frame))
            comparisons = [(f'{p}_{r}','Global') for p in ('ADC','Query-soft') for r in RESOLUTIONS]
            comparisons += [(f'ADC_{r}',f'Query-soft_{r}') for r in RESOLUTIONS]
            comparisons += [(f'{p}_{r}',f'{p}_grid_005') for p in ('ADC','Query-soft') for r in ('grid_001','continuous')]
            for left,right in comparisons:
                delta = 1/frame['rank_'+left] - 1/frame['rank_'+right]
                effects.append(dict(pair=pair,split=phase,left=left,right=right,n=len(frame),delta_mrr=delta.mean(),
                                    **{f'seed_{s}':delta[frame.seed==s].mean() for s in (1,2,3)},
                                    **{f'direction_{d}':delta[frame.direction==d].mean() for d in ('head','tail')}))
            checks.append(dict(pair=pair, split=phase, observations=len(frame), cells=6, original_grid_rank_mismatches=0,
                               original_locks_and_history_verified=True, weights_and_execution_replayed=True))
    require(receipt_sources == review['sources'], 'Report omitted/changed a source')
    require(datetime.fromisoformat(plan['prepared_at_utc']) < min(times['dev'])
            and max(times['dev']) < min(times['test']), 'Stage timestamp order mismatch')
    recomputed = {}
    specifications = [('summary',summary,['pair','split','method'],56),
                      ('by_seed_direction',by_cell,['pair','split','seed','direction','method'],336),
                      ('paired_effects',effects,['pair','split','left','right'],104),
                      ('weight_diagnostics',diagnostics,['pair','split','method'],56)]
    require(set(review['outputs']) == {name+'.csv' for name,_,_,_ in specifications}, 'Unexpected summary inventory')
    for name, records, keys, count in specifications:
        frame = pd.DataFrame(records)
        require(len(frame)==count, 'Incomplete derived output')
        saved = csv(folder/(name+'.csv'), review['outputs'][name+'.csv'])
        equal_frame(frame,saved,keys)
        recomputed[name] = frame
    return dict(status='grid_sensitivity_return_checks_passed', failures=[], sources=sources,
                cells=48, observations=sum(c['observations'] for c in checks), checkpoints=18, checks=checks,
                source_byte_checks=code_checks, original_grid_rank_mismatches=0,
                server_plan_sha256=plan_sha, server_commit=plan['current_git_commit'], server_selector_runtime=plan['runtime'],
                recorded_stage_times={'prepared':plan['prepared_at_utc'], 'dev_last':max(times['dev']).isoformat(), 'test_first':min(times['test']).isoformat(), 'test_last':max(times['test']).isoformat()},
                history_note='Recorded stage ordering and fixed code verified; prior TEST exposure remains. No restored holdout independence.',
                new_grid_selection=False, historical_results_replaced=False, new_confidence_intervals=False,
                local_scorer_runs=0, b09_status='closed_for_fixed_conditional_grid_sensitivity'), recomputed


def main():
    audit, outputs = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in ('scripts/review_paper_a_grid_sensitivity.py','tests/test_grid_sensitivity_return.py','scripts/review_paper_a_raw_score_contract.py'):
        audit['sources'][rel] = sha(ROOT/rel)
    audit['outputs'] = {}
    for name, frame in outputs.items():
        path = OUT/(name+'.csv')
        frame.to_csv(path,index=False)
        audit['outputs'][name+'.csv'] = sha(path)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources','source_byte_checks','checks')},indent=2))


if __name__ == '__main__':
    main()
