"""B09 staged sensitivity: prepare on CPU, score on server, return auditable ranks."""
import argparse
import gc
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.grid_sensitivity import METHODS, RESOLUTIONS, actions
from scripts.analyze_paper_a_conservative_radius import INPUT, PAIRS, KEY, GRID_COLUMNS, sha, require, close
from scripts.ablate_anchored_dynamic import FEATURE_GROUPS, feature_matrix, fit_geometry_model, model_outputs
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha

OUT = ROOT / 'outputs/paper_a_safe_correction/grid_sensitivity_review_v1'
PROTOCOL = 'docs/protocols/paper_a_grid_sensitivity_review.md'
CODE = ('router/grid_sensitivity.py', 'scripts/paper_a_grid_sensitivity.py',
        'scripts/run_paper_a_grid_sensitivity.ps1', 'tests/test_grid_sensitivity.py',
        'scripts/eval_heterogeneous_complementarity.py', 'scripts/ablate_anchored_dynamic.py',
        'scripts/crossfit_heterogeneous_dev_policies.py', 'scripts/crossfit_anchored_dynamic.py',
        'scripts/analyze_paper_a_conservative_radius.py', 'router/information_boundary.py',
        'router/score_combination.py', 'router/query_geometry.py')
TRIPLE = ['head_id', 'relation_id', 'tail_id']
REPLAY = ('Primary', 'Secondary', 'Global', 'ADC_grid_005', 'Query-soft_grid_005')


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def frame_write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_bytes(gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
    tmp.replace(path)


def read_frame(path):
    return pd.read_csv(path, float_precision='round_trip')


def check_phase_path(path, phase):
    if phase == 'dev':
        require(not any(p.lower().startswith('test') for p in Path(path).parts), 'DEV preparation cannot read TEST rows')


def checked_frame(path, phase, expected, sources):
    check_phase_path(path, phase)
    rel = path.relative_to(ROOT).as_posix()
    require(expected.get(rel) == sha(path), 'Changed input: ' + rel)
    sources[rel] = sha(path)
    return read_frame(path)


def validate_observations(frame, split):
    require(set(frame['split']) == {split}, 'Wrong split')
    require(not frame.duplicated(KEY).any(), 'Duplicate observation')
    require((frame.groupby(TRIPLE).size() == 6).all(), 'Incomplete seed/direction grouping')
    require(set(zip(frame.seed, frame.direction)) == {(s, d) for s in (1, 2, 3) for d in ('head', 'tail')}, 'Wrong cells')


def planned_frame(frame, g, p, invalid, anchor, beta, tau, history):
    result = frame[KEY].copy().reset_index(drop=True)
    weights, fallback = actions(g, p, invalid, anchor, beta, tau)
    for key, value in [('g', g), ('p', p), ('nonfinite', invalid), ('anchor', anchor),
                       ('beta', beta), ('tau', tau), ('fallback', fallback)]:
        result[key] = value
    for method, weight in weights.items():
        result['alpha_' + method] = weight
    for method, rr in history.items():
        result['historical_rank_' + method] = np.rint(1 / np.asarray(rr)).astype(np.int64)
        close(1 / result['historical_rank_' + method].to_numpy(), rr, 'Invalid historical RR')
    return result


def weight_diagnostics(frame):
    records = []
    # OOF signals differ across folds; repeatability is assessed within a fit.
    groups = ['seed', 'relation_id'] + (['fold'] if 'fold' in frame else [])
    for method in METHODS:
        alpha = frame['alpha_' + method].to_numpy()
        anchor = frame.anchor.to_numpy()
        raw = anchor if method == 'Global' else frame['alpha_' + method.split('_')[0] + '_continuous'].to_numpy()
        spans = []
        for direction, known in [('head', 'tail_id'), ('tail', 'head_id')]:
            f = frame[frame.direction == direction]
            by = f.groupby(groups + [known])['alpha_' + method].agg(['min', 'max'])
            spans.extend((by['max'] - by['min']).tolist())
        records.append(dict(method=method, rows=len(frame), action_rate=float(np.mean(np.abs(alpha-anchor)>1e-12)),
                            quantization_mean_abs_error=float(np.abs(alpha-raw).mean()),
                            quantization_max_abs_error=float(np.abs(alpha-raw).max()),
                            deadzone_suppressed_rate=float(np.mean((np.abs(raw-anchor)>1e-12)&(np.abs(alpha-anchor)<=1e-12))),
                            max_same_observable_query_span=max(spans, default=0.)))
    return records


def verify_plan():
    path = OUT / 'plan.json'
    plan = json.loads(path.read_text())
    require(plan['status'] == 'prepared_before_new_scoring', 'Incomplete plan')
    require(plan['methods'] == list(METHODS) and plan['resolutions'] == RESOLUTIONS, 'Configuration drift')
    for rel, value in plan['sources'].items():
        require(sha(ROOT / rel) == value, 'Plan input/code changed: ' + rel)
    for entry in plan['dev_plans'].values():
        require(sha(ROOT / entry['path']) == entry['sha256'], 'DEV action plan changed')
    return plan


def prepare():
    if (OUT / 'plan.json').exists():
        verify_plan()
        print('Existing immutable plan verified; no refitting.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'paper_a_draft/rerun_source_manifest.json'
    expected = json.loads(manifest_path.read_text())['sources']
    cp_path = ROOT / 'paper_a_draft/data_checkpoint_source_manifest.json'
    cp_expected = json.loads(cp_path.read_text())['sources']
    sources = {p: sha(ROOT / p) for p in (*CODE, PROTOCOL)}
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'ml/training/src').rglob('*.py')})
    for path in (manifest_path, cp_path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    ledger_path = ROOT / 'outputs/paper_a_safe_correction/data_checkpoint_review_v1/checkpoint_diagnostics.csv'
    require(sha(ledger_path) == cp_expected[ledger_path.relative_to(ROOT).as_posix()], 'Checkpoint ledger changed')
    sources[ledger_path.relative_to(ROOT).as_posix()] = sha(ledger_path)
    ledger = read_frame(ledger_path)
    checkpoints = {}
    for _, row in ledger.iterrows():
        config = row['run'] + '/config_merged.json'
        require(sha(ROOT / config) == cp_expected[config], 'Checkpoint config changed')
        sources[config] = sha(ROOT / config)
        checkpoints[row['run']] = dict(dataset=row.dataset, model=row.model, seed=int(row.seed),
                                       checkpoint_sha256=row.checkpoint_sha256, config_sha256=cp_expected[config])
    full_locks, dev_plans, fits, diagnostics = {}, {}, [], []
    for pair, label in PAIRS.items():
        folder = INPUT / pair
        rows_frame = checked_frame(folder / 'baseline_crossfit/dev_crossfit_query_rows.csv', 'dev', expected, sources)
        validate_observations(rows_frame, 'dev')
        previous = checked_frame(folder / 'p3_ablation/dev_p3_selected_query_rows.csv', 'dev', expected, sources).set_index('query_id')
        settings = checked_frame(folder / 'p3_ablation/dev_p3_selected_by_fold.csv', 'dev', expected, sources).set_index('fold')
        lock_path = folder / 'dev_lock/anchored_dev_lock.json'
        rel = lock_path.relative_to(ROOT).as_posix()
        require(sha(lock_path) == expected[rel], 'DEV lock changed')
        sources[rel] = sha(lock_path)
        lock = json.loads(lock_path.read_text())
        full_locks[pair] = {k: lock[k] for k in ('dataset', 'expert_a_name', 'expert_b_name', 'alpha0', 'beta', 'confidence_threshold', 'model_sha256')}
        rows = rows_frame.to_dict('records')
        assignment, _ = assign_grouped_folds(rows, 5, 20260901)
        fold_ids = np.asarray([assignment[triple_key(r)] for r in rows])
        parts = []
        for fold in range(5):
            train = [r for r, f in zip(rows, fold_ids) if f != fold]
            held = rows_frame.loc[fold_ids == fold].reset_index(drop=True)
            anchor = best_alpha(train, tuple(np.arange(21)/20))[0]
            setting = settings.loc[fold + 1]
            close(anchor, setting.alpha0, 'Fold anchor mismatch')
            model, _, _ = fit_geometry_model(train, fields=tuple(FEATURE_GROUPS['full_geometry']), random_state=20260902+10*fold+2)
            x, invalid = feature_matrix(held.to_dict('records'), tuple(FEATURE_GROUPS['full_geometry']))
            g, p = model_outputs(model, x)
            old = previous.loc[held.query_id]
            require((old.fold == fold + 1).all(), 'Grouped fold mismatch')
            anchors = np.full(len(held), anchor)
            current = planned_frame(held, g, p, invalid, anchors, setting.selected_beta, setting.selected_confidence_threshold,
                                    {'Primary': held.rr_a, 'Secondary': held.rr_b, 'Global': old.rr_global_crossfit,
                                     'ADC_grid_005': old.rr_method, 'Query-soft_grid_005': old.rr_query_soft})
            close(current.alpha_ADC_grid_005, old.alpha_applied, 'OOF ADC action replay failed')
            require(np.array_equal(current.fallback, old.fallback.astype(bool)), 'OOF fallback replay failed')
            grid = held[GRID_COLUMNS].to_numpy()
            for method, target in [('ADC', old.rr_method), ('Query-soft', old.rr_query_soft)]:
                idx = np.rint(20 * current['alpha_' + method + '_grid_005']).astype(int)
                close(grid[np.arange(len(held)), idx], target, 'OOF RR replay failed')
            current['fold'] = fold + 1
            parts.append(current)
            fits.append(dict(pair=pair, fold=fold+1, n_iter=int(model[-1].n_iter_[0]), oof_replayed=True))
        result = pd.concat(parts, ignore_index=True)
        path = OUT / 'prepared' / (pair + '_dev.csv.gz')
        frame_write(path, result)
        dev_plans[pair] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), rows=len(result))
        diagnostics.extend(dict(pair=pair, split='dev_oof', **r) for r in weight_diagnostics(result))
        print('[PREPARED] ' + pair, flush=True)
    plan = dict(status='prepared_before_new_scoring', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                methods=list(METHODS), resolutions=RESOLUTIONS,
                sources=sources, dev_plans=dev_plans, full_locks=full_locks, checkpoints=checkpoints,
                fold_fits=fits, weight_diagnostics=diagnostics, selector_fits=20,
                current_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                identity_note='Source hashes identify executed code, including any uncommitted changes; HEAD alone does not.',
                runtime=dict(python=platform.python_version(), numpy=np.__version__, sklearn=__import__('sklearn').__version__),
                test_rows_opened=False, new_grid_selection=False, original_grid_retained=True,
                dev_filter_scope='train_dev_test evaluation facts; no TEST outcome selection',
                notes='OOF refits replay original grid actions/RR; no claim of cross-runtime bitwise logits. Fixed full-DEV locks used on TEST.')
    json_write(OUT / 'plan.json', plan)
    print('Prepared all four pairs; finer/continuous RR still requires server scoring.', flush=True)


def test_plan(pair, plan):
    expected = json.loads((ROOT / 'paper_a_draft/rerun_source_manifest.json').read_text())['sources']
    path = INPUT / pair / 'test_anchored/test_locked_query_rows.csv'
    require(sha(path) == expected[path.relative_to(ROOT).as_posix()], 'TEST source changed')
    frame = read_frame(path)
    validate_observations(frame, 'test')
    lock = plan['full_locks'][pair]
    invalid = ~np.isfinite(frame[list(FEATURE_GROUPS['full_geometry'])]).all(axis=1)
    result = planned_frame(frame, frame.anchored_decision, frame.anchored_probability_a, invalid,
                           np.full(len(frame), lock['alpha0']), lock['beta'], lock['confidence_threshold'],
                           {'Primary': frame.rr_a, 'Secondary': frame.rr_b, 'Global': frame.rr_global,
                            'ADC_grid_005': frame.rr_anchored_locked, 'Query-soft_grid_005': frame.rr_query_soft_locked})
    close(result.alpha_ADC_grid_005, frame.alpha_anchored_locked, 'TEST ADC weight replay failed')
    close(result['alpha_Query-soft_grid_005'], frame.alpha_query_soft_locked, 'TEST Query-soft weight replay failed')
    destination = OUT / 'prepared' / (pair + '_test.csv.gz')
    if destination.exists():
        old = read_frame(destination)
        pd.testing.assert_frame_equal(result, old, check_dtype=False, rtol=0, atol=1e-12)
    else:
        frame_write(destination, result)
    return result, dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                        prepared_path=destination.relative_to(ROOT).as_posix(), prepared_sha256=sha(destination))


def rank_actions(raw_a, raw_b, gold_a, gold_b, queries, direction, facts, weights):
    import torch
    from scripts.eval_heterogeneous_complementarity import (
        filtered_copy, query_zscore_with_reference, ranks_against_reference, endpoint_safe_mixed_ranks)
    for value in (raw_a, raw_b, gold_a, gold_b):
        require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'Non-finite or unexpected raw score dtype')
    za, ga = query_zscore_with_reference(raw_a, gold_a)
    zb, gb = query_zscore_with_reference(raw_b, gold_b)
    for value in (za, zb, ga, gb):
        require(torch.isfinite(value).all().item(), 'Invalid normalized score')
    ra = ranks_against_reference(filtered_copy(raw_a, queries, direction, facts), gold_a)
    rb = ranks_against_reference(filtered_copy(raw_b, queries, direction, facts), gold_b)
    za, zb = filtered_copy(za, queries, direction, facts), filtered_copy(zb, queries, direction, facts)
    results = {'Primary': ra.numpy(), 'Secondary': rb.numpy()}
    for method, alpha in weights.items():
        results[method] = endpoint_safe_mixed_ranks(za, zb, ga, gb, alpha, ra, rb).numpy()
    return results


def complete_phase(phase, plan_sha):
    receipts = {}
    drift = 0
    for pair in PAIRS:
        for seed in (1, 2, 3):
            for direction in ('head', 'tail'):
                path = OUT / 'cells' / f'{phase}_{pair}_s{seed}_{direction}.json'
                receipt = json.loads(path.read_text())
                require(receipt['plan_sha256'] == plan_sha, 'Cell plan mismatch')
                require(sha(ROOT / receipt['rows_path']) == receipt['rows_sha256'], 'Cell output changed')
                receipts[path.relative_to(ROOT).as_posix()] = sha(path)
                drift += sum(receipt['historical_rank_mismatch_counts'].values())
    return dict(status='all_cells_scored' if drift == 0 else 'all_cells_scored_replay_review_required',
                phase=phase, plan_sha256=plan_sha, cells=len(receipts), receipts=receipts,
                historical_rank_mismatches=drift, new_grid_selection=False)


def score(phase, device):
    import torch
    from scripts.eval_heterogeneous_complementarity import load_expert, validate_pair, evaluation_fact_indexes, score_expert_block
    plan = verify_plan()
    plan_sha = sha(OUT / 'plan.json')
    require(device == 'cuda' and torch.cuda.is_available(), 'Full scoring requires the authorized CUDA server')
    if phase == 'test':
        dev = json.loads((OUT / 'dev_complete.json').read_text())
        require(dev == complete_phase('dev', plan_sha), 'DEV scoring is incomplete or changed')
    torch.set_num_threads(1)
    for pair in PAIRS:
        if phase == 'dev':
            info = plan['dev_plans'][pair]
            frame = read_frame(ROOT / info['path'])
        else:
            frame, info = test_plan(pair, plan)
        setting = plan['full_locks'][pair]
        for seed in (1, 2, 3):
            entries = []
            for name in (setting['expert_a_name'], setting['expert_b_name']):
                matches = [(run, c) for run, c in plan['checkpoints'].items()
                           if (c['dataset'], c['model'], c['seed']) == (setting['dataset'], name, seed)]
                require(len(matches) == 1, 'Ambiguous checkpoint role')
                run, entry = matches[0]
                require(sha(ROOT / run / 'best.ckpt') == entry['checkpoint_sha256'], 'Checkpoint changed')
                entries.append((name, run, entry))
            pending = []
            for direction in ('head', 'tail'):
                base = OUT / 'cells' / f'{phase}_{pair}_s{seed}_{direction}'
                receipt_path = base.with_suffix('.json')
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    require(receipt['plan_sha256'] == plan_sha and receipt['input'] == info, 'Cannot resume changed cell')
                    require(sha(base.with_suffix('.csv.gz')) == receipt['rows_sha256'], 'Completed cell changed')
                else:
                    pending.append(direction)
            if not pending:
                print(f'[RESUME] {phase} {pair} seed {seed}', flush=True)
                continue
            a, b = [load_expert(name, ROOT / run, device) for name, run, _ in entries]
            validate_pair(a, b)
            for expert in (a, b):
                require(expert.bundle.manifest['hashes'] == expert.cfg['_dataset_manifest']['hashes'],
                        'Loaded split/mapping/feature manifest differs from frozen checkpoint config')
            triples = a.bundle.valid_triples if phase == 'dev' else a.bundle.test_triples
            facts = evaluation_fact_indexes(a.bundle, include_test=True)
            for direction in pending:
                base = OUT / 'cells' / f'{phase}_{pair}_s{seed}_{direction}'
                local = frame[(frame.seed == seed) & (frame.direction == direction)].set_index(TRIPLE)
                order = pd.MultiIndex.from_tuples([tuple(t) for t in triples], names=TRIPLE)
                require(len(local) == len(order) and local.index.is_unique and order.is_unique and set(local.index) == set(order), 'Scorer/query-plan split mismatch')
                local = local.loc[order].reset_index()
                ranks = {method: [] for method in ('Primary', 'Secondary', *METHODS)}
                batch = max(a.query_batch_size, b.query_batch_size)
                for start in range(0, len(triples), batch):
                    stop = min(start + batch, len(triples))
                    q = torch.tensor(triples[start:stop], dtype=torch.long)
                    _, ga, raw_a = score_expert_block(a, q, direction, {}, device, retain_unfiltered=True)
                    _, gb, raw_b = score_expert_block(b, q, direction, {}, device, retain_unfiltered=True)
                    weights = {m: local['alpha_' + m].to_numpy()[start:stop] for m in METHODS}
                    current = rank_actions(raw_a, raw_b, ga, gb, q, direction, facts[direction], weights)
                    for method, value in current.items():
                        ranks[method].extend(value.tolist())
                for method, value in ranks.items():
                    local['rank_' + method] = value
                for method in METHODS:
                    local['execution_alpha_' + method] = local['alpha_' + method].to_numpy(dtype=np.float32).astype(float)
                output = base.with_suffix('.csv.gz')
                frame_write(output, local)
                mismatch = {m: int((local['rank_' + m] != local['historical_rank_' + m]).sum()) for m in REPLAY}
                receipt = dict(pair=pair, phase=phase, seed=seed, direction=direction, input=info,
                               completed_at_utc=datetime.now(timezone.utc).isoformat(), num_entities=a.num_entities,
                               plan_sha256=plan_sha, rows_path=output.relative_to(ROOT).as_posix(), rows_sha256=sha(output),
                               rows=len(local), methods=list(METHODS), historical_rank_mismatch_counts=mismatch,
                               triple_order_sha256=hashlib.sha256(json.dumps([list(t) for t in triples], separators=(',', ':')).encode()).hexdigest(),
                               outer_batch=batch, scorer_query_batches=[a.query_batch_size,b.query_batch_size],
                               runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                            device=torch.cuda.get_device_name(), execution_weight_dtype='float32'),
                               candidates_exported=False, new_grid_selection=False)
                json_write(base.with_suffix('.json'), receipt)
                print(f'[SCORED] {phase} {pair} seed {seed} {direction}: historical mismatches {mismatch}', flush=True)
            del a, b
            gc.collect()
            torch.cuda.empty_cache()
    completion = complete_phase(phase, plan_sha)
    json_write(OUT / (phase + '_complete.json'), completion)
    print(json.dumps(completion), flush=True)


def summarize_frame(frame):
    from scripts.analyze_paper_a_conservative_radius import metrics
    reference = 1 / frame.rank_Global.to_numpy()
    result = []
    for method in METHODS:
        rr = 1 / frame['rank_' + method].to_numpy()
        result.append(dict(method=method, **metrics(rr, reference, frame['alpha_' + method].to_numpy(), frame.anchor.to_numpy())))
    return result


def report(pack):
    plan = verify_plan()
    plan_sha = sha(OUT / 'plan.json')
    summaries, cells, effects, diagnostics, verified, sources = [], [], [], [], [], {}
    for phase in ('dev', 'test'):
        complete = json.loads((OUT / (phase + '_complete.json')).read_text())
        require(complete == complete_phase(phase, plan_sha), 'Changed phase completion')
        verified.append(complete)
        for rel, value in complete['receipts'].items():
            sources[rel] = value
            receipt = json.loads((ROOT / rel).read_text())
            sources[receipt['rows_path']] = receipt['rows_sha256']
        for pair in PAIRS:
            parts = []
            for seed in (1, 2, 3):
                for direction in ('head', 'tail'):
                    f = read_frame(OUT / 'cells' / f'{phase}_{pair}_s{seed}_{direction}.csv.gz')
                    receipt = json.loads((OUT / 'cells' / f'{phase}_{pair}_s{seed}_{direction}.json').read_text())
                    require(len(f) == receipt['rows'] and set(f.seed) == {seed} and set(f.direction) == {direction}, 'Cell coverage mismatch')
                    rank_array = f[['rank_' + m for m in ('Primary','Secondary',*METHODS)]].to_numpy()
                    require(np.isfinite(rank_array).all() and (rank_array == np.rint(rank_array)).all()
                            and (rank_array >= 1).all() and (rank_array <= receipt['num_entities']+1).all(), 'Invalid rank')
                    cells.extend(dict(pair=pair, split=phase, seed=seed, direction=direction, **r) for r in summarize_frame(f))
                    parts.append(f)
            frame = pd.concat(parts, ignore_index=True)
            require(not frame.duplicated(KEY).any() and (frame.groupby(TRIPLE).size()==6).all(), 'Result coverage mismatch')
            summaries.extend(dict(pair=pair, split=phase, **r) for r in summarize_frame(frame))
            diagnostics.extend(dict(pair=pair, split=phase, **r) for r in weight_diagnostics(frame))
            comparisons = [(f'{p}_{r}', 'Global') for p in ('ADC','Query-soft') for r in RESOLUTIONS]
            comparisons += [(f'ADC_{r}', f'Query-soft_{r}') for r in RESOLUTIONS]
            comparisons += [(f'{p}_{r}',f'{p}_grid_005') for p in ('ADC','Query-soft') for r in ('grid_001','continuous')]
            for left, right in comparisons:
                delta = 1/frame['rank_'+left] - 1/frame['rank_'+right]
                effects.append(dict(pair=pair,split=phase,left=left,right=right,n=len(frame),delta_mrr=float(delta.mean()),
                                    **{f'seed_{s}':float(delta[frame.seed==s].mean()) for s in (1,2,3)},
                                    **{f'direction_{d}':float(delta[frame.direction==d].mean()) for d in ('head','tail')}))
    outputs = {}
    for name, rows in [('summary',summaries),('by_seed_direction',cells),('paired_effects',effects),('weight_diagnostics',diagnostics)]:
        path = OUT / (name + '.csv')
        pd.DataFrame(rows).to_csv(path,index=False)
        outputs[path.name] = sha(path)
    replay_ok = all(v['historical_rank_mismatches']==0 for v in verified)
    review = dict(status='grid_scoring_review_ready' if replay_ok else 'grid_scoring_complete_replay_review_required',
                  plan_sha256=plan_sha,phase_checks=verified,sources=sources,outputs=outputs,
                  historical_grid_replay_exact=replay_ok,selector_fits=20,new_grid_selection=False,
                  current_paper_results_replaced=False,new_confidence_intervals=False,
                  note='Fixed-grid sensitivity point estimates; same new candidate scores for all comparisons. Continuous weights are float32 in ranking. Historical TEST exposure remains.')
    json_write(OUT/'review.json',review)
    if pack:
        target=OUT.parent/'grid_sensitivity_review_v1_return.zip'
        temporary=target.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUT.rglob('*')):
                if path.is_file() and not path.name.endswith('.partial') and path.name != 'last_error.json':
                    z.write(path,path.relative_to(OUT).as_posix())
        temporary.replace(target)
        print(f'Return {target} ({target.stat().st_size/1024**2:.2f} MiB)',flush=True)
    print(review['status'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','score-dev','score-test','report'))
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--pack',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='prepare': prepare()
        elif args.phase=='report': report(args.pack)
        else: score(args.phase.split('-')[1],args.device)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        json_write(OUT/'last_error.json',dict(error_type=type(error).__name__,message=str(error),command=sys.argv[1:]))
        raise
