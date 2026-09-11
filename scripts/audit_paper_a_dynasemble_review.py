"""Audit returned D01-D03 evidence without scorers, training, or large bootstrap jobs.

The review bundle omits candidate caches. This audits their recorded bindings, not
their absent bytes. Reported bootstrap intervals are hash-bound server outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from router.dynasemble_controls import ControlSelector, VARIANTS, grouped_folds

BASES = (1, 2, 3)
SEEDS = (11, 23, 37)
LRS = (1e-5, 5e-5, 1e-4)
EPOCHS = (1, 3, 5, 10)
KEY = ['base_seed', 'direction', 'head_id', 'relation_id', 'tail_id']
TRIPLE = KEY[2:]
DEFAULT = ROOT / 'outputs/paper_a_safe_correction/dynasemble_controls_v1_review'
OUTPUT = ROOT / 'outputs/paper_a_safe_correction/dynasemble_review_audit'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True), message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def selected_cv(cv):
    columns = ['base_seed', 'selector_seed', 'fold', 'learning_rate', 'epoch']
    expected = set(itertools.product(BASES, SEEDS, range(3), LRS, EPOCHS))
    require(len(cv) == len(expected) and set(map(tuple, cv[columns].to_numpy())) == expected,
            'CV matrix missing/duplicated: every predeclared seed/fold/lr/epoch is required')
    require(set(cv.variant) == {'R3_softplus'} and np.isfinite(cv.mrr).all(), 'Invalid CV scores')
    require(((cv.mrr > 0) & (cv.mrr <= 1) & (cv.n > 0)).all(), 'Invalid CV range')
    pooled = cv.assign(total=cv.mrr*cv.n).groupby(['learning_rate', 'epoch']).agg(total=('total', 'sum'), n=('n', 'sum')).reset_index()
    pooled['heldout_dev_mrr'] = pooled.total/pooled.n
    winner = pooled.sort_values(['heldout_dev_mrr', 'epoch', 'learning_rate'], ascending=[False, True, True]).iloc[0]
    return {'learning_rate': float(winner.learning_rate), 'epochs': int(winner.epoch),
            'heldout_dev_mrr': float(winner.heldout_dev_mrr)}, pooled


def check_split(signature, n, folds, fold=None):
    fit, val = signature['fit_ids'], signature['validation_ids']
    expected_fit = np.arange(n) if fold is None else np.flatnonzero(folds != fold)
    held = np.arange(n) if fold is None else np.flatnonzero(folds == fold)
    require(fit == expected_fit.tolist(), 'DEV fit IDs differ from declared grouped split')
    require(val == np.concatenate((held, held+n)).tolist(), 'DEV held-out head/tail or split mismatch')
    if fold is not None:
        require(set(fit).isdisjoint(i % n for i in val), 'DEV fit/validation leakage')


def check_trace(trace, done):
    sig = done['signature']
    require(trace['status'] == 'complete', 'Incomplete training trace')
    for field in ('variant', 'selector_seed', 'learning_rate'):
        require(trace[field] == sig[field], 'Trace identity mismatch')
    require([e['epoch'] for e in trace['epochs']] == list(range(1, sig['epochs']+1)), 'Missing epoch')
    metrics = [dict(epoch=e['epoch'], **e['heldout']) for e in trace['epochs'] if 'heldout' in e]
    require(metrics == done['metrics'], 'Checkpoint metric/trace mismatch')
    expected_params = set(ControlSelector(sig['variant']).state_dict())
    all_updated, min_update, min_grad = True, float('inf'), float('inf')
    for e in trace['epochs']:
        require(set(e['gradients']) == set(e['parameter_update_norms']) == expected_params, 'Missing layer diagnostics')
        vals = [e['loss_mean'], e['loss_first'], e['loss_last'], *e['parameter_update_norms'].values()]
        vals += [v for p in e['gradients'].values() for v in p.values()]
        require(np.isfinite(vals).all() and min(vals) >= 0, 'Invalid loss/gradient/update')
        min_update = min(min_update, *e['parameter_update_norms'].values())
        min_grad = min(min_grad, *(p['max_norm'] for p in e['gradients'].values()))
        all_updated &= min(e['parameter_update_norms'].values()) > 0
    permanent = all(max(p['max_norm'] for p in e['gradients'].values()) == 0 for e in trace['epochs'])
    zero = trace['epochs'][-1]['train_diagnostics']['zero_weight_fraction'] == 1
    require(done['health'] == {'permanent_zero_gradient': permanent, 'final_all_zero_weight': zero}, 'Health flag mismatch')
    require(trace['permanent_zero_gradient'] == permanent and trace['final_all_zero_weight'] == zero, 'Trace health mismatch')
    return {'all_layers_updated_every_epoch': bool(all_updated), 'min_epoch_parameter_update': min_update,
            'min_epoch_layer_max_gradient': min_grad, 'permanent_zero_gradient': permanent,
            'final_all_zero_weight': zero}


def check_rows(frame, reference, bs, ss, variant):
    require(not frame.duplicated(KEY).any(), 'Duplicate TEST observation')
    require(set(frame.base_seed) == {bs} and set(frame.selector_seed) == {ss} and set(frame.variant) == {variant}, 'TEST identity mismatch')
    ref = reference[reference.base_seed == bs]
    joined = frame.merge(ref[KEY+['rr_a', 'rr_b', 'rr_global', 'rr_anchored_locked']], on=KEY,
                         validate='one_to_one', suffixes=('', '_reference'))
    require(len(frame) == len(ref) == len(joined), 'Missing TEST observations')
    for side in ('a', 'b'):
        require(np.array_equal(joined['rr_'+side], joined['rr_'+side+'_reference']), 'Shared evaluator endpoint mismatch')
    for side in ('a', 'b', 'method'):
        ranks = frame['rank_'+side]
        require((ranks >= 1).all() and (ranks == ranks.astype(int)).all(), 'Invalid rank')
        close(frame['rr_'+side], 1/ranks, 'RR inconsistent with rank')
    require(np.array_equal(frame.target_entity_id, np.where(frame.direction == 'tail', frame.tail_id, frame.head_id)), 'Gold identity mismatch')
    w, z = frame.learned_weight.to_numpy(), frame.preactivation.to_numpy()
    require(np.isfinite(w).all() and np.isfinite(z).all() and (w >= 0).all(), 'Invalid weight')
    expected = np.maximum(z, 0) if variant in ('R1_source', 'R2_init') else np.logaddexp(0, z)
    close(w, expected, 'Activation mismatch', atol=4e-6)
    close(frame.primary_ratio, (1 if variant == 'R4_swap' else w)/(1+w), 'Coefficient role mismatch', atol=1e-7)
    require(np.array_equal(frame.gold_rank_matches_primary, frame.rank_method == frame.rank_a), 'Gold-rank diagnostic mismatch')
    probes = frame.complete_unfiltered_order_matches_normalized_primary.notna()
    expected_probes = []
    for direction, d in (('head', 0), ('tail', 1)):
        candidates = frame.index[frame.direction == direction].tolist()
        candidates.sort(key=lambda i: hashlib.sha256(str((int(frame.at[i, 'relation_id']), int(frame.at[i, 'head_id' if d else 'tail_id']), d)).encode()).digest())
        expected_probes += candidates[:32]
    require(set(frame.index[probes]) == set(expected_probes), 'Complete-order probe selection differs')
    return joined


def audit(root, output, cache_evidence=None):
    torch.set_num_threads(1)
    config_path = ROOT / 'configs/paper_a_dynasemble_controls.json'
    config = read(config_path)
    completion, manifest = read(root/'completion.json'), read(root/'summary_manifest.json')
    require(completion['pairs'] == 4 and set(manifest) == {p['pair'] for p in config['pairs']}, 'Incomplete pair matrix')
    for name in ('summary', 'seed_direction'):
        require(digest(root/(name+'.csv')) == completion[name+'_sha256'], 'Completion hash mismatch')
    summary = pd.read_csv(root/'summary.csv')
    cells, health = pd.read_csv(root/'seed_direction.csv'), pd.read_csv(root/'health_summary.csv')
    require(len(health) == 180 and not health.duplicated(['pair', 'variant', 'base_seed', 'selector_seed']).any(), 'Health coverage mismatch')
    require(len(summary) == 36 and not summary.duplicated(['pair', 'method']).any(), 'Summary coverage mismatch')
    require(len(cells) == 456 and not cells.duplicated(['pair', 'method', 'base_seed', 'selector_seed', 'direction']).any(), 'Seed/direction coverage mismatch')
    sources, pair_results, fit_rows, test_rows, cv_rows = {}, [], [], [], []
    def bind(path, expected=None):
        sha = digest(path)
        if expected is not None:
            require(sha == expected, f'Hash mismatch: {path}')
        sources[path.relative_to(ROOT).as_posix()] = sha
        return sha
    for p in root.glob('*.*'):
        bind(p)
    common_provenance = None
    for pair in config['pairs']:
        name, folder = pair['pair'], root/pair['pair']
        lock = read(folder/'dev_lock.json')
        lock_sha = bind(folder/'dev_lock.json', manifest[name]['lock_sha256'])
        require(lock['pair'] == name and lock['schema'] == 'dynasemble_controls_v1', 'Lock identity mismatch')
        prov = lock['provenance']
        if common_provenance is None:
            common_provenance = prov
        require(common_provenance == prov, 'Runtime/source mismatch across pairs')
        bind(config_path, prov['config_sha256'])
        for path, sha in prov['sources'].items():
            historical = subprocess.check_output(['git', 'show', prov['repository_commit']+':'+path], cwd=ROOT)
            require(hashlib.sha256(historical.replace(b'\r\n', b'\n')).hexdigest() == sha, 'Recorded commit/source mismatch: '+path)
        require(lock['dev_evaluation_filter'] == 'train_dev', 'DEV evaluation scope mismatch')
        bind(folder/'test_started.json')
        require(read(folder/'test_started.json')['dev_lock_sha256'] == lock_sha, 'DEV lock changed after TEST')
        bind(folder/'dev_cv.csv', lock['dev_cv_sha256'])
        cv = pd.read_csv(folder/'dev_cv.csv')
        selected, pooled = selected_cv(cv)
        for k, value in selected.items():
            close(value, lock['selection'][k], 'DEV winner differs: '+k)
        pooled['pair'] = name
        cv_rows += pooled.to_dict('records')
        old = ROOT/'outputs/paper_a_safe_correction/information_boundary_v2'/name
        ref_path = old/'test_anchored/test_locked_query_rows.csv'
        bind(ref_path, manifest[name]['reference_sha256'])
        reference = pd.read_csv(ref_path).rename(columns={'seed': 'base_seed'})
        require(set(reference.score_information_contract) == {'unfiltered_features_and_normalization_v2'}, 'Old boundary contract')
        # The older CSV was sorted by identity, so it cannot recover cache row order.
        dev_path = old/'full_ranking/dev_query_rows.csv'
        bind(dev_path)
        dev = pd.read_csv(dev_path, usecols=['seed', 'direction']+TRIPLE)
        triples = dev[(dev.seed == 1) & (dev.direction == 'tail')][TRIPLE].to_numpy()
        n = len(triples)
        small_caches = {}
        if cache_evidence is not None:
            for split, bs in itertools.product(('dev', 'test'), BASES):
                directory = cache_evidence/name/'cache'/f'{split}_seed{bs}'
                m = read(directory/'manifest.json')
                sha = lock['cache_manifests'][str(bs)] if split == 'dev' else read(folder/'test'/f'R3_softplus_b{bs}_s11.json')['identity']['cache_manifest_sha256']
                bind(directory/'manifest.json', sha)
                require(m['provenance'] == prov and m['pair'] == name and m['split'] == split, 'Supplemental cache identity mismatch')
                require(m['filter_scope'] == ('train_dev' if split == 'dev' else 'train_dev_test'), 'Supplemental cache filter mismatch')
                require(('test.tsv' in m['inputs']['dataset']) == (split == 'test'), 'TEST dataset entered DEV cache inputs')
                require(m['inputs']['experts'] == [next(a for a in lock['assets'] if a['seed'] == bs)], 'Cache expert asset mismatch')
                arrays = {}
                for key in ('queries', 'features'):
                    bind(directory/(key+'.npy'), m['array_sha256'][key])
                    arrays[key] = np.load(directory/(key+'.npy'), allow_pickle=False)
                q, features = arrays['queries'], arrays['features']
                count = m['n_triples']
                require(q.shape == features.shape == (2*count, 4), 'Cache array shape mismatch')
                require(np.isfinite(features).all(), 'Non-finite cached features')
                require(np.array_equal(q[:count, :3], q[count:, :3]) and (q[:count, 3] == 1).all() and (q[count:, 3] == 0).all(), 'Cache head/tail identity mismatch')
                expected_triples = triples if split == 'dev' else reference[reference.base_seed == bs][TRIPLE].drop_duplicates().to_numpy()
                require(set(map(tuple, q[:count, :3])) == set(map(tuple, expected_triples)), 'Cache triple identities differ from reviewed exports')
                small_caches[(split, bs)] = arrays
        folds = np.full(n, -1, dtype=int)
        for fold in range(3):
            declared = read(folder/'fits'/f'cv_b1_s11_f{fold}_lr1e-05'/'complete.json')['signature']['validation_ids']
            held = [i for i in declared if i < n]
            require(all(0 <= i < n for i in held) and len(held) == len(set(held)), 'Invalid DEV group IDs')
            require((folds[held] == -1).all(), 'Overlapping held-out folds')
            folds[held] = fold
        require((folds >= 0).all(), 'Missing held-out DEV groups')
        require(np.array_equal(np.bincount(folds), np.bincount(grouped_folds(triples))), 'Relation-stratified fold counts differ')
        if small_caches:
            for bs in BASES:
                require(np.array_equal(folds, grouped_folds(small_caches[('dev', bs)]['queries'][:n, :3])), 'DEV row-to-triple fold reconstruction mismatch')
        test_set = set(map(tuple, reference[TRIPLE].to_numpy()))
        require(set(map(tuple, triples)).isdisjoint(test_set), 'DEV/TEST original triple overlap')
        expected_selectors = set(itertools.product(VARIANTS, BASES, SEEDS))
        selectors = {(s['variant'], s['base_seed'], s['selector_seed']): s for s in lock['selectors']}
        require(len(lock['selectors']) == 45 and set(selectors) == expected_selectors, 'Missing final selector')
        paths = [(f'cv_b{bs}_s{ss}_f{f}_lr{lr:g}', 'R3_softplus', bs, ss, lr, 10, f)
                 for bs, ss, f, lr in itertools.product(BASES, SEEDS, range(3), LRS)]
        paths += [(s['relative_dir'].split('/')[-1], v, bs, ss, s['learning_rate'], s['epochs'], None)
                  for (v, bs, ss), s in selectors.items()]
        require({p.name for p in (folder/'fits').iterdir() if p.is_dir()} == {p[0] for p in paths}, 'Unexpected/missing fit run')
        for basename, variant, bs, ss, lr, epochs, fold in paths:
            path = folder/'fits'/basename
            bind(path/'complete.json')
            done = read(path/'complete.json')
            sig = done['signature']
            require(sig['provenance'] == prov and sig['cache_manifest_sha256'] == lock['cache_manifests'][str(bs)], 'Fit input binding differs')
            require((sig['variant'], sig['selector_seed'], sig['learning_rate'], sig['epochs']) == (variant, ss, lr, epochs), 'Fit configuration differs')
            require(sig['checkpoints'] == (list(EPOCHS) if fold is not None else [epochs]), 'Checkpoint budget differs')
            check_split(sig, n, folds, fold)
            bind(path/'model.pt', done['model_sha256'])
            bind(path/'trace.json', done['trace_sha256'])
            state = torch.load(path/'model.pt', weights_only=True, map_location='cpu')
            model = ControlSelector(variant)
            model.load_state_dict(state)
            require(all(torch.isfinite(v).all() for v in state.values()), 'Non-finite model parameter')
            trace = read(path/'trace.json')
            diagnostic = check_trace(trace, done)
            fit_rows.append(dict(pair=name, run=basename, stage='cv' if fold is not None else 'final', variant=variant,
                                 base_seed=bs, selector_seed=ss, learning_rate=lr, epochs=epochs, **diagnostic))
            if fold is not None:
                rows = cv[(cv.base_seed == bs) & (cv.selector_seed == ss) & (cv.fold == fold) & (cv.learning_rate == lr)].sort_values('epoch')
                close(rows[['epoch', 'mrr', 'n']], [[m['epoch'], m['mrr'], m['n']] for m in done['metrics']], 'CV CSV differs from trace')
                require(all(m['n'] == len(sig['validation_ids']) for m in done['metrics']), 'Held-out count mismatch')
            else:
                selector = selectors[(variant, bs, ss)]
                expected_lr, expected_ep = (5e-5, 1) if variant in VARIANTS[:3] else (selected['learning_rate'], selected['epochs'])
                require((lr, epochs) == (expected_lr, expected_ep), 'R4 budget or fixed-control setting mismatch')
                for k in ('model_sha256', 'trace_sha256', 'health'):
                    require(selector[k] == done[k], 'Selector lock differs from final fit')
                close(selector['full_dev_resubstitution_mrr'], done['metrics'][-1]['mrr'], 'Full DEV value mismatch')
                h = health[(health.pair == name) & (health.variant == variant) & (health.base_seed == bs) & (health.selector_seed == ss)].iloc[0]
                last = trace['epochs'][-1]
                for col, val in dict(learning_rate=lr, epochs=epochs, initial_zero_weight_fraction=trace['initial']['zero_weight_fraction'],
                    final_zero_weight_fraction=last['train_diagnostics']['zero_weight_fraction'], initial_weight_std=trace['initial']['weight']['std'],
                    final_weight_std=last['train_diagnostics']['weight']['std'], first_epoch_mean_loss=trace['epochs'][0]['loss_mean'],
                    last_epoch_mean_loss=last['loss_mean'], last_epoch_max_gradient_norm=max(v['max_norm'] for v in last['gradients'].values()),
                    full_dev_resubstitution_mrr=selector['full_dev_resubstitution_mrr'], **done['health']).items():
                    close(h[col], val, 'Health CSV differs: '+col)
        method_frames, test_cache_hashes = {}, {}
        expected_tests = {f'{v}_b{bs}_s{ss}.csv' for v, bs, ss in expected_selectors}
        require(set(manifest[name]['test_files']) == expected_tests == {p.name for p in (folder/'test').glob('*.csv')}, 'TEST file matrix mismatch')
        for variant in VARIANTS:
            parts = []
            for bs, ss in itertools.product(BASES, SEEDS):
                path = folder/'test'/f'{variant}_b{bs}_s{ss}.csv'
                sidecar = read(path.with_suffix('.json'))
                bind(path.with_suffix('.json'))
                bind(path, sidecar['rows_sha256'])
                require(sidecar['rows_sha256'] == manifest[name]['test_files'][path.name], 'Summary binding differs')
                require(sidecar['identity']['dev_lock_sha256'] == lock_sha and sidecar['selector'] == selectors[(variant, bs, ss)], 'TEST selector identity mismatch')
                sha = sidecar['identity']['cache_manifest_sha256']
                require(test_cache_hashes.setdefault(bs, sha) == sha, 'Different TEST cache for matched controls')
                frame = pd.read_csv(path)
                require(set(frame.pair) == {name}, 'TEST pair mismatch')
                parts.append(check_rows(frame, reference, bs, ss, variant))
                replay_error = None
                if small_caches:
                    cache = small_caches[('test', bs)]
                    require(np.array_equal(frame[TRIPLE].to_numpy(), cache['queries'][:, :3]), 'TEST cache row identity mismatch')
                    model = ControlSelector(variant)
                    model.load_state_dict(torch.load(folder/selectors[(variant, bs, ss)]['relative_dir']/'model.pt', map_location='cpu', weights_only=True))
                    with torch.no_grad():
                        predicted = model(torch.from_numpy(cache['features'])).numpy()
                    replay_error = float(np.max(np.abs(predicted-frame.learned_weight.to_numpy())))
                    require(np.allclose(predicted, frame.learned_weight, atol=2e-5, rtol=2e-6), 'CPU selector replay differs from GPU export')
                close(frame.rr_method.mean(), sidecar['metrics']['mrr'], 'TEST sidecar MRR differs')
                require(len(frame) == sidecar['metrics']['n'] and sidecar['complete_order_probe_rows'] == 64, 'TEST count mismatch')
                w = frame.learned_weight
                # Small GPU reduction/batch differences are reported, never averaged away.
                for k, series in [('weight', w), ('primary_ratio', frame.primary_ratio), ('preactivation', frame.preactivation)]:
                    for stat, value in [('min', series.min()), ('max', series.max()), ('median', series.median()), ('std', series.std(ddof=0))]:
                        close(value, sidecar['diagnostics'][k][stat], 'TEST diagnostic differs', atol=5e-6)
                observed = frame.assign(observed=np.where(frame.direction == 'tail', frame.head_id, frame.tail_id))
                observable_groups = observed.groupby(['direction', 'relation_id', 'observed']).learned_weight.agg(['max', 'min'])
                spans = observable_groups['max']-observable_groups['min']
                probe = frame.complete_unfiltered_order_matches_normalized_primary.dropna()
                test_rows.append(dict(pair=name, variant=variant, base_seed=bs, selector_seed=ss,
                    n=len(frame), zero_fraction=float((w == 0).mean()), weight_std=float(w.std(ddof=0)),
                    primary_ratio_mean=float(frame.primary_ratio.mean()), primary_ratio_min=float(frame.primary_ratio.min()),
                    primary_ratio_max=float(frame.primary_ratio.max()), primary_ratio_std=float(frame.primary_ratio.std(ddof=0)),
                    gold_rank_matches_primary=float(frame.gold_rank_matches_primary.mean()),
                    complete_order_matches=int(probe.eq(True).sum()), complete_order_probes=len(probe),
                    max_same_observable_weight_span=float(spans.max()), cpu_selector_replay_max_abs_error=replay_error))
            method_frames[variant] = pd.concat(parts, ignore_index=True)
        r0_path = old/'dynasemble/test_query_rows.csv'
        bind(r0_path, manifest[name]['historical_r0_sha256'])
        r0 = pd.read_csv(r0_path).rename(columns={'seed': 'base_seed', 'rr_dynasemble': 'rr_method'})
        r0['selector_seed'] = r0.base_seed
        r0 = r0.merge(reference[KEY+['rr_anchored_locked']], on=KEY, validate='one_to_one')
        require(len(r0) == len(reference) and set(r0.base_seed) == set(BASES), 'Historical failed seeds omitted')
        method_frames['R0_historical'] = r0
        for method, col in [('Global', 'rr_global'), ('ADC', 'rr_anchored_locked'), ('Query-soft', 'rr_query_soft_locked')]:
            method_frames[method] = reference.assign(rr_method=reference[col], selector_seed=-1)
        for method, frame in method_frames.items():
            row = summary[(summary.pair == name) & (summary.method == method)].iloc[0]
            delta = frame.rr_method-frame.rr_global
            harm, benefit = delta < -1e-12, delta > 1e-12
            values = dict(n_observations=len(frame), mrr=frame.rr_method.mean(), delta_mrr_vs_global=delta.mean(),
                delta_mrr_vs_adc=(frame.rr_method-frame.rr_anchored_locked).mean(), harmful_query_rate=harm.mean(),
                beneficial_query_rate=benefit.mean(), unchanged_query_rate=(~(harm | benefit)).mean(),
                mean_harm=-delta[harm].mean(), mean_benefit=delta[benefit].mean(), median_delta_rr=delta.median(),
                n_harmful=harm.sum(), n_beneficial=benefit.sum(), n_unchanged=(~(harm | benefit)).sum(),
                n_original_triple_clusters=len(frame[TRIPLE].drop_duplicates()),
                n_selector_seeds_per_base=frame.groupby('base_seed').selector_seed.nunique().max())
            for k, v in values.items():
                close(row[k], v, f'Summary point mismatch: {name}/{method}/{k}')
            for (bs, ss, d), group in frame.groupby(['base_seed', 'selector_seed', 'direction']):
                cell = cells[(cells.pair == name) & (cells.method == method) & (cells.base_seed == bs) & (cells.selector_seed == ss) & (cells.direction == d)]
                require(len(cell) == 1, 'Missing seed/direction cell')
                close(cell.iloc[0][['n', 'mrr', 'delta_vs_global']].to_numpy(dtype=float),
                      [len(group), group.rr_method.mean(), (group.rr_method-group.rr_global).mean()], 'Seed/direction value mismatch')
        pair_results.append(dict(pair=name, selection=selected, dev_triples=n, test_triples=len(test_set),
            r0_zero_weight_fraction=r0.assign(zero=r0.weight_expert_a == 0).groupby('base_seed').zero.mean().to_dict(),
            test_cache_manifest_sha256=test_cache_hashes, cv_trajectories=81, final_selectors=45))
        print('[AUDITED] '+name, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('fit_checks', fit_rows), ('test_diagnostics', test_rows), ('dev_candidates', cv_rows)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    # Small, verified tables are copied without changing line endings or numerics.
    for name in ('summary.csv', 'seed_direction.csv', 'health_summary.csv'):
        (output/name).write_bytes((root/name).read_bytes())
    report = dict(status='review_artifact_checks_passed', failures=[], small_cache_evidence_verified=cache_evidence is not None, provenance=common_provenance,
        counts=dict(cv_trajectories=324, final_selectors=180, test_exports=180, summary_rows=36, seed_direction_cells=456),
        pairs=pair_results,
        limitations=[
            'Full candidate score arrays are absent; candidate ranks and normalization cannot be independently replayed. Cache manifests/queries/features are verified only if small_cache_evidence_verified is true.',
            'If small_cache_evidence_verified is false, DEV fit/validation integer IDs are disjoint, exhaustive and identical across repeats, but only fold counts (not the cache row-to-triple mapping) are independently reconstructed.',
            'DEV/TEST timing is supported by code barrier and lock bindings, not an external timestamp attestation.',
            'Server bootstrap intervals are preserved with verified hashes; this read-only local audit recomputes point metrics and every seed/direction cell, not the large bootstrap.',
            'Continuous same-observable-query weight spans are recorded; nonzero spans do not by themselves identify a gold-dependent cause.',
            'These are problem-driven supplementary experiments after historical TEST inspection, not fresh blind confirmation.'],
        sources=sources, output_hashes={p.name: digest(p) for p in sorted(output.glob('*.csv'))})
    write(output/'audit.json', report)
    print(json.dumps({'status': report['status'], **report['counts'], 'bound_sources': len(sources)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--cache-evidence-root', type=Path, help='Unzipped small cache-evidence root, containing pair/cache/...')
    args = parser.parse_args()
    audit(args.input_root.resolve(), args.output_dir.resolve(), args.cache_evidence_root.resolve() if args.cache_evidence_root else None)
