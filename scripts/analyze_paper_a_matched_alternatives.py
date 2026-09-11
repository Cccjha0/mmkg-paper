"""Two-phase, CPU-only matched-map review for A02/D04-D06.

lock-dev creates all 24 family locks before apply-test can open TEST query rows.
The historical full-DEV model is reused, with twenty original fold fits replayed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.matched_actions import Action, ALPHAS, FAMILIES, configurations, grid_indices, weights, select_candidate
from scripts.analyze_paper_a_conservative_radius import sha, require, close, metrics, PAIRS, INPUT, KEY, GRID_COLUMNS
from scripts.ablate_anchored_dynamic import feature_matrix, fit_geometry_model, model_outputs, FEATURE_GROUPS
from scripts.crossfit_anchored_dynamic import read_csv
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha

OUT = ROOT/'outputs/paper_a_safe_correction/matched_alternatives_v1'
PROTOCOL = ROOT/'docs/protocols/paper_a_matched_alternatives.md'
FIELDS = tuple(FEATURE_GROUPS['full_geometry'])
ALL_PAIRS = {**PAIRS, 'mkgw_native_adamf': 'W-NA', 'db15k_native_adamf': 'D-NA'}
CODE = ('router/matched_actions.py', 'scripts/analyze_paper_a_matched_alternatives.py',
        'scripts/analyze_paper_a_conservative_radius.py', 'scripts/ablate_anchored_dynamic.py',
        'scripts/crossfit_anchored_dynamic.py', 'scripts/crossfit_heterogeneous_dev_policies.py',
        'scripts/lock_apply_anchored_dynamic.py', 'scripts/eval_heterogeneous_complementarity.py')


class Inputs:
    def __init__(self, phase):
        self.phase = phase
        self.expected = json.loads((ROOT/'paper_a_draft/rerun_source_manifest.json').read_text())['sources']
        self.sources = {}

    def bind(self, path, existing=True):
        path = Path(path)
        if self.phase == 'DEV':
            require(not any(part.lower().startswith('test') for part in path.parts), 'DEV phase cannot open TEST paths')
        rel = path.relative_to(ROOT).as_posix()
        value = sha(path)
        if existing:
            require(self.expected.get(rel) == value, 'Unverified source: '+rel)
        self.sources[rel] = value
        return path

    def frame(self, path):
        return pd.read_csv(self.bind(path), float_precision='round_trip')

    def obj(self, path):
        return json.loads(self.bind(path).read_text(encoding='utf-8'))

    def model(self, path):
        with self.bind(path).open('rb') as handle:
            return pickle.load(handle)


def preference(frame, model):
    x = frame[list(FIELDS)].to_numpy(copy=True)
    invalid = ~np.isfinite(x).all(axis=1)
    x[~np.isfinite(x)] = np.nan
    g, p = model_outputs(model, x)
    return g, p, invalid


def check_frame(frame, split):
    require(set(frame.split) == {split}, 'Wrong split')
    require(set(frame.score_information_contract) == {'unfiltered_features_and_normalization_v2'}, 'Wrong boundary contract')
    require(not frame.duplicated(KEY).any(), 'Duplicate observation key')
    require(set(frame.seed) == {1, 2, 3} and set(frame.direction) == {'head', 'tail'}, 'Incomplete seeds/directions')
    require((frame.groupby(['head_id', 'relation_id', 'tail_id']).size() == 6).all(), 'Incomplete original-triple grouping')


def apply(grid, signals, anchor, action):
    alpha, fallback = weights(*signals, anchor, action)
    return grid[np.arange(len(grid)), grid_indices(alpha, anchor)], alpha, fallback


def evaluate(grid, signals, anchor, family):
    candidates = []
    for action in configurations(family):
        rr, alpha, _ = apply(grid, signals, anchor, action)
        candidates.append({**asdict(action), 'mrr': float(rr.mean()),
                           'action_sha256': hashlib.sha256(alpha.astype('<f8').tobytes()).hexdigest()})
    selected = select_candidate(candidates)
    unique = len({r['action_sha256'] for r in candidates})
    return Action(selected['family'], selected['strength'], selected['tau']), selected, candidates, unique


def summaries(pair, frame, results, reference, anchor):
    pooled, cells = [], []
    for method, (rr, alpha) in results.items():
        pooled.append(dict(pair=pair, method=method, **metrics(rr, reference, alpha, anchor)))
        for seed in (1, 2, 3):
            for direction in ('head', 'tail'):
                mask = ((frame.seed == seed) & (frame.direction == direction)).to_numpy()
                local_anchor = anchor[mask] if isinstance(anchor, np.ndarray) else anchor
                cells.append(dict(pair=pair, method=method, seed=seed, direction=direction,
                                  **metrics(rr[mask], reference[mask], alpha[mask], local_anchor)))
    return pooled, cells


def check_repeat_weights(frame, results):
    for direction, known in [('tail', 'head_id'), ('head', 'tail_id')]:
        mask = (frame.direction == direction).to_numpy()
        groups = frame.loc[mask, ['seed', 'relation_id', known]].copy()
        for method, (_, alpha) in results.items():
            groups['alpha'] = alpha[mask]
            require(groups.groupby(['seed', 'relation_id', known]).alpha.nunique().max() == 1,
                    f'Gold-repeat grid weight mismatch: {method}/{direction}')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def lock_dev():
    require(not (OUT/'test_audit.json').exists(), 'TEST has already been applied; cannot replace DEV locks in this review')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = Inputs('DEV')
    inputs.bind(PROTOCOL, False)
    for file in CODE:
        inputs.bind(ROOT/file, False)
    candidate_rows, choices, dev_results, dev_cells, replay, locks = [], [], [], [], [], {}
    for pair in PAIRS:
        folder = INPUT/pair
        old_lock = inputs.obj(folder/'dev_lock/anchored_dev_lock.json')
        dev = inputs.frame(folder/'dev_lock/dev_locked_query_rows.csv')
        check_frame(dev, 'dev')
        model_path = folder/'dev_lock'/old_lock['model_file']
        require(sha(model_path) == old_lock['model_sha256'], 'Model differs from old lock')
        model = inputs.model(model_path)
        signals = preference(dev, model)
        close(signals[0], dev.anchored_decision, 'Full DEV decision replay mismatch')
        close(signals[1], dev.anchored_probability_a, 'Full DEV probability replay mismatch')
        grid, anchor = dev[GRID_COLUMNS].to_numpy(), old_lock['alpha0']
        original = Action('ADC+Global', old_lock['beta'], old_lock['confidence_threshold'])
        old_rr, old_alpha, old_fb = apply(grid, signals, anchor, original)
        close(old_rr, dev.rr_anchored_locked, 'Full DEV ADC mismatch')
        close(old_alpha, dev.alpha_anchored_locked, 'Full DEV ADC alpha mismatch')
        qs_rr, qs_alpha, _ = apply(grid, signals, anchor, Action('Shrink', 1., 0.))
        close(qs_rr, dev.rr_query_soft_locked, 'Full DEV Query-soft mismatch')
        close(qs_alpha, dev.alpha_query_soft_locked, 'Full DEV Query-soft alpha mismatch')
        settings = {}
        full_results = {'ADC-original': (old_rr, old_alpha), 'Query-soft': (qs_rr, qs_alpha)}
        for family in FAMILIES:
            action, selected, candidates, unique = evaluate(grid, signals, anchor, family)
            settings[family] = asdict(action)
            candidate_rows.extend(dict(pair=pair, fold=0, scope='full_dev_selection', **r) for r in candidates)
            choices.append(dict(pair=pair, fold=0, scope='full_dev_selection', anchor=anchor,
                                distinct_action_vectors=unique, **asdict(action), selection_mrr=selected['mrr']))
            rr, alpha, _ = apply(grid, signals, anchor, action)
            full_results[family] = rr, alpha
        check_repeat_weights(dev, full_results)
        locks[pair] = dict(alpha0=anchor, actions=settings, model_file=model_path.relative_to(ROOT).as_posix(),
                           model_sha256=sha(model_path), historical_lock_sha256=sha(folder/'dev_lock/anchored_dev_lock.json'),
                           classifier_n_iter=model[-1].n_iter_.tolist())
        path = inputs.bind(folder/'baseline_crossfit/dev_crossfit_query_rows.csv')
        rows = read_csv(path)
        assignment, _ = assign_grouped_folds(rows, 5, 20260901)
        old_oof = inputs.frame(folder/'p3_ablation/dev_p3_selected_query_rows.csv').set_index('query_id')
        old_folds = inputs.frame(folder/'p3_ablation/dev_p3_selected_by_fold.csv').set_index('fold')
        pieces, piece_frames, refs, anchors = {}, [], [], []
        for fold in range(5):
            train = [r for r in rows if assignment[triple_key(r)] != fold]
            held = [r for r in rows if assignment[triple_key(r)] == fold]
            a0, _ = best_alpha(train, tuple(ALPHAS))
            close(a0, old_folds.loc[fold+1, 'alpha0'], 'Fold anchor mismatch')
            fold_model, tx, tn = fit_geometry_model(train, fields=FIELDS, random_state=20260902+fold*10+2)
            hx, hn = feature_matrix(held, FIELDS)
            train_signals, held_signals = (*model_outputs(fold_model, tx), tn), (*model_outputs(fold_model, hx), hn)
            tg = np.array([[float(r[c]) for c in GRID_COLUMNS] for r in train])
            hg = np.array([[float(r[c]) for c in GRID_COLUMNS] for r in held])
            previous = old_oof.loc[[r['query_id'] for r in held]]
            require((previous.fold == fold+1).all(), 'OOF identities mismatch')
            ref = hg[:, int(round(a0*20))]
            close(ref, previous.rr_global_crossfit, 'OOF Global mismatch')
            original_fold = Action('ADC+Global', float(old_folds.loc[fold+1, 'selected_beta']),
                                   float(old_folds.loc[fold+1, 'selected_confidence_threshold']))
            r, a, fb = apply(hg, held_signals, a0, original_fold)
            close(r, previous.rr_method, 'OOF ADC mismatch')
            close(a, previous.alpha_applied, 'OOF ADC alpha mismatch')
            require(np.array_equal(fb, previous.fallback.astype(bool)), 'OOF fallback mismatch')
            qr, qa, _ = apply(hg, held_signals, a0, Action('Shrink', 1., 0.))
            close(qr, previous.rr_query_soft, 'OOF same-object Query-soft mismatch')
            result = {'ADC-original': (r, a), 'Query-soft': (qr, qa), 'Global': (ref, np.full(len(ref), a0))}
            for family in FAMILIES:
                action, selected, candidates, unique = evaluate(tg, train_signals, a0, family)
                if family == 'ADC+Global':
                    # Match the original exact (nonzero) argmax independently of the new zero/tie rule.
                    active = [c for c in candidates if c['strength'] > 0]
                    original_choice = max(active, key=lambda c: (c['mrr'], -c['strength'], c['tau']))
                    require((original_choice['strength'], original_choice['tau']) ==
                            (original_fold.strength, original_fold.tau), 'Historical fold selection mismatch')
                candidate_rows.extend(dict(pair=pair, fold=fold+1, scope='outer_train_selection', **c) for c in candidates)
                rr, alpha, _ = apply(hg, held_signals, a0, action)
                choices.append(dict(pair=pair, fold=fold+1, scope='outer_train_selection', anchor=a0,
                                    distinct_action_vectors=unique, **asdict(action), selection_mrr=selected['mrr'],
                                    heldout_mrr=float(rr.mean()), classifier_n_iter=int(fold_model[-1].n_iter_[0])))
                result[family] = rr, alpha
            for method, values in result.items():
                pieces.setdefault(method, []).append(values)
            piece_frames.append(previous.reset_index())
            refs.append(ref); anchors.append(np.full(len(ref), a0))
        pooled = {method: tuple(np.concatenate(v) for v in zip(*parts)) for method, parts in pieces.items()}
        frame = pd.concat(piece_frames, ignore_index=True)
        summary, cells = summaries(pair, frame, pooled, np.concatenate(refs), np.concatenate(anchors))
        dev_results.extend(summary); dev_cells.extend(cells)
        replay.append(dict(pair=pair, fold_models_replayed=5, original_fold_choices_replayed=True,
                           oof_adc_querysoft_replayed=True, full_dev_same_model_replayed=True,
                           dev_grid_gold_repeat_invariant=True, model_sha256=sha(model_path)))
        print('[DEV LOCKED] '+pair+' '+json.dumps(settings), flush=True)
    files = {'dev_candidates.csv': candidate_rows, 'dev_choices.csv': choices,
             'dev_oof_summary.csv': dev_results, 'dev_oof_seed_direction.csv': dev_cells}
    for name, records in files.items():
        pd.DataFrame(records).to_csv(OUT/name, index=False)
    lock = dict(version='matched_alternatives_v1', status='all_dev_locks_complete', failures=[],
                protocol_sha256=sha(PROTOCOL), phase='DEV only; no TEST query rows opened',
                families=list(FAMILIES), configurations_per_family=41, pairs=locks, replay=replay,
                runtime=dict(python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__),
                sources=inputs.sources, outputs={name: sha(OUT/name) for name in files})
    write_json(OUT/'dev_locks.json', lock)
    print(json.dumps({'status': lock['status'], 'models_reconstructed': 20, 'final_models_reused': 4,
                      'family_locks': 24, 'candidate_evaluations': len(candidate_rows)}))


def apply_test():
    lock_path = OUT/'dev_locks.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    require(lock['status'] == 'all_dev_locks_complete' and set(lock['pairs']) == set(PAIRS), 'Incomplete DEV locks')
    for rel, value in lock['sources'].items():
        require(sha(ROOT/rel) == value, 'DEV source changed after locking: '+rel)
    for name, value in lock['outputs'].items():
        require(sha(OUT/name) == value, 'DEV output changed after locking: '+name)
    inputs = Inputs('TEST')
    inputs.bind(lock_path, False)
    all_summary, all_cells, paired, static, static_cells, checks = [], [], [], [], [], []
    per_query = OUT/'query_rows'; per_query.mkdir(exist_ok=True)
    query_files = []
    for pair in ALL_PAIRS:
        folder = INPUT/pair
        frame = inputs.frame(folder/'test_anchored/test_locked_query_rows.csv')
        check_frame(frame, 'test')
        anchor = float(frame.alpha0_locked.iloc[0])
        require(frame.alpha0_locked.nunique() == 1, 'Inconsistent anchor')
        ref = frame.rr_global.to_numpy()
        close(frame.rr_oracle, np.maximum(frame.rr_a, frame.rr_b), 'Oracle is not expert max')
        require(set(frame.rrf_k) == {60.}, 'RRF k mismatch')
        close(frame.rr_rrf, 1/frame.rank_rrf, 'RRF rank/RR mismatch')
        selection = inputs.obj(folder/'full_ranking/selection.json')
        require(selection['global_alpha'] == anchor and selection['relation_min_support'] == 60, 'Static DEV lock mismatch')
        # The evaluator creates a torch tensor in z_a.dtype (float32 here), then
        # serializes each tensor scalar. Reproduce that cast; do not relax RR checks.
        expected_relation = frame.relation_id.astype(str).map(selection['relation_alpha']).fillna(anchor).to_numpy().astype(np.float32).astype(np.float64)
        close(frame.alpha_relation, expected_relation, 'Relation TEST policy differs from DEV lock')
        for method, col in [('Primary', 'rr_a'), ('Secondary', 'rr_b'), ('Equal-z', 'rr_equal'),
                            ('RRF', 'rr_rrf'), ('Global', 'rr_global'), ('Relation', 'rr_relation'),
                            ('Expert-oracle', 'rr_oracle')]:
            values = frame[col].to_numpy()
            r = dict(pair=pair, method=method, deployable=method != 'Expert-oracle',
                     **metrics(values, ref), **{f'hits_at_{k}': float((values >= 1/k).mean()) for k in (1, 3, 10)})
            static.append(r)
            for (seed, direction), part in frame.groupby(['seed', 'direction']):
                static_cells.append(dict(pair=pair, method=method, seed=seed, direction=direction,
                                         **metrics(part[col].to_numpy(), part.rr_global.to_numpy())))
        if pair not in PAIRS:
            continue
        setting = lock['pairs'][pair]
        model_path = ROOT/setting['model_file']
        require(sha(model_path) == setting['model_sha256'], 'Frozen model mismatch')
        model = inputs.model(model_path)
        signals = preference(frame, model)
        close(signals[0], frame.anchored_decision, 'TEST decision replay mismatch')
        close(signals[1], frame.anchored_probability_a, 'TEST probability replay mismatch')
        grid = frame[GRID_COLUMNS].to_numpy()
        result = {'Global': (ref, np.full(len(ref), anchor)),
                  'ADC-original': (frame.rr_anchored_locked.to_numpy(), frame.alpha_anchored_locked.to_numpy()),
                  'Query-soft': (frame.rr_query_soft_locked.to_numpy(), frame.alpha_query_soft_locked.to_numpy())}
        qr, qa, _ = apply(grid, signals, anchor, Action('Shrink', 1., 0.))
        close(qr, result['Query-soft'][0], 'TEST Query-soft RR differs from same object')
        close(qa, result['Query-soft'][1], 'TEST Query-soft weights differ from same object')
        for family in FAMILIES:
            rr, alpha, fb = apply(grid, signals, anchor, Action(**setting['actions'][family]))
            result[family] = rr, alpha
        check_repeat_weights(frame, result)
        summary, cells = summaries(pair, frame, result, ref, anchor)
        all_summary.extend(summary); all_cells.extend(cells)
        adc = result['ADC+Global'][0]
        for family in FAMILIES[1:]:
            delta = adc - result[family][0]
            seed = pd.Series(delta).groupby(frame.seed).mean()
            direction = pd.Series(delta).groupby(frame.direction).mean()
            paired.append(dict(pair=pair, comparator=family, delta_mrr_adc_minus_comparator=float(delta.mean()),
                               positive_seeds=int((seed > 1e-12).sum()), positive_directions=int((direction > 1e-12).sum()),
                               **{f'seed_{s}': float(seed.loc[s]) for s in (1, 2, 3)},
                               **{f'direction_{d}': float(direction.loc[d]) for d in ('head', 'tail')}))
        out = frame[KEY].copy()
        for method, (rr, alpha) in result.items():
            out['rr_'+method] = rr; out['alpha_'+method] = alpha
        path = per_query/(pair+'.csv.gz')
        out.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
        query_files.append(path)
        checks.append(dict(pair=pair, same_fitted_model_and_outputs=True, grid_weight_gold_repeat_invariant=True,
                           n_observations=len(frame), family_locks_applied=len(FAMILIES)))
        print('[TEST APPLIED] '+pair, flush=True)
    files = {'test_summary.csv': all_summary, 'test_seed_direction.csv': all_cells,
             'test_paired_differences.csv': paired, 'static_summary.csv': static, 'static_seed_direction.csv': static_cells}
    for name, rows in files.items():
        pd.DataFrame(rows).to_csv(OUT/name, index=False)
    all_sources = {**lock['sources'], **inputs.sources}
    audit = dict(version='matched_alternatives_v1', status='matched_alternative_checks_passed', failures=[],
                 dev_lock_sha256=sha(lock_path), test_used_for_selection=False, checks=checks,
                 sources=all_sources, outputs={name: sha(OUT/name) for name in files},
                 query_outputs={p.relative_to(OUT).as_posix(): sha(p) for p in query_files},
                 limitations=['Supplementary study after historical TEST inspection; no new blind confirmation.',
                              'Exact grid ranks reused; no scorer or full candidate rerun.',
                              'Point estimates conditional on fitted seeds; no new confidence intervals.',
                              'Six families match candidate count, not identical attainable action sets.',
                              'Expert oracle is not a bound on score fusion.'])
    write_json(OUT/'test_audit.json', audit)
    print(json.dumps({'status': audit['status'], 'test_family_cells': len(all_summary), 'static_cells': len(static)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock-dev', 'apply-test'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (lock_dev if args.phase == 'lock-dev' else apply_test)()
