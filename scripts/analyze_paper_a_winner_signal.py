"""B13/B14: small DEV logistic control, then fixed-policy cache diagnostics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.winner_signal_diagnostics import direction_rows, probability_metrics, utility_summary, TOL
from router.matched_actions import Action, ALPHAS
from scripts.analyze_paper_a_matched_alternatives import Inputs, check_frame, apply, evaluate
from scripts.analyze_paper_a_conservative_radius import INPUT, KEY, GRID_COLUMNS, sha, close, require, metrics
from scripts.ablate_anchored_dynamic import FEATURE_GROUPS, fit_geometry_model, feature_matrix, model_outputs
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha

OUT = ROOT / 'outputs/paper_a_safe_correction/winner_signal_review_v1'
PAIRS = {'mkgw_mhyper_native': 'W-N', 'mkgw_mhyper_adamf': 'W-A', 'db15k_mhyper_native': 'D-N',
         'db15k_mhyper_adamf': 'D-A', 'mkgw_native_adamf': 'W-NA', 'db15k_native_adamf': 'D-NA'}
FIELDS = tuple(FEATURE_GROUPS['full_geometry'])
CODE = ('scripts/analyze_paper_a_winner_signal.py', 'router/winner_signal_diagnostics.py',
        'tests/test_winner_signal_diagnostics.py', 'docs/protocols/paper_a_winner_signal_review.md',
        'scripts/analyze_paper_a_matched_alternatives.py', 'scripts/analyze_paper_a_conservative_radius.py',
        'scripts/ablate_anchored_dynamic.py', 'scripts/crossfit_anchored_dynamic.py',
        'scripts/crossfit_heterogeneous_dev_policies.py', 'router/matched_actions.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_frames(frames):
    outputs = {}
    for name, rows in frames.items():
        path = OUT / (name + '.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        outputs[path.relative_to(ROOT).as_posix()] = sha(path)
    return outputs


def joined_oof(inputs, folder):
    base = inputs.frame(folder / 'baseline_crossfit/dev_crossfit_query_rows.csv')
    check_frame(base, 'dev')
    old = inputs.frame(folder / 'p3_ablation/dev_p3_selected_query_rows.csv').set_index('query_id').loc[base.query_id]
    require(len(old) == len(base) and not old.index.duplicated().any(), 'OOF join mismatch')
    for key in KEY:
        require(np.array_equal(old[key].to_numpy(), base[key].to_numpy()), 'OOF metadata differs: ' + key)
    settings = inputs.frame(folder / 'p3_ablation/dev_p3_selected_by_fold.csv').set_index('fold')
    return base, old.reset_index(), settings


def dev():
    require(not (OUT / 'dev_complete.json').exists(), 'DEV receipt already exists; keep this version immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = Inputs('DEV')
    for rel in CODE:
        # This explicit inventory includes tests/ source code, not TEST data.
        # Dataset reads below still use the strict DEV path guard.
        inputs.sources[rel] = sha(ROOT / rel)
    inputs.bind(ROOT / 'paper_a_draft/rerun_source_manifest.json', False)
    scores, bins, policy_rows, cells, choices, candidates, fits, states, checks = [], [], [], [], [], [], [], [], []
    for pair, label in PAIRS.items():
        base, old, settings = joined_oof(inputs, INPUT / pair)
        rows = base.to_dict('records')
        assignment, _ = assign_grouped_folds(rows, 5, 20260901)
        fold_ids = np.array([assignment[triple_key(row)] for row in rows])
        require(np.array_equal(fold_ids + 1, old.fold), 'Original triple fold mismatch')
        n = len(base)
        probs = {name: np.full(n, np.nan) for name in ('balanced', 'unweighted', 'train_prior')}
        policies = {name: [np.full(n, np.nan), np.full(n, np.nan)] for name in
                    ('balanced_ADC41', 'balanced_Query-soft', 'unweighted_ADC41', 'unweighted_Query-soft', 'Global')}
        refs, anchors = np.full(n, np.nan), np.full(n, np.nan)
        for fold in range(5):
            train = base.loc[fold_ids != fold].reset_index(drop=True)
            held = base.loc[fold_ids == fold].reset_index(drop=True)
            mask = fold_ids == fold
            train_rows = train.to_dict('records')
            a0 = best_alpha(train_rows, tuple(ALPHAS))[0]
            close(a0, settings.loc[fold + 1, 'alpha0'], 'Fold anchor replay mismatch')
            state = 20260902 + fold * 10 + 2
            balanced, tx, tn = fit_geometry_model(train_rows, fields=FIELDS, random_state=state)
            y = (train.rr_a > train.rr_b).to_numpy().astype(int)
            non_tied = (train.rr_a != train.rr_b).to_numpy()
            unweighted = clone(balanced).set_params(logisticregression__class_weight=None)
            unweighted.fit(tx[non_tied], y[non_tied])
            require(not tn.any() and np.isfinite(tx).all(), 'Unexpected incomplete feature scope')
            for attr in ('statistics_',):
                close(getattr(balanced[0], attr), getattr(unweighted[0], attr), 'Imputation mismatch')
            for attr in ('mean_', 'scale_', 'var_'):
                close(getattr(balanced[1], attr), getattr(unweighted[1], attr), 'Scaling mismatch')
            hx, hn = feature_matrix(held.to_dict('records'), FIELDS)
            require(not hn.any(), 'Invalid held-out features')
            tg, hg = train[GRID_COLUMNS].to_numpy(), held[GRID_COLUMNS].to_numpy()
            refs[mask], anchors[mask] = hg[:, int(round(a0 * 20))], a0
            close(refs[mask], old.loc[mask, 'rr_global_crossfit'], 'OOF reference replay mismatch')
            probs['train_prior'][mask] = y[non_tied].mean()
            for name, model in [('balanced', balanced), ('unweighted', unweighted)]:
                ts = (*model_outputs(model, tx), tn)
                hs = (*model_outputs(model, hx), hn)
                probs[name][mask] = hs[1]
                action, selected, all_candidates, unique = evaluate(tg, ts, a0, 'ADC+Global')
                require(len(all_candidates) == 41, 'Unequal action search budget')
                choices.append(dict(pair=pair, fold=fold + 1, learner=name, anchor=a0,
                                    **asdict(action), selection_mrr=selected['mrr'], distinct_actions=unique))
                candidates.extend(dict(pair=pair, fold=fold + 1, learner=name, **r) for r in all_candidates)
                for policy_name, setting in [('ADC41', action), ('Query-soft', Action('Shrink', 1., 0.))]:
                    rr, alpha, _ = apply(hg, hs, a0, setting)
                    policies[name + '_' + policy_name][0][mask] = rr
                    policies[name + '_' + policy_name][1][mask] = alpha
                if name == 'balanced':
                    original = Action('ADC+Global', float(settings.loc[fold + 1, 'selected_beta']),
                                      float(settings.loc[fold + 1, 'selected_confidence_threshold']))
                    restricted = max((r for r in all_candidates if r['strength'] > 0), key=lambda r: (r['mrr'], -r['strength'], r['tau']))
                    require((restricted['strength'], restricted['tau']) == (original.strength, original.tau), 'Original 40-candidate choice differs')
                    rr, alpha, _ = apply(hg, hs, a0, original)
                    close(rr, old.loc[mask, 'rr_method'], 'Original ADC OOF RR replay mismatch')
                    close(alpha, old.loc[mask, 'alpha_applied'], 'Original ADC OOF action replay mismatch')
                    close(policies['balanced_Query-soft'][0][mask], old.loc[mask, 'rr_query_soft'], 'Original Query-soft replay mismatch')
                count0, count1 = np.bincount(y[non_tied], minlength=2)
                fit_info = dict(pair=pair, fold=fold + 1, learner=name, train_triples=len(train) // 6,
                                heldout_triples=len(held) // 6, fit_rows=int(non_tied.sum()), n0=int(count0), n1=int(count1),
                                class_weight0=float(non_tied.sum() / (2 * count0)) if name == 'balanced' else 1.,
                                class_weight1=float(non_tied.sum() / (2 * count1)) if name == 'balanced' else 1.,
                                random_state=state, n_iter=int(model[-1].n_iter_[0]))
                fits.append(fit_info)
                states.append(dict(**fit_info, fields=FIELDS, imputer=model[0].statistics_.tolist(),
                                   mean=model[1].mean_.tolist(), scale=model[1].scale_.tolist(),
                                   coefficients=model[-1].coef_.tolist(), intercept=model[-1].intercept_.tolist()))
            print(f'[DEV] {label} fold {fold+1}/5: replay and matched weighting fits passed', flush=True)
        policies['Global'] = [refs.copy(), anchors.copy()]
        target_mask = (base.rr_a != base.rr_b).to_numpy()
        y = (base.rr_a > base.rr_b).to_numpy().astype(int)
        for name, values in probs.items():
            require(np.isfinite(values).all(), 'Incomplete OOF preferences')
            stat, reliability = probability_metrics(y[target_mask], values[target_mask])
            scores.append(dict(pair=pair, label=label, learner=name, **stat))
            bins.extend(dict(pair=pair, learner=name, **r) for r in reliability)
        for name, (rr, alpha) in policies.items():
            require(np.isfinite(rr).all() and np.isfinite(alpha).all(), 'Incomplete held-out actions')
            policy_rows.append(dict(pair=pair, label=label, method=name, **metrics(rr, refs, alpha, anchors)))
            for seed in (1, 2, 3):
                for direction in ('head', 'tail'):
                    sub = ((base.seed == seed) & (base.direction == direction)).to_numpy()
                    cells.append(dict(pair=pair, method=name, seed=seed, direction=direction,
                                      **metrics(rr[sub], refs[sub], alpha[sub], anchors[sub])))
        checks.append(dict(pair=pair, observations=n, original_adc_and_query_soft_replayed=True,
                           original_grouping_and_anchors_replayed=True, preprocessing_equal=True))
    outputs = save_frames(dict(probability_summary=scores, reliability_bins=bins, dev_policies=policy_rows,
                               dev_policy_cells=cells, dev_choices=choices, dev_candidates=candidates, dev_fits=fits))
    path = OUT / 'fold_model_states.json'; write_json(path, states); outputs[path.relative_to(ROOT).as_posix()] = sha(path)
    require(len(fits) == 60 and len(candidates) == 2460 and sum(c['observations'] for c in checks) == 219564, 'DEV inventory incomplete')
    write_json(OUT / 'dev_complete.json', dict(status='winner_signal_dev_checks_passed', sources=inputs.sources, outputs=outputs,
        checks=checks, small_model_fits=60, balanced_reconstructions=30, new_unweighted_fits=30, final_models=0,
        candidate_evaluations=2460, test_opened=False, new_test_policy=False, sklearn_version=sklearn.__version__,
        max_iterations=max(f['n_iter'] for f in fits)))


def diagnose():
    receipt_path = OUT / 'dev_complete.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'winner_signal_dev_checks_passed' and not receipt['test_opened'], 'DEV phase incomplete')
    sources = {**receipt['sources'], **receipt['outputs']}
    for rel, value in sources.items():
        require(sha(ROOT / rel) == value, 'Changed DEV input/artifact: ' + rel)
    sources[receipt_path.relative_to(ROOT).as_posix()] = sha(receipt_path)
    inputs = Inputs('DIAGNOSTIC')
    contingency, summary, utilities, neighbor, checks = [], [], [], [], []
    for pair, label in PAIRS.items():
        folder = INPUT / pair
        for split in ('dev_oof', 'test'):
            if split == 'dev_oof':
                frame, old, settings = joined_oof(inputs, folder)
                anchor, applied, actual = old.alpha0.to_numpy(), old.alpha_applied.to_numpy(), old.rr_method.to_numpy()
                radius = old.fold.map(settings.selected_beta).to_numpy()
            else:
                frame = inputs.frame(folder / 'test_anchored/test_locked_query_rows.csv')
                check_frame(frame, 'test')
                lock = inputs.obj(folder / 'dev_lock/anchored_dev_lock.json')
                anchor, radius = np.full(len(frame), lock['alpha0']), np.full(len(frame), lock['beta'])
                close(anchor, frame.alpha0_locked, 'TEST anchor differs from lock')
                close(radius, frame.anchored_beta_locked, 'TEST radius differs from lock')
                applied, actual = frame.alpha_anchored_locked.to_numpy(), frame.rr_anchored_locked.to_numpy()
            d = direction_rows(frame[GRID_COLUMNS], anchor, radius, frame.rr_a, frame.rr_b, applied, actual)
            context = dict(pair=pair, label=label, split=split)
            for winner in ('A', 'B', 'tie'):
                part = d[d.winner == winner]
                counts = {key: int(part.opportunity.eq(key).sum()) for key in ('down_only', 'up_only', 'both', 'neither')}
                require(sum(counts.values()) == len(part), 'Contingency omits observations')
                contingency.append(dict(**context, winner=winner, n=len(part), **counts,
                                        suggested_available=int(part.suggested_available.sum())))
                for step in ('down_step', 'up_step', 'suggested_step'):
                    values = part[step].dropna().to_numpy()
                    neighbor.append(dict(**context, winner=winner, step=step, n=len(values),
                        mean_delta=float(values.mean()) if len(values) else None,
                        benefit_rate=float((values > TOL).mean()) if len(values) else None,
                        harm_rate=float((values < -TOL).mean()) if len(values) else None))
            for group in ('all', 'unchanged', 'tied_endpoints', 'winner_aligned', 'winner_opposed'):
                part = d if group == 'all' else d[d.alignment == group]
                utilities.append(dict(**context, alignment=group, **utility_summary(part, len(d))))
            eligible = (d.winner != 'tie') & d.any_gain & d.suggested_available
            misses = eligible & ~d.suggested_gain
            blocked = (d.winner != 'tie') & d.any_gain & ~d.suggested_available
            feasible = (d.winner != 'tie') & d.suggested_available
            aligned = d[d.alignment == 'winner_aligned']
            summary.append(dict(**context, n=len(d), opportunities=int(d.any_gain.sum()),
                feasible_suggestions=int(feasible.sum()), feasible_suggestions_without_gain=int((feasible & ~d.suggested_gain).sum()),
                eligible_opportunities=int(eligible.sum()), direction_misses=int(misses.sum()), blocked_opportunities=int(blocked.sum()),
                miss_rate=float(misses.sum() / eligible.sum()) if eligible.any() else None,
                suggested_profitable_but_step_nonpositive=int((feasible & d.suggested_gain & (d.suggested_step <= TOL)).sum()),
                aligned_n=len(aligned), aligned_harm_rate=float((aligned.actual_delta < -TOL).mean()) if len(aligned) else None,
                overall_utility=float(d.actual_delta.mean())))
            parts = [r for r in utilities if r['pair'] == pair and r['split'] == split and r['alignment'] != 'all']
            require(sum(r['n'] for r in parts) == len(d), 'Alignment partition incomplete')
            close(sum(r['utility_contribution'] for r in parts), d.actual_delta.mean(), 'Alignment utility decomposition failed')
            checks.append(dict(**context, observations=len(d), endpoint_anchor_action_rr_verified=True, partitions_verified=True))
            print(f'[DIAG] {label} {split}: {misses.sum()}/{eligible.sum()} feasible opportunity misses; {len(aligned)} aligned actions', flush=True)
    sources.update(inputs.sources)
    outputs = save_frames(dict(direction_contingency=contingency, direction_summary=summary,
                               alignment_utility=utilities, neighbor_step_utility=neighbor))
    require(len(checks) == 12 and sum(c['observations'] for c in checks) == 474732, 'Diagnostic coverage incomplete')
    audit = dict(status='winner_signal_checks_passed', failures=[], sources=sources, outputs=outputs, checks=checks,
                 small_model_fits=receipt['small_model_fits'], scorer_runs=0, new_test_policy=False,
                 test_used_for_selection=False, historical_results_replaced=False, calibrated_harm_probability=False,
                 diagnostic_opportunities_are_gold_aware=True, runtime=receipt['sklearn_version'])
    path = OUT / 'audit.json'; write_json(path, audit)
    sources.update(outputs); sources[path.relative_to(ROOT).as_posix()] = sha(path)
    write_json(ROOT / 'paper_a_draft/winner_signal_source_manifest.json', dict(version='winner_signal_review_v1', sources=sources))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['dev', 'diagnose'])
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        {'dev': dev, 'diagnose': diagnose}[parser.parse_args().phase]()
