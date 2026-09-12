"""B17/B18: fixed-budget, query-consistent rejection and direct paired effects."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.matched_actions import Action, weights
from router.rejection_diagnostics import MatchedGate, observable_units, stratum_ids, paired_intervals, TOL
from scripts.analyze_paper_a_conservative_radius import INPUT, GRID_COLUMNS, sha, require, close
from scripts.analyze_paper_a_matched_alternatives import Inputs, check_frame
from scripts.analyze_paper_a_winner_signal import PAIRS, FIELDS, joined_oof

OUT = ROOT / 'outputs/paper_a_safe_correction/rejection_controls_v1'
METHODS = ('ADC', 'Random-I', 'Head-first', 'Tail-first', 'Low-A-gap', 'Gap-agree', 'Random-shape', 'No-gate')
CODE = ('scripts/analyze_paper_a_rejection_controls.py', 'router/rejection_diagnostics.py',
    'tests/test_rejection_diagnostics.py', 'docs/protocols/paper_a_rejection_controls.md',
    'scripts/analyze_paper_a_winner_signal.py', 'scripts/analyze_paper_a_matched_alternatives.py',
    'scripts/analyze_paper_a_conservative_radius.py', 'router/matched_actions.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def replay(inputs, folder, pair, split, states):
    if split == 'dev_oof':
        frame, old, settings = joined_oof(inputs, folder)
        anchor, alpha, rr, ref = [old[k].to_numpy() for k in ('alpha0', 'alpha_applied', 'rr_method', 'rr_global_crossfit')]
        folds = old.fold.to_numpy()
        raw, applied, fallback = [np.full(len(frame), np.nan) for _ in range(3)]
        thresholds = []
        for fold in range(1, 6):
            mask = folds == fold
            s = [s for s in states if s['pair'] == pair and s['fold'] == fold and s['learner'] == 'balanced']
            require(len(s) == 1 and s[0]['fields'] == list(FIELDS), 'Missing balanced fold state')
            s = s[0]
            train = ~mask
            require(int((frame.rr_a[train] != frame.rr_b[train]).sum()) == s['fit_rows'], 'Training tie exclusion differs')
            x = frame.loc[mask, list(FIELDS)].to_numpy()
            g = ((x-np.asarray(s['mean']))/np.asarray(s['scale'])) @ np.asarray(s['coefficients'])[0]+s['intercept'][0]
            require(np.isfinite(x).all() and np.isfinite(g).all(), 'Invalid OOF diagnostic scope')
            setting = settings.loc[fold]
            a0, beta, tau = float(setting.alpha0), float(setting.selected_beta), float(setting.selected_confidence_threshold)
            close(anchor[mask], a0, 'Fold anchor differs')
            signal = g, expit(g), np.zeros(mask.sum(), dtype=bool)
            raw[mask], _ = weights(*signal, a0, Action('ADC+Global', beta, 0.))
            applied[mask], fallback[mask] = weights(*signal, a0, Action('ADC+Global', beta, tau))
            if tau == 0:
                require(np.array_equal(raw[mask], applied[mask]), 'Zero gate not identical')
            thresholds.append(dict(fold=fold, anchor=a0, beta=beta, tau=tau))
        require(np.array_equal(fallback.astype(bool), old.fallback.to_numpy(dtype=bool)), 'OOF fallback differs')
    else:
        frame = inputs.frame(folder/'test_anchored/test_locked_query_rows.csv')
        check_frame(frame, 'test')
        lock = inputs.obj(folder/'dev_lock/anchored_dev_lock.json')
        anchor = np.full(len(frame), lock['alpha0'])
        alpha, rr, ref = [frame[k].to_numpy() for k in ('alpha_anchored_locked', 'rr_anchored_locked', 'rr_global')]
        g, p = frame.anchored_decision.to_numpy(), frame.anchored_probability_a.to_numpy()
        close(p, expit(g), 'TEST sigmoid differs')
        close(anchor, frame.alpha0_locked, 'TEST anchor differs')
        close(lock['beta'], frame.anchored_beta_locked, 'TEST radius differs')
        close(lock['confidence_threshold'], frame.anchored_confidence_threshold_locked, 'TEST gate differs')
        signal = g, p, np.zeros(len(frame), dtype=bool)
        raw, _ = weights(*signal, lock['alpha0'], Action('ADC+Global', lock['beta'], 0.))
        applied, fallback = weights(*signal, lock['alpha0'], Action('ADC+Global', lock['beta'], lock['confidence_threshold']))
        require(np.array_equal(fallback, frame.anchored_fallback.to_numpy(dtype=bool)), 'TEST fallback differs')
        folds = np.zeros(len(frame), dtype=int)
        thresholds = [dict(fold=0, anchor=lock['alpha0'], beta=lock['beta'], tau=lock['confidence_threshold'])]
    require(np.isfinite(frame[list(FIELDS)].to_numpy()).all(), 'Invalid feature scope')
    grid = frame[GRID_COLUMNS].to_numpy()
    close(alpha, applied, 'Original action differs')
    close(rr, grid[np.arange(len(frame)), np.rint(alpha*20).astype(int)], 'Original RR differs')
    close(ref, grid[np.arange(len(frame)), np.rint(anchor*20).astype(int)], 'Anchor RR differs')
    close(grid[:,0], frame.rr_b, 'B endpoint differs'); close(grid[:,-1], frame.rr_a, 'A endpoint differs')
    raw_rr = grid[np.arange(len(frame)), np.rint(raw*20).astype(int)]
    require(not np.any((abs(alpha-anchor) > TOL) & (abs(raw-anchor) <= TOL)), 'Active action not a proposal')
    close(np.where(abs(alpha-anchor) > TOL, raw_rr, ref), rr, 'Raw/active action RR differs')
    return frame, folds, anchor, alpha, rr, ref, raw, raw_rr, thresholds


def summary(delta, keep, movement, reference):
    return dict(n=len(delta), mrr=float(reference.mean()+np.mean(delta*keep)), utility=float(np.mean(delta*keep)),
        harm_rate=float(np.mean((delta < -TOL)*keep)), mean_loss=float(np.mean(np.maximum(-delta,0)*keep)),
        action_rate=float(np.mean(keep)), mean_movement=float(np.mean(movement*keep)))


def main():
    require(not (OUT/'audit.json').exists(), 'Completed review is immutable; use a new version')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = Inputs('DIAGNOSTIC')
    manifest_path = ROOT/'paper_a_draft/ties_harm_source_manifest.json'
    expected = json.loads(manifest_path.read_text())['sources']
    sources = {rel:sha(ROOT/rel) for rel in CODE}
    sources[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
    def old(rel):
        require(sha(ROOT/rel) == expected[rel], 'Changed B15/B16 evidence: '+rel)
        sources[rel] = sha(ROOT/rel)
        return ROOT/rel
    states = json.loads(old('outputs/paper_a_safe_correction/winner_signal_review_v1/fold_model_states.json').read_text())
    old_harm = pd.read_csv(old('outputs/paper_a_safe_correction/ties_harm_review_v1/harm_discrimination.csv'), float_precision='round_trip').set_index(['pair','split','population'])
    old_full = pd.read_csv(old('outputs/paper_a_safe_correction/ties_harm_review_v1/tie_summary.csv'), float_precision='round_trip').set_index(['pair','split','population'])
    results, fallback_rows, cells, random_rows, allocations, checks = [], [], [], [], [], []
    started = time.monotonic()
    for pi, (pair, label) in enumerate(PAIRS.items()):
        for si, split in enumerate(('dev_oof', 'test')):
            context = dict(pair=pair, label=label, split=split)
            frame, folds, anchor, alpha, rr, ref, raw, raw_rr, thresholds = replay(inputs, INPUT/pair, pair, split, states)
            raw_delta, full_delta, movement = raw_rr-ref, rr-ref, abs(raw-anchor)
            full_keep = abs(alpha-anchor) > TOL
            close(raw_delta.mean(), old_harm.loc[pair,split,'all']['utility'], 'Previous no-gate result differs')
            close(full_delta.mean(), old_full.loc[pair,split,'all']['utility'], 'Previous full result differs')
            known = np.where(frame.direction == 'tail', frame.head_id, frame.tail_id)
            units, inverse = observable_units(frame.seed, folds, frame.direction, frame.relation_id, known)
            # Changing the specified gold leaves the observable unit map intact.
            head, tail = frame.head_id.to_numpy(copy=True), frame.tail_id.to_numpy(copy=True)
            head[frame.direction == 'head'] = -999; tail[frame.direction == 'tail'] = -999
            other_units, other_inverse = observable_units(frame.seed, folds, frame.direction, frame.relation_id,
                                                         np.where(frame.direction == 'tail', head, tail))
            require(units.equals(other_units) and np.array_equal(inverse,other_inverse), 'Gold affects query grouping')
            first = np.full(len(units), len(frame)); np.minimum.at(first, inverse, np.arange(len(frame)))
            for values in (raw, alpha, anchor):
                close(values[first][inverse], values, 'Same observable query gets different action')
            eligible_units = np.flatnonzero(movement[first] > TOL)
            u = units.iloc[eligible_units].reset_index(drop=True)
            indices = first[eligible_units]
            keep = full_keep[indices]
            steps = np.rint((raw[indices]-anchor[indices])*20).astype(int)
            base = dict(seed=u.seed, fold=u.fold, repetitions=u.repetitions)
            rate = MatchedGate(stratum_ids(base), keep)
            shape_ids = stratum_ids(dict(**base, direction=u.direction, steps=steps))
            shape = MatchedGate(shape_ids, keep)
            seed = 20260927+2*pi+si
            tie_key = np.random.default_rng(seed).random(len(u))
            gap_a = frame.geometry_a_top1_top2_margin.to_numpy()[indices]/np.maximum(frame.geometry_a_score_std.to_numpy()[indices],1e-12)
            gap_b = frame.geometry_b_top1_top2_margin.to_numpy()[indices]/np.maximum(frame.geometry_b_score_std.to_numpy()[indices],1e-12)
            priorities = {'Head-first':(u.direction == 'head').to_numpy().astype(int),
                'Tail-first':(u.direction == 'tail').to_numpy().astype(int), 'Low-A-gap':-gap_a,
                'Gap-agree':np.sign(steps)*(gap_a-gap_b)}
            decisions = {'ADC':keep.astype(float), 'Random-I':rate.probabilities(),
                **{name:rate.ordered(values,tie_key).astype(float) for name,values in priorities.items()},
                'Random-shape':shape.probabilities(), 'No-gate':np.ones(len(u))}
            row_keep = {}
            target_count = int(u.repetitions.to_numpy() @ keep)
            for name, retained in decisions.items():
                if name != 'No-gate':
                    require(abs(u.repetitions.to_numpy() @ retained-target_count) < 1e-8, 'Unmatched actual intervention count')
                unit_prob = np.zeros(len(units)); unit_prob[eligible_units] = retained
                row_keep[name] = unit_prob[inverse]
            close(row_keep['ADC'], full_keep, 'Original query keep mask differs')
            close(np.mean(movement*row_keep['Random-shape']), abs(alpha-anchor).mean(), 'Unmatched expected mean movement')
            # Independently check realizations with integer count/step histograms.
            for name, gate in [('Random-I',rate),('Random-shape',shape)]:
                draw = gate.draw(np.random.default_rng(seed))
                require(int(u.repetitions.to_numpy() @ draw) == target_count, 'Random draw changes intervention count')
                if name == 'Random-shape':
                    require(np.array_equal(np.bincount(shape_ids,weights=u.repetitions.to_numpy()*draw),
                                           np.bincount(shape_ids,weights=u.repetitions.to_numpy()*keep)), 'Random draw changes signed/direction shape')
                unit_outcomes = np.column_stack([np.bincount(inverse,weights=x,minlength=len(units))[eligible_units]
                    for x in (raw_delta, (raw_delta < -TOL).astype(float), np.maximum(-raw_delta,0))])
                spread = gate.random_spread(unit_outcomes, len(frame), seed+(100 if name == 'Random-shape' else 0))
                r = dict(**context, method=name, **{k:spread[k] for k in ('partial_units','partial_strata','total_strata','replicates')},
                    eligible_units=len(u), seed=seed+(100 if name == 'Random-shape' else 0))
                for j, key in enumerate(('utility','harm_rate','mean_loss')):
                    r.update({key+'_expected':float(spread['expected'][j]),key+'_random_lo':float(spread['low'][j]),key+'_random_hi':float(spread['high'][j])})
                random_rows.append(r)
                close(r['utility_expected'], np.mean(raw_delta*row_keep[name]), 'Random expectation differs')
            clusters, _ = pd.factorize(pd.MultiIndex.from_frame(frame[['head_id','relation_id','tail_id']]),sort=True)
            differences = np.column_stack([full_delta-raw_delta*row_keep[name] for name in METHODS])
            point, low, high = paired_intervals(differences,clusters,20260929+2*pi+si)
            for j,name in enumerate(METHODS):
                results.append(dict(**context, method=name, **summary(raw_delta,row_keep[name],movement,ref),
                    full_minus_method=point[j],paired_lo=low[j],paired_hi=high[j],bootstrap_replicates=2000,
                    bootstrap_seed=20260929+2*pi+si,random_expectation=name.startswith('Random')))
                for base_seed in (1,2,3):
                    for direction in ('head','tail'):
                        m = ((frame.seed == base_seed)&(frame.direction == direction)).to_numpy()
                        cells.append(dict(**context,method=name,seed=base_seed,direction=direction,
                            **summary(raw_delta[m],row_keep[name][m],movement[m],ref[m]),full_minus_method=float(differences[m,j].mean())))
            j = METHODS.index('No-gate')
            fallback_rows.append(dict(**context,n=len(frame),tau_min=min(t['tau'] for t in thresholds),tau_max=max(t['tau'] for t in thresholds),
                full_utility=float(full_delta.mean()),no_gate_utility=float(raw_delta.mean()),paired_effect=point[j],paired_lo=low[j],paired_hi=high[j],
                full_interventions=int(full_keep.sum()),no_gate_interventions=int((movement>TOL).sum()),
                full_harm_rate=float((full_delta < -TOL).mean()),no_gate_harm_rate=float((raw_delta < -TOL).mean())))
            allocations.extend(dict(**context,**t) for t in thresholds)
            checks.append(dict(**context,observations=len(frame),observable_units=len(units),eligible_units=len(u),
                nonzero_full_units=int(keep.sum()),all_actions_and_rr_replayed=True,whole_query_decisions_verified=True,
                gold_replacement_invariant=True,intervention_counts_exact=True,shape_histograms_exact=True,
                prior_results_reconciled=True,no_gate_uses_original_radius=True))
            print(f'{label} {split}: {len(u):,} eligible query units; {target_count:,} matched interventions; all controls complete ({time.monotonic()-started:.1f}s)',flush=True)
    require(len(checks)==12 and sum(r['observations'] for r in checks)==474732, 'Incomplete inventory')
    require(len(results)==96 and len(cells)==576 and len(random_rows)==24 and len(fallback_rows)==12, 'Incomplete outputs')
    outputs={}
    for name,rows in [('matched_summary',results),('paired_fallback',fallback_rows),('seed_direction_cells',cells),
                      ('randomization_spread',random_rows),('original_settings',allocations)]:
        path=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(path,index=False);outputs[path.relative_to(ROOT).as_posix()]=sha(path)
    sources.update(inputs.sources)
    audit=dict(status='rejection_controls_checks_passed',failures=[],checks=checks,sources=sources,outputs=outputs,
        selector_fits=0,scorer_runs=0,test_used_for_selection=False,new_gate_selected=False,
        historical_results_replaced=False,controls_condition_on_observed_workload=True,all_random_draws_match_intervention_count=True,
        shape_random_draws_match_signed_direction_histogram=True,random_rows_are_exact_expectations=True,
        bootstrap_replicates=2000,random_gate_replicates=2000,elapsed_seconds=round(time.monotonic()-started,2))
    path=OUT/'audit.json';write_json(path,audit)
    sources.update(outputs);sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write_json(ROOT/'paper_a_draft/rejection_source_manifest.json',dict(version='rejection_controls_v1',sources=sources))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
