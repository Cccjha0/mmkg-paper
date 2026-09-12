"""B15/B16 fixed-policy cache analysis: no model fit, scorer run, or selection."""
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
from router.ties_harm_diagnostics import RankedBinary, action_populations, cluster_intervals, tie_summary, TOL
from scripts.analyze_paper_a_conservative_radius import INPUT, GRID_COLUMNS, sha, require, close
from scripts.analyze_paper_a_matched_alternatives import Inputs, check_frame
from scripts.analyze_paper_a_winner_signal import PAIRS, FIELDS, joined_oof

OUT = ROOT / 'outputs/paper_a_safe_correction/ties_harm_review_v1'
PREVIOUS = ROOT / 'outputs/paper_a_safe_correction/winner_signal_review_v1'
CODE = ('scripts/analyze_paper_a_ties_harm.py', 'router/ties_harm_diagnostics.py',
        'tests/test_ties_harm_diagnostics.py', 'docs/protocols/paper_a_ties_harm_review.md',
        'scripts/analyze_paper_a_winner_signal.py', 'scripts/analyze_paper_a_matched_alternatives.py',
        'scripts/analyze_paper_a_conservative_radius.py', 'router/matched_actions.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    require(not (OUT / 'audit.json').exists(), 'Completed review is immutable; use a new version')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = Inputs('DIAGNOSTIC')
    previous_manifest = ROOT / 'paper_a_draft/winner_signal_source_manifest.json'
    expected = json.loads(previous_manifest.read_text())['sources']
    sources = {rel: sha(ROOT / rel) for rel in CODE}
    sources[previous_manifest.relative_to(ROOT).as_posix()] = sha(previous_manifest)

    def previous(name):
        path = PREVIOUS / name
        rel = path.relative_to(ROOT).as_posix()
        require(sha(path) == expected[rel], 'Changed previous diagnostic: '+rel)
        sources[rel] = sha(path)
        return path

    states = json.loads(previous('fold_model_states.json').read_text())
    counts = pd.read_csv(previous('direction_contingency.csv')).set_index(['pair', 'split', 'winner'])
    utilities = pd.read_csv(previous('alignment_utility.csv'), float_precision='round_trip').set_index(['pair', 'split', 'alignment'])
    ties, cells, discrimination, checks = [], [], [], []
    started = time.monotonic()
    for pair_index, (pair, label) in enumerate(PAIRS.items()):
        for split_index, split in enumerate(('dev_oof', 'test')):
            folder = INPUT / pair
            if split == 'dev_oof':
                frame, old, settings = joined_oof(inputs, folder)
                anchor, applied, actual = old.alpha0.to_numpy(), old.alpha_applied.to_numpy(), old.rr_method.to_numpy()
                reference, fallback = old.rr_global_crossfit.to_numpy(), old.fallback.to_numpy(dtype=bool)
                g, p, raw_alpha, replay, replay_fallback = [np.full(len(frame), np.nan) for _ in range(5)]
                for fold in range(1, 6):
                    select = old.fold.to_numpy() == fold
                    state = [s for s in states if s['pair'] == pair and s['fold'] == fold and s['learner'] == 'balanced']
                    require(len(state) == 1, 'Missing or duplicate fold state')
                    state = state[0]
                    require(state['fields'] == list(FIELDS), 'Feature order changed')
                    x = frame.loc[select, list(FIELDS)].to_numpy()
                    require(np.isfinite(x).all(), 'Unexpected invalid OOF features; do not impute into this diagnostic')
                    z = (x-np.asarray(state['mean']))/np.asarray(state['scale'])
                    g[select] = z @ np.asarray(state['coefficients'])[0] + state['intercept'][0]
                    p[select] = expit(g[select])
                    setting = settings.loc[fold]
                    a0, beta, tau = float(setting.alpha0), float(setting.selected_beta), float(setting.selected_confidence_threshold)
                    close(anchor[select], a0, 'OOF anchor mismatch')
                    signals = g[select], p[select], np.zeros(select.sum(), dtype=bool)
                    raw_alpha[select], _ = weights(*signals, a0, Action('ADC+Global', beta, 0.))
                    replay[select], replay_fallback[select] = weights(*signals, a0, Action('ADC+Global', beta, tau))
                    soft, _ = weights(*signals, a0, Action('Shrink', 1., 0.))
                    grid = frame.loc[select, GRID_COLUMNS].to_numpy()
                    close(grid[np.arange(len(grid)), np.rint(soft*20).astype(int)], old.loc[select, 'rr_query_soft'], 'OOF Query-soft RR mismatch')
            else:
                frame = inputs.frame(folder / 'test_anchored/test_locked_query_rows.csv')
                check_frame(frame, 'test')
                lock = inputs.obj(folder / 'dev_lock/anchored_dev_lock.json')
                anchor = np.full(len(frame), lock['alpha0'])
                applied, actual = frame.alpha_anchored_locked.to_numpy(), frame.rr_anchored_locked.to_numpy()
                g, p = frame.anchored_decision.to_numpy(), frame.anchored_probability_a.to_numpy()
                close(p, expit(g), 'TEST preference/margin mismatch')
                close(anchor, frame.alpha0_locked, 'TEST anchor mismatch')
                close(lock['beta'], frame.anchored_beta_locked, 'TEST radius mismatch')
                close(lock['confidence_threshold'], frame.anchored_confidence_threshold_locked, 'TEST gate mismatch')
                signals = g, p, np.zeros(len(frame), dtype=bool)
                raw_alpha, _ = weights(*signals, lock['alpha0'], Action('ADC+Global', lock['beta'], 0.))
                replay, replay_fallback = weights(*signals, lock['alpha0'], Action('ADC+Global', lock['beta'], lock['confidence_threshold']))
                reference = frame.rr_global.to_numpy()
                fallback = frame.anchored_fallback.to_numpy(dtype=bool)
            grid = frame[GRID_COLUMNS].to_numpy()
            require(np.isfinite(frame[list(FIELDS)].to_numpy()).all() and np.isfinite(g).all() and np.isfinite(p).all(), 'Nonfinite diagnostic scope')
            close(applied, replay, 'Recorded policy weight replay mismatch')
            require(np.array_equal(fallback, replay_fallback.astype(bool)), 'Fallback replay mismatch')
            close(grid[:, 0], frame.rr_b, 'Endpoint B differs')
            close(grid[:, -1], frame.rr_a, 'Endpoint A differs')
            close(grid[np.arange(len(frame)), np.rint(anchor*20).astype(int)], reference, 'Reference RR differs')
            close(grid[np.arange(len(frame)), np.rint(applied*20).astype(int)], actual, 'Policy RR differs')
            raw_rr = grid[np.arange(len(frame)), np.rint(raw_alpha*20).astype(int)]
            tied = (frame.rr_a == frame.rr_b).to_numpy()
            require(np.array_equal(tied, frame.rank_a == frame.rank_b), 'Rank/RR tie definitions differ')
            require(int(tied.sum()) == int(counts.loc[pair, split, 'tie']['n']), 'Previous tie count differs')
            context = dict(pair=pair, label=label, split=split)
            summaries = {}
            for name, population in [('all', None), ('ties', True), ('non_ties', False)]:
                r = tie_summary(tied, applied, anchor, actual, reference, population)
                summaries[name] = r
                ties.append(dict(**context, population=name, **r))
                for seed in (1, 2, 3):
                    for direction in ('head', 'tail'):
                        m = ((frame.seed == seed) & (frame.direction == direction)).to_numpy()
                        cells.append(dict(**context, seed=seed, direction=direction, population=name,
                            **tie_summary(tied[m], applied[m], anchor[m], actual[m], reference[m], population)))
            require(summaries['ties']['n']+summaries['non_ties']['n'] == len(frame), 'Tie partition incomplete')
            close(summaries['ties']['contribution']+summaries['non_ties']['contribution'], summaries['all']['utility'], 'Tie utility decomposition differs')
            close(summaries['all']['utility'], utilities.loc[pair, split, 'all']['utility'], 'Previous overall utility differs')
            close(summaries['ties']['contribution'], utilities.loc[pair, split, 'tied_endpoints']['utility_contribution'], 'Previous active-tie utility contribution differs')
            populations = action_populations(np.ones(len(frame), dtype=bool), raw_alpha, applied, anchor, raw_rr, actual)
            clusters, cluster_keys = pd.factorize(pd.MultiIndex.from_frame(frame[['head_id', 'relation_id', 'tail_id']]), sort=True)
            k = len(cluster_keys)
            require(np.array_equal(np.bincount(clusters), np.full(k, 6)), 'Incomplete triple clusters')
            confidence = np.abs(2*p-1)
            evaluators, meta = {}, {}
            for name, mask in populations.items():
                delta = (actual if name == 'executed' else raw_rr)-reference
                harm = delta < -TOL
                evaluators[name] = RankedBinary(harm[mask], 1-confidence[mask], clusters[mask])
                meta[name] = dict(n=int(mask.sum()), harm_n=int(harm[mask].sum()), nonharm_n=int((~harm[mask]).sum()),
                    total_n=len(frame), fraction=float(mask.mean()), full_triples=k,
                    subset_triples=int(len(np.unique(clusters[mask]))), harmful_triples=int(len(np.unique(clusters[mask & harm]))),
                    noharm_triples=int(len(np.unique(clusters[mask & ~harm]))),
                    mean_confidence=float(confidence[mask].mean()) if mask.any() else np.nan,
                    utility=float(delta[mask].mean()) if mask.any() else np.nan)
            require(meta['executed']['harm_n'] == summaries['all']['harm_n'], 'Inactive deployed rows have harm')
            require(meta['all']['harm_n'] == meta['proposed']['harm_n'], 'Inactive raw proposals have harm')
            intervals = cluster_intervals(evaluators, k, replicates=2000, seed=20260915+2*pair_index+split_index)
            for name in populations:
                discrimination.append(dict(**context, population=name, **meta[name], **intervals[name],
                    bootstrap_replicates=2000, bootstrap_seed=20260915+2*pair_index+split_index))
            checks.append(dict(**context, observations=len(frame), original_policy_and_fallback_replayed=True,
                grid_endpoints_and_rr_replayed=True, tie_partition_and_previous_results_reconciled=True,
                raw_executed_subset_verified=True, full_six_observation_clusters_verified=True, invalid_rows=0))
            print(f'{label} {split}: {len(frame):,} rows replayed; 3 populations x 2,000 cluster replicates complete ({time.monotonic()-started:.1f}s)', flush=True)
    require(len(checks) == 12 and sum(c['observations'] for c in checks) == 474732, 'Incomplete review')
    require(len(ties) == 36 and len(cells) == 216 and len(discrimination) == 36, 'Incomplete result inventory')
    outputs = {}
    for name, rows in [('tie_summary', ties), ('tie_cells', cells), ('harm_discrimination', discrimination)]:
        path = OUT / (name+'.csv'); pd.DataFrame(rows).to_csv(path, index=False)
        outputs[path.relative_to(ROOT).as_posix()] = sha(path)
    sources.update(inputs.sources)
    audit = dict(status='ties_harm_checks_passed', failures=[], checks=checks, sources=sources, outputs=outputs,
        selector_fits=0, scorer_runs=0, new_policy_selection=False, test_used_for_selection=False,
        historical_results_replaced=False, calibrated_harm_probability=False,
        diagnostic_ties_are_gold_aware=True, full_triple_cluster_bootstrap=True,
        bootstrap_replicates=2000, split_cells=12, seed_direction_cells=72, population_cells=36,
        elapsed_seconds=round(time.monotonic()-started, 2))
    path = OUT / 'audit.json'; write_json(path, audit)
    sources.update(outputs); sources[path.relative_to(ROOT).as_posix()] = sha(path)
    write_json(ROOT / 'paper_a_draft/ties_harm_source_manifest.json', dict(version='ties_harm_review_v1', sources=sources))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
