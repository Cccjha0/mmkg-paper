"""A03/B04/D08: DEV tradeoffs and matched TEST risks using audited ranking grids.

Only twenty 13-feature logistic fold models are reconstructed on CPU. No scorer,
GPU training, new TEST model selection, or large bootstrap is run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ablate_anchored_dynamic import feature_matrix, fit_geometry_model, model_outputs, FEATURE_GROUPS
from scripts.crossfit_anchored_dynamic import read_csv
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha

INPUT = ROOT/'outputs/paper_a_safe_correction/information_boundary_v2'
DYNA = ROOT/'outputs/paper_a_safe_correction/dynasemble_controls_v1_review'
OUT = ROOT/'outputs/paper_a_safe_correction/conservative_radius_review'
PAIRS = dict(mkgw_mhyper_native='W-N', mkgw_mhyper_adamf='W-A', db15k_mhyper_native='D-N', db15k_mhyper_adamf='D-A')
ALPHAS = np.arange(21, dtype=float)/20
BETAS = np.arange(21, dtype=float)/20
THRESHOLDS = (0., .1, .2, .3)
KEY = ['seed', 'direction', 'head_id', 'relation_id', 'tail_id']
GRID_COLUMNS = [f'rr_alpha_{a:.2f}'.replace('.', '_') for a in ALPHAS]
TOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right, message):
    require(np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True), message)


def project(values, anchor):
    """Exact nearest-grid lexicographic tie rule; vectorized adjacent candidates."""
    x = np.asarray(values, dtype=float)
    hi = np.searchsorted(ALPHAS, x).clip(0, len(ALPHAS)-1)
    lo = (hi-1).clip(0)
    dl, dh = np.abs(ALPHAS[lo]-x), np.abs(ALPHAS[hi]-x)
    use_hi = (dh < dl) | ((dh == dl) & (np.abs(ALPHAS[hi]-anchor) < np.abs(ALPHAS[lo]-anchor)))
    return np.where(use_hi, hi, lo)


def policy(grid, decision, probability, nonfinite, anchor, beta, tau):
    fallback = nonfinite | (np.abs(2*probability-1) < tau)
    continuous = np.where(fallback, anchor, np.clip(anchor+beta*np.tanh(decision), 0, 1))
    idx = project(continuous, anchor)
    alpha = ALPHAS[idx]
    require(np.max(np.abs(alpha-anchor)) <= beta+TOL, 'Action radius exceeded on aligned grid')
    return grid[np.arange(len(grid)), idx], alpha, fallback


def metrics(rr, reference, alpha=None, anchor=None):
    rr, reference = np.asarray(rr), np.asarray(reference)
    delta = rr-reference
    bad, good = delta < -TOL, delta > TOL
    loss, gain = np.maximum(-delta, 0).mean(), np.maximum(delta, 0).mean()
    close(delta.mean(), gain-loss, 'Loss/gain decomposition failed')
    return dict(n=len(rr), mrr=float(rr.mean()), global_mrr=float(reference.mean()), delta_mrr=float(delta.mean()),
        harm_rate=float(bad.mean()), benefit_rate=float(good.mean()), rank_change_rate=float((bad | good).mean()),
        conditional_loss=float(-delta[bad].mean()) if bad.any() else None,
        mean_loss=float(loss), mean_gain=float(gain),
        action_rate=float((np.abs(alpha-anchor) > TOL).mean()) if alpha is not None else None,
        mean_abs_alpha_deviation=float(np.abs(alpha-anchor).mean()) if alpha is not None else None)


def nondominated(frame):
    """Empirical utility/loss frontier; equal points are both retained."""
    utility, loss = frame.delta_mrr.to_numpy(), frame.mean_loss.to_numpy()
    return np.asarray([not np.any((utility >= u-TOL) & (loss <= l+TOL) & ((utility > u+TOL) | (loss < l-TOL)))
                       for u, l in zip(utility, loss)])


def choose(candidates, cap):
    valid = [r for r in candidates if 0 < r['beta'] <= cap+TOL]
    return max(valid, key=lambda r: (r['mrr'], -r['beta'], r['tau']))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT/'paper_a_draft/rerun_source_manifest.json').read_text())['sources']
    expected.update(json.loads((ROOT/'outputs/paper_a_safe_correction/dynasemble_review_audit/audit.json').read_text())['sources'])
    sources = {}
    def bind(path, existing=True):
        rel = path.relative_to(ROOT).as_posix()
        value = sha(path)
        if existing:
            require(rel in expected and expected[rel] == value, 'Unverified input: '+rel)
        sources[rel] = value
    def csv(path):
        bind(path)
        return pd.read_csv(path, float_precision='round_trip')
    def obj(path):
        bind(path)
        return json.loads(path.read_text(encoding='utf-8'))
    bind(ROOT/'docs/protocols/paper_a_conservative_radius_review.md', False)
    bind(Path(__file__), False)
    for p in ('scripts/ablate_anchored_dynamic.py', 'scripts/crossfit_anchored_dynamic.py', 'scripts/crossfit_heterogeneous_dev_policies.py'):
        bind(ROOT/p, False)
    dev_curves, fold_selections, dev_family, test_table, test_cells, checks = [], [], [], [], [], []
    for pair in PAIRS:
        folder = INPUT/pair
        lock = obj(folder/'dev_lock/anchored_dev_lock.json')
        dev = csv(folder/'dev_lock/dev_locked_query_rows.csv')
        require(set(dev.split) == {'dev'} and set(dev.score_information_contract) == {'unfiltered_features_and_normalization_v2'}, 'Wrong DEV scope')
        grid = dev[GRID_COLUMNS].to_numpy()
        anchor, locked_beta, tau = lock['alpha0'], lock['beta'], lock['confidence_threshold']
        nonfinite = ~np.isfinite(dev[list(FEATURE_GROUPS['full_geometry'])].to_numpy()).all(1)
        nonfinite |= ~np.isfinite(dev.anchored_decision) | ~np.isfinite(dev.anchored_probability_a)
        rr0 = grid[:, int(round(anchor*20))]
        r, a, fb = policy(grid, dev.anchored_decision.to_numpy(), dev.anchored_probability_a.to_numpy(), nonfinite, anchor, locked_beta, tau)
        close(r, dev.rr_anchored_locked, 'Locked DEV reconstruction differs')
        close(a, dev.alpha_anchored_locked, 'Locked DEV action differs')
        candidates = []
        for beta in BETAS:
            for threshold in THRESHOLDS:
                rr, alpha, fallback = policy(grid, dev.anchored_decision.to_numpy(), dev.anchored_probability_a.to_numpy(), nonfinite, anchor, beta, threshold)
                row = dict(pair=pair, scope='full_dev_resubstitution', beta=float(beta), tau=threshold,
                    at_locked_tau=threshold == tau, locked_configuration=beta == locked_beta and threshold == tau,
                    **metrics(rr, rr0, alpha, anchor))
                dev_curves.append(row); candidates.append(row)
        selected = choose(candidates, .5)
        require((selected['beta'], selected['tau']) == (locked_beta, tau), 'Full DEV selection differs from immutable lock')

        # Reconstruct only the original small fold classifier, and gate all new
        # diagnostics on reproduction of the old held-out predictions.
        path = folder/'baseline_crossfit/dev_crossfit_query_rows.csv'
        bind(path)
        rows = read_csv(path)
        require({r['split'] for r in rows} == {'dev'}, 'TEST row in DEV reconstruction')
        assignment, _ = assign_grouped_folds(rows, 5, 20260901)
        previous = csv(folder/'p3_ablation/dev_p3_selected_query_rows.csv').set_index('query_id')
        prev_folds = csv(folder/'p3_ablation/dev_p3_selected_by_fold.csv').set_index('fold')
        heldout_parts = {key: [] for key in ['cap_0.50', 'cap_1.00', 'Global', 'Query-soft']}
        curve_parts = {float(beta): [] for beta in BETAS}
        for fold in range(5):
            train = [r for r in rows if assignment[triple_key(r)] != fold]
            held = [r for r in rows if assignment[triple_key(r)] == fold]
            a0, _ = best_alpha(train, tuple(ALPHAS))
            close(a0, prev_folds.loc[fold+1, 'alpha0'], 'Fold anchor differs')
            model, tx, tn = fit_geometry_model(train, fields=tuple(FEATURE_GROUPS['full_geometry']), random_state=20260902+fold*10+2)
            hx, hn = feature_matrix(held, tuple(FEATURE_GROUPS['full_geometry']))
            td, tp = model_outputs(model, tx); hd, hp = model_outputs(model, hx)
            tg = np.array([[float(r[c]) for c in GRID_COLUMNS] for r in train])
            hg = np.array([[float(r[c]) for c in GRID_COLUMNS] for r in held])
            hg0 = hg[:, int(round(a0*20))]
            train_candidates = []
            for beta in BETAS[1:]:
                for threshold in THRESHOLDS:
                    rr, alpha, fallback = policy(tg, td, tp, tn, a0, beta, threshold)
                    train_candidates.append(dict(beta=float(beta), tau=threshold, mrr=float(rr.mean())))
            small, wide = choose(train_candidates, .5), choose(train_candidates, 1.)
            require((small['beta'], small['tau']) == (prev_folds.loc[fold+1,'selected_beta'], prev_folds.loc[fold+1,'selected_confidence_threshold']), 'Fold restricted selection mismatch')
            idx = [r['query_id'] for r in held]
            old = previous.loc[idx]
            require((old.fold == fold+1).all(), 'Original-triple grouping mismatch')
            rr, alpha, fb = policy(hg, hd, hp, hn, a0, small['beta'], small['tau'])
            close(rr, old.rr_method, 'Fold held-out RR differs; cannot replace historical policy')
            close(alpha, old.alpha_applied, 'Fold held-out action differs')
            require(np.array_equal(fb, old.fallback.astype(bool)), 'Fold fallback differs')
            close(float(rr.mean()), prev_folds.loc[fold+1,'selected_heldout_mrr'], 'Fold MRR differs')
            for family, setting in [('cap_0.50',small),('cap_1.00',wide)]:
                r, a, f = policy(hg, hd, hp, hn, a0, setting['beta'], setting['tau'])
                heldout_parts[family].append((r,hg0,a,np.full(len(a),a0)))
                fold_selections.append(dict(pair=pair,fold=fold+1,family=family,beta=setting['beta'],tau=setting['tau'],
                    train_mrr=setting['mrr'],alpha0=a0,train_triples=len(train)//6,heldout_triples=len(held)//6,
                    **metrics(r,hg0,a,a0)))
            soft_idx = project(np.where(hn,a0,hp),a0)
            soft_rr = hg[np.arange(len(hg)),soft_idx]
            close(soft_rr,old.rr_query_soft,'Held-out Query-soft differs')
            heldout_parts['Global'].append((hg0,hg0,np.full(len(hg0),a0),np.full(len(hg0),a0)))
            heldout_parts['Query-soft'].append((soft_rr,hg0,ALPHAS[soft_idx],np.full(len(hg0),a0)))
            for beta in BETAS:
                r,a,f = policy(hg,hd,hp,hn,a0,beta,small['tau'])
                curve_parts[float(beta)].append((r,hg0,a,np.full(len(a),a0)))
        def pooled(parts):
            return metrics(*[np.concatenate(v) for v in zip(*parts)])
        for family, parts in heldout_parts.items():
            dev_family.append(dict(pair=pair, method=family, **pooled(parts)))
        for beta,parts in curve_parts.items():
            dev_curves.append(dict(pair=pair, scope='grouped_oof_fixed_fold_tau',beta=beta,tau=None,
                at_locked_tau=False,locked_configuration=False,**pooled(parts)))
        checks.append(dict(pair=pair,locked_beta=locked_beta,locked_tau=tau,full_dev_reproduced=True,
            five_fold_actions_rr_fallback_reproduced=True,dev_labels_only=True))
        print('[DEV VERIFIED] '+pair, flush=True)

    # Every new radius/threshold choice above is DEV-only and complete before
    # this code opens TEST. Wider-family fitted policies are never applied to TEST.
    for pair in PAIRS:
        folder = INPUT/pair
        lock = obj(folder/'dev_lock/anchored_dev_lock.json')
        frame = csv(folder/'test_anchored/test_locked_query_rows.csv')
        require(set(frame.split) == {'test'}, 'Wrong TEST split')
        require(not frame.duplicated(KEY).any(), 'Duplicate TEST keys')
        anchor = lock['alpha0']
        grid = frame[GRID_COLUMNS].to_numpy()
        reference = frame.rr_global.to_numpy()
        parts = {'Global': (reference,np.full(len(frame),anchor)),
            'Relation': (frame.rr_relation.to_numpy(),frame.alpha_relation.to_numpy()),
            'Query-soft': (frame.rr_query_soft_locked.to_numpy(),frame.alpha_query_soft_locked.to_numpy()),
            'ADC': (frame.rr_anchored_locked.to_numpy(),frame.alpha_anchored_locked.to_numpy())}
        nonfinite = ~np.isfinite(frame[list(FEATURE_GROUPS['full_geometry'])].to_numpy()).all(1)
        nonfinite |= ~np.isfinite(frame.anchored_decision) | ~np.isfinite(frame.anchored_probability_a)
        for label,beta,tau in [('ADC',lock['beta'],lock['confidence_threshold']),('beta=1',1.,lock['confidence_threshold']),('tau=0',lock['beta'],0.)]:
            rr,alpha,fallback = policy(grid,frame.anchored_decision.to_numpy(),frame.anchored_probability_a.to_numpy(),nonfinite,anchor,beta,tau)
            if label == 'ADC':
                close(rr,frame.rr_anchored_locked,'TEST ADC reconstruction differs')
                close(alpha,frame.alpha_anchored_locked,'TEST ADC action differs')
            parts[label] = rr,alpha
        for label,(rr,alpha) in parts.items():
            test_table.append(dict(pair=pair,method=label,alpha_coordinate='zscore_primary',**metrics(rr,reference,alpha,anchor)))
            for seed,direction in [(s,d) for s in (1,2,3) for d in ('head','tail')]:
                mask = ((frame.seed == seed)&(frame.direction == direction)).to_numpy()
                test_cells.append(dict(pair=pair,method=label,seed=seed,selector_seed=-1,direction=direction,
                    **metrics(rr[mask],reference[mask],alpha[mask],anchor)))
        for method in ('R0_historical','R1_source','R2_init','R3_softplus_fixed','R3_softplus','R4_swap'):
            variants = []
            if method == 'R0_historical':
                f=csv(folder/'dynasemble/test_query_rows.csv').rename(columns={'rr_dynasemble':'rr_method'})
                f['selector_seed']=f.seed
                variants=[f]
            else:
                for bs in (1,2,3):
                    for ss in (11,23,37):
                        f=csv(DYNA/pair/'test'/f'{method}_b{bs}_s{ss}.csv').rename(columns={'base_seed':'seed'})
                        variants.append(f)
            f=pd.concat(variants,ignore_index=True)
            require(not f.duplicated(KEY+['selector_seed']).any(),'Duplicate Dyna observations')
            # Reconcile every reference explicitly, including the historical R0 column.
            f=f.drop(columns=['rr_global'],errors='ignore').merge(frame[KEY+['rr_global']],on=KEY,validate='many_to_one')
            expected_rows=len(frame)*(1 if method=='R0_historical' else 3)
            require(len(f)==expected_rows,'Missing Dyna observations')
            test_table.append(dict(pair=pair,method=method,alpha_coordinate='minmax_not_comparable',**metrics(f.rr_method.to_numpy(),f.rr_global.to_numpy())))
            for (seed,ss,direction),g in f.groupby(['seed','selector_seed','direction']):
                test_cells.append(dict(pair=pair,method=method,seed=seed,selector_seed=ss,direction=direction,
                    **metrics(g.rr_method.to_numpy(),g.rr_global.to_numpy())))
        print('[TEST COMPARED] '+pair,flush=True)
    curves=pd.DataFrame(dev_curves)
    curves['empirical_nondominated_utility_loss']=False
    for _,indices in curves.groupby(['pair','scope']).groups.items():
        curves.loc[indices,'empirical_nondominated_utility_loss']=nondominated(curves.loc[indices])
    tables={'dev_radius_grid.csv':curves,'dev_family_comparison.csv':pd.DataFrame(dev_family),
        'dev_fold_selections.csv':pd.DataFrame(fold_selections),'test_fair_comparison.csv':pd.DataFrame(test_table),
        'test_seed_direction.csv':pd.DataFrame(test_cells)}
    # Existing reported MRR/harm are constraints on this analysis, not new estimates.
    known_path=ROOT/'outputs/paper_a_safe_correction/dynasemble_review_audit/summary.csv'
    known_audit=json.loads((known_path.parent/'audit.json').read_text())
    require(sha(known_path)==known_audit['output_hashes']['summary.csv'],'Dyna summary hash mismatch')
    bind(known_path,False)
    known=pd.read_csv(known_path).set_index(['pair','method'])
    frozen=csv(ROOT/'outputs/paper_a_safe_correction/information_boundary_rerun_audit/frozen_ablation.csv').set_index(['pair','method'])
    for r in test_table:
        if (r['pair'],r['method']) in known.index:
            k=known.loc[(r['pair'],r['method'])]
            close([r['mrr'],r['harm_rate'],r['conditional_loss'] if r['conditional_loss'] is not None else np.nan],
                  [k.mrr,k.harmful_query_rate,k.mean_harm],'Existing TEST summary differs')
        if r['method'] in ('ADC','beta=1','tau=0'):
            k=frozen.loc[(r['pair'],{'ADC':'full','beta=1':'no_bound','tau=0':'no_fallback'}[r['method']])]
            close([r['delta_mrr'],r['harm_rate'],r['conditional_loss']],
                  [k.delta_mrr_vs_global,k.harmful_query_rate,k.mean_harm],'Frozen TEST ablation differs')
    for name,frame in tables.items():
        frame.to_csv(OUT/name,index=False)
    manifest=dict(status='conservative_radius_checks_passed',failures=[],test_used_for_new_selection=False,
        scope='problem-driven DEV family analysis and frozen TEST comparison; no new deployed policy',
        definitions=dict(mean_loss='E[max(-DeltaRR,0)]',rank_change_rate='answer-rank change, not full-order intervention',
                         action_rate='same z-score coefficient coordinate only; unavailable for min-max DynaSemble'),
        runtime=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__),
        grids=dict(beta=BETAS.tolist(),tau=list(THRESHOLDS),original_cap=.5,diagnostic_cap=1.),
        pair_checks=checks,sources=sources,outputs={name:sha(OUT/name) for name in tables},
        limitations=['DEV full-fit surface is resubstitution; fold selection uses training portions and evaluated held-out portions.',
          'Pointwise descriptive risks, no risk guarantee or new confidence intervals.',
          'The 0.50 cap is a historical design restriction, not a calibrated risk threshold.',
          'Min-max and z-score coefficient distances cannot be compared directly.'])
    (OUT/'audit.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'fold_models':20,'test_methods':len(test_table),'dev_grid_points':len(curves)}))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
