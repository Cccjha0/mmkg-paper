"""B05/B06: display-only relabeling and frozen-weight coupling checks; no fitting."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paper_a_conservative_radius import ALPHAS, PAIRS, project
from scripts.ablate_anchored_dynamic import FEATURE_GROUPS

OUT = ROOT / 'outputs/paper_a_safe_correction/action_semantics_review_v1'
TABLES = ROOT / 'paper_a_draft/tables/action_semantics'
FIGURES = ROOT / 'paper_a_draft/figures/action_semantics'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def proposal_floor(beta, tau):
    """Unclipped |beta*tanh(g)| floor among finite rows with c >= tau."""
    return beta * 2 * tau / (1 + tau * tau)


def check_weights(frame, anchor, beta, tau):
    g = frame.anchored_decision.to_numpy()
    p = frame.anchored_probability_a.to_numpy()
    assert np.isfinite(g).all() and np.isfinite(p).all()
    c = np.abs(2 * p - 1)
    identity_error = float(np.max(np.abs(np.abs(np.tanh(g)) - 2 * c / (1 + c * c))))
    np.testing.assert_allclose(c, np.abs(np.tanh(g / 2)), rtol=0, atol=1e-12)
    assert identity_error < 1e-12
    np.testing.assert_allclose(c, frame.anchored_confidence, rtol=0, atol=1e-12)
    nonfinite = ~np.isfinite(frame[list(FEATURE_GROUPS['full_geometry'])].to_numpy()).all(axis=1)
    fallback = nonfinite | (c < tau)
    np.testing.assert_array_equal(fallback, frame.anchored_fallback.astype(str).str.lower().isin(['true', '1']))
    proposal = beta * np.tanh(g)
    retained = ~fallback
    assert np.all(np.abs(proposal[retained]) >= proposal_floor(beta, tau) - 1e-12)
    raw = np.clip(anchor + proposal, 0, 1)
    continuous = np.where(fallback, anchor, raw)
    alpha = ALPHAS[project(continuous, anchor)]
    np.testing.assert_allclose(continuous, frame.alpha_anchored_continuous, rtol=0, atol=1e-12)
    np.testing.assert_allclose(alpha, frame.alpha_anchored_locked, rtol=0, atol=1e-12)
    displacement = np.abs(alpha - anchor)
    changed = displacement > 1e-12
    return dict(rows=len(frame), anchor=anchor, beta=beta, tau=tau,
                unclipped_retained_floor=proposal_floor(beta, tau), identity_max_error=identity_error,
                retained=int(retained.sum()), changed=int(changed.sum()),
                retained_zero_grid_change=int((retained & ~changed).sum()),
                observed_min_nonzero_grid_displacement=float(displacement[changed].min()) if changed.any() else None,
                observed_nonzero_grid_displacements=[float(x) for x in np.unique(displacement[changed])],
                confidence_and_full_weights_replayed=True)


def draw_fair_plot(data):
    # Same values, scales, markers and inset membership as the bound original.
    labels = {'Global': 'Global', 'Query-soft': 'Query-soft', 'ADC': 'ADC',
              'beta=1': r'Expanded-radius ($\beta=1$)', 'R3_softplus': 'Dyna R3'}
    colors = {'Global': '#303030', 'Query-soft': '#d57c19', 'ADC': '#1764ab',
              'beta=1': '#b64036', 'R3_softplus': '#8055a1'}
    markers = {'Global': 's', 'Query-soft': '^', 'ADC': 'o', 'beta=1': 'D', 'R3_softplus': 'P'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.grid': True, 'grid.alpha': .18})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    for ax, (pair, label) in zip(axes.flat, PAIRS.items()):
        for method, color in colors.items():
            row = data.loc[pair, method]
            ax.scatter(row.mean_loss, row.delta_mrr, s=35 + 150 * row.rank_change_rate,
                       color=color, marker=markers[method], edgecolor='white', linewidth=.5,
                       label=labels[method], zorder=3)
        ax.axhline(0, color='#707070', linewidth=.6)
        ax.set(title=label, xlabel='Unconditional mean RR loss', ylabel='Net RR utility vs Global')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        ax.margins(.15)
        if label != 'W-N':
            detail = ax.inset_axes([.13, .13, .50, .43])
            for method in ('Global', 'ADC', 'beta=1'):
                row = data.loc[pair, method]
                detail.scatter(row.mean_loss, row.delta_mrr, s=20 + 85 * row.rank_change_rate,
                               color=colors[method], marker=markers[method], zorder=3)
            detail.axhline(0, color='#707070', linewidth=.5)
            detail.margins(.25)
            detail.tick_params(labelsize=6)
            detail.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
            detail.xaxis.offsetText.set_fontsize(6)
            detail.yaxis.offsetText.set_fontsize(6)
            detail.set_title('Three-policy detail', fontsize=7, pad=3)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside upper center', ncol=5, frameon=False, fontsize=8)
    fig.savefig(FIGURES / 'test_utility_loss.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGURES / 'test_utility_loss.png', dpi=180)
    plt.close(fig)


def main():
    for path in (OUT, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    sources = {}

    def bind(path, expected=None):
        rel = path.relative_to(ROOT).as_posix()
        value = sha(path)
        if expected is not None:
            assert expected.get(rel) == value, rel
        sources[rel] = value
        return path

    rerun = json.loads(bind(ROOT / 'paper_a_draft/rerun_source_manifest.json').read_text())
    conservative = json.loads(bind(ROOT / 'paper_a_draft/conservative_source_manifest.json').read_text())
    expected = {**rerun['sources'], **conservative['sources']}
    checks = []
    columns = ['anchored_decision', 'anchored_probability_a', 'anchored_confidence', 'anchored_fallback',
               'alpha_anchored_continuous', 'alpha_anchored_locked', 'seed', 'direction',
               *FEATURE_GROUPS['full_geometry']]
    for pair, label in PAIRS.items():
        folder = ROOT / 'outputs/paper_a_safe_correction/information_boundary_v2' / pair
        lock = json.loads(bind(folder / 'dev_lock/anchored_dev_lock.json', expected).read_text())
        for split, filename in [('dev', 'dev_lock/dev_locked_query_rows.csv'),
                                ('test', 'test_anchored/test_locked_query_rows.csv')]:
            frame = pd.read_csv(bind(folder / filename, expected), usecols=columns, float_precision='round_trip')
            assert set(zip(frame.seed, frame.direction)) == {(s, d) for s in (1, 2, 3) for d in ('head', 'tail')}
            result = check_weights(frame, lock['alpha0'], lock['beta'], lock['confidence_threshold'])
            checks.append(dict(pair=pair, label=label, split=split, **result))
            if label == 'D-A':
                assert (lock['alpha0'], lock['beta'], lock['confidence_threshold']) == (1., .5, .3)
                assert abs(result['observed_min_nonzero_grid_displacement'] - .30) < 1e-12

    # Produce new display assets; original result tables/manifests remain frozen.
    for name, original, manifest in [('ablation', 'rerun/ablation.tex', rerun),
                                     ('fair', 'conservative/fair.tex', conservative)]:
        path = ROOT / 'paper_a_draft/tables' / original
        key = path.name if name == 'ablation' else path.relative_to(ROOT / 'paper_a_draft').as_posix()
        assert sha(path) == manifest['tables'][key]
        bind(path)
        text = path.read_text(encoding='utf-8')
        assert text.count(r'& $\beta=1$ &') == 4
        text = text.replace(r'& $\beta=1$ &', '& Expanded-radius &')
        definition = (r'Expanded-radius fixes $\beta=1$ while retaining the fitted signal, anchor, '
                      r'threshold, tanh, $[0,1]$ clipping and grid. ')
        text = text.replace(r'\caption{', r'\caption{' + definition, 1)
        (TABLES / (name + '.tex')).write_text(text, encoding='utf-8')
    risk_path = ROOT / 'outputs/paper_a_safe_correction/conservative_radius_review/test_fair_comparison.csv'
    risk = pd.read_csv(bind(risk_path, expected)).set_index(['pair', 'method'])
    draw_fair_plot(risk)
    for rel in ('scripts/build_paper_a_action_semantics.py', 'tests/test_action_semantics.py',
                'scripts/analyze_paper_a_conservative_radius.py', 'scripts/ablate_anchored_dynamic.py'):
        bind(ROOT / rel)
    audit = dict(status='action_semantics_checks_passed', checks=checks,
                 definition='Expanded-radius: beta=1, same signal/anchor/tau/tanh/clipping/grid/nonfinite fallback.',
                 coupling='c=abs(tanh(g/2)); abs(tanh(g))=2c/(1+c*c)',
                 sources=sources, base_model_runs=0, selector_fits=0, new_policy_selection=False,
                 historical_metrics_replaced=False, test_used_for_selection=False,
                 interpretation='Observed displacement floors verify a mechanical consequence; they are not independent evidence of calibrated confidence or safe ranking.')
    path = OUT / 'audit.json'
    path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    sources = {**sources, path.relative_to(ROOT).as_posix(): sha(path)}
    manifest = dict(version='action_semantics_review_v1', sources=sources,
                    tables={p.relative_to(ROOT / 'paper_a_draft').as_posix(): sha(p) for p in TABLES.glob('*.tex')},
                    figures={p.relative_to(ROOT / 'paper_a_draft').as_posix(): sha(p) for p in FIGURES.glob('*.pdf')})
    (ROOT / 'paper_a_draft/action_semantics_source_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
