"""Build source-bound manuscript tables for the matched alternatives review."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paper_a_matched_alternatives import OUT as DATA, PAIRS, ALL_PAIRS, INPUT, FAMILIES, sha

OUT = ROOT/'paper_a_draft/tables/matched'
LABELS = {'ADC+Global': 'ADC + Global', 'ADC-original': 'ADC original', 'Query-soft': 'Query-soft',
          'Global': 'Global', **{f: f for f in FAMILIES[1:]}}


def table(name, caption, columns, header, rows):
    text = ('\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n'
            '\\caption{'+caption+'}\n\\label{tab:matched-'+name+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
            +' & '.join(header)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+'\\\\' if row else '\\midrule' for row in rows)
            +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (OUT/(name+'.tex')).write_text(text, encoding='utf-8')


def main():
    audit = json.loads((DATA/'test_audit.json').read_text(encoding='utf-8'))
    lock = json.loads((DATA/'dev_locks.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'matched_alternative_checks_passed' and not audit['failures']
    assert not audit['test_used_for_selection'] and audit['dev_lock_sha256'] == sha(DATA/'dev_locks.json')
    for rel, expected in audit['sources'].items():
        assert sha(ROOT/rel) == expected, rel
    for source in (audit['outputs'], audit['query_outputs'], lock['outputs']):
        for name, expected in source.items():
            assert sha(DATA/name) == expected, name
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(DATA/'dev_oof_summary.csv').set_index(['pair', 'method'])
    test = pd.read_csv(DATA/'test_summary.csv').set_index(['pair', 'method'])
    choices = pd.read_csv(DATA/'dev_choices.csv')
    full = choices[choices.fold == 0].set_index(['pair', 'family'])
    static = pd.read_csv(DATA/'static_summary.csv').set_index(['pair', 'method'])
    paired = pd.read_csv(DATA/'test_paired_differences.csv').set_index(['pair', 'comparator'])
    methods = ('Global', 'Query-soft', 'ADC-original', *FAMILIES)
    for name, frame, caption in (
        ('dev', dev, r'Grouped held-out DEV MRR with shared fitted preference objects. Each new family selects from 41 configurations on the outer training portion, including explicit Global. Full-DEV selection scores are excluded here.'),
        ('test', test, r'TEST MRR for matched action maps. Six new families share the same frozen preference object, features, anchor and ranking grid, with 41 DEV candidate configurations each. All four pairs are locked before TEST application. ADC + Global admits a zero-action candidate; its final choices match original ADC. These new comparisons are descriptive point estimates.')):
        rows = [[LABELS[m], *[f'{frame.loc[(p,m)].mrr:.6f}' for p in PAIRS]] for m in methods]
        table(name, caption, 'lrrrr', ['Method', *PAIRS.values()], rows)
    rows = [[f, *[f'{paired.loc[(p,f)].delta_mrr_adc_minus_comparator:+.6f}' for p in PAIRS]] for f in FAMILIES[1:]]
    table('paired', r'Paired TEST MRR difference: ADC + Global minus each alternative. Positive values favor ADC. No family is selected or omitted using TEST; full seed/direction differences accompany these point estimates, without new significance claims.',
          'lrrrr', ['Comparator', *PAIRS.values()], rows)
    rows = []
    for family in FAMILIES:
        values = []
        for pair in PAIRS:
            r = full.loc[(pair,family)]
            values.append(f'{r.strength:.3f}, {r.tau:.1f} [{int(r.distinct_action_vectors)}]')
        rows.append([LABELS[family], *values])
    table('locks', r'Full-DEV action locks: strength, threshold [number of distinct applied-weight vectors among 41 tested configurations]. Strength zero denotes explicit Global. Strength means $\lambda$ for shrinkage, $\gamma$ for Linear-g and $\beta$ otherwise. Distinct-vector counts expose configurations that coincide after grid projection; all candidate scores and all 120 fold selections are supplied.',
          'lrrrr', ['Family', *PAIRS.values()], rows)
    rows = []
    for pair, label in PAIRS.items():
        for method in ('Global', *FAMILIES):
            r = test.loc[(pair,method)]
            rows.append([label, LABELS[method], f'{r.delta_mrr:+.6f}', f'{r.mean_loss:.6f}', f'{100*r.harm_rate:.3f}',
                         '--' if pd.isna(r.conditional_loss) else f'{r.conditional_loss:.5f}',
                         f'{100*r.action_rate:.2f}', f'{r.mean_abs_alpha_deviation:.4f}'])
        if label != 'D-A': rows.append([])
    table('risk', r'Matched TEST utility, loss and action measures relative to Global. Definitions are those of Section~\ref{sec:fair-risk}: $U$ is net utility, $L_-$ unconditional loss, $H$ harm frequency, $C$ conditional loss, $I_\alpha$ weight-change frequency and $D_\alpha$ mean displacement. All displayed maps use the same z-score coordinate. Global has undefined $C$.',
          'llrrrrrr', ['Pair','Method',r'$U$',r'$L_-$',r'$H$ (\%)',r'$C$',r'$I_\alpha$ (\%)',r'$D_\alpha$'], rows)
    static_labels = {'Primary': 'Primary', 'Secondary': 'Secondary', 'Equal-z': 'Equal z-score',
                     'RRF': 'Equal RRF', 'Global': 'Global', 'Relation': 'Relation'}
    table('static', r'Complete deployable static comparisons, TEST MRR. Equal z-score fixes $\alpha=0.5$; equal RRF fixes $k=60$ and constructs candidate ranks before answer filtering. Global and Relation use their original DEV locks. NA denotes NativE + AdaMF-MAT, with NativE primary. Supplementary CSVs include Hits@1/3/10 and every seed/direction.',
          'lrrrrrr', ['Method', *ALL_PAIRS.values()], [[title, *[f'{static.loc[(p,m)].mrr:.6f}' for p in ALL_PAIRS]] for m,title in static_labels.items()])
    table('oracle', r'Answer-aware expert-selection diagnostic, reported separately from deployable methods. Oracle is the per-observation maximum of the two standalone RRs. It is neither deployable nor an upper bound on score fusion: an interior mixture can rank the gold above both experts. The final column is only the observed gap to Global.',
          'lrr', ['Pair', 'Expert oracle MRR', r'Oracle$-$Global'],
          [[label, f'{static.loc[(p,"Expert-oracle")].mrr:.6f}', f'{static.loc[(p,"Expert-oracle")].delta_mrr:+.6f}'] for p,label in ALL_PAIRS.items()])
    relation_counts = {}
    for pair in ALL_PAIRS:
        selection = json.loads((INPUT/pair/'full_ranking/selection.json').read_text())
        relation_counts[pair] = sum(r['source'] == 'relation_dev' for r in selection['relation_details'].values())
    budget = dict(shared_features=13, preference_model='same median imputer + scaler + balanced liblinear logistic C=1',
                  max_iter=2000, full_model_n_iter={p:lock['pairs'][p]['classifier_n_iter'] for p in PAIRS},
                  reconstructed_fold_models=20, shared_global_grid_points=21,
                  new_family_configurations={f:41 for f in FAMILIES}, family_specific_refits=0,
                  original_adc_configurations=40, query_soft_configurations=1,
                  relation_eligible_counts=relation_counts, relation_grid_points_per_eligible_relation=21,
                  dynasemble_r3=dict(learning_rates=3, checkpoints=4, folds=3, base_selector_combinations=9,
                                     cv_trajectories_per_pair=81, epoch_cap=10),
                  stopping='logistic convergence or max_iter; no family-specific early stopping; Dyna fixed epochs/checkpoints')
    write = DATA/'budget.json'
    write.write_text(json.dumps(budget, indent=2)+'\n', encoding='utf-8')
    budget_rows = [
        ['Fixed scorers / Equal / RRF', 'None', 'Fixed parameters; no search'],
        ['Global', '21 weights', 'Max DEV MRR; shared by all maps'],
        ['Relation', r'$21R$ weights', r'Per-relation DEV MRR; support $\geq60$'],
        ['Query-soft', '1 map', 'Same fitted logistic; no action search'],
        ['Original ADC', r'$10\times4=40$', 'Strength / gate; no explicit Global'],
        ['Each new family', '41 maps', '40 active + Global; same-object DEV MRR'],
        ['R0 / R1 / R2 / R3-fixed', '1 setting each', r'One epoch; LR $5\times10^{-5}$'],
        ['R3', r'$3\times4=12$ settings', 'Grouped DEV CV; 81 trajectories/pair'],
        ['R4', 'Inherited R3 setting', 'No orientation-specific retuning']]
    table('budget', r'Combiner search budgets, excluding frozen base-model training. $R$ is the count of eligible relations. The six matched families share the 13-feature logistic fit and 21-point anchor search; they perform no extra learner tuning or early stopping. Logistic convergence uses max\_iter=2000; R3 records all checkpoints through epoch 10 and selects by DEV. Dyna comparisons differ in inputs and learners and are not mapping-only controls.',
          'lll', ['Method', 'Action / training search', 'Selection and stopping'], budget_rows)
    manifest = dict(version='matched_alternatives_v1', sources=dict(audit['sources']), tables={})
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    for p in DATA.glob('*'):
        if p.is_file(): manifest['sources'][p.relative_to(ROOT).as_posix()] = sha(p)
    for rel, value in audit['query_outputs'].items():
        manifest['sources'][(DATA/rel).relative_to(ROOT).as_posix()] = value
    for p in (DATA/'provenance').glob('*'):
        if p.is_file(): manifest['sources'][p.relative_to(ROOT).as_posix()] = sha(p)
    for p in OUT.glob('*.tex'):
        manifest['tables'][p.relative_to(ROOT/'paper_a_draft').as_posix()] = sha(p)
    (ROOT/'paper_a_draft/matched_source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'tables': len(manifest['tables']), 'sources': len(manifest['sources'])}))


if __name__ == '__main__':
    main()
