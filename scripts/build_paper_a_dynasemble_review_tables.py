"""Generate manuscript tables from the reviewed D01-D03 summaries, without fitting."""
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_paper_a_dynasemble_review import digest

AUDIT = ROOT/'outputs/paper_a_safe_correction/dynasemble_review_audit'
OUT = ROOT/'paper_a_draft/tables/dynasemble'
PAIRS = dict(mkgw_mhyper_native='W-N', mkgw_mhyper_adamf='W-A', db15k_mhyper_native='D-N', db15k_mhyper_adamf='D-A')


def table(name, caption, columns, header, rows, size='small'):
    text = ('\\begin{table}[htbp]\n\\centering\\'+size+'\n\\setlength{\\tabcolsep}{4pt}\n'
            '\\caption{'+caption+'}\n\\label{tab:dyna-'+name+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
            +' & '.join(header)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+'\\\\' for row in rows)
            +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (OUT/(name+'.tex')).write_text(text, encoding='utf-8')


def main():
    audit = json.loads((AUDIT/'audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'review_artifact_checks_passed' and not audit['failures']
    for file, sha in audit['output_hashes'].items():
        assert digest(AUDIT/file) == sha, file
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(AUDIT/'summary.csv').set_index(['pair', 'method'])
    old_path = ROOT/'outputs/paper_a_safe_correction/information_boundary_rerun_audit/method_summary.csv'
    old = pd.read_csv(old_path).set_index(['pair', 'split', 'method'])
    diagnostics = pd.read_csv(AUDIT/'test_diagnostics.csv')
    rows = []
    for p, label in PAIRS.items():
        vals = [summary.loc[(p, 'Global')].mrr, old.loc[(p, 'test', 'Relation')].mrr,
                *[summary.loc[(p, m)].mrr for m in ('Query-soft', 'R0_historical', 'R3_softplus', 'ADC')]]
        rows.append([label, *[f'{v:.6f}' for v in vals]])
    table('main', r'TEST MRR with both historical (R0) and healthy DEV-selected (R3) DynaSemble transfers. R0 retains all three paired runs; R3 averages three base seeds $\times$ three selector seeds. Other methods use three base seeds. W: MKG-W; D: DB15K; N: M-Hyper + NativE; A: M-Hyper + AdaMF-MAT.',
          'lrrrrrr', ['Pair', 'Global', 'Relation', 'Query-soft', 'R0', 'R3', 'ADC'], rows, 'footnotesize')
    rows = []
    for p, label in PAIRS.items():
        r = summary.loc[(p, 'R3_softplus')]
        lo, hi = ast.literal_eval(r.delta_mrr_vs_adc_ci95)
        rows.append([label, f'{r.delta_mrr_vs_adc:+.6f}', f'[{lo:+.6f}, {hi:+.6f}]',
                     f'{100*r.harmful_query_rate:.3f}', f'{100*summary.loc[(p,"ADC")].harmful_query_rate:.3f}'])
    table('effects', r'Healthy DynaSemble R3 versus ADC. Intervals are 10,000 original-triple bootstrap resamples conditional on fitted seeds, averaging selector repetitions before resampling. Harm is relative to Global and uses realized selector observations.',
          'lr lrr', ['Pair', r'$\Delta$MRR (R3$-$ADC)', r'95\% interval', r'R3 harm (\%)', r'ADC harm (\%)'], rows)
    variants = [('R0_historical', 'R0 historical'), ('R1_source', 'R1 source core'), ('R2_init', 'R2 initialization'),
                ('R3_softplus_fixed', 'R3 fixed Softplus'), ('R3_softplus', 'R3 DEV-selected'), ('R4_swap', 'R4 role swap')]
    table('variants', r'All predeclared DynaSemble variants, TEST MRR. No seed or orientation is selected using TEST. R1/R2/R3-fixed use $5\times10^{-5}$ and one epoch; R4 inherits the R3 budget.',
          'lrrrr', ['Variant', *PAIRS.values()], [[title, *[f'{summary.loc[(p,v)].mrr:.6f}' for p in PAIRS]] for v, title in variants])
    rows = []
    for p in audit['pairs']:
        d = diagnostics[(diagnostics.pair == p['pair']) & (diagnostics.variant == 'R3_softplus')]
        s = p['selection']
        lr = {1e-5:r'$10^{-5}$', 5e-5:r'$5\times10^{-5}$', 1e-4:r'$10^{-4}$'}[s['learning_rate']]
        rows.append([PAIRS[p['pair']], lr, str(s['epochs']), f'{s["heldout_dev_mrr"]:.6f}',
                     f'{d.weight_std.min():.4f}--{d.weight_std.max():.4f}', f'{d.primary_ratio_std.mean():.4f}'])
    table('selection', r'R3 DEV-selected settings and TEST variation. Weight SD is the range over nine selectors, each measured across its queries; ratio SD is the mean of the nine within-selector SDs. Selected CV MRR is a selection score, not an unbiased estimate after model selection. All 180 final fits and 324 CV trajectories have finite parameters and positive parameter updates in every layer in every epoch.',
          'llrrlr', ['Pair', 'LR', 'Epochs', 'CV MRR', 'Weight SD range', 'Ratio SD'], rows)
    rows = []
    for p, label in PAIRS.items():
        a = diagnostics[(diagnostics.pair == p) & (diagnostics.variant == 'R3_softplus')]
        b = diagnostics[(diagnostics.pair == p) & (diagnostics.variant == 'R4_swap')]
        rows.append([label, f'{a.primary_ratio_mean.mean():.4f}', f'{b.primary_ratio_mean.mean():.4f}',
            f'{100*a.gold_rank_matches_primary.mean():.2f}', f'{100*b.gold_rank_matches_primary.mean():.2f}',
            f'{summary.loc[(p,"R4_swap")].mrr-summary.loc[(p,"R3_softplus")].mrr:+.6f}'])
    table('roles', r'Role-swap diagnostic. Primary ratios are $w/(1+w)$ for R3 and $1/(1+w)$ for R4. Gold-rank equality compares filtered gold ranks with the raw primary; it does not imply complete-order equality. R4$-$R3 differences are descriptive, with no new orientation selection.',
          'lrrrrr', ['Pair', 'R3 ratio', 'R4 ratio', r'R3 equal (\%)', r'R4 equal (\%)', r'R4$-$R3 MRR'], rows)
    manifest = dict(version='dynasemble_controls_v1_review', sources={old_path.relative_to(ROOT).as_posix(): digest(old_path)}, tables={})
    manifest['sources'].update(audit['sources'])
    for path in sorted(AUDIT.glob('*')):
        if path.is_file():
            manifest['sources'][path.relative_to(ROOT).as_posix()] = digest(path)
    for path in sorted(OUT.glob('*.tex')):
        manifest['tables'][path.relative_to(ROOT/'paper_a_draft').as_posix()] = digest(path)
    (ROOT/'paper_a_draft/dynasemble_source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'tables': len(manifest['tables']), 'sources': len(manifest['sources'])}))


if __name__ == '__main__':
    main()
