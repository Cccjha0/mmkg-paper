"""Build all B15/B16 formal tables from verified small numerical summaries."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paper_a_conservative_radius import sha, require

OUT = ROOT / 'outputs/paper_a_safe_correction/ties_harm_review_v1'
TABLES = ROOT / 'paper_a_draft/tables/ties_harm'
LABELS = ('W-N', 'W-A', 'D-N', 'D-A', 'W-NA', 'D-NA')


def fmt(v, scale=1, digits=2, signed=False):
    return '--' if pd.isna(v) else format(v*scale, ('+' if signed else '')+f'.{digits}f')


def interval(r, key, scale=1, digits=3, signed=False):
    return (r'\shortstack{' + fmt(r[key], scale, digits, signed) + r'\\{\scriptsize ['
            + fmt(r[key+'_lo'], scale, digits, signed)+','+fmt(r[key+'_hi'], scale, digits, signed) + r']}}')


def table(rows, caption, label, columns, heading):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{'+caption+'}', r'\label{'+label+'}', r'\begin{tabular}{'+columns+'}',
        r'\toprule', heading+r'\\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def main():
    path = ROOT / 'paper_a_draft/ties_harm_source_manifest.json'
    manifest = json.loads(path.read_text())
    audit_path = OUT / 'audit.json'
    require(sha(audit_path) == manifest['sources'][audit_path.relative_to(ROOT).as_posix()], 'Changed audit')
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'ties_harm_checks_passed' and not audit['failures'], 'Unverified analysis')
    # Bind the historical adverse finding as an explicitly separate version.
    for rel in ('docs/reports/paper_a_confidence_harm_report.md',
                'outputs/paper_a_safe_correction/confidence_harm/confidence_harm_summary.csv',
                'docs/reports/paper_a_ties_harm_review_2026-09-12.md'):
        manifest['sources'][rel] = sha(ROOT / rel)

    def frame(name):
        p = OUT / (name+'.csv')
        require(sha(p) == manifest['sources'][p.relative_to(ROOT).as_posix()], 'Changed numbers')
        return pd.read_csv(p, float_precision='round_trip')

    ties = frame('tie_summary').query('population == "ties"').set_index(['label', 'split'])
    cells = frame('tie_cells').query('population == "ties"').set_index(['label', 'split', 'seed', 'direction'])
    harm = frame('harm_discrimination').set_index(['label', 'split', 'population'])
    rows, assets = [], {}
    for label in LABELS:
        for split, display in [('dev_oof', 'DEV'), ('test', 'TEST')]:
            r = ties.loc[label, split]
            rows.append(f'{label} & {display} & {int(r.n):,} & {fmt(r.fraction,100)} & {fmt(r.action_rate,100)} & '
                        f'{fmt(r.harm_rate,100)} & {fmt(r.benefit_rate,100)} & {fmt(r.utility,1e4,2,True)} & '
                        f'{fmt(r.mean_loss,1e4)} & {fmt(r.contribution,1e4,2,True)}'+r'\\')
    assets['ties_pooled.tex'] = table(rows,
        r'Endpoint RR ties under the original ADC policies. $n_T$ counts tied observations, $T$ is their percentage of all observations, and $I_T,H_T,B_T$ are weight-change, harm and benefit percentages within all ties, including unchanged ties. $U_T$ is mean ADC-minus-anchor RR within ties, $L_T$ their unconditional mean loss, and $C_T=(n_T/n)U_T$ their contribution to overall utility; these three columns use $10^{-4}$ RR units. DEV uses original grouped held-out policies, TEST original locks. Three seeds and both directions are pooled; equality is the exact endpoint RR criterion excluded from classifier fitting.',
        'tab:ties-pooled', 'llrrrrrrrr', r'Pair & Split & $n_T$ & $T$ (\%) & $I_T$ (\%) & $H_T$ (\%) & $B_T$ (\%) & $U_T$ & $L_T$ & $C_T$')
    for split, display in [('dev_oof', 'DEV'), ('test', 'TEST')]:
        rows = []
        for label in LABELS:
            for seed in (1, 2, 3):
                for direction in ('head', 'tail'):
                    r = cells.loc[label, split, seed, direction]
                    rows.append(f'{label} & {seed} & {direction[0].upper()} & {int(r.n):,} & {fmt(r.fraction,100)} & '
                                f'{fmt(r.action_rate,100)} & {fmt(r.harm_rate,100)} & {fmt(r.utility,1e4,2,True)} & {fmt(r.mean_loss,1e4)}'+r'\\')
        denominator = '4,276 and 7,922' if split == 'dev_oof' else '4,274 and 9,902'
        assets[f'ties_{split}.tex'] = table(rows,
            f'Complete {display} tie cells by pair, base seed and direction (H=head, T=tail). Each W/D cell contains {denominator} total observations, respectively. '
            r'$n_T$ is the tied count and $T$ its percentage of that cell. $I_T,H_T$ use $n_T$ as denominator; $U_T,L_T$ are net utility and unconditional loss within ties, in $10^{-4}$ RR units. All 36 cells are retained, including zero and adverse outcomes.',
            'tab:ties-'+split, 'llcrrrrrr', r'Pair & Seed & Dir. & $n_T$ & $T$ (\%) & $I_T$ (\%) & $H_T$ (\%) & $U_T$ & $L_T$')
        rows = []
        for label in LABELS:
            for name, short in [('all', 'All'), ('proposed', 'Prop.'), ('executed', 'Exec.')]:
                r = harm.loc[label, split, name]
                values = [interval(r, 'prevalence', 100, 2), interval(r, 'auroc'), interval(r, 'ap', 100, 2), interval(r, 'ap_lift', 100, 2, True)]
                rows.append(f'{label} & {short} & {int(r.n):,}/{int(r.harm_n):,} & ' + ' & '.join(values)+r'\\[3pt]')
        valid = harm.xs(split, level='split')
        excluded = sum(int((valid[key+'_valid_replicates'] != 2000).sum()) for key in ('prevalence', 'auroc', 'ap', 'ap_lift'))
        validity = ('All displayed metric intervals have 2,000 valid replicates. ' if excluded == 0 else
                    'Some intervals omit undefined empty/one-class bootstrap replicates; exact valid counts accompany the CSV. ')
        assets[f'harm_{split}.tex'] = table(rows,
            f'{display} harm discrimination by fixed score $1-c$. All=all valid ungated grid proposals; Prop.=their nonzero subset; Exec.=actual nonzero corrections after the original gate. '
            r'$n/h$ gives observations/harm events; $P_h$ is the same-subset prevalence and constant-score AP reference. The AUROC reference is 0.5. AP is non-interpolated average precision; lift is AP minus $P_h$ in percentage points. Brackets are conditional 95\% original-triple bootstrap intervals (2,000 draws), retaining seed/direction clusters before subset restriction. '
            + validity + 'These are descriptive discrimination intervals, not calibrated harm probabilities or newly selected deployment rules.',
            'tab:harm-'+split, 'llrrrrr', r'Pair & Set & $n/h$ & $P_h$ (\%) & AUROC & AP (\%) & Lift (pp)')
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest['tables'] = {}
    for name, value in assets.items():
        p = TABLES / name; p.write_text(value, encoding='utf-8')
        manifest['tables'][p.relative_to(ROOT / 'paper_a_draft').as_posix()] = sha(p)
    asset = dict(status='ties_harm_assets_verified', pooled_tie_rows=len(ties), seed_direction_tie_rows=len(cells),
                 harm_population_rows=len(harm), tables=manifest['tables'], test_used_for_selection=False)
    p = OUT / 'asset_audit.json'; p.write_text(json.dumps(asset, indent=2)+'\n', encoding='utf-8')
    manifest['sources'][p.relative_to(ROOT).as_posix()] = sha(p)
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(asset, indent=2))


if __name__ == '__main__':
    main()
