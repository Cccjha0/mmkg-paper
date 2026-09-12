"""Build source-bound B13/B14 tables without fitting or selecting any policy."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paper_a_conservative_radius import sha, require

OUT = ROOT / 'outputs/paper_a_safe_correction/winner_signal_review_v1'
TABLES = ROOT / 'paper_a_draft/tables/winner_signal'
LABELS = ('W-N', 'W-A', 'D-N', 'D-A', 'W-NA', 'D-NA')


def table(rows, caption, label, columns, heading):
    heading = heading.replace('%', chr(92) + '%')
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
        r'\caption{' + caption + '}', r'\label{' + label + '}', r'\begin{tabular}{' + columns + '}',
        r'\toprule', heading + r'\\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def fmt(value, scale=1, digits=2, signed=False):
    if pd.isna(value):
        return '--'
    return format(scale * value, ('+' if signed else '') + f'.{digits}f')


def main():
    manifest_path = ROOT / 'paper_a_draft/winner_signal_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('audit.json', 'dev_complete.json'):
        path = OUT / name
        require(sha(path) == manifest['sources'][path.relative_to(ROOT).as_posix()], 'Changed receipt')
    audit = json.loads((OUT / 'audit.json').read_text())
    dev = json.loads((OUT / 'dev_complete.json').read_text())
    require(audit['status'] == 'winner_signal_checks_passed' and not audit['failures'], 'Unverified diagnostics')

    def frame(name):
        path = OUT / (name + '.csv')
        require(sha(path) == manifest['sources'][path.relative_to(ROOT).as_posix()], 'Changed numerical source')
        return pd.read_csv(path, float_precision='round_trip')

    counts = frame('direction_contingency')
    directions = frame('direction_summary')
    utility = frame('alignment_utility').set_index(['label', 'split', 'alignment'])
    probabilities = frame('probability_summary')
    policies = frame('dev_policies').set_index(['label', 'method'])
    rows = []
    for label in LABELS:
        for split, split_label in [('dev_oof', 'DEV'), ('test', 'TEST')]:
            sub = counts[(counts.label == label) & (counts.split == split)].set_index('winner')
            for winner in ('A', 'B', 'tie'):
                r = sub.loc[winner]
                cells = [fmt(r[key] / r.n, 100) if r.n else '--' for key in ('down_only', 'up_only', 'both', 'neither')]
                rows.append(f'{label} & {split_label} & {winner} & {int(r.n):,} & ' + ' & '.join(cells) + r'\\')
    contingency = table(rows,
        r'Endpoint winner versus profitable correction directions. Each row reports percentages within its winner class, including ties. Down/up mean lower/higher primary weight. A side is profitable if some grid point within the original fixed radius improves the anchor RR by more than $10^{-12}$. Both and neither are retained; an unavailable direction cannot be profitable. DEV uses grouped held-out anchors/radii; TEST uses the original locks. These gold-aware opportunities are diagnostic, not selector inputs or policies.',
        'tab:winner-contingency', 'lllrrrrr', r'Pair & Split & Winner & $n$ & Down only & Up only & Both & Neither')
    rows = []
    for label in LABELS:
        for split, split_label in [('dev_oof', 'DEV'), ('test', 'TEST')]:
            r = directions[(directions.label == label) & (directions.split == split)].iloc[0]
            u = utility.loc[label, split, 'winner_aligned']
            rows.append(f'{label} & {split_label} & {int(r.eligible_opportunities):,} & {fmt(r.miss_rate,100)} & '
                f'{int(r.blocked_opportunities):,} & {int(r.aligned_n):,} & {fmt(r.aligned_harm_rate,100)} & '
                f'{fmt(u.conditional_loss,1,4)} & {fmt(u.utility,1e4,2,True)}' + r'\\')
    alignment = table(rows,
        r'Direction and magnitude mismatch. $E$ counts non-tied observations with at least one profitable side and a feasible winner-suggested side; miss is the percentage of $E$ for which only the opposite side is profitable. Block counts opportunity observations whose suggested side is unavailable. $A$ counts actual nonzero ADC actions aligned with the endpoint winner; $H_A$ and $C_A$ are their harm percentage and conditional mean RR loss. $U_A$ is their mean ADC-minus-anchor RR in $10^{-4}$ units. Empty groups are undefined, not zero. Blocked suggestions are excluded from the miss denominator; actual aligned actions can still harm through their chosen magnitude.',
        'tab:winner-alignment', 'llrrrrrrr', r'Pair & Split & $E$ & Miss (%) & Block & $A$ & $H_A$ (%) & $C_A$ & $U_A$')
    rows = []
    for label in LABELS:
        for learner, display in [('balanced', 'Balanced'), ('unweighted', 'Unweighted'), ('train_prior', 'Prior')]:
            r = probabilities[(probabilities.label == label) & (probabilities.learner == learner)].iloc[0]
            if learner == 'train_prior':
                outcomes = ['--'] * 4
            else:
                adc, qs = policies.loc[label, learner + '_ADC41'], policies.loc[label, learner + '_Query-soft']
                outcomes = [fmt(adc.delta_mrr,1e4,2,True), fmt(adc.mean_loss,1e4,2), fmt(adc.action_rate,100), fmt(qs.delta_mrr,1e4,2,True)]
            rows.append(f'{label} & {display} & {fmt(r.brier,1,4)} & {fmt(r.ece10,100)} & {fmt(r.auc,1,3)} & '
                        + ' & '.join(outcomes) + r'\\')
    weighting = table(rows,
        r'Class-weight control on grouped held-out DEV. Balanced and unweighted use the same 13 features, non-tied training rows, preprocessing, $C=1$, folds and random states: five fits per learner per pair, with no final control model or new TEST application. Brier, 10-bin ECE and AUC use naturally weighted non-tied held-out winner labels; Prior is the training-fold winner frequency, a probability-only baseline. Lower Brier is not calibration alone, and ECE is bin-dependent. Policy outcomes use all held-out observations: $U$ is ADC+Global utility, $L_-$ its unconditional mean loss, $I$ its weight-change percentage, and $U_Q$ Query-soft utility. Utility/loss use $10^{-4}$ RR units relative to the common fold Global. Each learner gets the same 41 training-portion ADC candidates; Query-soft uses its same fitted object. All outcomes are retained, without selecting a weighting scheme from TEST.',
        'tab:winner-weighting', 'llrrrrrrr', r'Pair & Fit & Brier & ECE (%) & AUC & $U$ & $L_-$ & $I$ (%) & $U_Q$')
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest['tables'] = {}
    for name, value in [('contingency.tex', contingency), ('alignment.tex', alignment), ('weighting.tex', weighting)]:
        path = TABLES / name; path.write_text(value, encoding='utf-8')
        manifest['tables'][path.relative_to(ROOT / 'paper_a_draft').as_posix()] = sha(path)
    asset = dict(status='winner_signal_assets_verified', contingency_rows=len(counts), diagnostic_cells=len(directions),
                 probability_rows=len(probabilities), policy_rows=len(policies), small_fits=dev['small_model_fits'],
                 tables=manifest['tables'], test_used_for_selection=False, historical_results_replaced=False)
    path = OUT / 'asset_audit.json'; path.write_text(json.dumps(asset, indent=2) + '\n', encoding='utf-8')
    manifest['sources'][path.relative_to(ROOT).as_posix()] = sha(path)
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(asset, indent=2))


if __name__ == '__main__':
    main()
