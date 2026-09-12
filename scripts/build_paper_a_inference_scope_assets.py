"""Build B11/B12 fitting-scope tables from the verified model inventory."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paper_a_conservative_radius import require, sha

OUT = ROOT / 'outputs/paper_a_safe_correction/inference_training_scope_v1'
TABLES = ROOT / 'paper_a_draft/tables/inference_scope'


def main():
    manifest_path = ROOT / 'paper_a_draft/inference_scope_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    audit_path = OUT / 'audit.json'
    require(sha(audit_path) == manifest['sources'][audit_path.relative_to(ROOT).as_posix()], 'Changed scope audit')
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'inference_training_scope_checks_passed', 'Unverified scope')
    path = OUT / 'adc_pairs.csv'
    require(sha(path) == audit['sources'][path.relative_to(ROOT).as_posix()], 'Changed ADC counts')
    counts = pd.read_csv(path)
    methods = [
        ('Primary / Secondary', 'Base checkpoints trained on TRAIN; checkpoint and primary-role selection on DEV.',
         '0 selectors; 3 base checkpoints per expert.', '1 base checkpoint.'),
        ('Equal / Equal RRF', r'No combiner fitting; fixed $\alpha=0.5$ or RRF $k=60$.',
         '0 fitted combiners.', '1 checkpoint pair and fixed rule.'),
        ('Global / Relation', r'Pooled $6N$ DEV rows, including endpoint ties; MRR selection.',
         '1 weight / relation map; 5 corresponding OOF states.', '1 checkpoint pair and its weight / map.'),
        ('ADC / Query-soft', r'Pooled $6N$ DEV rows; fit preprocessing and logistic on non-tied endpoint winners.',
         '1 shared final model; 5 shared OOF models.', '1 checkpoint pair and the shared model; apply the chosen map.'),
        ('Action / grid variants', 'Same fitted signal and training rows as full ADC, including matched shrinkage and linear maps.',
         'Reuse the ADC model within each fold and full fit; no fit per action candidate.', '1 checkpoint pair and the shared model with one locked action rule.'),
        ('Feature variants', 'Same pooled non-tied rows; change feature coordinates only.',
         '5 OOF fits per feature family; final 9D/13D comparison: 1 model per dimension.',
         '1 checkpoint pair and the chosen feature model. P3-only feature variants have no TEST fit.'),
        ('Dyna R0', r'One base seed per network; $N$ directional loss rows in its single epoch.',
         '3 final networks: 1 per base seed; no selector repeats.', '1 checkpoint pair and its corresponding network.'),
        ('Dyna R1 / R2 / R3-fixed', r'One base seed per network; $2N$ cache rows, $N$ loss rows in one epoch.',
         '9 final networks per variant: 3 base seeds times 3 selector seeds.', '1 checkpoint pair and 1 network; repetitions are separate runs.'),
        ('Dyna R3 / R4', r'One base seed per network; $N$ loss rows per epoch. R3 pools held-out metrics for setting selection; R4 inherits them.',
         '9 final networks per variant; R3 additionally has 81 CV trajectories per pair.',
         '1 checkpoint pair and 1 network; no averaging of networks at serving.'),
        ('Endpoint Oracle', 'Uses the gold-dependent better standalone RR per observation.',
         '0 selectors; diagnostic only.', 'Not deployable.'),
    ]
    rows = [' & '.join(row) + r'\\' for row in methods]
    scope = '\n'.join([
        r'\begin{table}[htbp]', r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{Fitting information and deployment units. Counts are per pair, per named variant, not across all pairs. $N$ is the number of original DEV triples; $6N$ combines three paired base seeds and both directions. An OOF model is used on held-out DEV triples only. Final-model counts include every reported initialization, not a serving ensemble. Core ADC, static and P3 feature scopes cover six pairs; matched maps, grid and 9D/13D sensitivity, and Dyna controls cover the four primary pairs.}',
        r'\label{tab:inference-method-scope}',
        r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{.16\linewidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}',
        r'\toprule Method & Fitting / selection information & Fitted objects per pair & One deployment unit \\',
        r'\midrule', *rows, r'\bottomrule', r'\end{tabularx}', r'\end{table}', ''])
    count_rows = [
        f'{r.label} & {int(r.dev_triples):,} & {int(r.available_directional_rows):,} & '
        f'{int(r.fit_rows):,} & {int(r.excluded_ties):,}' + r'\\'
        for _, r in counts.iterrows()
    ]
    samples = '\n'.join([
        r'\begin{table}[htbp]', r'\centering\footnotesize\setlength{\tabcolsep}{5pt}',
        r'\caption{Actual final ADC training scope. The imputer medians, scaler means and stored sample counts match the pooled non-tied rows in every saved model. Each row corresponds to one final classifier, shared across three base seeds and two directions. Excluding ties applies to classifier fitting, not MRR selection or evaluation. For Dyna controls on the first four pairs, each individual network has only its own base-seed scores and uses $N$ directional loss rows per epoch; these are a different loss and training unit, not directly equivalent samples.}',
        r'\label{tab:inference-fit-counts}', r'\begin{tabular}{lrrrr}',
        r'\toprule Pair & DEV triples $N$ & Pooled rows $6N$ & Classifier fit rows & Excluded ties \\',
        r'\midrule', *count_rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest['tables'] = {}
    for name, value in [('methods.tex', scope), ('samples.tex', samples)]:
        path = TABLES / name
        path.write_text(value, encoding='utf-8')
        manifest['tables'][path.relative_to(ROOT / 'paper_a_draft').as_posix()] = sha(path)
    asset = dict(status='inference_scope_assets_verified', adc_pairs=len(counts), method_groups=len(methods),
                 adc_fit_rows=counts.fit_rows.astype(int).tolist(), methods=methods, tables=manifest['tables'],
                 rank_cache_required_for_inference=False, training_information_matched_between_adc_and_dyna=False)
    path = OUT / 'asset_audit.json'
    path.write_text(json.dumps(asset, indent=2) + '\n', encoding='utf-8')
    manifest['sources'][path.relative_to(ROOT).as_posix()] = sha(path)
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in asset.items() if k != 'methods'}, indent=2))


if __name__ == '__main__':
    main()
