"""E02 formal fixed-setting design, paired effects and loss/action tables."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_component_ablation import OUT,sha,PAIRS,METHODS,COMPARATORS
from scripts.build_paper_a_selection_seed_assets import table


def num(x,scale=1,digits=2):return '--' if pd.isna(x) else f'{scale*x:.{digits}f}'
def ci(r):return f'{r.delta*1e4:+.2f} [{r.lo*1e4:+.2f},{r.hi*1e4:+.2f}]'


def main():
    p=OUT/'audit.json';audit=json.loads(p.read_text());assert audit['status']=='component_ablation_checks_passed'
    sources={**audit['sources'],**audit['outputs']}
    for rel,h in sources.items():assert sha(ROOT/rel)==h,rel
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
    summary=pd.read_csv(OUT/'summary.csv');effects=pd.read_csv(OUT/'effects.csv')
    target=ROOT/'paper_a_draft/tables/component_ablation';target.mkdir(parents=True,exist_ok=True);tables={}
    def save(name,rows,caption,cols,head):
        p=target/(name+'.tex');p.write_text(table(rows,caption,'tab:component-'+name,cols,head),encoding='utf-8')
        tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    rows=[
        r'ADC & $a+b\tanh(g)$ & Original $b,t$\\',
        r'Center-0.5 & $0.5+b\tanh(g)$ & Accepted proposal origin only\\',
        r'Linear-p & $a+b(2p-1)$ & Bounded response shape; same $b,t$\\',
        r'Clip-g & $a+b\clip(g,-1,1)$ & Saturation shape; same $b,t$\\',
        r'Shrink-fixed & $(1-b)a+bp$ & Linear shrinkage with fixed $\lambda=b$, same $t$\\',
        r'Radius-1 & $a+\tanh(g)$ & Radius only: $b\mapsto1$, same $t$\\',
        r'No-gate & $a+b\tanh(g)$ & Threshold only: $t\mapsto0$, same $b$\\',
        r'Global & $a$ & Static reference\\']
    save('design',rows,
        r'Fixed-setting component contrasts. The original fitted object supplies $g,p=\sigma(g)$ and settings $a=\alpha_0,b=\beta,t=\tau$. All proposals share clipping, 0.05 projection with original-anchor tie-breaking, and return to $a$ when rejected/invalid. Center-0.5 removes the selected center only from the accepted proposal; fallback and projection still use $a$. Shrink-fixed changes the shrinkage target/direction and envelope, so it is not a pure shape or origin ablation. No new parameter search is performed; these controls complement the separately DEV-tuned families.',
        'lll',r'Variant & Accepted proposal & Controlled change')
    for split in ('test','dev_oof'):
        rows=[]
        f=effects[effects.split==split].set_index(['label','comparator','regime'])
        for label in PAIRS.values():
            for method in COMPARATORS:
                r=f.loc[label,method,'Native'];m=f.loc[label,method,'Matched-I']
                rows.append(f'{label} & {method} & {ci(r)} & {ci(m)} & {int(m.match_active_n):,} & '+
                    num(m.adc_fraction_retained,100,1)+'/'+num(m.other_fraction_retained,100,1)+r'\\')
        save('effects_'+split,rows,
            ('TEST' if split=='test' else 'Original grouped held-out DEV')+
            r' ADC-minus-variant RR differences in $10^{-4}$ units with 95\% original-triple percentile intervals (10,000 common draws). Native retains each original gate/action count. Matched-I independently and uniformly thins each side to the same nonzero count in each seed/fold/repetition stratum. $n_*$ is the common observation count; retained gives ADC/variant percentages of their original nonzero counts. Each row has its own budget, so matched ADC is not one common policy across rows. Exact discrete-action expectations are compared; no expected alpha is ranked. Intervals condition on fixed fits and workload allocations, omit refitting/reallocation uncertainty and are unadjusted for multiplicity. Zero inclusion is not equivalence.',
            'llllrl',r'Pair & Variant & Native [95\% CI] & Matched-I [95\% CI] & $n_*$ & Retained (\%)')
    for scope,labels in [('primary',list(PAIRS.values())[:4]),('additional',list(PAIRS.values())[4:])]:
        rows=[]
        f=summary[summary.split=='test'].set_index(['label','method'])
        for label in labels:
            for method in METHODS:
                r=f.loc[label,method]
                rows.append(f'{label} & {method} & '+ ' & '.join([num(r.utility,1e4),num(r.intervention,100),
                    num(r.harm,100),num(r.harm_active,100),num(r.mean_loss,1e4),num(r.movement,1,4)])+r'\\')
        save('risk_'+scope,rows,
            r'TEST native component policies: $U$ and unconditional RR loss $L_-$ use $10^{-4}$ units; intervention $I$, all-observation harm $H$ and intervention-conditional harm $H_I$ are percentages; $D$ is mean absolute weight displacement over all observations. Undefined Global conditional harm is marked --, not zero. All policies use the same original Global reference. These are fixed-setting sensitivity points, not independently DEV-optimized competitors. The supplement retains full counts, MRR, conditional matched metrics and all DEV/seed/direction results.',
            'llrrrrrr',r'Pair & Variant & $U$ & $I$ (\%) & $H$ (\%) & $H_I$ (\%) & $L_-$ & $D$')
    for rel in ('scripts/build_paper_a_component_ablation_assets.py','scripts/build_paper_a_selection_seed_assets.py',
                'docs/reports/paper_a_component_ablation_review_2026-09-12.md'):
        if (ROOT/rel).exists():sources[rel]=sha(ROOT/rel)
    (ROOT/'paper_a_draft/component_ablation_source_manifest.json').write_text(json.dumps(dict(
        version='component_ablation_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),sources=len(sources))))


if __name__=='__main__':main()
