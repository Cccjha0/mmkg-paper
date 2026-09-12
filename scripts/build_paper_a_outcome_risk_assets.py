"""Formal harm-denominator, gain/loss and tail tables from frozen evidence."""
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_outcome_risk import OUT,sha,PAIRS
from scripts.build_paper_a_selection_seed_assets import table


def number(v,scale=1,digits=2):return '--' if pd.isna(v) else f'{v*scale:.{digits}f}'
def interval(r,name,scale=100,digits=2):
    return number(r[name],scale,digits)+' ['+number(r[name+'_lo'],scale,digits)+', '+number(r[name+'_hi'],scale,digits)+']'


def main():
    p=OUT/'audit.json';audit=json.loads(p.read_text());assert audit['status']=='outcome_risk_checks_passed'
    sources={**audit['sources'],**audit['outputs']}
    for rel,h in sources.items():assert sha(ROOT/rel)==h,rel
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
    summary=pd.read_csv(OUT/'summary.csv').set_index(['label','split']);examples=pd.read_csv(OUT/'worst_examples.csv')
    assets={}
    for split in ('test','dev_oof'):
        display='TEST' if split=='test' else 'Grouped held-out DEV'
        rows=[]
        for label in PAIRS.values():
            r=summary.loc[label,split]
            rows.append(f'{label} & {int(r.n):,} & {int(r.active_n):,} & {int(r.harm_n):,} & '
                f'{interval(r,"harm",100,3)} & {interval(r,"harm_active")} & {int(r.active_unchanged_n):,}'+r'\\')
        assets['harm_'+split+'.tex']=table(rows,
            display+r' harm with two denominators under original ADC. $N$ counts all seed/direction/triple observations; $n_I$ counts actual nonzero projected weight changes and $n_h$ harmed observations. Every harm occurs within $I$ in this replay. $H=n_h/N$ and $H_I=n_h/n_I$ are percentages, each with its own 95\% original-triple percentile interval (10,000 draws). $n_{I,0}$ counts changed weights with unchanged gold RR. Bootstrap draws retain all six observations before restricting to interventions and recompute both denominators; intervals cannot be obtained by dividing endpoints. All intervals are conditional, exploratory and unadjusted for multiplicity.',
            'tab:outcome-harm-'+split,'lrrrllr',r'Pair & $N$ & $n_I$ & $n_h$ & $H$ [95\% CI] & $H_I$ [95\% CI] & $n_{I,0}$')
        rows=[]
        for label in PAIRS.values():
            r=summary.loc[label,split]
            rows.append(label+' & '+' & '.join([number(r[n],100,3) for n in ('benefit','harm','unchanged')]+
                [number(r[n],1,5) for n in ('conditional_gain','conditional_loss')]+
                [number(r[n],1e4,3) for n in ('mean_gain','mean_loss','utility')])+r'\\')
        assets['decomposition_'+split+'.tex']=table(rows,
            display+r' complete gain--loss decomposition against the corresponding original Global anchor. $B,H,Z$ are benefit, harm and exactly unchanged-RR percentages of all $N$ observations in Table~\ref{tab:outcome-harm-'+split+r'}. $G$ and $C$ are conditional mean RR gain and loss in raw RR units. $BG$, $HC$ and net $U$ use $10^{-4}$ RR units; $U=BG-HC$. Computation precedes rounding: all pooled and seed/direction decompositions agree within $2.7\times10^{-17}$. Displayed rounding may leave a last-digit residual. Unchanged RR includes inactive and active-but-rank-unchanged rows. Empty conditional means would be undefined. Complete counts, full precision and conditional intervals are supplied with the table.',
            'tab:outcome-decomposition-'+split,'lrrrrrrrr',r'Pair & $B$ (\%) & $H$ (\%) & $Z$ (\%) & $G$ & $C$ & $BG$ & $HC$ & $U$')
    rows=[]
    for label in PAIRS.values():
        for split,display in [('dev_oof','DEV'),('test','TEST')]:
            r=summary.loc[label,split]
            rows.append(f'{label} & {display} & {int(r.harm_n):,} & '+' & '.join(number(r[n],1,6) for n in ('loss_q50','loss_q90','loss_q95','loss_q99','loss_max'))+
                ' & '+number(r.worst_tenth_loss_share,100,2)+r'\\')
    assets['tails.tex']=table(rows,
        r'Conditional distribution of $-\Delta$ among harmed ADC observations, in raw RR units. Quantiles use linear interpolation and exclude zero/beneficial changes; they are descriptive sample summaries. The final column is the percentage of total loss supplied by the largest $\lceil0.10n_h\rceil$ harmful observations, so its count can slightly exceed 10\% after rounding upward. All pairs and both splits are retained. A small median or unconditional loss does not bound the maximum, and the observed maximum is not a population bound.',
        'tab:outcome-tails','llrrrrrrr',r'Pair & Split & $n_h$ & $q_{50}$ & $q_{90}$ & $q_{95}$ & $q_{99}$ & Max & Top 10\% share')
    rows=[]
    for label in PAIRS.values():
        r=summary.loc[label,'test']
        for threshold,name in [('.10','severe_10'),('.50','severe_50')]:
            rows.append(f'{label} & {threshold} & {int(r[name+"_n"])} & {interval(r,name,100,3)} & {interval(r,name+"_active",100,3)}'+r'\\')
    assets['severe.tex']=table(rows,
        r'Predefined TEST RR-loss thresholds, not selected for significance. The event is $-\Delta\geq t$; counts use exact integer-rank cross-products to handle threshold ties. All and active percentages use $N$ and $n_I$ from Table~\ref{tab:outcome-harm-test}. The .50 threshold includes a rank-1-to-2 loss; .10 is a smaller absolute RR loss, not an acceptability calibration. Brackets are conditional 95\% original-triple percentile intervals from the common 10,000 draws. A zero observed event or degenerate empirical-bootstrap interval cannot certify future safety.',
        'tab:outcome-severe','lrrll',r'Pair & $t$ & Event count & All [95\% CI] & Active [95\% CI]')
    rows=[]
    for label in PAIRS.values():
        for split,display in [('dev_oof','DEV'),('test','TEST')]:
            r=summary.loc[label,split]
            rows.append(f'{label} & {display} & {int(r.top1_retained_n)}/{int(r.top1_n)} & {interval(r,"top1_retention")} & '
                f'{int(r.active_top1_retained_n)}/{int(r.active_top1_n)} & {interval(r,"top1_active_retention")} & {int(r.top1_outside10_n)}'+r'\\')
    assets['top1.tex']=table(rows,
        r'Filtered gold rank-1 retention under ADC. All columns restrict to observations where the corresponding Global gold rank is 1; active columns additionally require a nonzero weight change. Count cells give retained/eligible observations, followed by retention percentage and conditional 95\% original-triple percentile interval. The final column counts reference-rank-1 answers falling below the first ten positions, using the same all-eligible denominator. These are shared-evaluator filtered gold outcomes, not unfiltered serving-list accuracy. The CSV also reports lost/new rank-1 counts, net Hits@1, active-subset losses, all denominators and their intervals.',
        'tab:outcome-top1','lll lllr',r'Pair & Split & All count & Retention [95\% CI] & Active count & Retention [95\% CI] & To $>10$')
    rows=[]
    for r in examples[examples.order==1].itertuples():
        rows.append(f'{r.label} & {r.seed}/{r.direction} & ({r.head_id},{r.relation_id},{r.tail_id}) & '
            rf'{r.rank_global}$\to${r.rank_adc} & {r.alpha0:.2f}$\to${r.alpha_adc:.2f} & {r.loss:.6f}'+r'\\')
    assets['examples.tex']=table(rows,
        r'Largest observed TEST RR loss per pair under the predeclared ordering: loss descending, then seed, direction and $(h,r,t)$. IDs are the canonical dataset IDs. Ranks are filtered gold ranks and weights are original locked actions. These selected extremes illustrate the tail and are neither representative queries nor independent confirmations. The full supplement retains the three largest losses per pair (18 observations), without changing selector inputs or choosing a new policy.',
        'tab:outcome-examples','lllllr',r'Pair & Seed/dir. & $(h,r,t)$ & Global$\to$ADC rank & $\alpha_0\to\hat\alpha$ & Loss')
    target=ROOT/'paper_a_draft/tables/outcome_risk';target.mkdir(parents=True,exist_ok=True);tables={}
    for name,value in assets.items():
        p=target/name;p.write_text(value,encoding='utf-8');tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for rel in ('scripts/build_paper_a_outcome_risk_assets.py','scripts/build_paper_a_selection_seed_assets.py','docs/reports/paper_a_outcome_risk_review_2026-09-12.md'):
        p=ROOT/rel
        if p.exists():sources[rel]=sha(p)
    (ROOT/'paper_a_draft/outcome_risk_source_manifest.json').write_text(json.dumps(dict(version='outcome_risk_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),sources=len(sources))))


if __name__=='__main__':main()
