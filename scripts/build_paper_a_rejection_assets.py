"""Formal B17/B18 tables, including every fixed rule and unfavorable effect."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_conservative_radius import sha, require

OUT=ROOT/'outputs/paper_a_safe_correction/rejection_controls_v1'
TABLES=ROOT/'paper_a_draft/tables/rejection'
LABELS=('W-N','W-A','D-N','D-A','W-NA','D-NA')
METHODS=('ADC','Random-I','Head-first','Tail-first','Low-A-gap','Gap-agree','Random-shape')


def f(v,scale=1,digits=2,signed=False):
    return '--' if pd.isna(v) else format(v*scale,('+' if signed else '')+f'.{digits}f')


def effect(point,lo,hi):
    return f(point,1e4,2,True)+' ['+f(lo,1e4,2,True)+','+f(hi,1e4,2,True)+']'


def table(rows,caption,label,cols,head):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',head+r'\\',r'\midrule',
        *rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    path=ROOT/'paper_a_draft/rejection_source_manifest.json'
    manifest=json.loads(path.read_text())
    def source(name):
        p=OUT/name
        require(sha(p)==manifest['sources'][p.relative_to(ROOT).as_posix()],'Changed result: '+name)
        return p
    audit=json.loads(source('audit.json').read_text())
    require(audit['status']=='rejection_controls_checks_passed' and not audit['failures'],'Unverified controls')
    matched=pd.read_csv(source('matched_summary.csv'),float_precision='round_trip').set_index(['label','split','method'])
    fallback=pd.read_csv(source('paired_fallback.csv'),float_precision='round_trip').set_index(['label','split'])
    random=pd.read_csv(source('randomization_spread.csv'),float_precision='round_trip').set_index(['label','split','method'])
    assets={}; rows=[]
    for label in LABELS:
        for split,display in [('dev_oof','DEV'),('test','TEST')]:
            r=fallback.loc[label,split]
            tau=f(r.tau_min) if r.tau_min==r.tau_max else f(r.tau_min)+'--'+f(r.tau_max)
            rows.append(f'{label} & {display} & {tau} & {f(r.full_interventions/r.n,100)}/{f(r.no_gate_interventions/r.n,100)} & '
                f'{f(r.full_utility,1e4,2,True)} & {f(r.no_gate_utility,1e4,2,True)} & {effect(r.paired_effect,r.paired_lo,r.paired_hi)}'+r'\\')
    assets['fallback.tex']=table(rows,
        r'Direct gate effects, retaining all six pairs and both splits. F is original full ADC; 0 sets only $\tau=0$ with the same fitted margin, anchor, radius, clipping and projection. $I_F/I_0$ gives actual nonzero weight percentages; $U_F,U_0$ are mean RR changes relative to the common anchor in $10^{-4}$ units. The last column is the paired $\RR_F-\RR_0$ effect in those units, with a conditional 95\% interval from 2,000 original-triple bootstrap draws retaining all seeds/directions. DEV uses original held-out folds, hence threshold ranges; TEST uses original full-DEV locks. Intervals concern the direct difference, not two separate effects versus Global. Positive, null and negative values are all retained.',
        'tab:rejection-fallback','lllrrrr',r'Pair & Split & $\tau$ & $I_F/I_0$ (\%) & $U_F$ & $U_0$ & F$-$0 [95\% CI]')
    for split,display in [('dev_oof','DEV'),('test','TEST')]:
        rows=[]
        for label in LABELS:
            for method in METHODS:
                r=matched.loc[label,split,method]
                contrast='--' if method=='ADC' else effect(r.full_minus_method,r.paired_lo,r.paired_hi)
                rows.append(f'{label} & {method} & {f(r.action_rate,100)} & {f(r.mean_movement,1,4)} & '
                    f'{f(r.utility,1e4,2,True)} & {f(r.harm_rate,100)} & {f(r.mean_loss,1e4)} & {contrast}'+r'\\')
        assets['matched_'+split+'.tex']=table(rows,
            display+r' rejection controls at exactly matched nonzero intervention counts. Every realized gate retains whole observable queries and matches ADC within seed, fold and query-repetition-count strata; all rows therefore have the same $I$ within a pair. Random-I and Random-shape are exact expectations, not selected draws. Random-shape also matches direction-by-signed-displacement histograms and mean $D=|\Delta\alpha|$ in every realization. Head-/Tail-first use direction only; Low-A-gap and Gap-agree use unfiltered normalized score gaps. All four fixed rules are reported without selecting a winner. $U,L_-$ and paired ADC-minus-row effects use $10^{-4}$ RR units; $H$ is harm percentage over all observations. Brackets are conditional 95\% original-triple bootstrap intervals, with observed-workload allocations fixed.',
            'tab:rejection-'+split,'llrrrrrr',r'Pair & Gate & $I$ (\%) & $D$ & $U$ & $H$ (\%) & $L_-$ & ADC$-$row [95\% CI]')
    rows=[]
    for label in LABELS:
        for split,display in [('dev_oof','DEV'),('test','TEST')]:
            a,b=[random.loc[label,split,m] for m in ('Random-I','Random-shape')]
            c=matched.loc[label,split,'ADC']
            rows.append(f'{label} & {display} & {f(c.utility,1e4,2,True)} & '
                f'{effect(a.utility_expected,a.utility_random_lo,a.utility_random_hi)} & '
                f'{effect(b.utility_expected,b.utility_random_lo,b.utility_random_hi)} & {int(b.partial_units):,}/{int(b.eligible_units):,}'+r'\\')
    assets['random_spread.tex']=table(rows,
        r'Random-gate spread on the fixed observed workload. Random columns give exact expected utility and 2.5/97.5 percentiles of 2,000 random gate realizations, in $10^{-4}$ RR units. These are randomization ranges, not population confidence intervals. No draw is selected. Mixed/eligible counts query units in Random-shape strata with $0<k<n$, where rejection identities can vary, versus all nonzero proposal units. In wholly retained/rejected strata the allocation is deterministic. Shape matching can leave little freedom because margin strength and proposed magnitude are coupled.',
        'tab:rejection-random','llrrrl',r'Pair & Split & ADC $U$ & Random-I $U$ [draw range] & Random-shape $U$ [draw range] & Mixed/eligible')
    TABLES.mkdir(parents=True,exist_ok=True)
    manifest['tables']={}
    for name,value in assets.items():
        p=TABLES/name;p.write_text(value,encoding='utf-8');manifest['tables'][p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    asset=dict(status='rejection_assets_verified',matched_rows=84,fallback_rows=12,random_rows=24,
               tables=manifest['tables'],test_used_for_selection=False)
    p=OUT/'asset_audit.json';p.write_text(json.dumps(asset,indent=2)+'\n',encoding='utf-8')
    manifest['sources'][p.relative_to(ROOT).as_posix()]=sha(p)
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    report=ROOT/'docs/reports/paper_a_rejection_controls_review_2026-09-12.md'
    if report.exists():
        manifest['sources'][report.relative_to(ROOT).as_posix()]=sha(report)
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(asset,indent=2))


if __name__=='__main__':
    main()
