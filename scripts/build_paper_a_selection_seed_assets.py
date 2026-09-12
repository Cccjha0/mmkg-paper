"""C09/C10 formal tables with separate training and conditional query uncertainty."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_selection_seed import OUT,sha,PAIRS,METHODS


def table(rows,caption,label,cols,head):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',head+r'\\',r'\midrule',
        *rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def setting(r):return f'{r.beta:.2f}/{r.tau:.1f}'
def signed(x):return f'{x*1e4:+.2f}'
def interval(r):return f'{signed(r.delta)} [{signed(r.lo)},{signed(r.hi)}]'
def meansd(mean,sd,scale=1,digits=6):return f'{mean*scale:.{digits}f}'+r'$\pm$'+f'{sd*scale:.{digits}f}'


def main():
    sources={}
    for name,status in [('dev_complete.json','selection_hierarchy_checks_passed'),('seed_complete.json','seed_dispersion_checks_passed')]:
        p=OUT/name;a=json.loads(p.read_text());assert a['status']==status
        for rel,digest in {**a['sources'],**a['outputs']}.items():
            assert sha(ROOT/rel)==digest,rel
            if rel in sources:assert sources[rel]==digest
            sources[rel]=digest
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    dev=pd.read_csv(OUT/'dev_summary.csv').set_index(['label','method'])
    effect=pd.read_csv(OUT/'dev_paired_effects.csv').set_index(['label','left','right'])
    choices=pd.read_csv(OUT/'dev_choices.csv').set_index(['label','outer_fold','method'])
    seed=pd.read_csv(OUT/'test_seed_dispersion.csv').set_index(['label','method'])
    paired=pd.read_csv(OUT/'test_paired_seed_dispersion.csv').set_index(['label','comparator'])
    assets={};rows=[]
    for label in PAIRS.values():
        for count,a,b in [(40,'Original-40','Inner-40'),(41,'Resub-41','Inner-41')]:
            x,y=dev.loc[label,a],dev.loc[label,b];e=effect.loc[label,b,a]
            rows.append(f'{label} & {count} & {x.mrr:.6f} & {y.mrr:.6f} & {interval(e)} & '
                f'{x.intervention*100:.2f}/{y.intervention*100:.2f} & {x.mean_loss*1e4:.2f}/{y.mean_loss*1e4:.2f}'+r'\\')
    assets['dev_comparison.tex']=table(rows,
        r'Fixed DEV selection-layer sensitivity. R selects radius/gate from fitted outer-training preferences; I uses three inner grouped holdouts within the same outer training portion. Both are evaluated on the same original five outer holdouts. Forty means the original 40 configurations; 41 adds explicit Global. The I$-$R effect and unconditional losses $L_R/L_I$ use $10^{-4}$ RR units; intervention $I_R/I_I$ is a percentage. Brackets are 95\% paired original-triple bootstrap intervals (2,000 draws), conditional on frozen experts, fits, choices and folds. They omit model-development and fitting uncertainty. No new TEST policy is applied.',
        'tab:selection-dev','llrrlrr',r'Pair & Grid & R MRR & I MRR & I$-$R [95\% CI] & $I_R/I_I$ & $L_R/L_I$')
    for dataset,prefix in [('mkgw','W'),('db15k','D')]:
        rows=[]
        for pair,label in PAIRS.items():
            if not pair.startswith(dataset):continue
            for fold in range(6):
                r=choices.loc[label,fold,'Original-40' if fold else 'Resub-40']
                s=choices.loc[label,fold,'Resub-41'];a=choices.loc[label,fold,'Inner-40'];b=choices.loc[label,fold,'Inner-41']
                display=str(fold) if fold else 'Full'
                rows.append(f'{label} & {display} & {r.anchor:.2f} & {setting(r)} & {setting(a)} & {setting(s)} & {setting(b)}'+r'\\')
        assets['choices_'+dataset+'.tex']=table(rows,
            (r'MKG-W' if prefix=='W' else r'DB15K')+r' selection choices. Cells give $\beta/\tau$. R is original training resubstitution; I is inner-OOF selection. Folds 1--5 are the original outer holdouts. Full uses full DEV, with I choices reported only as diagnostics and never applied to TEST. The anchor and final preference object are shared across the four settings in a row; each inner fold fits its own anchor/preprocessing/classifier for selecting I. Complete inner scopes, sample counts, identities and all 2,952 selection scores accompany the tables.',
            'tab:selection-choices-'+dataset,'llrrrrr',r'Pair & Fold & $\alpha_0$ & R-40 & I-40 & R-41 & I-41')
    rows=[]
    for label in PAIRS.values():
        a,g=seed.loc[label,'ADC'],seed.loc[label,'Global'];r=paired.loc[label,'Global']
        ci=f'[{signed(r.query_ci_low)},{signed(r.query_ci_high)}]'
        rows.append(f'{label} & {meansd(g["mean"],g.sd)} & {meansd(a["mean"],a.sd)} & '
            f'{signed(r.seed1)} & {signed(r.seed2)} & {signed(r.seed3)} & {meansd(r["mean"],r.sd,1e4,2)} & {ci}'+r'\\')
    assets['query_vs_seed.tex']=table(rows,
        r'Two distinct uncertainty summaries for the retained TEST policies. Absolute MRR columns give mean $\pm$ sample SD over three base-checkpoint seeds. The three paired effects and their mean $\pm$ sample SD use $10^{-4}$ RR units. The last column retains the existing 10,000 original-triple clustered-bootstrap 95\% interval for the pooled effect, in those units. It samples query triples while fixing the trained models and seeds; seed SD describes observed checkpoint dispersion and is not a training-population confidence interval. ADC uses one pooled selector per pair, so these are not three independent selector fits. Directions and related pairs are not independent replications.',
        'tab:selection-query-seed','lllrrrll',r'Pair & Global MRR $\pm$ SD & ADC MRR $\pm$ SD & $\Delta_1$ & $\Delta_2$ & $\Delta_3$ & $\bar\Delta\pm$ SD & Query CI')
    for dataset,title in [('mkgw','MKG-W'),('db15k','DB15K')]:
        rows=[]
        for pair,label in PAIRS.items():
            if not pair.startswith(dataset):continue
            for method in METHODS:
                if (label,method) not in seed.index:continue
                r=seed.loc[label,method]
                rows.append(f'{label} & {method} & {r.seed1:.6f} & {r.seed2:.6f} & {r.seed3:.6f} & {r["mean"]:.6f} & {r.sd:.6f}'+r'\\')
            rows.append(r'\addlinespace')
        assets['seed_'+dataset+'.tex']=table(rows,
            title+r' absolute TEST MRR for every available method and all three base seeds. Each seed averages both directions and all original triples; mean is the unweighted average of the three seed MRRs and SD uses divisor $3-1$. R3 first averages its three selector repetitions within a base seed, so it still contributes three base-seed MRRs. R0 retains the failed transfer. Methods absent from an additional pair remain absent; no seed is deleted. The accompanying CSV reports all 456 seed/direction cells and 70 paired ADC-minus-comparator seed dispersions. The mean of every row reproduces the complete main comparison.',
            'tab:selection-seeds-'+dataset,'llrrrrr',r'Pair & Method & Seed 1 & Seed 2 & Seed 3 & Mean & SD')
    # Reuse the already audited development history, without rewriting old evidence.
    history=ROOT/'outputs/paper_a_safe_correction/test_history_review_v1/timeline.json'
    hm=json.loads((ROOT/'paper_a_draft/history_source_manifest.json').read_text())['sources']
    assert sha(history)==hm[history.relative_to(ROOT).as_posix()]
    sources[history.relative_to(ROOT).as_posix()]=sha(history)
    target=ROOT/'paper_a_draft/tables/selection_seed';target.mkdir(parents=True,exist_ok=True);tables={}
    for name,value in assets.items():
        p=target/name;p.write_text(value,encoding='utf-8');tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for p in (Path(__file__),ROOT/'docs/reports/paper_a_selection_seed_review_2026-09-12.md'):
        if p.exists():sources[p.relative_to(ROOT).as_posix()]=sha(p)
    manifest=dict(version='selection_seed_v1',sources=sources,tables=tables)
    (ROOT/'paper_a_draft/selection_seed_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),source_count=len(sources)),indent=2))


if __name__=='__main__':main()
