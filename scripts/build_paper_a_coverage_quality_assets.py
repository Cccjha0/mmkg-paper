"""Formal coverage diagnostics; no outcome-driven selection of panels or points."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_coverage_quality import OUT,sha,PAIRS
from scripts.build_paper_a_selection_seed_assets import table


def num(x,scale=100,digits=2):return '--' if pd.isna(x) else f'{x*scale:.{digits}f}'


def main():
    p=OUT/'audit.json';audit=json.loads(p.read_text());assert audit['status']=='coverage_quality_checks_passed'
    sources={**audit['sources'],**audit['outputs']}
    for rel,h in sources.items():assert sha(ROOT/rel)==h,rel
    sources[p.relative_to(ROOT).as_posix()]=sha(p)
    curves=pd.read_csv(OUT/'curves.csv');agg=pd.read_csv(OUT/'aggregates.csv')
    ap=pd.read_csv(OUT/'average_precision.csv');weights=pd.read_csv(OUT/'weights.csv')
    target=ROOT/'paper_a_draft/tables/coverage_quality';target.mkdir(parents=True,exist_ok=True)
    figures_dir=ROOT/'paper_a_draft/figures/coverage_quality';figures_dir.mkdir(parents=True,exist_ok=True)
    tables={};figures={}
    def save_table(name,rows,caption,columns,header):
        path=target/(name+'.tex');path.write_text(table(rows,caption,'tab:coverage-'+name,columns,header),encoding='utf-8')
        tables[path.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(path)
    rows=[]
    for r in weights.itertuples():
        rows.append(f'{r.label} & '+('DEV' if r.split=='dev_oof' else 'TEST')+
                    f' & {r.n:,} & {r.original_triples:,} & {r.micro_full_weight*100:.4f} & 25.00'+r'\\')
    save_table('weights',rows,r'Four-primary-pair aggregation weights. Each original triple supplies three seeds and two directions. Same-dataset pairs reuse exactly the same observation keys (100\% overlap); repeated pair evaluations are not independent samples. Full-denominator micro weights are $N_j/\sum N_j$; macro weights are $1/4$. Conditional micro weights instead use each accepted or active subset count, at each coverage and random draw. DEV has 146,376 pair-observations from 12,198 distinct dataset triples; TEST has 170,112 from 14,176. Additional pairs are displayed separately and do not enter these four-pair summaries.',
               'llrrrr',r'Pair & Split & $N_j$ & Triples & Micro (\%) & Macro (\%)')
    for pool in ('All','Active'):
        rows=[]
        for label in PAIRS.values():
            for d in (1,5,10):
                c=curves[(curves.label==label)&(curves.split=='test')&(curves.pool==pool)&(curves.decile==d)].set_index('method')
                r=c.loc['Confidence'];rand=c.loc['Random']
                rows.append(f'{label} & {d*10} & {int(r.accepted_n):,} & {int(r.active_n):,} & {int(r.harm_n):,} & '+
                    ' & '.join(num(r[k]) for k in ('harm_all','harm_accepted','harm_active'))+
                    f' & {num(rand.harm_active)} & {num(r.utility,1e4,2)} & {num(rand.utility,1e4,2)}'+r'\\')
        save_table('counts_'+pool.lower(),rows,
            r'TEST '+pool+r'-pool diagnostic at predeclared 10/50/100\% per-stratum unit quotas. $n_S,n_I,n_h$ and $H,H_S,H_I$ describe the confidence order; all three harm rates are percentages. $R_I$ is mean random-order conditional active harm, recomputed in each of 512 realizations; $U_c,U_r$ are confidence and mean random utility in $10^{-4}$ RR units. Random acceptance counts match exactly; in the Active pool actual interventions also match exactly, and $n_S=n_I$. All-pool random interventions can differ. Quotas are rounded down within strata, so requested quota is distinct from observation coverage. The zero endpoint is Global and all intermediate deciles, randomization ranges and exact expected additive totals are retained in the CSV. No point is a newly selected deployment setting.',
            'lrrrrrrrrrr',r'Pair & Quota & $n_S$ & $n_I$ & $n_h$ & $H$ & $H_S$ & $H_I$ & $R_I$ & $U_c$ & $U_r$')
    rows=[]
    for r in ap[ap.scope!='pair'].itertuples():
        rows.append(('DEV' if r.split=='dev_oof' else 'TEST')+' & '+r.population+' & '+r.scope+
                    ' & '+' & '.join(num(v) for v in (r.prevalence,r.ap,r.ap_lift))+r'\\')
    save_table('ap',rows,
        r'Primary-pair AP aggregation, descriptive points only. Macro averages four pair APs and four pair prevalences with weight $1/4$. Micro AP concatenates the four score/label arrays and recomputes a single threshold ranking; it is not a weighted average of pair APs. Micro prevalence weights by each population count. All/Prop./Exec. retain the definitions in Appendix~\ref{app:ties-harm}. AP is non-interpolated average precision, not trapezoidal PR area. Lift is AP minus same-population prevalence in percentage points, not a ratio. Pooling uncalibrated preference scores can add cross-pair ranking effects; neither aggregate replaces per-pair results or represents four independent datasets.',
        'lllrrr',r'Split & Population & Aggregate & Prevalence (\%) & AP (\%) & AP lift (pp)')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    blue='#2463a5';orange='#bf6b20'
    def panel(ax,part,metric):
        c=part[part.method=='Confidence'].sort_values('decile');r=part[part.method=='Random'].sort_values('decile')
        x=c.quota_fraction.to_numpy()*100
        ax.plot(x,c[metric]*100,color=blue,marker='o',ms=2.6,lw=1.4,label='Confidence order')
        ax.plot(x,r[metric]*100,color=orange,ls='--',lw=1.3,label='Random mean')
        ax.fill_between(x,r[metric+'_random_lo']*100,r[metric+'_random_hi']*100,color=orange,alpha=.19,label='Random 95% range')
        ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    def save_fig(fig,name):
        path=figures_dir/(name+'.pdf');fig.savefig(path,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        figures[path.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(path);plt.close(fig)
    for split in ('test','dev_oof'):
        fig,axes=plt.subplots(2,3,figsize=(7.1,4.65),layout='constrained')
        for row,mode in enumerate(('micro','macro')):
            for col,metric in enumerate(('harm_all','harm_accepted','harm_active')):
                ax=axes[row,col];panel(ax,agg[(agg.split==split)&(agg.pool=='All')&(agg.aggregation==mode)],metric)
                ax.set_title(mode.capitalize()+' / '+['all observations','accepted subset','active subset'][col])
                ax.set_ylabel('Harm (%)');ax.set_xlabel('Per-stratum unit quota (%)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
        save_fig(fig,'aggregate_'+split)
        for pool in ('All','Active'):
            fig,axes=plt.subplots(2,3,figsize=(7.1,4.65),layout='constrained')
            for ax,label in zip(axes.ravel(),PAIRS.values()):
                panel(ax,curves[(curves.split==split)&(curves.pool==pool)&(curves.label==label)],'harm_active')
                ax.set_title(label);ax.set_ylabel('Harm among active (%)');ax.set_xlabel('Per-stratum unit quota (%)')
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
            save_fig(fig,'pairs_'+pool.lower()+'_'+split)
    for rel in ('scripts/build_paper_a_coverage_quality_assets.py','scripts/build_paper_a_selection_seed_assets.py',
                'docs/reports/paper_a_coverage_quality_review_2026-09-12.md'):
        if (ROOT/rel).exists():sources[rel]=sha(ROOT/rel)
    manifest=dict(version='coverage_quality_v1',sources=sources,tables=tables,figures=figures)
    (ROOT/'paper_a_draft/coverage_quality_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),figures=len(figures),sources=len(sources))))


if __name__=='__main__':main()
