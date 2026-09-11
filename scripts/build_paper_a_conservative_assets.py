"""Tables and publication figures for the audited A03/B04/D08 analysis."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_conservative_radius import sha, PAIRS, INPUT

DATA=ROOT/'outputs/paper_a_safe_correction/conservative_radius_review'
TABLES=ROOT/'paper_a_draft/tables/conservative'
FIGURES=ROOT/'paper_a_draft/figures/conservative'
LABELS={'Global':'Global','Relation':'Relation','Query-soft':'Query-soft','ADC':'ADC','beta=1':r'$\beta=1$',
        'tau=0':r'$\tau=0$','R3_softplus':'Dyna R3'}


def table(name,caption,columns,header,rows):
    text=('\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n'
          '\\caption{'+caption+'}\n\\label{tab:conservative-'+name+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
          +' & '.join(header)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+'\\\\' if r else '\\midrule' for r in rows)
          +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (TABLES/(name+'.tex')).write_text(text,encoding='utf-8')


def main():
    TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    audit=json.loads((DATA/'audit.json').read_text())
    assert audit['status']=='conservative_radius_checks_passed' and not audit['failures']
    for name,value in audit['outputs'].items():
        assert sha(DATA/name)==value,name
    for name,value in audit['sources'].items():
        assert sha(ROOT/name)==value,name
    dev=pd.read_csv(DATA/'dev_family_comparison.csv').set_index(['pair','method'])
    test=pd.read_csv(DATA/'test_fair_comparison.csv').set_index(['pair','method'])
    curves=pd.read_csv(DATA/'dev_radius_grid.csv')
    folds=pd.read_csv(DATA/'dev_fold_selections.csv')
    rows=[]
    for pair,label in PAIRS.items():
        a=dev.loc[(pair,'cap_0.50')];b=dev.loc[(pair,'cap_1.00')]
        rows.append([label,f'{a.delta_mrr:+.6f}',f'{b.delta_mrr:+.6f}',f'{a.mean_loss:.6f}',f'{b.mean_loss:.6f}',
                     f'{100*a.harm_rate:.2f}',f'{100*b.harm_rate:.2f}'])
    table('dev',r'Grouped held-out DEV comparison of restricted ($\beta\leq0.50$) and wider ($\beta\leq1.00$) finite families. Each fold selects $\beta,\tau$ by training-portion MRR with the same fitted classifier and anchor. $U$ is net RR utility, $L_-$ is unconditional mean RR loss, and $H$ is harm frequency. No new policy is applied to TEST.',
          'lrrrrrr',['Pair',r'$U_{.50}$',r'$U_{1.00}$',r'$L_{-,.50}$',r'$L_{-,1.00}$',r'$H_{.50}$ (\%)',r'$H_{1.00}$ (\%)'],rows)
    rows=[]
    for pair,label in PAIRS.items():
        for method in LABELS:
            r=test.loc[(pair,method)]
            rows.append([label,LABELS[method],f'{r.delta_mrr:+.6f}',f'{r.mean_loss:.6f}',f'{100*r.harm_rate:.3f}',
                '--' if pd.isna(r.conditional_loss) else f'{r.conditional_loss:.5f}',
                '--' if pd.isna(r.action_rate) else f'{100*r.action_rate:.2f}',
                '--' if pd.isna(r.mean_abs_alpha_deviation) else f'{r.mean_abs_alpha_deviation:.4f}'])
        if label!='D-A': rows.append([])
    table('fair',r'Common TEST reference and risk definitions. $U=\Delta\mathrm{MRR}$; $L_-=E[\max(-\Delta\mathrm{RR},0)]$; $H$ is harm frequency; $C$ is conditional mean loss; $I_\alpha$ is weight-change frequency; $D_\alpha$ is mean absolute weight displacement. Global has zero risk relative to itself and undefined $C$. Dyna R3 uses min-max rather than z-score coefficients, so $I_\alpha,D_\alpha$ are not comparable and are omitted. All historical and repaired Dyna variants are included in the supplementary CSV.',
          'llrrrrrr',['Pair','Method',r'$U$',r'$L_-$',r'$H$ (\%)',r'$C$',r'$I_\alpha$ (\%)',r'$D_\alpha$'],rows)
    rows=[]
    for pair,label in PAIRS.items():
        lock=json.loads((INPUT/pair/'dev_lock/anchored_dev_lock.json').read_text())
        small=folds[(folds.pair==pair)&(folds.family=='cap_0.50')]
        wide=folds[(folds.pair==pair)&(folds.family=='cap_1.00')]
        rows.append([label,f'{lock["beta"]:.2f}',', '.join(f'{b:.2f}' for b in small.beta),', '.join(f'{b:.2f}' for b in wide.beta)])
    table('folds',r'Full-DEV locked radius and all five training-fold selections. Columns list folds 1--5 in order. The wider family is a supplementary DEV diagnostic; original TEST locks remain unchanged.',
          'lrll',['Pair','Locked',r'Fold $\beta$, cap 0.50',r'Fold $\beta$, cap 1.00'],rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for ax,(pair,label) in zip(axes.flat,PAIRS.items()):
        d=curves[(curves.pair==pair)&(curves.scope=='grouped_oof_fixed_fold_tau')].sort_values('beta')
        ax.plot(d.beta,d.delta_mrr,color='#1764ab',label='Net utility')
        ax.scatter(d.beta,d.delta_mrr,s=12+80*d.action_rate,color='#1764ab',zorder=3)
        ax.plot(d.beta,d.mean_loss,color='#b64036',linestyle='--',label='Mean loss')
        ax.axvline(.5,color='#606060',linestyle=':',linewidth=1)
        ax.axhline(0,color='#707070',linewidth=.6)
        ax.set(title=label,xlabel=r'Radius $\beta$',ylabel='Mean RR change / loss')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axes[0,0].legend(loc='best',frameon=False)
    fig.savefig(FIGURES/'dev_radius.pdf');fig.savefig(FIGURES/'dev_radius.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    colors={'Global':'#303030','Query-soft':'#d57c19','ADC':'#1764ab','beta=1':'#b64036','R3_softplus':'#8055a1'}
    markers={'Global':'s','Query-soft':'^','ADC':'o','beta=1':'D','R3_softplus':'P'}
    for ax,(pair,label) in zip(axes.flat,PAIRS.items()):
        for method,color in colors.items():
            r=test.loc[(pair,method)]
            ax.scatter(r.mean_loss,r.delta_mrr,s=35+150*r.rank_change_rate,color=color,marker=markers[method],
                       edgecolor='white',linewidth=.5,label=LABELS[method],zorder=3)
        ax.axhline(0,color='#707070',linewidth=.6)
        ax.set(title=label,xlabel='Unconditional mean RR loss',ylabel='Net RR utility vs Global')
        ax.ticklabel_format(axis='both',style='sci',scilimits=(-2,2))
        ax.margins(.15)
        if label!='W-N':
            # Same three-policy detail in all endpoint-anchor pairs: retain the
            # full overview while making Global/ADC/beta=1 distinguishable.
            detail=ax.inset_axes([.13,.13,.50,.43])
            for method in ('Global','ADC','beta=1'):
                r=test.loc[(pair,method)]
                detail.scatter(r.mean_loss,r.delta_mrr,s=20+85*r.rank_change_rate,
                               color=colors[method],marker=markers[method],zorder=3)
            detail.axhline(0,color='#707070',linewidth=.5)
            detail.margins(.25)
            detail.tick_params(labelsize=6)
            detail.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
            detail.xaxis.offsetText.set_fontsize(6);detail.yaxis.offsetText.set_fontsize(6)
            detail.set_title(r'Global / ADC / $\beta=1$ detail',fontsize=7,pad=3)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=5,frameon=False)
    fig.savefig(FIGURES/'test_utility_loss.pdf');fig.savefig(FIGURES/'test_utility_loss.png',dpi=180);plt.close(fig)
    manifest=dict(version='conservative_radius_review_v1',sources=dict(audit['sources']),tables={},figures={})
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    for p in DATA.glob('*'):
        if p.is_file(): manifest['sources'][p.relative_to(ROOT).as_posix()]=sha(p)
    for p in TABLES.glob('*.tex'):manifest['tables'][p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for p in FIGURES.glob('*.pdf'):manifest['figures'][p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    (ROOT/'paper_a_draft/conservative_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tables':len(manifest['tables']),'figures':len(manifest['figures']),'sources':len(manifest['sources'])}))


if __name__=='__main__': main()
