"""Build B09 formal tables from independently verified returned rank summaries."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_conservative_radius import PAIRS, sha, require
from scripts.review_paper_a_grid_sensitivity import OUT

TABLES=ROOT/'paper_a_draft/tables/grid_sensitivity'


def table(name,caption,columns,header,rows):
    text=('\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
          +'\\caption{'+caption+'}\n\\label{tab:grid-'+name+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
          +' & '.join(header)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+'\\\\' if isinstance(r,list) else r for r in rows)
          +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (TABLES/(name+'.tex')).write_text(text,encoding='utf-8')


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    require(audit['status']=='grid_sensitivity_return_checks_passed' and not audit['failures'],'Return not verified')
    require(audit['cells']==48 and audit['observations']==316488 and audit['original_grid_rank_mismatches']==0,'Incomplete replay')
    sources=dict(audit['sources'])
    for rel,value in sources.items():
        require(sha(ROOT/rel)==value,'Changed audited source: '+rel)
    for name,value in audit['outputs'].items():
        require(sha(OUT/name)==value,'Changed aggregate: '+name)
        sources[(OUT/name).relative_to(ROOT).as_posix()]=value
    for path in (OUT/'audit.json',Path(__file__)):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    TABLES.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(OUT/'summary.csv').set_index(['pair','split','method'])
    paired=pd.read_csv(OUT/'paired_effects.csv').set_index(['pair','split','left','right'])
    diagnostics=pd.read_csv(OUT/'weight_diagnostics.csv').set_index(['pair','split','method'])
    for phase in ('dev','test'):
        rows=[]
        for pair,label in PAIRS.items():
            if rows: rows.append(r'\midrule')
            for method,display,resolution in [('Global','Global','--')]+[(f'{p}_{r}',p,s) for p in ('ADC','Query-soft') for r,s in [('grid_005','0.05'),('grid_001','0.01'),('continuous','Cont.')]]:
                row=summary.loc[pair,phase,method]
                rows.append([label if method=='Global' else '',display,resolution,f'{row.mrr:.6f}',f'{1e4*row.delta_mrr:+.2f}',
                             f'{1e4*row.mean_loss:.2f}',f'{100*row.harm_rate:.2f}',f'{100*row.action_rate:.2f}'])
        scope='Grouped five-fold OOF DEV' if phase=='dev' else 'Fixed TEST'
        table(phase,scope+r' quantization sensitivity on all four primary pairs. Each policy changes only projection, sharing its original fitted signal and DEV-chosen parameters across resolutions. All six dynamic settings were fixed before new scoring; no grid selection or grid-specific retuning occurs. $\Delta_G$ is MRR minus Global, $L_-$ unconditional RR loss relative to Global, $H$ harm frequency and $I_\alpha$ weight-change frequency. $\Delta_G,L_-$ are scaled by $10^4$; $H,I_\alpha$ are percentages. Three seeds and both directions receive equal weight. Continuous execution retains float32 ranking. These retrospective point estimates carry no new confidence intervals.',
              'lllrrrrr',['Pair','Policy','Step','MRR',r'$10^4\Delta_G$',r'$10^4L_-$',r'$H$ (\%)',r'$I_\alpha$ (\%)'],rows)
    rows=[]
    for pair,label in PAIRS.items():
        for policy in ('ADC','Query-soft'):
            values=[]
            for phase in ('dev','test'):
                for resolution in ('grid_001','continuous'):
                    values.append(f'{1e4*paired.loc[(pair,phase,policy+"_"+resolution,policy+"_grid_005")].delta_mrr:+.2f}')
            rows.append([label,policy,*values])
    table('effects',r'Paired MRR changes from the original 0.05 grid, multiplied by $10^4$. All listed comparisons were fixed in advance; positive values favor the finer or continuous action. DEV uses OOF signals and TEST uses the full-DEV locks. Aggregate changes can conceal opposing seed or direction effects; complete cell effects accompany the numerical supplement. No significance or optimal-grid claim is made.',
          'llrrrr',['Pair','Policy','DEV 0.01','DEV cont.','TEST 0.01','TEST cont.'],rows)
    report=dict(status='grid_sensitivity_assets_verified',tables=['dev','test','effects'],methods_per_pair_split=7,
                dev_test_summary_rows=56,paired_effect_rows=104,seed_direction_rows=336,
                new_grid_selection=False,original_paper_grid=.05,historical_results_replaced=False,new_confidence_intervals=False,
                adc_test_minus_global={label:{res:float(summary.loc[(pair,'test','ADC_'+res)].delta_mrr) for res in ('grid_005','grid_001','continuous')} for pair,label in PAIRS.items()},
                adc_test_max_abs_delta_vs_005=float(pd.read_csv(OUT/'paired_effects.csv').query("split == 'test' and right == 'ADC_grid_005'").delta_mrr.abs().max()),
                adc_deadzone_suppressed_rate_max=float(diagnostics.reset_index().query("method.str.startswith('ADC')",engine='python').deadzone_suppressed_rate.max()))
    path=OUT/'asset_audit.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    sources[path.relative_to(ROOT).as_posix()]=sha(path)
    tables={p.relative_to(ROOT/'paper_a_draft').as_posix():sha(p) for p in TABLES.glob('*.tex')}
    (ROOT/'paper_a_draft/grid_sensitivity_source_manifest.json').write_text(json.dumps(dict(version='grid_sensitivity_return_review_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
