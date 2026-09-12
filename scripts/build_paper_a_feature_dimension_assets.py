"""Build and hash-bind B02 tables without fitting or selecting any policy."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_feature_dimension import OUT as DATA, PAIRS, sha

TABLES=ROOT/'paper_a_draft/tables/feature_dimension'


def table(name,caption,columns,header,rows):
    text=('\\begin{table}[htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n'
          +'\\caption{'+caption+'}\n\\label{tab:dimension-'+name+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
          +' & '.join(header)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+'\\\\' for r in rows)
          +'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (TABLES/(name+'.tex')).write_text(text,encoding='utf-8')


def sci(value):
    coefficient,exponent=f'{value:+.2e}'.split('e')
    return f'${coefficient}\\times10^{{{int(exponent)}}}$'


def main():
    audit=json.loads((DATA/'test_audit.json').read_text())
    assert audit['status']=='feature_dimension_checks_passed' and not audit['failures']
    assert not audit['test_used_for_selection'] and audit['dev_lock_sha256']==sha(DATA/'dev_locks.json')
    sources=dict(audit['sources'])
    for rel,value in sources.items():assert sha(ROOT/rel)==value,rel
    for group in ('outputs','dev_outputs'):
        for name,value in audit[group].items():
            assert sha(DATA/name)==value,name
            sources[(DATA/name).relative_to(ROOT).as_posix()]=value
    for path in (DATA/'test_audit.json',ROOT/'tests/test_score_definition_and_redundancy.py',Path(__file__)):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    replay=json.loads((DATA/'runtime_fit_replay.json').read_text())
    assert replay['status']=='full_13d_runtime_fit_replayed' and not replay['test_opened']
    for rel,value in replay['sources'].items():
        assert sha(ROOT/rel)==value,rel
        sources[rel]=value
    sources[(DATA/'runtime_fit_replay.json').relative_to(ROOT).as_posix()]=sha(DATA/'runtime_fit_replay.json')
    TABLES.mkdir(parents=True,exist_ok=True)
    dev=pd.read_csv(DATA/'dev_oof_summary.csv').set_index(['pair','method'])
    test=pd.read_csv(DATA/'test_summary.csv').set_index(['pair','method'])
    dp=pd.read_csv(DATA/'dev_paired.csv').set_index(['pair','policy'])
    tp=pd.read_csv(DATA/'test_paired.csv').set_index(['pair','policy'])
    choices=pd.read_csv(DATA/'dev_choices.csv')
    locks=choices[choices.scope=='full_dev_selection'].set_index(['pair','dimension'])
    diag=pd.read_csv(DATA/'dev_dependency_diagnostics.csv')
    test_diag=pd.read_csv(DATA/'test_dependency_diagnostics.csv')
    assert diag.incomplete_rows.sum()==test_diag.incomplete_rows.sum()==0
    fits=pd.read_csv(DATA/'fit_diagnostics.csv')
    assert len(fits)==48 and fits.reused.sum()==4 and len(choices)==48
    assert len(pd.read_csv(DATA/'dev_candidates.csv'))==1968
    rows=[]
    for pair,label in PAIRS.items():
        for policy in ('ADC','Query-soft'):
            rows.append([label,policy,*[f'{frame.loc[(pair,policy+"-"+str(dim))].mrr:.6f}'
                                       for frame in (dev,test) for dim in (9,13)],
                         sci(dp.loc[(pair,policy)].delta_13_minus_9),sci(tp.loc[(pair,policy)].delta_13_minus_9)])
    table('mrr',r'Matched feature dimensions on four primary pairs. Both use the same grouped five-fold DEV scheme, random states, labels, $C=1$ fitting and static anchors. Each ADC dimension selects from 41 identical action configurations, including Global; Query-soft shares its dimension-specific fit. DEV is held-out OOF, TEST follows all eight final locks. Differences are 13D minus 9D, without new confidence intervals. Historical TEST inspection makes this a retrospective diagnostic.',
          'llrrrrrr',['Pair','Policy','DEV 9D','DEV 13D','TEST 9D','TEST 13D',r'$\Delta$DEV',r'$\Delta$TEST'],rows)
    rows=[]
    for pair,label in PAIRS.items():
        for dim in (9,13):
            r=test.loc[(pair,'ADC-'+str(dim))];c=locks.loc[(pair,dim)]
            rows.append([label,str(dim),f'{c.strength:.2f}',f'{c.tau:.1f}',f'{r.mean_loss:.6f}',
                         f'{100*r.harm_rate:.3f}',f'{100*r.action_rate:.2f}'])
    table('risk',r'Full-DEV ADC locks and TEST losses for the matched dimensions. $L_-$ is unconditional RR loss relative to Global; $H$ is harm frequency and $I_\alpha$ the weight-change frequency. Complete seed/direction and Query-soft risk records accompany the tables. Equal action budgets need not produce identical policies.',
          'llrrrrr',['Pair','Dim.',r'$\beta$',r'$\tau$',r'$L_-$',r'$H$ (\%)',r'$I_\alpha$ (\%)'],rows)
    report=dict(status='feature_dimension_assets_verified',incomplete_feature_rows=0,
                dev_candidate_evaluations=1968,fitted_models=44,reused_full_models=4,
                full_13d_runtime_verification_fits=4,
                max_n_iter=int(fits.n_iter.max()),
                max_difference_residual=float(max(diag.max_difference_residual.max(),test_diag.max_difference_residual.max())),
                max_collapsed_decision_error=float(max(diag.max_collapsed_decision_error.max(),test_diag.max_collapsed_decision_error.max())),
                raw_score_finiteness_certified=False,test_used_for_selection=False,
                runtime=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
                historical_model_sklearn='1.7.2',
                runtime_note='Historical pickle version differs; full DEV/TEST outputs and original OOF ADC ranks were replayed with strict tolerances. No cross-version bitwise guarantee.')
    path=DATA/'asset_audit.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    sources[path.relative_to(ROOT).as_posix()]=sha(path)
    tables={p.relative_to(ROOT/'paper_a_draft').as_posix():sha(p) for p in TABLES.glob('*.tex')}
    (ROOT/'paper_a_draft/feature_dimension_source_manifest.json').write_text(json.dumps(dict(
        version='feature_dimension_review_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
