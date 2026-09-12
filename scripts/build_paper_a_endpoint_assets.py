"""Render the independently checked B10 result into compact manuscript tables."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_conservative_radius import sha, require

OUT=ROOT/'outputs/paper_a_safe_correction/endpoint_contract_return_review_v1'
TABLES=ROOT/'paper_a_draft/tables/endpoint_contract'


def table(body,caption,label,columns,head):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
                      r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',
                      r'\toprule',head+r'\\',r'\midrule',*body,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    manifest_path=ROOT/'paper_a_draft/endpoint_contract_source_manifest.json'
    manifest=json.loads(manifest_path.read_text());audit=json.loads((OUT/'audit.json').read_text())
    require(sha(OUT/'audit.json')==manifest['sources'][(OUT/'audit.json').relative_to(ROOT).as_posix()],'Changed return review')
    require(audit['status']=='endpoint_return_checks_passed' and audit['primary_test_effects_unchanged_at_1e_12'],'Unverified primary effects')
    def frame(name):
        path=OUT/(name+'.csv');require(sha(path)==audit['sources'][path.relative_to(ROOT).as_posix()],'Changed summary')
        return pd.read_csv(path,float_precision='round_trip')
    experts=frame('expert_summary').set_index(['dataset','model','split'])
    policies=frame('policy_sensitivity'); anchors=frame('dev_anchor_sensitivity')
    rows=[]
    for dataset,name in [('mkg_w','MKG-W'),('db15k','DB15K')]:
        for model in ('M-Hyper','NativE','AdaMF-MAT'):
            dev,test=(experts.loc[dataset,model,s] for s in ('dev','test'))
            rows.append(f'{name} & {model} & {int(dev.normalized_vs_raw_mismatches)} & {1e4*dev.delta_mrr:+.3f} & {int(test.normalized_vs_raw_mismatches)} & {1e4*test.delta_mrr:+.3f}'+r'\\')
    TABLES.mkdir(parents=True,exist_ok=True)
    tables={}
    tables['experts.tex']=table(rows,
        r'Un-overridden normalized versus raw standalone endpoints. $n_{\ne}$ counts changed gold ranks; $\Delta$ is normalized-minus-raw MRR in $10^{-4}$ units. Each MKG-W model has 25,656/25,644 DEV/TEST observations; DB15K has 47,532/59,412. All 1,293 differences improve normalized RR. Shared raw and deployed endpoint ranks agree on all 474,732 rows; all raw-score exception counts are zero. Values displayed as zero can contain changes below the shown precision.',
        'tab:endpoint-experts','llrrrr',r'Dataset & Expert & DEV $n_{\ne}$ & DEV $\Delta$ & TEST $n_{\ne}$ & TEST $\Delta$')
    rows=[]
    for label in ('W-N','W-A','D-N','D-A','W-NA','D-NA'):
        sub=policies[(policies.label==label)&(policies.split=='test')].set_index(['method','convention'])
        values=[1e4*sub.loc[m,c].delta_mrr for m in ('ADC','Query-soft','Relation') for c in ('raw_endpoint','normalized_endpoint')]
        rows.append(label+' & '+' & '.join(f'{v:+.3f}' for v in values)+r'\\')
    tables['policies.tex']=table(rows,
        r'Fixed-policy TEST sensitivity to exact endpoint dispatch. Entries are MRR changes relative to the corresponding Global, in $10^{-4}$ units. R uses reported raw standalone endpoints; Z substitutes audited normalized ranks only at actions 0 or 1. Every action, anchor and interior rank remains fixed. The four primary ADC--Global row effects agree within $10^{-12}$; additional pairs and comparators can change. This is not retraining or new significance evidence. All six policies, both splits, seed/direction cells and loss measures are supplied in the numerical supplement.',
        'tab:endpoint-policies','lrrrrrr',r'Pair & ADC R & ADC Z & Q-soft R & Q-soft Z & Relation R & Relation Z')
    rows=[f'{r.label} & {r.original_anchor:.2f} & {r.normalized_endpoint_grid_anchor:.2f} & {r.supervision_category_changes}'+r'\\' for _,r in anchors.iterrows()]
    tables['anchors.tex']=table(rows,
        r'DEV-only convention diagnostic on the complete 21-weight grid. Replace endpoint columns only, then apply the same Global tie rule. Winner/tie changes count differences in the three standalone-supervision categories if those endpoint ranks also defined the labels. These full-DEV diagnostics do not estimate OOF performance. The alternative anchor is not applied to TEST, and no selector is refitted.',
        'tab:endpoint-anchors','lrrr',r'Pair & Original anchor & Z-grid anchor & Winner/tie changes')
    manifest['tables']={}
    for name,value in tables.items():
        path=TABLES/name;path.write_text(value,encoding='utf-8')
        manifest['tables'][path.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(path)
    manifest['sources'][Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    asset=dict(status='endpoint_assets_verified',expert_split_rows=len(experts),policy_convention_rows=len(policies),
               dev_anchor_rows=len(anchors),primary_test_effects_unchanged_at_1e_12=True,
               normalized_endpoint_equivalence=False,training_runs=0,test_used_for_selection=False,tables=manifest['tables'])
    path=OUT/'asset_audit.json';path.write_text(json.dumps(asset,indent=2)+'\n',encoding='utf-8')
    manifest['sources'][path.relative_to(ROOT).as_posix()]=sha(path)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(asset,indent=2))


if __name__=='__main__':main()
