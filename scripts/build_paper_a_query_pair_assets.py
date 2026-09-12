"""Bind C11/C12 small summaries and generate current paired-comparison tables."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_query_pair import OUT,sha,PAIRS,MAIN
from scripts.build_paper_a_selection_seed_assets import table,meansd,signed


def main():
    sources={};assets={}
    for name in ('dev_complete.json','test_complete.json'):
        p=OUT/name;a=json.loads(p.read_text());assert a['status']=='query_pair_checks_passed'
        for rel,h in {**a['sources'],**a['outputs']}.items():
            assert sha(ROOT/rel)==h,rel
            if rel in sources:assert sources[rel]==h
            sources[rel]=h
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    pairs=pd.read_csv(OUT/'paired_intervals.csv').set_index(['label','comparator'])
    dev=pd.read_csv(OUT/'dev_summary.csv').set_index(['label','method'])
    effects=pd.read_csv(OUT/'dev_effects.csv').set_index(['label','left','right'])
    fits=pd.read_csv(OUT/'dev_fits.csv')
    inv=pd.concat([pd.read_csv(OUT/('query_inventory_'+s+'.csv')) for s in ('dev','test')])
    def ci(r):return f'{signed(r.delta)} [{signed(r.lo)}, {signed(r.hi)}]'
    def pct(n,d):return f'{100*n/d:.2f}'
    rows=[]
    for r in inv.itertuples():
        cross=f'{int(r.cross_fold_keys)} / {pct(r.cross_fold_keys,r.unique_keys)}' if r.split=='dev' else '--'
        rowcross=pct(r.rows_on_cross_fold_keys,r.answer_rows) if r.split=='dev' else '--'
        seen=pct(r.rows_with_dev_key,r.answer_rows) if r.split=='test' else '--'
        dataset_label={'mkgw':'MKG-W','db15k':'DB15K'}[r.dataset]
        rows.append(f'{dataset_label} & {r.split} & {r.direction} & {r.answer_rows} & {r.unique_keys} & '
            f'{r.repeated_keys} / {pct(r.repeated_keys,r.unique_keys)} & {pct(r.rows_on_repeated_keys,r.answer_rows)} & {cross} & {rowcross} & {seen}'+r'\\')
    assets['query_inventory.tex']=table(rows,
        r'Semantic query inventory, identical across each dataset\textquotesingle s three pairs. A key is (direction, relation, known entity); the three checkpoint seeds repeat these requests and do not triple these counts. $N$ counts labeled directional answer rows, $K$ unique keys. Repeated keys have multiple gold answers. Repeated and cross-fold key cells give count / percentage of $K$; row columns use $N$. Cross-fold keys occur in multiple original-triple DEV folds. TEST has no outer folds; its final column is the fraction of answer rows with a key also in DEV. These are graph/query overlap statistics, not evidence that gold enters inference features.',
        'tab:query-inventory','lllrrrrrrr',r'Dataset & Split & Dir. & $N$ & $K$ & Repeat $K$/\% & Repeat rows\% & Cross $K$/\% & Cross rows\% & DEV key\%')
    rows=[]
    for label in PAIRS.values():
        a,b,c=[dev.loc[label,m] for m in ('Original','Purged','Random-size')]
        e=effects.loc[label,'Purged','Original'];g=effects.loc[label,'Purged','Random-size']
        rows.append(f'{label} & {a.mrr:.6f} & {b.mrr:.6f} & {c.mrr:.6f} & {ci(e)} & {ci(g)}'+r'\\')
    assets['purge.tex']=table(rows,
        r'DEV query-isolation sensitivity on unchanged original outer holdouts. O is the original combiner; P removes whole training triples sharing either head or tail query keys with the holdout. R retains the same number of training triples per relation as P using one fixed hash draw. All refit anchor, preprocessing, logistic and the original 40 beta/gate choices within their own training portions. Differences and pointwise 95\% percentile intervals use $10^{-4}$ RR units and 2,000 original-triple draws. They condition on fitted experts, folds and fixed subsets; R does not average over random subset selection. Lower sample size and changed training composition remain relevant. No new TEST policy is applied.',
        'tab:query-purge','lrrrll',r'Pair & O MRR & P MRR & R MRR & P$-$O [95\% CI] & P$-$R [95\% CI]')
    caption=(r'Unified paired TEST effects, ADC minus comparator, in $10^{-4}$ MRR units. All 82 available method/component contrasts are retained across the two datasets. Brackets are two-sided pointwise 95\% percentile intervals from 10,000 original-triple resamples, retaining six seed/direction observations. The same PCG64 draws are used for every pair and comparator within a dataset. R3 first averages its three selector repetitions within an observation. Expanded-radius changes only $\beta$ to 1; no-fallback changes only $\tau$ to 0. All intervals are conditional, retrospective and unadjusted for multiplicity. A zero-containing interval is not equivalence. Displayed zero endpoints can reflect rounding; the CSV retains full precision. ')
    for dataset in ('mkgw','db15k'):
        rows=[]
        for pair,label in PAIRS.items():
            if not pair.startswith(dataset):continue
            for comparator,r in pairs.loc[label].iterrows():
                rows.append(f'{label} & {comparator} & {signed(r.delta)} & [{signed(r.lo)}, {signed(r.hi)}] & {r.seed_sd*1e4:.2f}'+r'\\')
            rows.append(r'\addlinespace')
        assets['paired_'+dataset+'.tex']=table(rows,caption+'SD is the sample SD of the three paired base-seed effects, not a standard error.',
            'tab:query-paired-'+dataset,'llrlr',r'Pair & Comparator & $\Delta$ & 95\% CI & Seed SD')
    for name,comparators,label in [('nearest',('Query-soft','Relation'),'tab:query-nearest'),('components',('Expanded-radius','No-fallback'),'tab:query-components')]:
        rows=[key+' & '+' & '.join(ci(pairs.loc[key,c]) for c in comparators)+r'\\' for key in PAIRS.values()]
        assets[name+'.tex']=table(rows,caption.split('All 82')[0]+r'This view highlights '+
            ('the direct Query-soft and Relation comparisons. ' if name=='nearest' else 'full-minus-expanded-radius and full-minus-no-fallback comparisons. ')+
            r'All use the same 10,000-draw paired original-triple percentile procedure as the central Global comparison. Intervals are exploratory, conditional and pointwise; neither zero inclusion nor a small point estimate establishes equivalence.',
            label,'lll','Pair & '+comparators[0]+r' [95\% CI] & '+comparators[1]+r' [95\% CI]')
    # Current replacements use new intervals without rewriting old bound evidence.
    rows=[]
    for pair,label in PAIRS.items():
        r=pairs.loc[label,'Global']
        rows.append(f'{label} & '+('Primary' if pair in MAIN else 'Additional')+f' & {r.delta:+.6f} & [{r.lo:+.6f}, {r.hi:+.6f}]'+r'\\')
    assets['intervals.tex']=table(rows,
        r'ADC minus Global on all six retained TEST pairs. These current pointwise 95\% original-triple percentile intervals use the unified 10,000-draw run, with common draws across all comparisons in each dataset (Appendix~\ref{app:claims-effects}). All six seed/direction observations remain together. D-N and both additional intervals span zero. Central primary-pair comparisons are retrospective, without established TEST blindness or multiplicity correction.',
        'tab:claims-intervals','llrl',r'Pair & Role & $\Delta$MRR & 95\% interval')
    old_manifest=json.loads((ROOT/'paper_a_draft/selection_seed_source_manifest.json').read_text())
    seed_path=ROOT/'outputs/paper_a_safe_correction/selection_seed_v1/test_seed_dispersion.csv'
    rel=seed_path.relative_to(ROOT).as_posix();assert sha(seed_path)==old_manifest['sources'][rel];sources[rel]=sha(seed_path)
    seed=pd.read_csv(seed_path).set_index(['label','method']);rows=[]
    for label in PAIRS.values():
        a,g=seed.loc[label,'ADC'],seed.loc[label,'Global'];r=pairs.loc[label,'Global']
        rows.append(f'{label} & {meansd(g["mean"],g.sd)} & {meansd(a["mean"],a.sd)} & {signed(r.seed1)} & {signed(r.seed2)} & {signed(r.seed3)} & '
            f'{meansd(r.delta,r.seed_sd,1e4,2)} & [{signed(r.lo)}, {signed(r.hi)}]'+r'\\')
    assets['query_vs_seed.tex']=table(rows,
        r'Training-seed dispersion and conditional query intervals for the retained TEST policies. Absolute MRR gives mean $\pm$ sample SD over three base seeds. Paired effects, their mean $\pm$ SD and the query CI use $10^{-4}$ RR units. The final column uses the current unified 10,000-draw original-triple percentile run. Seed SD describes these three checkpoint pairs; the query CI holds their models fixed while resampling triples. Neither is a training-population CI. ADC shares one pooled selector, and directions/related pairs are not independent replications.',
        'tab:selection-query-seed','lllrrrll',r'Pair & Global MRR $\pm$ SD & ADC MRR $\pm$ SD & $\Delta_1$ & $\Delta_2$ & $\Delta_3$ & $\bar\Delta\pm$ SD & Query CI')
    # Keep the compact 70-point matrix but direct readers to the now-complete CIs.
    old=ROOT/'paper_a_draft/tables/claims_cost/paired.tex';m=json.loads((ROOT/'paper_a_draft/claims_cost_source_manifest.json').read_text())
    assert sha(old)==m['tables'][old.relative_to(ROOT/'paper_a_draft').as_posix()]
    sources[old.relative_to(ROOT).as_posix()]=sha(old)
    text=old.read_text().replace('No additional intervals were computed.',r'The current unified intervals for every row appear in Tables~\ref{tab:query-paired-mkgw}--\ref{tab:query-paired-db15k}.')
    text=text.replace(r'R3 intervals appear in Table~\ref{tab:dyna-effects}.',r'The original R3 diagnostic intervals are retained in Table~\ref{tab:dyna-effects}.')
    assets['paired_points.tex']=text
    target=ROOT/'paper_a_draft/tables/query_pair';target.mkdir(parents=True,exist_ok=True);tables={}
    for name,value in assets.items():
        p=target/name;p.write_text(value,encoding='utf-8');tables[p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for rel in ('scripts/build_paper_a_query_pair_assets.py','scripts/build_paper_a_selection_seed_assets.py','docs/reports/paper_a_query_pair_review_2026-09-12.md'):
        p=ROOT/rel
        if p.exists():sources[rel]=sha(p)
    (ROOT/'paper_a_draft/query_pair_source_manifest.json').write_text(json.dumps(dict(version='query_pair_v1',sources=sources,tables=tables),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(tables),sources=len(sources))))


if __name__=='__main__':main()
