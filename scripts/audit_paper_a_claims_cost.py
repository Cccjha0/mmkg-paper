"""A05/F01/E07: aggregate existing results and audit historical cost; no model runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/paper_a_safe_correction/claims_cost_review_v1'
PAPER = ROOT / 'paper_a_draft'
TABLES = PAPER / 'tables/claims_cost'
BASE = ROOT / 'outputs/paper_a_safe_correction'
PAIRS = dict(mkgw_mhyper_native='W-N', mkgw_mhyper_adamf='W-A',
             db15k_mhyper_native='D-N', db15k_mhyper_adamf='D-A',
             mkgw_native_adamf='W-NA', db15k_native_adamf='D-NA')
MAIN = tuple(PAIRS)[:4]
KEY = ['seed', 'direction', 'head_id', 'relation_id', 'tail_id']
METHODS = ['Primary', 'Secondary', 'Equal-z', 'RRF', 'Global', 'Relation',
           'Query-soft', 'Shrink', 'Shrink-gate', 'Linear-p', 'Clip-g', 'Linear-g',
           'R0', 'R3', 'ADC']
LABELS = {**{m:m for m in METHODS}, 'Equal-z':'Equal z-score', 'RRF':'Equal RRF',
          'R0':'Dyna R0 (historical)', 'R3':'Dyna R3 (healthy)'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def close(a, b):
    if not np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError('Numerical source mismatch')


def align(reference, other, columns):
    """Pair exact observations, rejecting duplicates or dropped/extra observations."""
    if reference.duplicated(KEY).any() or other.duplicated(KEY).any():
        raise ValueError('Duplicate observation')
    left = pd.MultiIndex.from_frame(reference[KEY])
    right = pd.MultiIndex.from_frame(other[KEY])
    if len(left) != len(right) or set(left) != set(right):
        raise ValueError('Observation coverage mismatch')
    return other.set_index(KEY).reindex(left)[columns].reset_index(drop=True)


def reverse_interval(interval):
    lo, hi = interval
    if lo > hi: raise ValueError('Reversed interval')
    return [-hi, -lo]


def time_ratio(numerator, denominator):
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0 or numerator < 0:
        raise ValueError('Invalid timing')
    return float(numerator / denominator)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    sources, expected = {}, {}
    for name in ('rerun', 'matched', 'dynasemble'):
        path = PAPER / (name+'_source_manifest.json')
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        for rel, digest in json.loads(path.read_text())['sources'].items():
            if rel in expected: assert expected[rel] == digest, rel
            expected[rel] = digest

    def bind(path, recorded=True):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha(path)
        if recorded: assert expected.get(rel) == digest, rel
        sources[rel] = digest
        return path

    def read(path):
        return pd.read_csv(bind(path), float_precision='round_trip')

    old = read(BASE/'information_boundary_rerun_audit/method_summary.csv')
    old = old[old.split == 'test'].set_index(['pair', 'method'])
    static = read(BASE/'matched_alternatives_v1/static_summary.csv').set_index(['pair', 'method'])
    matched = read(BASE/'matched_alternatives_v1/test_summary.csv').set_index(['pair', 'method'])
    dyna = read(BASE/'dynasemble_review_audit/summary.csv').set_index(['pair', 'method'])
    pooled, effects, cells, checks = [], [], [], []
    for pair, label in PAIRS.items():
        frame = read(BASE/'information_boundary_v2'/pair/'test_anchored/test_locked_query_rows.csv')
        assert set(frame.split) == {'test'}
        assert set(frame.score_information_contract) == {'unfiltered_features_and_normalization_v2'}
        assert set(frame.seed) == {1,2,3} and set(frame.direction) == {'head','tail'}
        assert not frame.duplicated(KEY).any()
        assert frame.groupby(['head_id','relation_id','tail_id']).size().eq(6).all()
        cols = dict(Primary='rr_a', Secondary='rr_b', **{'Equal-z':'rr_equal', 'RRF':'rr_rrf',
                    'Global':'rr_global', 'Relation':'rr_relation', 'Query-soft':'rr_query_soft_locked',
                    'ADC':'rr_anchored_locked'})
        rr = {m:frame[c].to_numpy() for m,c in cols.items()}
        for method in ('Primary','Secondary','Equal-z','RRF','Global','Relation'):
            close(rr[method].mean(), static.loc[(pair,method)].mrr)
        for method in ('Global','Query-soft','ADC'):
            close(rr[method].mean(), old.loc[(pair,method)].mrr)
        if pair in MAIN:
            maps = read(BASE/'matched_alternatives_v1/query_rows'/(pair+'.csv.gz'))
            columns = ['rr_'+m for m in ('Global','Query-soft','ADC-original','ADC+Global',*METHODS[7:12])]
            aligned = align(frame, maps, columns)
            close(aligned['rr_ADC-original'],rr['ADC'])
            close(aligned['rr_ADC+Global'],rr['ADC'])
            close(aligned['rr_Global'],rr['Global'])
            close(aligned['rr_Query-soft'],rr['Query-soft'])
            for m in METHODS[7:12]:
                rr[m] = aligned['rr_'+m].to_numpy()
                close(rr[m].mean(), matched.loc[(pair,m)].mrr)
            historical = read(BASE/'information_boundary_v2'/pair/'dynasemble/test_query_rows.csv')
            aligned = align(frame,historical,['rr_a','rr_b','rr_dynasemble'])
            close(aligned.rr_a,rr['Primary']); close(aligned.rr_b,rr['Secondary'])
            rr['R0'] = aligned.rr_dynasemble.to_numpy()
            reps = []
            for ss in (11,23,37):
                part = pd.concat([read(BASE/'dynasemble_controls_v1_review'/pair/'test'/
                              f'R3_softplus_b{bs}_s{ss}.csv') for bs in (1,2,3)], ignore_index=True)
                part = part.rename(columns={'base_seed':'seed'})
                assert set(part.selector_seed) == {ss} and set(part.variant) == {'R3_softplus'}
                aligned = align(frame,part,['rr_a','rr_b','rr_method'])
                close(aligned.rr_a,rr['Primary']); close(aligned.rr_b,rr['Secondary'])
                reps.append(aligned.rr_method.to_numpy())
            rr['R3'] = np.mean(reps,axis=0)
            for m, original in [('R0','R0_historical'),('R3','R3_softplus')]:
                close(rr[m].mean(),dyna.loc[(pair,original)].mrr)
        maximum = max(v.mean() for v in rr.values())
        for method in METHODS:
            if method not in rr: continue
            v = rr[method]
            assert np.isfinite(v).all() and ((v>0)&(v<=1)).all()
            pooled.append(dict(pair=pair,label=label,role='primary' if pair in MAIN else 'additional',
                method=method,n_observations=len(frame),base_seeds=3,selector_repetitions=3 if method=='R3' else 1,
                mrr=float(v.mean()),best_displayed_mrr=abs(v.mean()-maximum)<=1e-12))
            if method == 'ADC': continue
            delta = rr['ADC']-v
            lo,hi,status = None,None,'not computed; descriptive point estimate'
            if method == 'Global':
                lo,hi = json.loads(old.loc[(pair,'ADC')].delta_mrr_ci95)
                status = 'retained original-triple bootstrap; 10000 resamples'
            elif method == 'R3':
                lo,hi = reverse_interval(json.loads(dyna.loc[(pair,'R3_softplus')].delta_mrr_vs_adc_ci95))
                status = 'retained original-triple bootstrap; selector repetitions averaged; 10000 resamples'
            effects.append(dict(pair=pair,label=label,comparator=method,delta_adc_minus_comparator=float(delta.mean()),
                                ci_low=lo,ci_high=hi,interval_status=status))
            annotated = frame[KEY].copy(); annotated['delta']=delta
            for group in (['seed'],['direction'],['seed','direction']):
                for keys,part in annotated.groupby(group):
                    keys = keys if isinstance(keys,tuple) else (keys,)
                    cells.append(dict(pair=pair,comparator=method,scope='_'.join(group),
                                      **dict(zip(group,keys)),n=len(part),delta_adc_minus_comparator=part.delta.mean()))
        checks.append(dict(pair=pair,matched_observations=len(frame),methods=len(rr),
                           endpoint_and_summary_verified=True,complete_seed_direction_coverage=True))
        print('[PAIRED] '+pair,flush=True)

    # Preserve the returned historical benchmark without editing the user's original files.
    history = OUT/'historical_timing'; history.mkdir(exist_ok=True)
    original = BASE/'efficiency'
    for name in ('benchmark_manifest.json','efficiency_raw_repetitions.csv','efficiency_summary.csv',
                 'efficiency_by_seed.csv','hardware.json'):
        shutil.copyfile(original/name,history/name)
        bind(history/name,False)
    timing_manifest = json.loads((history/'benchmark_manifest.json').read_text())
    for name,digest in timing_manifest['output_hashes'].items(): assert sha(history/name)==digest,name
    assert timing_manifest['warmup']==3 and timing_manifest['repetitions']==5
    assert timing_manifest['queries_per_direction']==256
    raw = pd.read_csv(history/'efficiency_raw_repetitions.csv',float_precision='round_trip')
    summary = pd.read_csv(history/'efficiency_summary.csv',float_precision='round_trip')
    byseed = pd.read_csv(history/'efficiency_by_seed.csv',float_precision='round_trip')
    assert len(raw)==240 and len(summary)==16 and len(byseed)==48
    assert not raw.duplicated(['dataset','pair','method','seed','repetition']).any()
    assert set(raw.n_queries)=={512} and set(raw.seed)=={1,2,3} and set(raw.repetition)=={1,2,3,4,5}
    mapping = {'scoring_s':'scoring_time','feature_computation_s':'feature_time',
               'combiner_inference_s':'combiner_inference_time','score_combination_s':'score_combination_time',
               'combiner_only_overhead_s':'combiner_only_overhead','total_s':'total_time'}
    close(raw.combiner_only_overhead_s,raw.feature_computation_s+raw.combiner_inference_s+raw.score_combination_s)
    close(raw.queries_per_second,raw.n_queries/raw.total_s)
    for groups,reported in ((['dataset','pair','method'],summary),(['seed','dataset','pair','method'],byseed)):
        indexed = reported.set_index(groups)
        for key,part in raw.groupby(groups):
            for col,prefix in mapping.items():
                close(part[col].mean(),indexed.loc[key,prefix+'_mean_s'])
                close(part[col].std(ddof=1),indexed.loc[key,prefix+'_std_s'])
    cost = []
    for pair in MAIN:
        dataset = 'MKG-W' if pair.startswith('mkgw') else 'DB15K'
        pair_name = 'M-Hyper + NativE' if pair.endswith('native') else 'M-Hyper + AdaMF-MAT'
        block = summary[(summary.dataset==dataset)&(summary.pair==pair_name)].set_index('method')
        p,g,d,a = [block.loc[x] for x in ('Primary only','Primary + Secondary + Global alpha',
                  'Primary + Secondary + DynaSemble','Primary + Secondary + Anchored Dynamic')]
        assert a.combiner_params==14 and d.combiner_params==369
        cost.append(dict(pair=pair,label=PAIRS[pair],version='historical pre-information-boundary repair',
            primary_total_s=p.total_time_mean_s,global_total_s=g.total_time_mean_s,
            historical_dyna_total_s=d.total_time_mean_s,adc_total_s=a.total_time_mean_s,
            adc_total_std_s=a.total_time_std_s,adc_feature_s=a.feature_time_mean_s,
            adc_policy_s=a.combiner_inference_time_mean_s,adc_combination_s=a.score_combination_time_mean_s,
            adc_overhead_s=a.combiner_only_overhead_mean_s,dyna_overhead_s=d.combiner_only_overhead_mean_s,
            adc_over_primary=time_ratio(a.total_time_mean_s,p.total_time_mean_s),
            adc_over_global=time_ratio(a.total_time_mean_s,g.total_time_mean_s),
            adc_overhead_over_dyna=time_ratio(a.combiner_only_overhead_mean_s,d.combiner_only_overhead_mean_s)))

    outputs = {'main_results.csv':pd.DataFrame(pooled),'paired_effects.csv':pd.DataFrame(effects),
               'paired_seed_direction.csv':pd.DataFrame(cells),'historical_cost.csv':pd.DataFrame(cost)}
    for name,frame in outputs.items(): frame.to_csv(OUT/name,index=False,lineterminator='\n')
    assert len(pooled)==76 and len(effects)==70 and len(cells)==770
    audit = dict(version='claims_cost_review_v1',status='claims_cost_checks_passed',failures=[],
                 primary_pairs=4,additional_pairs=2,method_pair_cells=len(pooled),paired_effects=len(effects),
                 paired_seed_direction_cells=len(cells),checks=checks,training_runs=0,new_bootstrap_runs=0,
                 test_used_for_new_selection=False,pair_partition_before_test_established=False,
                 historical_timing_rows_verified=240,current_corrected_timing_available=False,
                 historical_dyna_is_healthy_r3=False,sources=sources,
                 outputs={n:sha(OUT/n) for n in outputs})
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    build(outputs,audit)


def build(outputs,audit):
    tables = {}
    def table(name,caption,columns,header,rows):
        text = ('\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
                '\\caption{'+caption+'}\n\\label{tab:claims-'+name+'}\n\\begin{tabular}{'+columns+'}\n'
                '\\toprule\n'+header+'\\\\\n\\midrule\n'+
                '\n'.join(' & '.join(r)+r'\\' if r else r'\midrule' for r in rows)+
                '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
        path=TABLES/(name+'.tex'); path.write_text(text,encoding='utf-8',newline='\n')
        tables[path.relative_to(PAPER).as_posix()]=sha(path)
    main = outputs['main_results.csv'].set_index(['pair','method'])
    effects = outputs['paired_effects.csv'].set_index(['pair','comparator'])
    rows=[]
    for m in METHODS:
        row=[LABELS[m]]
        for p in PAIRS:
            if (p,m) not in main.index: row.append('--'); continue
            r=main.loc[(p,m)]; value=f'{r.mrr:.6f}'
            row.append(r'\textbf{'+value+'}' if r.best_displayed_mrr else value)
        rows.append(row)
        if m in ('Relation','Linear-g','R3'): rows.append([])
    table('main',r'Complete corrected TEST MRR on four primary pairs and two additional architecture pairs. '
          r'All use three equally weighted base seeds and both directions; R3 first averages three selector repetitions per base observation. '
          r'Equal/RRF are fixed; Global, Relation and adaptive settings use DEV choices (budgets in Table~\ref{tab:matched-budget}). '
          r'ADC + Global coincides with ADC. R0 retains failed seeds; R3 is healthy. '
          r'Bold marks the greatest unrounded MRR among displayed deployable methods, including ties within $10^{-12}$; it is not a significance claim. '
          r'Displayed ties can differ below six decimals. -- means not run. Column roles are retrospective and were not established as TEST-blind.',
          'lrrrr|rr',r' & \multicolumn{4}{c|}{Primary} & \multicolumn{2}{c}{Additional}\\'+
          '\nMethod & '+' & '.join(PAIRS.values()),rows)
    rows=[]
    for m in METHODS[:-1]:
        rows.append([LABELS[m],*[f'{effects.loc[(p,m)].delta_adc_minus_comparator:+.6f}' if (p,m) in effects.index else '--' for p in PAIRS]])
    table('paired',r'Complete paired effects: ADC minus each displayed comparator on identical TEST observations. '
          r'Positive favors ADC; all 70 available comparisons are retained. These are descriptive point estimates. '
          r'Original-triple intervals for Global appear in Table~\ref{tab:claims-intervals}; R3 intervals appear in Table~\ref{tab:dyna-effects}. '
          r'No additional intervals were computed. The supplement supplies all three seed, two direction and six joint cells per comparison.',
          'lrrrr|rr','Comparator & '+' & '.join(PAIRS.values()),rows)
    rows=[]
    for p,label in PAIRS.items():
        r=effects.loc[(p,'Global')]
        rows.append([label,'Primary' if p in MAIN else 'Additional',f'{r.delta_adc_minus_comparator:+.6f}',
                     f'[{r.ci_low:+.6f}, {r.ci_high:+.6f}]'])
    table('intervals',r'ADC minus Global for all four primary and two additional pairs. '
          r'Retained 95\% original-triple bootstrap intervals use 10,000 resamples and include all three base seeds and both directions within a triple. '
          r'D-N and both additional pairs have intervals spanning zero. These pointwise conditional intervals do not account for TEST reuse or adaptive design.',
          'llrl',r'Pair & Role & $\Delta$MRR & 95\% interval',rows)
    rows=[]
    for r in outputs['historical_cost.csv'].itertuples():
        rows.append([r.label,f'{r.primary_total_s:.3f}',f'{r.global_total_s:.3f}',f'{r.adc_total_s:.3f}',
            f'{r.adc_over_primary:.2f}',f'{r.adc_over_global:.2f}',f'{r.adc_overhead_s:.3f}',f'{r.dyna_overhead_s:.3f}'])
    table('cost',r'Historical cost audit, before the information-boundary repair; not current corrected timing. '
          r'Times are mean seconds per 512 sampled directional queries over three seeds and five measured repetitions per seed on an A100 (three warm-ups). '
          r'$T$ is measured total wall time; $O$ includes features, policy inference and combination/ranking. '
          r'Dyna here is the historical selector, not healthy R3. Ratios divide mean wall times; full stage means/SDs and raw measurements are supplied.',
          'lrrrrrrr',r'Pair & $T_P$ & $T_G$ & $T_{ADC}$ & $T_{ADC}/T_P$ & $T_{ADC}/T_G$ & $O_{ADC}$ & $O_{Dyna}$',rows)
    sources=dict(audit['sources'])
    for p in OUT.rglob('*'):
        if p.is_file(): sources[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ('scripts/audit_paper_a_claims_cost.py','docs/protocols/paper_a_claims_cost_review.md',
                'docs/reports/paper_a_claims_cost_review_2026-09-12.md',
                'tests/test_claims_cost_audit.py','scripts/benchmark_paper_a_efficiency.py',
                'docs/protocols/paper_a_test_history_review.json'):
        sources[rel]=sha(ROOT/rel)
    manifest=dict(version='claims_cost_review_v1',sources=sources,tables=tables)
    (PAPER/'claims_cost_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources','outputs','checks')}))


if __name__=='__main__': main()
