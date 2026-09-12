"""D07/D09/D10: frozen ADC harm denominators, complete utility and loss tails."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.outcome_risk_diagnostics import observations,summarize,cluster_ratio_intervals,FIELDS as RF,RATIOS
from scripts.analyze_paper_a_rejection_controls import replay
from scripts.analyze_paper_a_conservative_radius import INPUT,sha,require,close
from scripts.analyze_paper_a_matched_alternatives import Inputs
from scripts.analyze_paper_a_winner_signal import PAIRS
from scripts.audit_paper_a_claims_cost import KEY

OUT=ROOT/'outputs/paper_a_safe_correction/outcome_risk_v1'
CODE=('scripts/analyze_paper_a_outcome_risk.py','router/outcome_risk_diagnostics.py',
    'tests/test_outcome_risk_diagnostics.py','docs/protocols/paper_a_outcome_risk_review.md',
    'scripts/analyze_paper_a_rejection_controls.py','scripts/analyze_paper_a_conservative_radius.py',
    'scripts/analyze_paper_a_matched_alternatives.py','scripts/analyze_paper_a_winner_signal.py',
    'router/matched_actions.py')


def assert_accounting(r):
    require(r['harm_n']+r['benefit_n']+r['unchanged_n']==r['n'],'Incomplete RR-sign partition')
    require(r['active_n']+r['inactive_n']==r['n'],'Incomplete action partition')
    require(r['active_unchanged_n']+r['harm_n']+r['benefit_n']==r['active_n'],'Action and outcome partitions differ')
    require(r['top1_retained_n']+r['top1_lost_n']==r['top1_n'],'Top-1 partition differs')
    residuals=[r['mean_gain']-r['mean_loss']-r['utility']]
    if r['harm_n']:residuals.append(r['harm']*r['conditional_loss']-r['mean_loss'])
    if r['benefit_n']:residuals.append(r['benefit']*r['conditional_gain']-r['mean_gain'])
    require(max(abs(v) for v in residuals)<1e-12,'Gain/loss identity differs')
    return max(abs(v) for v in residuals)


def main():
    require(not (OUT/'audit.json').exists(),'Completed outcome review is immutable')
    OUT.mkdir(parents=True,exist_ok=True);inputs=Inputs('DIAGNOSTIC');sources={r:sha(ROOT/r) for r in CODE}
    expected={}
    for name in ('ties_harm','query_pair'):
        path=ROOT/f'paper_a_draft/{name}_source_manifest.json';sources[path.relative_to(ROOT).as_posix()]=sha(path)
        expected.update(json.loads(path.read_text())['sources'])
    def bind(path):
        rel=path.relative_to(ROOT).as_posix();require(sha(path)==expected.get(rel),'Changed evidence '+rel)
        sources[rel]=sha(path);return path
    states=json.loads(bind(INPUT.parent/'winner_signal_review_v1/fold_model_states.json').read_text())
    previous=pd.read_csv(bind(INPUT.parent/'ties_harm_review_v1/tie_summary.csv'),float_precision='round_trip').set_index(['pair','split','population'])
    prior_harm=pd.read_csv(bind(INPUT.parent/'ties_harm_review_v1/harm_discrimination.csv'),float_precision='round_trip').set_index(['pair','split','population'])
    paired=pd.read_csv(bind(INPUT.parent/'query_pair_v1/paired_intervals.csv'),float_precision='round_trip').set_index(['pair','comparator'])
    rows=[];cells=[];examples=[];checks=[];residuals=[];start=time.monotonic()
    for pair,label in PAIRS.items():
        for split in ('dev_oof','test'):
            f,folds,anchor,alpha,rr,ref,raw,raw_rr,settings=replay(inputs,INPUT/pair,pair,split,states)
            x,rank,base=observations(rr,ref,alpha,anchor);r=summarize(x);residuals.append(assert_accounting(r))
            ids,unique=pd.factorize(pd.MultiIndex.from_frame(f[['head_id','relation_id','tail_id']]),sort=True)
            require(np.all(np.bincount(ids)==6),'Incomplete six-observation clusters')
            old=previous.loc[pair,split,'all'];executed=prior_harm.loc[pair,split,'executed']
            for col,newcol in [('n','n'),('active_n','active_n'),('harm_n','harm_n'),('benefit_n','benefit_n'),
                ('harm_rate','harm'),('benefit_rate','benefit'),('mean_loss','mean_loss'),('conditional_loss','conditional_loss'),('utility','utility')]:
                close(r[newcol],old[col],'Prior point mismatch: '+col)
            close(r['active_n'],executed.n,'Executed denominator mismatch')
            close(r['harm_active'],executed.prevalence,'Executed harm prevalence mismatch')
            support={}
            for name,col in [('full','n'),('active','active_n'),('harmful','harm_n'),('beneficial','benefit_n'),('top1','top1_n'),('active_top1','active_top1_n')]:
                support[name+'_triples']=len(np.unique(ids[x[:,RF.index(col)]>0]))
            seed=(2026091212 if split=='test' else 2026091214)+(pair.startswith('db15k'))
            ci=cluster_ratio_intervals(x,ids,seed)
            order_hash=hashlib.sha256(''.join(','.join(map(str,key))+'\n' for key in unique).encode()).hexdigest()
            if split=='test':
                p=paired.loc[pair,'Global'];close(r['utility'],p.delta,'Unified point mismatch')
                close([ci['utility_lo'],ci['utility_hi']],[p.lo,p.hi],'Unified C12 interval mismatch')
                require(order_hash==p.cluster_order_sha256 and seed==p.seed,'Unified C12 resampling scope differs')
                require(np.array_equal(base,f.rank_global.to_numpy()),'Recovered reference integer rank differs')
                case=f[KEY].copy();case['rank_global']=base;case['rank_adc']=rank;case['rr_global']=ref;case['rr_adc']=rr
                case['loss']=ref-rr;case['alpha0']=anchor;case['alpha_adc']=alpha
                case=case[case.loss>0].sort_values(['loss',*KEY],ascending=[False]+[True]*len(KEY),kind='stable').head(3)
                examples.extend(dict(pair=pair,label=label,order=i+1,**v) for i,v in enumerate(case.to_dict('records')))
            context=dict(pair=pair,label=label,split=split)
            rows.append(dict(**context,**r,**support,**ci,bootstrap_seed=seed,bootstrap_replicates=10000,
                cluster_order_sha256=order_hash,quantile='linear',rng='PCG64'))
            for base_seed in (1,2,3):
                for direction in ('head','tail'):
                    mask=((f.seed==base_seed)&(f.direction==direction)).to_numpy()
                    cell=summarize(x[mask]);residuals.append(assert_accounting(cell))
                    cells.append(dict(**context,seed=base_seed,direction=direction,**cell))
            checks.append(dict(**context,observations=len(f),clusters=len(unique),actions_and_rr_replayed=True,
                prior_counts_and_points_reconciled=True,inactive_rr_changes=0,exact_sign_matches_tolerance=True,
                integer_ranks_verified=True,query_ci_reconciled_with_c12=(split=='test'),
                minimum_valid_bootstrap_replicates=min(ci[n+'_valid_replicates'] for n in RATIOS)))
            print(f'[OUTCOME RISK] {label} {split}: harm {r["harm_n"]}/{r["n"]}; active {r["harm_n"]}/{r["active_n"]}; 10,000 draws ({time.monotonic()-start:.1f}s)',flush=True)
    require(len(rows)==12 and len(cells)==72 and len(examples)==18,'Incomplete risk inventory')
    require(sum(c['observations'] for c in checks)==474732,'Incomplete query coverage')
    outputs={}
    for name,values in [('summary',rows),('seed_direction',cells),('worst_examples',examples)]:
        path=OUT/(name+'.csv');pd.DataFrame(values).to_csv(path,index=False,lineterminator='\n')
        outputs[path.relative_to(ROOT).as_posix()]=sha(path)
    sources.update(inputs.sources)
    audit=dict(status='outcome_risk_checks_passed',failures=[],sources=sources,outputs=outputs,checks=checks,
        pairs=6,splits=2,pooled_rows=12,seed_direction_cells=72,worst_examples=18,bootstrap_replicates=10000,
        ratio_metrics=list(RATIOS),max_decomposition_residual=max(residuals),observations=474732,
        thresholds_rr=[.1,.5],threshold_event_arithmetic='exact integer-rank cross-products',
        conditional_quantiles=[.5,.9,.95,.99],top1_definition='filtered gold rank under the shared evaluator',
        selector_fits=0,scorer_runs=0,new_policy_selection=False,test_used_for_selection=False,
        historical_results_replaced=False,simultaneous_intervals=False,population_safety_guarantee=False,
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),elapsed_seconds=time.monotonic()-start)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
