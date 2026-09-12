"""D11/D12: corrected nested coverage, random order and AP/aggregation audit."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.coverage_quality import NestedOrder, COUNTS, METRICS, rates, aggregate, distribution
from router.rejection_diagnostics import observable_units, stratum_ids
from router.ties_harm_diagnostics import RankedBinary
from scripts.analyze_paper_a_confidence_harm import average_precision
from scripts.analyze_paper_a_rejection_controls import replay
from scripts.analyze_paper_a_conservative_radius import INPUT,sha,require,close
from scripts.analyze_paper_a_matched_alternatives import Inputs
from scripts.analyze_paper_a_winner_signal import PAIRS,FIELDS
from scripts.audit_paper_a_claims_cost import KEY

OUT=ROOT/'outputs/paper_a_safe_correction/coverage_quality_v1'
CODE=('scripts/analyze_paper_a_coverage_quality.py','router/coverage_quality.py',
      'tests/test_coverage_quality.py','docs/protocols/paper_a_coverage_quality_review.md',
      'scripts/analyze_paper_a_confidence_harm.py','router/ties_harm_diagnostics.py',
      'router/rejection_diagnostics.py','scripts/analyze_paper_a_rejection_controls.py',
      'scripts/analyze_paper_a_winner_signal.py','scripts/analyze_paper_a_conservative_radius.py',
      'scripts/analyze_paper_a_matched_alternatives.py','router/matched_actions.py')


def margin(f,folds,pair,split,states):
    if split=='test':return f.anchored_decision.to_numpy()
    g=np.full(len(f),np.nan)
    for fold in range(1,6):
        mask=folds==fold
        state=next(s for s in states if s['pair']==pair and s['fold']==fold and s['learner']=='balanced')
        require(state['fields']==list(FIELDS),'Feature order differs')
        x=f.loc[mask,list(FIELDS)].to_numpy()
        g[mask]=((x-np.array(state['mean']))/np.array(state['scale'])) @ np.array(state['coefficients'])[0]+state['intercept'][0]
    return g


def curve_records(context,point,expected,draws,n):
    rows=[];r=rates(point,n);random=rates(draws,n)
    for d in range(11):
        base=dict(**context,decile=d,quota_fraction=d/10,n=n)
        rows.append(dict(**base,method='Confidence',**dict(zip(COUNTS,point[d])),**dict(zip(METRICS,r[d]))))
        rand=dict(**base,method='Random',**dict(zip(COUNTS,draws[:,d].mean(axis=0))),**distribution(random[:,d]))
        rand.update({name+'_exact_expectation':float(expected[d,j]) for j,name in enumerate(COUNTS)})
        rows.append(rand)
    return rows


def main():
    require(not (OUT/'audit.json').exists(),'Completed coverage review is immutable')
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=Inputs('DIAGNOSTIC');sources={r:sha(ROOT/r) for r in CODE}
    manifest=ROOT/'paper_a_draft/ties_harm_source_manifest.json'
    expected=json.loads(manifest.read_text())['sources'];sources[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
    def bind(path):
        rel=path.relative_to(ROOT).as_posix();require(sha(path)==expected.get(rel),'Changed prior evidence '+rel)
        sources[rel]=sha(path);return path
    states=json.loads(bind(INPUT.parent/'winner_signal_review_v1/fold_model_states.json').read_text())
    prior=pd.read_csv(bind(INPUT.parent/'ties_harm_review_v1/harm_discrimination.csv'),float_precision='round_trip').set_index(['pair','split','population'])
    curves=[];aggregates=[];ap_rows=[];weights=[];checks=[];primary={};ap_primary={};observed={};start=time.monotonic()
    for pi,(pair,label) in enumerate(PAIRS.items()):
        for si,split in enumerate(('dev_oof','test')):
            f,folds,anchor,alpha,rr,ref,raw,raw_rr,settings=replay(inputs,INPUT/pair,pair,split,states)
            confidence=np.abs(2*expit(margin(f,folds,pair,split,states))-1)
            require(np.isfinite(confidence).all(),'Nonfinite confidence')
            raw_delta=raw_rr-ref;active=np.abs(raw-anchor)>1e-12;harm=raw_delta < -1e-12
            require(np.array_equal(harm,raw_delta<0),'Exact harm signs differ')
            close(raw_delta[~active],0,'Inactive raw proposal alters RR')
            units,inv=observable_units(f.seed,folds,f.direction,f.relation_id,
                                      np.where(f.direction=='tail',f.head_id,f.tail_id))
            first=np.full(len(units),len(f));np.minimum.at(first,inv,np.arange(len(f)))
            for x in (raw,anchor,alpha):close(x[first][inv],x,'Same observable query action differs')
            cmin=np.full(len(units),np.inf);cmax=np.full(len(units),-np.inf)
            np.minimum.at(cmin,inv,confidence);np.maximum.at(cmax,inv,confidence)
            # Repeated exports can have small numeric score drift. A unit mean
            # depends on observed repetitions, never on which gold occurs first.
            unit_confidence=np.bincount(inv,weights=confidence)/units.repetitions.to_numpy()
            # Gold replacement does not enter the unit keys or priorities.
            head,tail=f.head_id.to_numpy(copy=True),f.tail_id.to_numpy(copy=True)
            head[f.direction=='head']=-999;tail[f.direction=='tail']=-999
            other,oi=observable_units(f.seed,folds,f.direction,f.relation_id,np.where(f.direction=='tail',head,tail))
            require(other.equals(units) and np.array_equal(oi,inv),'Gold affects grouping')
            observations=np.column_stack([np.ones(len(f)),active,harm,raw_delta,np.maximum(-raw_delta,0),abs(raw-anchor)])
            values=np.column_stack([np.bincount(inv,weights=observations[:,j],minlength=len(units)) for j in range(6)])
            tie_key=np.random.default_rng(2026091250+2*pi+si).random(len(units))
            for ai,pool in enumerate(('All','Active')):
                eligible=np.ones(len(units),dtype=bool) if pool=='All' else active[first]
                u=units[eligible];v=values[eligible]
                strata=stratum_ids(dict(seed=u.seed,fold=u.fold,repetitions=u.repetitions))
                gate=NestedOrder(strata,unit_confidence[eligible],tie_key[eligible])
                point,expected_totals=gate.totals(v)
                seed=2026091200+4*pi+2*si+ai
                draws=gate.random_totals(v,seed)
                close(draws[:,:,0],np.broadcast_to(point[:,0],draws[:,:,0].shape),'Random acceptance budget differs')
                if pool=='Active':close(draws[:,:,1],np.broadcast_to(point[:,1],draws[:,:,1].shape),'Random intervention budget differs')
                require(np.all(np.diff(point[:,2])>=0) and np.all(np.diff(draws[:,:,2],axis=1)>=0),'Non-nested cumulative harm')
                close(point[0],0,'Zero coverage is not Global');close(draws[:,0],0,'Random zero differs')
                close(point[-1,1:3],observations.sum(axis=0)[1:3],'Full acceptance counts differ')
                close(point[-1,1:]/len(f),observations.sum(axis=0)[1:]/len(f),'Full acceptance differs from no-gate')
                close(draws[:,-1]/len(f),np.broadcast_to(point[-1]/len(f),draws[:,-1].shape),'Random endpoint differs')
                context=dict(pair=pair,label=label,split=split,pool=pool,eligible_units=int(eligible.sum()),
                    strata=len(gate.groups),random_seed=seed,random_draws=512)
                curves.extend(curve_records(context,point,expected_totals,draws,len(f)))
                if pi<4:primary.setdefault((split,pool),[]).append((pair,label,len(f),point,draws))
            for population,mask in [('all',np.ones(len(f),dtype=bool)),('proposed',active),('executed',abs(alpha-anchor)>1e-12)]:
                y=harm[mask];score=1-confidence[mask]
                sk=average_precision_score(y,score);legacy=average_precision(y,score)
                rb=RankedBinary(y,score,np.arange(len(y))).evaluate(np.ones(len(y)))
                old=prior.loc[pair,split,population]
                close([sk,legacy,rb[2]],[old.ap]*3,'AP implementation or historical point differs')
                close([y.mean(),sk-y.mean()],[old.prevalence,old.ap_lift],'AP lift/prevalence differs')
                ap_rows.append(dict(scope='pair',label=label,pair=pair,split=split,population=population,
                    n=len(y),harm_n=int(y.sum()),prevalence=float(y.mean()),ap=sk,ap_lift=sk-y.mean(),sklearn_error=sk-old.ap))
                if pi<4:ap_primary.setdefault((split,population),[]).append((y,score))
            if pi<4:
                keys=list(map(tuple,f[KEY].sort_values(KEY).to_numpy()))
                dataset='DB15K' if pair.startswith('db15k') else 'MKG-W'
                key=(dataset,split)
                if key in observed:require(keys==observed[key],'Primary pairs use different observations')
                else:observed[key]=keys
                weights.append(dict(pair=pair,label=label,dataset=dataset,split=split,n=len(f),original_triples=len(f)//6,macro_weight=.25))
            checks.append(dict(pair=pair,label=label,split=split,observations=len(f),units=len(units),
                gold_invariant_units=True,same_query_actions_verified=True,raw_and_locked_replayed=True,
                nested_harm_verified=True,random_budgets_exact=True,zero_and_full_endpoints_verified=True,
                ap_populations_verified=3,invalid_rows=0,
                repeated_confidence_max_range=float((cmax-cmin).max()),
                repeated_units_with_confidence_range_over_1e_12=int(((cmax-cmin)>1e-12).sum())))
            print(f'[COVERAGE] {label} {split}: two pools x 512 nested orders; AP verified ({time.monotonic()-start:.1f}s)',flush=True)
    for (split,pool),items in primary.items():
        require(len(items)==4,'Incomplete primary aggregate')
        sizes=[v[2] for v in items];points=np.stack([v[3] for v in items]);draws=np.stack([v[4] for v in items])
        for mode in ('macro','micro'):
            p=aggregate(points,sizes,mode);r=aggregate(draws,sizes,mode)
            for d in range(11):
                base=dict(split=split,pool=pool,aggregation=mode,decile=d,quota_fraction=d/10,pairs=4,n=sum(sizes))
                aggregates.append(dict(**base,method='Confidence',**dict(zip(METRICS,p[d]))))
                aggregates.append(dict(**base,method='Random',**distribution(r[:,d])))
    for (split,population),items in ap_primary.items():
        y=np.concatenate([v[0] for v in items]);s=np.concatenate([v[1] for v in items])
        for mode in ('macro','micro'):
            ap=average_precision_score(y,s) if mode=='micro' else np.mean([average_precision_score(a,b) for a,b in items])
            prev=y.mean() if mode=='micro' else np.mean([a.mean() for a,b in items])
            ap_rows.append(dict(scope=mode,label='Primary four',pair='',split=split,population=population,
                n=len(y),harm_n=int(y.sum()),prevalence=prev,ap=ap,ap_lift=ap-prev))
    for row in weights:
        total=sum(r['n'] for r in weights if r['split']==row['split'])
        row.update(micro_full_weight=row['n']/total,pooled_observations=total,
            unique_dataset_observations=total//2,unique_dataset_triples=total//12,
            same_dataset_pair_observation_overlap=1.)
    require(len(curves)==528 and len(aggregates)==176 and len(ap_rows)==48 and len(weights)==8,'Incomplete output inventory')
    require(sum(r['observations'] for r in checks)==474732,'Incomplete observation inventory')
    outputs={}
    for name,rows in [('curves',curves),('aggregates',aggregates),('average_precision',ap_rows),('weights',weights)]:
        p=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False,lineterminator='\n');outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    sources.update(inputs.sources)
    audit=dict(status='coverage_quality_checks_passed',failures=[],sources=sources,outputs=outputs,checks=checks,
        curve_rows=528,aggregate_rows=176,ap_rows=48,weight_rows=8,observations=474732,
        random_replicates=512,deciles=list(range(11)),pools=['All','Active'],selector_fits=0,scorer_runs=0,
        policy_selection=False,test_used_for_selection=False,historical_results_replaced=False,
        random_bands_are_population_ci=False,pairs_are_independent=False,
        ap_definition='non-interpolated tie-grouped average precision',ap_lift_definition='AP minus same-population prevalence',
        primary_pair_overlap_verified=True,random_conditional_ratios_computed_per_draw=True,
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        elapsed_seconds=time.monotonic()-start)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
