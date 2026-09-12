"""E02: change one setting at a time, then equalize actual intervention budgets."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.component_ablation import METHODS,COMPARATORS,component_weights,match_interventions,outcome_summary
from router.rejection_diagnostics import observable_units,stratum_ids
from scripts.analyze_paper_a_rejection_controls import replay
from scripts.analyze_paper_a_coverage_quality import margin
from scripts.analyze_paper_a_query_pair import cluster_intervals
from scripts.analyze_paper_a_conservative_radius import INPUT,GRID_COLUMNS,sha,require,close
from scripts.analyze_paper_a_matched_alternatives import Inputs
from scripts.analyze_paper_a_winner_signal import PAIRS

OUT=ROOT/'outputs/paper_a_safe_correction/component_ablation_v1'
CODE=('scripts/analyze_paper_a_component_ablation.py','router/component_ablation.py',
    'tests/test_component_ablation.py','docs/protocols/paper_a_component_ablation_review.md',
    'scripts/analyze_paper_a_rejection_controls.py','scripts/analyze_paper_a_coverage_quality.py',
    'scripts/analyze_paper_a_query_pair.py','router/rejection_diagnostics.py','router/matched_actions.py',
    'scripts/analyze_paper_a_conservative_radius.py','scripts/analyze_paper_a_matched_alternatives.py',
    'scripts/analyze_paper_a_winner_signal.py')


def main():
    require(not (OUT/'audit.json').exists(),'Completed component review is immutable')
    OUT.mkdir(parents=True,exist_ok=True);inputs=Inputs('DIAGNOSTIC')
    sources={r:sha(ROOT/r) for r in CODE};expected={}
    for name in ('ties_harm','query_pair','outcome_risk'):
        p=ROOT/f'paper_a_draft/{name}_source_manifest.json';sources[p.relative_to(ROOT).as_posix()]=sha(p)
        expected.update(json.loads(p.read_text())['sources'])
    def bind(p):
        rel=p.relative_to(ROOT).as_posix();require(sha(p)==expected.get(rel),'Changed prior evidence '+rel)
        sources[rel]=sha(p);return p
    states=json.loads(bind(INPUT.parent/'winner_signal_review_v1/fold_model_states.json').read_text())
    prior=pd.read_csv(bind(INPUT.parent/'query_pair_v1/paired_intervals.csv'),float_precision='round_trip').set_index(['pair','comparator'])
    outcomes=pd.read_csv(bind(INPUT.parent/'outcome_risk_v1/summary.csv'),float_precision='round_trip').set_index(['pair','split'])
    summaries=[];cells=[];effects=[];allocations=[];settings_rows=[];checks=[];started=time.monotonic()
    for pair,label in PAIRS.items():
        for split in ('dev_oof','test'):
            f,folds,a,old_alpha,old_rr,ref,raw,raw_rr,settings=replay(inputs,INPUT/pair,pair,split,states)
            g=margin(f,folds,pair,split,states);require(np.isfinite(g).all(),'Nonfinite margin')
            units,inv=observable_units(f.seed,folds,f.direction,f.relation_id,np.where(f.direction=='tail',f.head_id,f.tail_id))
            first=np.full(len(units),len(f));np.minimum.at(first,inv,np.arange(len(f)))
            ug=np.bincount(inv,weights=g)/units.repetitions.to_numpy();up=expit(ug)
            context=dict(pair=pair,label=label,split=split)
            unit_actions={m:np.full(len(units),np.nan) for m in METHODS}
            original_wide=np.full(len(f),np.nan)
            for s in settings:
                mask=units.fold.to_numpy()==s['fold'];rowmask=folds==s['fold']
                for m in METHODS:
                    unit_actions[m][mask]=component_weights(ug[mask],up[mask],np.zeros(mask.sum(),bool),s['anchor'],s['beta'],s['tau'],m)
                original_wide[rowmask]=component_weights(g[rowmask],expit(g[rowmask]),np.zeros(rowmask.sum(),bool),s['anchor'],s['beta'],s['tau'],'Radius-1')
                settings_rows.append(dict(**context,**s))
            actions={m:v[inv] for m,v in unit_actions.items()}
            require(np.array_equal(actions['ADC'],old_alpha),'Unit mean changes original ADC action')
            require(np.array_equal(actions['No-gate'],raw),'Unit mean changes original no-gate action')
            require(np.array_equal(actions['Radius-1'],original_wide),'Unit mean changes expanded-radius action')
            grid=f[GRID_COLUMNS].to_numpy();rrs={m:grid[np.arange(len(f)),np.rint(v*20).astype(int)] for m,v in actions.items()}
            require(np.array_equal(rrs['ADC'],old_rr) and np.array_equal(rrs['No-gate'],raw_rr),'Original RRs differ')
            require(np.array_equal(rrs['Global'],ref),'Static reference differs')
            require(all(np.array_equal(v!=a,abs(v-a)>1e-12) for v in actions.values()),'Movement sign tolerance mismatch')
            require(all(np.array_equal(r<ref,r-ref < -1e-12) for r in rrs.values()),'Harm sign tolerance mismatch')
            # Gold replacement leaves observable allocation inputs unchanged.
            h,t=f.head_id.to_numpy(copy=True),f.tail_id.to_numpy(copy=True)
            h[f.direction=='head']=-999;t[f.direction=='tail']=-999
            other,oi=observable_units(f.seed,folds,f.direction,f.relation_id,np.where(f.direction=='tail',h,t))
            require(other.equals(units) and np.array_equal(inv,oi),'Gold affects unit grouping')
            native={}
            for m in METHODS:
                r=outcome_summary(rrs[m],ref,actions[m],a);native[m]=r;summaries.append(dict(**context,method=m,**r))
                close(r['mean_gain']-r['mean_loss'],r['utility'],'Utility decomposition mismatch')
                for seed in (1,2,3):
                    for direction in ('head','tail'):
                        mask=((f.seed==seed)&(f.direction==direction)).to_numpy()
                        cells.append(dict(**context,method=m,seed=seed,direction=direction,
                            **outcome_summary(rrs[m][mask],ref[mask],actions[m][mask],a[mask])))
            previous=outcomes.loc[pair,split]
            for name in ('n','active_n','harm_n','benefit_n','mrr','utility','harm','harm_active','intervention','mean_loss','mean_gain'):
                if name in previous:close(native['ADC'][name],previous[name],'Prior ADC statistic differs: '+name)
            strata=stratum_ids(dict(seed=units.seed,fold=units.fold,repetitions=units.repetitions))
            difference_columns=[];pending=[]
            for m in COMPARATORS:
                lp,rp,ledger=match_interventions(strata,units.repetitions,unit_actions['ADC']!=a[first],unit_actions[m]!=a[first])
                allocations.extend(dict(**context,comparator=m,**row) for row in ledger)
                match_n=sum(r['retained_observations'] for r in ledger)
                for regime in ('Native','Matched-I'):
                    left,right=(np.ones(len(f)),np.ones(len(f))) if regime=='Native' else (lp[inv],rp[inv])
                    lstat=outcome_summary(rrs['ADC'],ref,actions['ADC'],a,left)
                    rstat=outcome_summary(rrs[m],ref,actions[m],a,right)
                    if regime=='Matched-I':
                        require(np.allclose([lstat['active_n'],rstat['active_n']],match_n,atol=1e-8,rtol=0),'Unmatched intervention count')
                        # The integer quota is exact in every realization; avoid
                        # exposing summation roundoff as fractional interventions.
                        for stat in (lstat,rstat):
                            stat.update(active_n=match_n,intervention=match_n/len(f),
                                harm_active=stat['harm_n']/match_n if match_n else np.nan)
                    difference_columns.append(left*(rrs['ADC']-ref)-right*(rrs[m]-ref))
                    pending.append(dict(**context,comparator=m,regime=regime,
                        **{'adc_'+k:v for k,v in lstat.items()},**{'other_'+k:v for k,v in rstat.items()},
                        native_adc_active_n=native['ADC']['active_n'],native_other_active_n=native[m]['active_n'],
                        match_active_n=match_n,match_strata=len(ledger),zero_quota_strata=sum(r['retained_units']==0 for r in ledger),
                        adc_fraction_retained=match_n/native['ADC']['active_n'] if native['ADC']['active_n'] else np.nan,
                        other_fraction_retained=match_n/native[m]['active_n'] if native[m]['active_n'] else np.nan))
            seed=(2026091212 if split=='test' else 2026091214)+int(pair.startswith('db15k'))
            point,lo,hi,metadata=cluster_intervals(f,np.column_stack(difference_columns),seed)
            for j,r in enumerate(pending):
                close(point[j],r['adc_utility']-r['other_utility'],'Paired expectation differs')
                r.update(delta=float(point[j]),lo=float(lo[j]),hi=float(hi[j]),**metadata)
                if split=='test' and r['regime']=='Native' and r['comparator'] in ('Radius-1','No-gate'):
                    old=prior.loc[pair,{'Radius-1':'Expanded-radius','No-gate':'No-fallback'}[r['comparator']]]
                    close([r['delta'],r['lo'],r['hi']],[old.delta,old.lo,old.hi],'Unified C12 interval differs')
                    require(r['cluster_order_sha256']==old.cluster_order_sha256 and seed==old.seed,'Unified cluster seed/order differs')
                effects.append(r)
            checks.append(dict(**context,observations=len(f),units=len(units),original_actions_and_rr_exact=True,
                unit_mean_preserves_original_three_paths=True,gold_invariant_units=True,matching_quotas_exact=True,
                prior_adc_points_reconciled=True,c12_radius_and_gate_intervals_reconciled=(split=='test'),
                fitted_objects=len(settings),selector_fits=0,scorer_runs=0))
            print(f'[COMPONENT] {label} {split}: 8 policies, 12 effects, 10,000 common draws ({time.monotonic()-started:.1f}s)',flush=True)
    require(len(summaries)==96 and len(cells)==576 and len(effects)==144 and len(settings_rows)==36,'Incomplete component inventory')
    require(sum(c['observations'] for c in checks)==474732,'Incomplete observation count')
    outputs={}
    for name,rows in [('summary',summaries),('effects',effects),('seed_direction',cells),('matching_quotas',allocations),('original_settings',settings_rows)]:
        p=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False,lineterminator='\n');outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    sources.update(inputs.sources)
    audit=dict(status='component_ablation_checks_passed',failures=[],sources=sources,outputs=outputs,checks=checks,
        native_policy_rows=96,seed_direction_rows=576,effect_rows=144,setting_rows=36,quota_rows=len(allocations),
        observations=474732,methods=list(METHODS),comparators=list(COMPARATORS),bootstrap_replicates=10000,
        selector_fits=0,scorer_runs=0,new_policy_selection=False,test_used_for_selection=False,
        historical_results_replaced=False,matched_rows_are_exact_discrete_expectations=True,
        matched_budgets_are_pairwise=True,thinning_reallocated_in_bootstrap=False,multiplicity_adjusted=False,
        center_variant_retains_original_fallback_and_projection_reference=True,
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),elapsed_seconds=time.monotonic()-started)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
