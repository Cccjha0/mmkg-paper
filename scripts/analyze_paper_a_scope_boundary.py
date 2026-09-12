"""E03/E04: original DEV local-strength strata and locked additional-pair stages."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.scope_boundary_diagnostics import TRIPLE,BINS,training_map,assign_strata,stage_summary,conditional_intervals
from scripts.analyze_paper_a_rejection_controls import replay
from scripts.analyze_paper_a_coverage_quality import margin
from scripts.analyze_paper_a_query_pair import triple_hash
from scripts.analyze_paper_a_conservative_radius import INPUT,GRID_COLUMNS,sha,require,close
from scripts.analyze_paper_a_matched_alternatives import Inputs,check_frame
from scripts.analyze_paper_a_winner_signal import PAIRS

OUT=ROOT/'outputs/paper_a_safe_correction/scope_boundary_v1'
CODE=('scripts/analyze_paper_a_scope_boundary.py','router/scope_boundary_diagnostics.py',
    'tests/test_scope_boundary_diagnostics.py','docs/protocols/paper_a_scope_boundary_review.md',
    'scripts/analyze_paper_a_rejection_controls.py','scripts/analyze_paper_a_coverage_quality.py',
    'scripts/analyze_paper_a_query_pair.py','scripts/analyze_paper_a_conservative_radius.py',
    'scripts/analyze_paper_a_matched_alternatives.py','scripts/analyze_paper_a_winner_signal.py',
    'router/matched_actions.py')


def main():
    require(not (OUT/'audit.json').exists(),'Completed scope review is immutable')
    OUT.mkdir(parents=True,exist_ok=True);inputs=Inputs('DIAGNOSTIC')
    sources={r:sha(ROOT/r) for r in CODE};expected={}
    for name in ('component_ablation','complementarity','outcome_risk'):
        p=ROOT/f'paper_a_draft/{name}_source_manifest.json';sources[p.relative_to(ROOT).as_posix()]=sha(p)
        expected.update(json.loads(p.read_text())['sources'])
    def old(rel):
        p=ROOT/rel;require(sha(p)==expected.get(rel),'Changed prior evidence '+rel)
        sources[rel]=sha(p);return p
    base='outputs/paper_a_safe_correction/'
    states=json.loads(old(base+'winner_signal_review_v1/fold_model_states.json').read_text())
    outcomes=pd.read_csv(old(base+'outcome_risk_v1/summary.csv'),float_precision='round_trip').set_index(['pair','split'])
    complement=pd.read_csv(old(base+'complementarity_review_v1/pair_summary.csv'),float_precision='round_trip').set_index(['pair','split'])
    effects=pd.read_csv(old(base+'component_ablation_v1/effects.csv'),float_precision='round_trip').set_index(['pair','split','comparator','regime'])
    summaries=[];cells=[];strata=[];strata_cells=[];maps=[];settings_rows=[];checks=[];started=time.monotonic()
    for pair,label in PAIRS.items():
        for split in ('dev_oof','test'):
            f,folds,a,alpha,rr,ref,raw,raw_rr,settings=replay(inputs,INPUT/pair,pair,split,states)
            check_frame(f,'dev' if split=='dev_oof' else 'test')
            g=margin(f,folds,pair,split,states);c=np.abs(2*expit(g)-1);reject=np.zeros(len(f),bool)
            for s in settings:
                reject[folds==s['fold']]=c[folds==s['fold']]<s['tau']
                settings_rows.append(dict(pair=pair,label=label,split=split,**s))
            r=stage_summary(alpha,a,raw,reject,rr,ref);prior=outcomes.loc[pair,split]
            for k in ('n','active_n','harm_n','benefit_n','unchanged_n','utility','mean_gain','mean_loss','conditional_gain','conditional_loss','intervention','harm_active'):
                close(r[k],prior[k],'Prior full outcome differs: '+k)
            close(r['mean_gain']-r['mean_loss'],r['utility'],'Gain/loss identity')
            gate=effects.loc[pair,split,'No-gate','Native'];matched=effects.loc[pair,split,'No-gate','Matched-I']
            close(r['utility']-np.mean(raw_rr-ref),gate.delta,'Prior direct gate effect differs')
            ra,rb=f.rr_a.to_numpy(),f.rr_b.to_numpy();non_tied=ra!=rb
            oracle=float(np.maximum(ra,rb).mean()-max(ra.mean(),rb.mean()))
            close(oracle,complement.loc[pair,split].oracle_gap,'Prior oracle differs')
            close(r['fallback'],complement.loc[pair,split].fallback_rate,'Prior fallback differs')
            r.update(utility_lo=float(prior.utility_lo),utility_hi=float(prior.utility_hi),
                harm_active_lo=float(prior.harm_active_lo),harm_active_hi=float(prior.harm_active_hi),
                primary_mrr=float(ra.mean()),secondary_mrr=float(rb.mean()),endpoint_gap=float(np.mean(ra-rb)),
                b_win_n=int((rb>ra).sum()),tie_n=int((ra==rb).sum()),b_win_rate=float((rb>ra).mean()),tie_rate=float((ra==rb).mean()),
                oracle_gap=oracle,grid_oracle_gap=float(np.mean(f[GRID_COLUMNS].max(axis=1).to_numpy()-ref)),
                winner_n=int(non_tied.sum()),winner_prevalence=float((ra[non_tied]>rb[non_tied]).mean()),
                winner_auc=float(roc_auc_score(ra[non_tied]>rb[non_tied],g[non_tied])),
                raw_utility=float(np.mean(raw_rr-ref)),gate_delta=float(gate.delta),gate_lo=float(gate.lo),gate_hi=float(gate.hi),
                matched_gate_delta=float(matched.delta),matched_gate_lo=float(matched.lo),matched_gate_hi=float(matched.hi),
                bootstrap_seed=int(prior.bootstrap_seed),bootstrap_replicates=int(prior.bootstrap_replicates),
                cluster_order_sha256=triple_hash(f),invalid_n=0)
            require(r['cluster_order_sha256']==prior.cluster_order_sha256,'Prior cluster order differs')
            summaries.append(dict(pair=pair,label=label,split=split,**r))
            for seed in (1,2,3):
                for direction in ('head','tail'):
                    mask=((f.seed==seed)&(f.direction==direction)).to_numpy()
                    cells.append(dict(pair=pair,label=label,split=split,seed=seed,direction=direction,
                        **stage_summary(alpha[mask],a[mask],raw[mask],reject[mask],rr[mask],ref[mask])))
            if split=='dev_oof':
                groups=np.full(len(f),'',object)
                for fold in range(1,6):
                    hold=folds==fold;train=f.loc[~hold]
                    require(not set(map(tuple,train[TRIPLE].to_numpy())) & set(map(tuple,f.loc[hold,TRIPLE].to_numpy())),
                        'Original triple crosses train/holdout')
                    mapping=training_map(train)
                    groups[hold]=assign_strata(mapping,f.loc[hold,'relation_id'],f.loc[hold,'direction'])
                    maps.extend(dict(pair=pair,label=label,fold=fold,**row) for row in mapping.to_dict('records'))
                ids,_=pd.factorize(pd.MultiIndex.from_frame(f[TRIPLE]),sort=True)
                seed=2026091214+int(pair.startswith('db15k'))
                intervals=conditional_intervals(ids,groups,np.column_stack([ra-rb,rr-ref]),seed)
                for name in BINS:
                    mask=groups==name;s=stage_summary(alpha[mask],a[mask],raw[mask],reject[mask],rr[mask],ref[mask])
                    q=intervals[name]
                    strata.append(dict(pair=pair,label=label,stratum=name,**s,mass=float(mask.mean()),
                        triple_n=len(f.loc[mask,TRIPLE].drop_duplicates()),
                        endpoint_gap=float(np.mean((ra-rb)[mask])) if mask.any() else np.nan,
                        endpoint_gap_lo=q[0][0],endpoint_gap_hi=q[0][1],utility_lo=q[1][0],utility_hi=q[1][1],
                        delta_contribution=float(np.sum((rr-ref)[mask])/len(f)),valid_replicates=q['valid_replicates'],
                        bootstrap_seed=seed,bootstrap_replicates=10000,cluster_order_sha256=triple_hash(f)))
                    for ss in (1,2,3):
                        for direction in ('head','tail'):
                            m=mask&((f.seed==ss)&(f.direction==direction)).to_numpy()
                            strata_cells.append(dict(pair=pair,label=label,stratum=name,seed=ss,direction=direction,
                                endpoint_gap=float(np.mean((ra-rb)[m])) if m.any() else np.nan,
                                **stage_summary(alpha[m],a[m],raw[m],reject[m],rr[m],ref[m])))
                own=[s for s in strata if s['pair']==pair]
                require(sum(s['n'] for s in own)==len(f),'Stratum count does not close')
                close(sum(s['delta_contribution'] for s in own),r['utility'],'Stratum utility does not close')
            checks.append(dict(pair=pair,label=label,split=split,observations=len(f),original_replay_passed=True,
                prior_outcomes_and_gate_effects_reconciled=True,prior_oracle_and_fallback_reconciled=True,
                four_stages_disjoint_exhaustive=True,invalid_rows=0,dev_map_uses_outer_training_only=(split=='dev_oof')))
            print(f'[SCOPE] {label} {split}: rejected {100*r["fallback"]:.3f}%, changed {100*r["intervention"]:.3f}% ({time.monotonic()-started:.1f}s)',flush=True)
    require(len(summaries)==12 and len(cells)==72 and len(strata)==24 and len(strata_cells)==144,'Incomplete inventory')
    require(sum(r['n'] for r in summaries)==474732,'Incomplete population')
    outputs={}
    for name,rows in [('summary',summaries),('seed_direction',cells),('dev_strata',strata),('dev_strata_seed_direction',strata_cells),
            ('dev_training_cells',maps),('original_settings',settings_rows)]:
        p=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False,lineterminator='\n');outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    sources.update(inputs.sources)
    audit=dict(status='scope_boundary_checks_passed',failures=[],sources=sources,outputs=outputs,checks=checks,
        summary_rows=12,seed_direction_rows=72,dev_strata_rows=24,dev_strata_seed_direction_rows=144,
        training_cell_rows=len(maps),original_setting_rows=len(settings_rows),observations=474732,
        band_mrr=.005,min_training_triples=50,bootstrap_replicates=10000,selector_fits=0,scorer_runs=0,
        test_used_for_selection=False,new_policy_selection=False,historical_results_replaced=False,
        global_necessity_established=False,local_reversals_are_globally_ineligible_pairs=False,
        strata_maps_refitted_in_bootstrap=False,intervals_multiplicity_adjusted=False,
        winner_auc_is_descriptive_not_calibration=True,elapsed_seconds=time.monotonic()-started,
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
