"""C09 DEV-only inner selection sensitivity; C10 separate fixed-TEST seed reporting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_matched_alternatives import Inputs, apply, check_frame
from scripts.analyze_paper_a_winner_signal import joined_oof, FIELDS, PAIRS
from scripts.analyze_paper_a_conservative_radius import INPUT, GRID_COLUMNS, sha, require, close
from scripts.ablate_anchored_dynamic import fit_geometry_model, feature_matrix, model_outputs
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key, best_alpha
from scripts.audit_paper_a_claims_cost import align, METHODS, KEY, MAIN
from router.matched_actions import configurations, ALPHAS
from router.rejection_diagnostics import paired_intervals

OUT=ROOT/'outputs/paper_a_safe_correction/selection_seed_v1'
ACTIONS=configurations('ADC+Global')
NAMES=('Original-40','Resub-41','Inner-40','Inner-41')
CODE=('scripts/analyze_paper_a_selection_seed.py','tests/test_selection_seed_diagnostics.py',
      'docs/protocols/paper_a_selection_seed_review.md','scripts/ablate_anchored_dynamic.py',
      'scripts/crossfit_heterogeneous_dev_policies.py','scripts/lock_apply_anchored_dynamic.py',
      'scripts/analyze_paper_a_winner_signal.py','scripts/analyze_paper_a_matched_alternatives.py',
      'router/matched_actions.py','router/rejection_diagnostics.py')


def save(name,value):
    path=OUT/name
    if isinstance(value,pd.DataFrame):value.to_csv(path,index=False,lineterminator='\n')
    else:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def folds_for(frame,k,seed):
    rows=frame[['head_id','relation_id','tail_id']].to_dict('records')
    assignment,_=assign_grouped_folds(rows,k,seed)
    return np.array([assignment[triple_key(r)] for r in rows])


def anchor_for(frame):
    return best_alpha(frame[GRID_COLUMNS].to_dict('records'),tuple(ALPHAS))[0]


def candidate_matrix(frame,signal,anchor):
    grid=frame[GRID_COLUMNS].to_numpy()
    return np.column_stack([apply(grid,signal,anchor,a)[0] for a in ACTIONS])


def select_index(scores,with_global):
    scores=np.asarray(scores,float)
    if scores.shape!=(41,) or not np.isfinite(scores).all():raise ValueError('Expected 41 finite candidate scores')
    return max(range(0 if with_global else 1,41),key=lambda i:(scores[i],-ACTIONS[i].strength,ACTIONS[i].tau))


def inner_oof(frame,scope_index,fit_fn=fit_geometry_model):
    """Accept only the current training portion; no outer-holdout argument exists."""
    ids=folds_for(frame,3,2026091209)
    scores=np.empty((len(frame),41)); covered=np.zeros(len(frame),int); records=[]
    for fold in range(3):
        fit=frame.loc[ids!=fold]; held=frame.loc[ids==fold]
        fit_keys=set(map(tuple,fit[['head_id','relation_id','tail_id']].to_numpy()))
        held_keys=set(map(tuple,held[['head_id','relation_id','tail_id']].to_numpy()))
        require(fit_keys.isdisjoint(held_keys) and len(fit)==6*len(fit_keys) and len(held)==6*len(held_keys),'Inner triple scope differs')
        state=20260902+scope_index*100+fold+1
        model,_,_=fit_fn(fit[list(FIELDS)+['rr_a','rr_b']].to_dict('records'),fields=FIELDS,random_state=state)
        x,invalid=feature_matrix(held.to_dict('records'),FIELDS)
        anchor=anchor_for(fit)
        scores[ids==fold]=candidate_matrix(held,(*model_outputs(model,x),invalid),anchor)
        covered[ids==fold]+=1
        records.append(dict(inner_fold=fold+1,fit_triples=len(fit_keys),heldout_triples=len(held_keys),
            fit_rows=int((fit.rr_a!=fit.rr_b).sum()),heldout_rows=len(held),anchor=anchor,random_state=state,
            n_iter=int(model[-1].n_iter_[0]),fit_triple_sha256=hashlib.sha256(repr(sorted(fit_keys)).encode()).hexdigest(),
            heldout_triple_sha256=hashlib.sha256(repr(sorted(held_keys)).encode()).hexdigest(),disjoint=True))
    require(np.all(covered==1),'Inner holdout coverage differs')
    return scores.mean(axis=0),records


def saved_signal(frame,state):
    x=frame[list(FIELDS)].to_numpy()
    require(np.isfinite(x).all(),'Unexpected non-finite inputs in retained scope')
    g=((x-np.asarray(state['mean']))/np.asarray(state['scale']))@np.asarray(state['coefficients'])[0]+state['intercept'][0]
    return g,expit(g),np.zeros(len(frame),bool)


def statistics(rr,reference,alpha,anchor):
    delta=np.asarray(rr)-np.asarray(reference)
    return dict(n=len(rr),mrr=float(np.mean(rr)),utility=float(np.mean(delta)),
        harm=float(np.mean(delta < -1e-12)),mean_loss=float(np.maximum(-delta,0).mean()),
        intervention=float(np.mean(np.abs(alpha-anchor)>1e-12)),movement=float(np.mean(np.abs(alpha-anchor))))


def observable_overlap(train,held):
    def keys(f):
        known=np.where(f.direction=='tail',f.head_id,f.tail_id)
        return list(zip(f.seed,f.direction,f.relation_id,known))
    seen=set(keys(train)); q=keys(held)
    return sum(k in seen for k in q),len(q)


def dev():
    require(not (OUT/'dev_complete.json').exists(),'Completed DEV review is immutable')
    OUT.mkdir(parents=True,exist_ok=True);inputs=Inputs('DEV')
    inputs.sources.update({r:sha(ROOT/r) for r in CODE})
    state_path=ROOT/'outputs/paper_a_safe_correction/winner_signal_review_v1/fold_model_states.json'
    expected=json.loads((ROOT/'paper_a_draft/winner_signal_source_manifest.json').read_text())['sources']
    require(sha(state_path)==expected[state_path.relative_to(ROOT).as_posix()],'Changed retained fold states')
    inputs.sources[state_path.relative_to(ROOT).as_posix()]=sha(state_path)
    states=json.loads(state_path.read_text()); choices=[]; candidates=[]; scopes=[]; summaries=[]; effects=[]; seeds=[]
    for pi,(pair,label) in enumerate(PAIRS.items()):
        base,old,settings=joined_oof(inputs,INPUT/pair)
        ids=folds_for(base,5,20260901);require(np.array_equal(ids+1,old.fold),'Outer grouping differs')
        predictions={name:[np.empty(len(base)),np.empty(len(base))] for name in NAMES}
        anchors=old.alpha0.to_numpy(); reference=old.rr_global_crossfit.to_numpy()
        for fold in range(1,6):
            mask=ids==fold-1;train=base.loc[~mask];held=base.loc[mask]
            state=next(s for s in states if s['pair']==pair and s['fold']==fold and s['learner']=='balanced')
            anchor=anchor_for(train);close(anchor,settings.loc[fold,'alpha0'],'Outer anchor differs')
            ts,hs=saved_signal(train,state),saved_signal(held,state)
            resub=candidate_matrix(train,ts,anchor).mean(axis=0)
            inner,inner_records=inner_oof(train,fold)
            scopes.extend(dict(pair=pair,label=label,outer_fold=fold,**r) for r in inner_records)
            original=select_index(resub,False)
            require((ACTIONS[original].strength,ACTIONS[original].tau)==
                (settings.loc[fold,'selected_beta'],settings.loc[fold,'selected_confidence_threshold']),'Original choice differs')
            close(resub[original],settings.loc[fold,'selected_train_mrr'],'Original selection MRR differs')
            overlap,n=observable_overlap(train,held)
            for score_name,values in [('resubstitution',resub),('inner_oof',inner)]:
                candidates.extend(dict(pair=pair,outer_fold=fold,selection=score_name,beta=a.strength,tau=a.tau,mrr=float(v)) for a,v in zip(ACTIONS,values))
            indices=(original,select_index(resub,True),select_index(inner,False),select_index(inner,True))
            for name,index in zip(NAMES,indices):
                a=ACTIONS[index];rr,alpha,_=apply(held[GRID_COLUMNS].to_numpy(),hs,anchor,a)
                predictions[name][0][mask]=rr;predictions[name][1][mask]=alpha
                values=resub if name in NAMES[:2] else inner
                choices.append(dict(pair=pair,label=label,outer_fold=fold,method=name,anchor=anchor,beta=a.strength,tau=a.tau,
                    selection_mrr=float(values[index]),heldout_shared_observable_rows=overlap,heldout_rows=n,
                    **statistics(rr,reference[mask],alpha,anchor)))
            close(predictions['Original-40'][0][mask],old.loc[mask,'rr_method'],'Outer heldout RR differs')
            close(predictions['Original-40'][1][mask],old.loc[mask,'alpha_applied'],'Outer heldout action differs')
            print(f'[INNER DEV] {label} outer {fold}/5',flush=True)
        for name,(rr,alpha) in predictions.items():
            summaries.append(dict(pair=pair,label=label,method=name,**statistics(rr,reference,alpha,anchors)))
            for seed in (1,2,3):
                m=base.seed.to_numpy()==seed
                seeds.append(dict(pair=pair,label=label,method=name,seed=seed,**statistics(rr[m],reference[m],alpha[m],anchors[m])))
        clusters=pd.factorize(pd.MultiIndex.from_frame(base[['head_id','relation_id','tail_id']]),sort=True)[0]
        comparisons=[('Inner-40','Original-40'),('Inner-41','Resub-41'),('Resub-41','Original-40'),('Inner-41','Original-40')]
        ds=np.column_stack([predictions[a][0]-predictions[b][0] for a,b in comparisons])
        point,lo,hi=paired_intervals(ds,clusters,2026091210+pi)
        effects.extend(dict(pair=pair,label=label,left=a,right=b,delta=float(point[j]),lo=float(lo[j]),hi=float(hi[j])) for j,(a,b) in enumerate(comparisons))
        # Same selection experiment on full DEV; never apply these new choices to TEST.
        full=inputs.frame(INPUT/pair/'dev_lock/dev_locked_query_rows.csv');check_frame(full,'dev')
        lock=inputs.obj(INPUT/pair/'dev_lock/anchored_dev_lock.json');anchor=anchor_for(full)
        close(anchor,lock['alpha0'],'Full DEV anchor differs')
        require(lock['beta_grid']==[i/20 for i in range(1,11)] and lock['confidence_threshold_grid']==[0.,.1,.2,.3],'Final and outer grids differ')
        signal=full.anchored_decision.to_numpy(),full.anchored_probability_a.to_numpy(),np.zeros(len(full),bool)
        resub=candidate_matrix(full,signal,anchor).mean(axis=0);inner,records=inner_oof(full,0)
        scopes.extend(dict(pair=pair,label=label,outer_fold=0,**r) for r in records)
        i=select_index(resub,False);require((ACTIONS[i].strength,ACTIONS[i].tau)==(lock['beta'],lock['confidence_threshold']),'Full DEV original selection differs')
        for score_name,values in [('resubstitution',resub),('inner_oof',inner)]:
            candidates.extend(dict(pair=pair,outer_fold=0,selection=score_name,beta=a.strength,tau=a.tau,mrr=float(v)) for a,v in zip(ACTIONS,values))
            for zero in (False,True):
                index=select_index(values,zero);a=ACTIONS[index]
                choices.append(dict(pair=pair,label=label,outer_fold=0,method=('Resub' if score_name=='resubstitution' else 'Inner')+('-41' if zero else '-40'),
                    anchor=anchor,beta=a.strength,tau=a.tau,selection_mrr=float(values[index])))
    frames={'dev_choices.csv':choices,'dev_candidates.csv':candidates,'inner_scopes.csv':scopes,
            'dev_summary.csv':summaries,'dev_paired_effects.csv':effects,'dev_seed_metrics.csv':seeds}
    outputs={}
    for name,rows in frames.items():p=save(name,pd.DataFrame(rows));outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    require(len(scopes)==108 and len(choices)==144 and len(candidates)==2952,'Unexpected selection budget')
    save('dev_complete.json',dict(status='selection_hierarchy_checks_passed',sources=inputs.sources,outputs=outputs,
        additional_logistic_fits=108,outer_fits_replayed=30,full_dev_locks_replayed=6,outer_folds=5,inner_folds=3,
        test_policy_applied=False,test_opened=False,base_scorer_runs=0,original_results_replaced=False,
        original_recipe_outer_holdout_label_leak_found=False,development_adaptivity_removed=False))


def seed_summary(values):
    values=np.asarray(values,float)
    if values.shape!=(3,) or not np.isfinite(values).all():raise ValueError('Require exactly three finite seed MRRs')
    return float(values.mean()),float(values.std(ddof=1))


def seed_report():
    require((OUT/'dev_complete.json').exists() and not (OUT/'seed_complete.json').exists(),'Need completed DEV first; keep reports immutable')
    expected={};sources={}
    for name in ('rerun','matched','dynasemble','claims_cost','history'):
        path=ROOT/f'paper_a_draft/{name}_source_manifest.json'
        if not path.exists() and name=='history':path=ROOT/'paper_a_draft/test_history_source_manifest.json'
        m=json.loads(path.read_text());expected.update(m['sources'])
    def read(path):
        rel=path.relative_to(ROOT).as_posix();require(sha(path)==expected[rel],'Unverified seed source: '+rel)
        sources[rel]=sha(path);return pd.read_csv(path,float_precision='round_trip')
    old=read(INPUT.parent/'claims_cost_review_v1/main_results.csv').set_index(['pair','method'])
    paired=read(INPUT.parent/'claims_cost_review_v1/paired_effects.csv').set_index(['pair','comparator'])
    results=[];dispersion=[];effects=[];cells=[]
    for pair,label in PAIRS.items():
        f=read(INPUT/pair/'test_anchored/test_locked_query_rows.csv');check_frame(f,'test')
        cols=dict(Primary='rr_a',Secondary='rr_b',**{'Equal-z':'rr_equal','RRF':'rr_rrf','Global':'rr_global',
            'Relation':'rr_relation','Query-soft':'rr_query_soft_locked','ADC':'rr_anchored_locked'})
        rr={m:f[c].to_numpy() for m,c in cols.items()}
        if pair in MAIN:
            maps=read(INPUT.parent/'matched_alternatives_v1/query_rows'/(pair+'.csv.gz'))
            aligned=align(f,maps,['rr_'+m for m in METHODS[7:12]])
            rr.update({m:aligned['rr_'+m].to_numpy() for m in METHODS[7:12]})
            r0=read(INPUT/pair/'dynasemble/test_query_rows.csv')
            rr['R0']=align(f,r0,['rr_dynasemble']).rr_dynasemble.to_numpy()
            reps=[]
            for ss in (11,23,37):
                rep=pd.concat([read(INPUT.parent/'dynasemble_controls_v1_review'/pair/'test'/f'R3_softplus_b{bs}_s{ss}.csv') for bs in (1,2,3)])
                rep=rep.rename(columns={'base_seed':'seed'});require(set(rep.selector_seed)=={ss},'Selector repetitions differ')
                reps.append(align(f,rep,['rr_method']).rr_method.to_numpy())
            rr['R3']=np.mean(reps,axis=0)
        for method,values in rr.items():
            means=[float(values[f.seed.to_numpy()==seed].mean()) for seed in (1,2,3)]
            mean,sd=seed_summary(means);close(mean,old.loc[(pair,method),'mrr'],'Seed means do not reproduce main table')
            dispersion.append(dict(pair=pair,label=label,method=method,seed1=means[0],seed2=means[1],seed3=means[2],mean=mean,sd=sd))
            for seed,value in zip((1,2,3),means):results.append(dict(pair=pair,label=label,method=method,seed=seed,mrr=value,n=int((f.seed==seed).sum())))
            for keys,part in f.groupby(['seed','direction']):
                cells.append(dict(pair=pair,label=label,method=method,seed=int(keys[0]),direction=keys[1],mrr=float(values[part.index].mean()),n=len(part)))
            if method=='ADC':continue
            delta=rr['ADC']-values;means=[float(delta[f.seed.to_numpy()==seed].mean()) for seed in (1,2,3)]
            mean,sd=seed_summary(means);close(mean,paired.loc[(pair,method),'delta_adc_minus_comparator'],'Paired seed difference differs')
            r=paired.loc[(pair,method)]
            effects.append(dict(pair=pair,label=label,comparator=method,seed1=means[0],seed2=means[1],seed3=means[2],mean=mean,sd=sd,
                query_ci_low=r.ci_low,query_ci_high=r.ci_high,query_interval_status=r.interval_status))
        print('[SEED MRR] '+label,flush=True)
    outputs={}
    for name,rows in [('test_seed_metrics.csv',results),('test_seed_dispersion.csv',dispersion),('test_paired_seed_dispersion.csv',effects),('test_seed_direction.csv',cells)]:
        p=save(name,pd.DataFrame(rows));outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    require(len(dispersion)==76 and len(results)==228 and len(effects)==70 and len(cells)==456,'Missing seed cells')
    save('seed_complete.json',dict(status='seed_dispersion_checks_passed',sources=sources,outputs=outputs,
        method_pair_cells=76,seed_metrics=228,paired_effects=70,seed_direction_cells=456,base_seeds=3,sd_ddof=1,
        new_training_runs=0,new_test_policy_applied=False,seed_sd_is_population_ci=False,selector_fits_independent_across_base_seeds=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('dev','seed-report'));args=p.parse_args()
    with threadpool_limits(limits=1):
        (dev if args.phase=='dev' else seed_report)()
