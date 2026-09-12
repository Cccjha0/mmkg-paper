"""C11 query-key inventory/purged DEV controls and C12 unified paired intervals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_paper_a_selection_seed import (FIELDS,GRID_COLUMNS,PAIRS,ACTIONS,
    folds_for,anchor_for,candidate_matrix,select_index,saved_signal,statistics,seed_summary)
from scripts.analyze_paper_a_winner_signal import joined_oof
from scripts.analyze_paper_a_matched_alternatives import Inputs,apply,check_frame
from scripts.analyze_paper_a_conservative_radius import INPUT,sha,require,close,policy
from scripts.ablate_anchored_dynamic import fit_geometry_model,feature_matrix,model_outputs
from scripts.audit_paper_a_claims_cost import align,KEY,METHODS,MAIN

OUT=ROOT/'outputs/paper_a_safe_correction/query_pair_v1'
TRIPLE=['head_id','relation_id','tail_id']
CODE=('scripts/analyze_paper_a_query_pair.py','tests/test_query_pair_diagnostics.py',
      'docs/protocols/paper_a_query_pair_review.md','scripts/analyze_paper_a_selection_seed.py',
      'scripts/analyze_paper_a_winner_signal.py','scripts/analyze_paper_a_matched_alternatives.py',
      'scripts/analyze_paper_a_conservative_radius.py','scripts/ablate_anchored_dynamic.py',
      'scripts/crossfit_heterogeneous_dev_policies.py','scripts/audit_paper_a_claims_cost.py',
      'router/matched_actions.py')


def query_keys(frame):
    require(set(frame.direction)<= {'head','tail'},'Unknown query direction')
    return list(zip(frame.direction,frame.relation_id,np.where(frame.direction=='tail',frame.head_id,frame.tail_id)))


def triple_hash(frame):
    triples=sorted(set(tuple(map(int,row)) for row in frame[TRIPLE].to_numpy()))
    return hashlib.sha256((''.join(','.join(map(str,row))+'\n' for row in triples)).encode()).hexdigest()


def purge_training(train,held):
    """Remove whole six-row triples, using query identities and no outcomes."""
    held_keys=set(query_keys(held))
    bad={tuple(row) for row,key in zip(train[TRIPLE].to_numpy(),query_keys(train)) if key in held_keys}
    keep=np.array([tuple(row) not in bad for row in train[TRIPLE].to_numpy()])
    result=train.loc[keep].copy()
    require(not set(query_keys(result)) & held_keys,'Purge left shared query keys')
    return result


def size_control(train,purged,fold):
    triples=train[TRIPLE].drop_duplicates()
    quotas=purged[TRIPLE].drop_duplicates().groupby('relation_id').size().to_dict()
    chosen=set()
    for relation,part in triples.groupby('relation_id'):
        rows=[tuple(map(int,r)) for r in part.to_numpy()]
        def order(row):return hashlib.sha256(f'2026091211|{fold}|'.encode()+('|'.join(map(str,row))).encode()).digest()
        chosen.update(sorted(rows,key=order)[:quotas.get(relation,0)])
    result=train.loc[[tuple(r) in chosen for r in train[TRIPLE].to_numpy()]].copy()
    require(len(result)==len(purged),'Size control mismatch')
    require(result[TRIPLE].drop_duplicates().groupby('relation_id').size().to_dict()==quotas,'Relation quota mismatch')
    return result


def inventory(frame,folds=None,dev=None):
    # Three seeds repeat the same labeled semantic requests.
    f=frame.loc[frame.seed==1].copy();f['query_key']=query_keys(f)
    if folds is not None:f['fold']=np.asarray(folds)[frame.seed.to_numpy()==1]
    result=[];dev_keys=set(query_keys(dev)) if dev is not None else set()
    for direction,part in f.groupby('direction'):
        counts=part.groupby('query_key').size()
        require(len(part)==len(part[TRIPLE].drop_duplicates()),'Repeated original triple')
        r=dict(direction=direction,answer_rows=len(part),unique_keys=len(counts),
               repeated_keys=int((counts>1).sum()),rows_on_repeated_keys=int(counts[counts>1].sum()),max_answers=int(counts.max()))
        if folds is not None:
            nf=part.groupby('query_key').fold.nunique();cross=set(nf[nf>1].index)
            r.update(cross_fold_keys=len(cross),rows_on_cross_fold_keys=int(part.query_key.isin(cross).sum()))
        if dev is not None:r['rows_with_dev_key']=sum(k in dev_keys for k in part.query_key)
        result.append(r)
    return result


def cluster_intervals(frame,differences,seed,replicates=10000):
    """Equal six-row triple clusters; common PCG64 draws for every column."""
    check_frame(frame,str(frame.split.iloc[0]))
    d=np.asarray(differences,float)
    require(d.ndim==2 and len(d)==len(frame) and np.isfinite(d).all(),'Bad paired differences')
    ids,unique=pd.factorize(pd.MultiIndex.from_frame(frame[TRIPLE]),sort=True)
    counts=np.bincount(ids);require(np.all(counts==6),'Need all six observations per triple')
    means=np.column_stack([np.bincount(ids,weights=d[:,j])/6 for j in range(d.shape[1])])
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.empty((replicates,d.shape[1]))
    for start in range(0,replicates,32):
        ix=rng.integers(len(counts),size=(min(32,replicates-start),len(counts)))
        draws[start:start+len(ix)]=means[ix].mean(axis=1)
    point=means.mean(axis=0);lo,hi=np.quantile(draws,[.025,.975],axis=0,method='linear')
    order_hash=hashlib.sha256(''.join(','.join(map(str,r))+'\n' for r in unique).encode()).hexdigest()
    return point,lo,hi,dict(clusters=len(counts),cluster_order_sha256=order_hash,seed=seed,
        replicates=replicates,rng='PCG64',quantile='linear',observations_per_cluster=6)


def write_outputs(frames,receipt,sources,**extra):
    outputs={}
    for name,rows in frames.items():
        p=OUT/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False,lineterminator='\n')
        outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    value=dict(status='query_pair_checks_passed',sources=sources,outputs=outputs,
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),**extra)
    (OUT/receipt).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def dev():
    require(not (OUT/'dev_complete.json').exists(),'Completed DEV receipt is immutable')
    OUT.mkdir(parents=True,exist_ok=True);inputs=Inputs('DEV')
    sources={r:sha(ROOT/r) for r in CODE}
    state_path=ROOT/'outputs/paper_a_safe_correction/winner_signal_review_v1/fold_model_states.json'
    expected=json.loads((ROOT/'paper_a_draft/winner_signal_source_manifest.json').read_text())['sources']
    require(sha(state_path)==expected[state_path.relative_to(ROOT).as_posix()],'Changed fold state')
    sources[state_path.relative_to(ROOT).as_posix()]=sha(state_path);states=json.loads(state_path.read_text())
    counts=[];fold_counts=[];fits=[];summaries=[];effects=[];cells=[];candidates=[];dataset_keys={};subsets=[]
    for pair,label in PAIRS.items():
        f,old,settings=joined_oof(inputs,INPUT/pair);ids=folds_for(f,5,20260901)+1
        require(np.array_equal(ids,old.fold),'Original fold mismatch')
        dataset=pair.split('_')[0];identity=(triple_hash(f),inventory(f,ids))
        if dataset in dataset_keys:require(identity==dataset_keys[dataset],'Pairs differ in dataset query inventory')
        else:
            dataset_keys[dataset]=identity;counts.extend(dict(dataset=dataset,split='dev',**r) for r in identity[1])
        names=('Original','Purged','Random-size')
        rr={n:np.empty(len(f)) for n in names};alpha={n:np.empty(len(f)) for n in names};anchor={n:np.empty(len(f)) for n in names}
        seen=np.zeros(len(f),bool)
        for fold in range(1,6):
            mask=ids==fold;train,held=f.loc[~mask],f.loc[mask]
            train_keys=set(query_keys(train));seen[mask]=[k in train_keys for k in query_keys(held)]
            purged=purge_training(train,held);random=size_control(train,purged,fold)
            state=next(s for s in states if s['pair']==pair and s['fold']==fold and s['learner']=='balanced')
            for name,fit in [('Original',train),('Purged',purged),('Random-size',random)]:
                a0=anchor_for(fit);random_state=20260902+(fold-1)*10+2
                if name=='Original':ts,hs=saved_signal(fit,state),saved_signal(held,state);n_iter=None
                else:
                    model,x,bad=fit_geometry_model(fit.to_dict('records'),fields=FIELDS,random_state=random_state)
                    ts=(*model_outputs(model,x),bad);hx,hbad=feature_matrix(held.to_dict('records'),FIELDS)
                    hs=(*model_outputs(model,hx),hbad);n_iter=int(model[-1].n_iter_[0])
                values=candidate_matrix(fit,ts,a0).mean(axis=0);index=select_index(values,False);action=ACTIONS[index]
                out,weights,_=apply(held[GRID_COLUMNS].to_numpy(),hs,a0,action)
                rr[name][mask]=out;alpha[name][mask]=weights;anchor[name][mask]=a0
                shared=set(query_keys(fit)) & set(query_keys(held))
                fits.append(dict(pair=pair,label=label,fold=fold,method=name,training_triples=len(fit)//6,
                    original_training_triples=len(train)//6,heldout_triples=len(held)//6,fit_rows=int((fit.rr_a!=fit.rr_b).sum()),
                    winner_a_rows=int((fit.rr_a>fit.rr_b).sum()),winner_b_rows=int((fit.rr_a<fit.rr_b).sum()),
                    shared_keys=len(shared),training_sha256=triple_hash(fit),heldout_sha256=triple_hash(held),
                    anchor=a0,beta=action.strength,tau=action.tau,selection_mrr=float(values[index]),n_iter=n_iter,random_state=random_state))
                candidates.extend(dict(pair=pair,fold=fold,method=name,beta=a.strength,tau=a.tau,mrr=float(v)) for a,v in zip(ACTIONS[1:],values[1:]))
                if name=='Original':
                    close(out,old.loc[mask,'rr_method'],'Original OOF RR mismatch');close(weights,old.loc[mask,'alpha_applied'],'Original action mismatch')
                    close(a0,settings.loc[fold,'alpha0'],'Original anchor mismatch')
                    require((action.strength,action.tau)==(settings.loc[fold,'selected_beta'],settings.loc[fold,'selected_confidence_threshold']),'Original selection mismatch')
            if pair in ('mkgw_mhyper_native','db15k_mhyper_native'):
                for direction in ('head','tail'):
                    h=held[(held.seed==1)&(held.direction==direction)];keys=query_keys(h);tk=set(query_keys(train))
                    fold_counts.append(dict(dataset=dataset,fold=fold,direction=direction,heldout_answer_rows=len(h),
                        heldout_unique_keys=len(set(keys)),shared_unique_keys=len(set(keys)&tk),shared_answer_rows=sum(k in tk for k in keys)))
            print(f'[QUERY DEV] {label} fold {fold}/5: {len(train)//6} -> {len(purged)//6} train triples',flush=True)
        for name in names:
            reference=f[GRID_COLUMNS].to_numpy()[np.arange(len(f)),np.rint(anchor[name]*20).astype(int)]
            summaries.append(dict(pair=pair,label=label,method=name,**statistics(rr[name],reference,alpha[name],anchor[name])))
            for seed in (1,2,3):
                m=f.seed.to_numpy()==seed
                cells.append(dict(pair=pair,label=label,method=name,seed=seed,mrr=float(rr[name][m].mean())))
        for name,mask in [('seen_key',seen),('unseen_key',~seen)]:
            subsets.append(dict(pair=pair,label=label,subset=name,n=int(mask.sum()),
                **{n+'_mrr':float(rr[n][mask].mean()) for n in names}))
        contrasts=[('Purged','Original'),('Random-size','Original'),('Purged','Random-size')]
        d=np.column_stack([rr[a]-rr[b] for a,b in contrasts]);p,lo,hi,meta=cluster_intervals(f,d,2026091214+(dataset=='db15k'),2000)
        effects.extend(dict(pair=pair,label=label,left=a,right=b,delta=float(p[j]),lo=float(lo[j]),hi=float(hi[j]),**meta) for j,(a,b) in enumerate(contrasts))
    sources.update(inputs.sources)
    write_outputs(dict(query_inventory_dev=counts,query_folds=fold_counts,dev_fits=fits,dev_summary=summaries,
        dev_effects=effects,dev_seed_metrics=cells,dev_key_subsets=subsets,dev_candidates=candidates),
        'dev_complete.json',sources,selector_fits=60,original_fits_replayed=30,test_opened=False,
        new_test_policy=False,base_scorer_runs=0,query_isolation_guarantee='purged combiner training only',
        end_to_end_unseen_query_generalization=False)


def test_report():
    require((OUT/'dev_complete.json').exists() and not (OUT/'test_complete.json').exists(),'Need immutable DEV first')
    expected={};sources={r:sha(ROOT/r) for r in CODE}
    for name in ('rerun','matched','dynasemble','claims_cost','selection_seed','rejection','conservative'):
        mpath=ROOT/f'paper_a_draft/{name}_source_manifest.json';sources[mpath.relative_to(ROOT).as_posix()]=sha(mpath)
        m=json.loads(mpath.read_text());expected.update(m['sources'])
    def read(path):
        rel=path.relative_to(ROOT).as_posix();require(sha(path)==expected.get(rel),'Changed source '+rel)
        sources[rel]=sha(path);return pd.read_csv(path,float_precision='round_trip')
    previous=read(INPUT.parent/'claims_cost_review_v1/paired_effects.csv').set_index(['pair','comparator'])
    old_fallback=read(INPUT.parent/'rejection_controls_v1/paired_fallback.csv').set_index(['pair','split'])
    old_components=read(INPUT.parent/'information_boundary_rerun_audit/frozen_ablation.csv').set_index(['pair','method'])
    rows=[];cells=[];inventory_rows=[];ci_checks=[];dataset_keys={}
    for pair,label in PAIRS.items():
        f=read(INPUT/pair/'test_anchored/test_locked_query_rows.csv');check_frame(f,'test')
        dataset=pair.split('_')[0]
        devframe=read(INPUT/pair/'baseline_crossfit/dev_crossfit_query_rows.csv')
        inv=inventory(f,dev=devframe)
        if dataset in dataset_keys:require((triple_hash(f),inv)==dataset_keys[dataset],'TEST query inventory differs across pairs')
        else:
            dataset_keys[dataset]=(triple_hash(f),inv);inventory_rows.extend(dict(dataset=dataset,split='test',**r) for r in inv)
        cols=dict(Primary='rr_a',Secondary='rr_b',**{'Equal-z':'rr_equal','RRF':'rr_rrf','Global':'rr_global',
            'Relation':'rr_relation','Query-soft':'rr_query_soft_locked','ADC':'rr_anchored_locked'})
        rr={m:f[c].to_numpy() for m,c in cols.items()}
        if pair in MAIN:
            maps=read(INPUT.parent/'matched_alternatives_v1/query_rows'/(pair+'.csv.gz'))
            aligned=align(f,maps,['rr_'+m for m in METHODS[7:12]])
            rr.update({m:aligned['rr_'+m].to_numpy() for m in METHODS[7:12]})
            r0=read(INPUT/pair/'dynasemble/test_query_rows.csv');rr['R0']=align(f,r0,['rr_dynasemble']).rr_dynasemble.to_numpy()
            reps=[]
            for ss in (11,23,37):
                rep=pd.concat([read(INPUT.parent/'dynasemble_controls_v1_review'/pair/'test'/f'R3_softplus_b{bs}_s{ss}.csv') for bs in (1,2,3)])
                rep=rep.rename(columns={'base_seed':'seed'});require(set(rep.selector_seed)=={ss},'Selector repeat differs')
                reps.append(align(f,rep,['rr_method']).rr_method.to_numpy())
            rr['R3']=np.mean(reps,axis=0)
        a0=float(f.alpha0_locked.iloc[0]);beta=float(f.anchored_beta_locked.iloc[0]);tau=float(f.anchored_confidence_threshold_locked.iloc[0])
        require(np.isfinite(f[list(FIELDS)].to_numpy()).all(),'Unexpected nonfinite TEST features')
        for name,b,t in [('ADC',beta,tau),('Expanded-radius',1.,tau),('No-fallback',beta,0.)]:
            v,alpha,_=policy(f[GRID_COLUMNS].to_numpy(),f.anchored_decision.to_numpy(),f.anchored_probability_a.to_numpy(),np.zeros(len(f),bool),a0,b,t)
            if name=='ADC':close(v,rr['ADC'],'TEST ADC grid replay mismatch')
            else:rr[name]=v
            variant={'ADC':'full','Expanded-radius':'no_bound','No-fallback':'no_fallback'}[name]
            close(v.mean(),old_components.loc[(pair,variant),'mrr'],'Old component point differs')
        methods=[m for m in METHODS if m!='ADC' and m in rr]+['Expanded-radius','No-fallback']
        d=np.column_stack([rr['ADC']-rr[m] for m in methods])
        p,lo,hi,meta=cluster_intervals(f,d,2026091212+(dataset=='db15k'))
        ci_checks.append(dict(pair=pair,comparisons=len(methods),**meta))
        for j,m in enumerate(methods):
            sm=[float(d[f.seed.to_numpy()==s,j].mean()) for s in (1,2,3)];mean,sd=seed_summary(sm)
            close(mean,p[j],'Paired seed mean mismatch')
            if (pair,m) in previous.index:close(p[j],previous.loc[(pair,m),'delta_adc_minus_comparator'],'Old paired point differs')
            if m=='No-fallback':close(p[j],old_fallback.loc[(pair,'test'),'paired_effect'],'Old direct fallback point differs')
            prev=previous.loc[(pair,m)] if (pair,m) in previous.index else None
            rows.append(dict(pair=pair,label=label,comparator=m,delta=float(p[j]),lo=float(lo[j]),hi=float(hi[j]),
                seed1=sm[0],seed2=sm[1],seed3=sm[2],seed_sd=sd,
                zero_observation_differences=bool(np.all(d[:,j]==0)),
                old_lo=float(prev.ci_low) if prev is not None and pd.notna(prev.ci_low) else None,
                old_hi=float(prev.ci_high) if prev is not None and pd.notna(prev.ci_high) else None,
                role='central retrospective' if m=='Global' and pair in MAIN else 'exploratory',**meta))
            for seed in (1,2,3):
                for direction in ('head','tail'):
                    mask=((f.seed==seed)&(f.direction==direction)).to_numpy()
                    cells.append(dict(pair=pair,label=label,comparator=m,seed=seed,direction=direction,n=int(mask.sum()),delta=float(d[mask,j].mean())))
        print(f'[UNIFIED CI] {label}: {len(methods)} contrasts, 10,000 common triple draws',flush=True)
    require(len(rows)==82 and len(cells)==492,'Incomplete paired inventory')
    write_outputs(dict(paired_intervals=rows,paired_seed_direction=cells,query_inventory_test=inventory_rows),
        'test_complete.json',sources,checks=ci_checks,contrasts=82,paired_seed_direction_cells=492,
        test_used_for_selection=False,new_test_policy=False,selector_fits=0,base_scorer_runs=0,
        simultaneous_intervals=False,confirmatory_status=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('dev','test-report'));args=parser.parse_args()
    with threadpool_limits(limits=1):(dev if args.phase=='dev' else test_report)()
