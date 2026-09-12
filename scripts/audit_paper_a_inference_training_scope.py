"""B11/B12 small-model and metadata audit; no scorer execution or model fitting."""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.analyze_paper_a_conservative_radius import sha,require,KEY
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds,triple_key
from router.dynasemble_controls import VARIANTS,grouped_folds

OUT=ROOT/'outputs/paper_a_safe_correction/inference_training_scope_v1'
PAIRS={'mkgw_mhyper_native':'W-N','mkgw_mhyper_adamf':'W-A','db15k_mhyper_native':'D-N',
       'db15k_mhyper_adamf':'D-A','mkgw_native_adamf':'W-NA','db15k_native_adamf':'D-NA'}
CODE=('scripts/audit_paper_a_inference_training_scope.py','tests/test_inference_training_scope.py',
      'docs/protocols/paper_a_inference_training_scope.md','scripts/apply_anchored_safe.py','router/anchored_inference.py',
      'router/query_geometry.py','scripts/lock_apply_anchored_dynamic.py','scripts/ablate_anchored_dynamic.py',
      'scripts/crossfit_anchored_dynamic.py','scripts/crossfit_heterogeneous_dev_policies.py',
      'scripts/eval_heterogeneous_complementarity.py','scripts/eval_openbg_dynasemble.py',
      'scripts/paper_a_dynasemble_controls.py','router/dynasemble_controls.py','scripts/analyze_paper_a_matched_alternatives.py')


def main():
    sources={p:sha(ROOT/p) for p in CODE}
    manifests={}
    for name in ('rerun','dynasemble','matched'):
        path=ROOT/f'paper_a_draft/{name}_source_manifest.json'
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        manifests[name]=json.loads(path.read_text())['sources']
    def bound(rel,version='rerun',expected=None):
        actual=sha(ROOT/rel)
        require(actual==(manifests[version][rel] if expected is None else expected),'Changed source: '+rel)
        sources[rel]=actual;return ROOT/rel
    def obj(rel,version='rerun',expected=None):return json.loads(bound(rel,version,expected).read_text())
    def frame(rel,version='rerun'):return pd.read_csv(bound(rel,version),float_precision='round_trip')
    adc=[]; adc_seed=[]; folds=[]; dyna=[]; cv=[]; version_warnings=set(); matched=[]
    matched_lock=obj('outputs/paper_a_safe_correction/matched_alternatives_v1/dev_locks.json','matched')
    for pair,label in PAIRS.items():
        base='outputs/paper_a_safe_correction/information_boundary_v2/'+pair
        lock=obj(base+'/dev_lock/anchored_dev_lock.json')
        f=frame(base+'/dev_lock/dev_locked_query_rows.csv')
        require(lock['seeds']==[1,2,3] and set(f.seed)=={1,2,3} and set(f.direction)=={'head','tail'},'Wrong ADC seed/direction scope')
        require(not f.duplicated(KEY).any() and (f.groupby(['head_id','relation_id','tail_id']).size()==6).all(),'Wrong triple repetitions')
        n=len(f)//6; mask=f.rr_a!=f.rr_b
        path=bound(base+'/dev_lock/'+lock['model_file'],expected=lock['model_sha256'])
        with warnings.catch_warnings(record=True) as caught,path.open('rb') as handle:model=pickle.load(handle)
        version_warnings.update(str(w.message) for w in caught)
        x=f.loc[mask,list(QUERY_GEOMETRY_FIELDS)].to_numpy()
        require(np.isfinite(x).all() and model.n_features_in_==13,'Unexpected fit feature schema')
        require(int(model[1].n_samples_seen_)==int(mask.sum()),'Saved preprocessing sample count is not pooled non-tied DEV')
        np.testing.assert_allclose(model[0].statistics_,np.median(x,axis=0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(model[1].mean_,x.mean(axis=0),rtol=1e-12,atol=1e-12)
        require(model[-1].coef_.size+model[-1].intercept_.size==14,'Wrong parameter count')
        adc.append(dict(pair=pair,label=label,dev_triples=n,available_directional_rows=len(f),fit_rows=int(mask.sum()),
                        excluded_ties=int((~mask).sum()),base_seeds=3,full_selectors=1,full_geometry_oof_selectors=5,
                        model_sha256=lock['model_sha256'],pooled_preprocessing_count_and_moments_verified=True))
        for (seed,direction),part in f.groupby(['seed','direction']):
            adc_seed.append(dict(pair=pair,seed=int(seed),direction=direction,available_rows=len(part),fit_rows=int(mask.loc[part.index].sum())))
        rows=f.to_dict('records'); assignment,_=assign_grouped_folds(rows,5,20260901)
        fold_ids=np.array([assignment[triple_key(r)] for r in rows])
        old=frame(base+'/p3_ablation/dev_p3_selected_by_fold.csv').set_index('fold')
        for fold in range(5):
            train=fold_ids!=fold
            require(train.sum()==6*int(old.loc[fold+1].train_triple_groups),'OOF training scope mismatch')
            require((~train).sum()==6*int(old.loc[fold+1].heldout_triple_groups),'OOF held-out scope mismatch')
            folds.append(dict(pair=pair,fold=fold+1,train_triples=int(train.sum()/6),available_rows=int(train.sum()),
                              non_tied_fit_rows=int((train&mask).sum()),heldout_rows=int((~train).sum()),base_seeds=3))
        if pair not in list(PAIRS)[:4]:continue
        settings=matched_lock['pairs'][pair]
        require(settings['model_sha256']==lock['model_sha256'] and len(settings['actions'])==6,'Matched maps do not share one original fit')
        matched.append(dict(pair=pair,action_families=sorted(settings['actions']),shared_model_sha256=lock['model_sha256']))
        r0=obj(base+'/dynasemble/dev_lock.json')
        require(set(r0['selectors'])=={'1','2','3'},'R0 selector count mismatch')
        for bs in (1,2,3):
            record=r0['selectors'][str(bs)];summary=record['training_summary']
            bound(base+f'/dynasemble/selectors/seed{bs}.pt',expected=record['sha256'])
            require(summary['seed']==bs and summary['num_dev_triples']==summary['num_training_directional_queries']==n,'R0 training scope mismatch')
            dyna.append(dict(pair=pair,variant='R0',base_seed=bs,selector_seed=bs,fit_triples=n,available_directional_rows=2*n,
                             loss_rows_per_epoch=n,epochs=1,optimization_rows=n,model_sha256=record['sha256']))
        folder='outputs/paper_a_safe_correction/dynasemble_controls_v1_review/'+pair
        dl=obj(folder+'/dev_lock.json','dynasemble')
        expected={(v,b,s) for v in VARIANTS for b in (1,2,3) for s in (11,23,37)}
        require(len(dl['selectors'])==45 and {(s['variant'],s['base_seed'],s['selector_seed']) for s in dl['selectors']}==expected,'Dyna final fit inventory mismatch')
        cache_orders={}
        for bs in (1,2,3):
            cache='outputs/paper_a_safe_correction/dynasemble_controls_v1_cache_evidence/'+pair+f'/cache/dev_seed{bs}'
            cm=obj(cache+'/manifest.json','dynasemble')
            require(sha(ROOT/cache/'manifest.json')==dl['cache_manifests'][str(bs)],'Dyna cache base binding mismatch')
            # Original export audit already binds these small observable query arrays.
            q_path=bound(cache+'/queries.npy','dynasemble')
            q=np.load(q_path,allow_pickle=False)
            require(cm['n_triples']==n and len(q)==2*n,'Wrong Dyna directional cache scope')
            require(np.array_equal(q[:n,:3],q[n:,:3]),'Directions differ in original triples')
            cache_orders[bs]=grouped_folds(q[:n,:3])
        for sel in dl['selectors']:
            rel=folder+'/'+sel['relative_dir'];done=obj(rel+'/complete.json','dynasemble');sig=done['signature']
            require(sig['provenance']==dl['provenance'] and sig['cache_manifest_sha256']==dl['cache_manifests'][str(sel['base_seed'])],
                    'Final network not bound to its single-base cache')
            require(sig['fit_ids']==list(range(n)) and sig['validation_ids']==list(range(2*n)),'Wrong final fit/diagnostic scope')
            require(sig['variant']==sel['variant'] and sig['selector_seed']==sel['selector_seed'] and sig['epochs']==sel['epochs'],'Selector identity differs')
            require(done['model_sha256']==sel['model_sha256'],'Final model identity differs')
            bound(rel+'/model.pt','dynasemble',sel['model_sha256'])
            dyna.append(dict(pair=pair,variant=sel['variant'],base_seed=sel['base_seed'],selector_seed=sel['selector_seed'],
                             fit_triples=n,available_directional_rows=2*n,loss_rows_per_epoch=n,epochs=sel['epochs'],
                             optimization_rows=n*sel['epochs'],model_sha256=sel['model_sha256']))
        for bs in (1,2,3):
            for ss in (11,23,37):
                for fold in range(3):
                    fit_ids=np.flatnonzero(cache_orders[bs]!=fold).tolist(); held=np.flatnonzero(cache_orders[bs]==fold)
                    for lr in (1e-5,5e-5,1e-4):
                        done=obj(folder+f'/fits/cv_b{bs}_s{ss}_f{fold}_lr{lr:g}/complete.json','dynasemble');sig=done['signature']
                        require(sig['cache_manifest_sha256']==dl['cache_manifests'][str(bs)] and sig['selector_seed']==ss,'CV base/selector scope mismatch')
                        require(sig['fit_ids']==fit_ids and sig['validation_ids']==np.concatenate((held,held+n)).tolist(),'CV split scope mismatch')
                        require(sig['variant']=='R3_softplus' and sig['epochs']==10 and sig['learning_rate']==lr,'CV trajectory settings mismatch')
                        cv.append(dict(pair=pair,base_seed=bs,selector_seed=ss,fold=fold,learning_rate=lr,
                                       fit_triples=len(fit_ids),available_fit_directional_rows=2*len(fit_ids),
                                       loss_rows_per_epoch=len(fit_ids),heldout_directional_rows=2*len(held),epochs=sig['epochs']))
        print('[SCOPED] '+pair,flush=True)
    require(len(adc)==6 and len(folds)==30 and len(dyna)==192 and len(cv)==324,'Scope inventory incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('adc_pairs',adc),('adc_seed_direction',adc_seed),('adc_folds',folds),('dynasemble_final',dyna),('dynasemble_cv',cv)]:
        path=OUT/(name+'.csv');pd.DataFrame(data).to_csv(path,index=False);sources[path.relative_to(ROOT).as_posix()]=sha(path)
    audit=dict(status='inference_training_scope_checks_passed',sources=sources,adc_pairs=adc,matched_shared_models=matched,
               adc_final_models=6,adc_full_geometry_oof_scopes=30,dyna_historical_final_models=12,dyna_new_final_models=180,dyna_cv_trajectories=324,
               version_warnings=sorted(version_warnings),scientific_model_fits=0,base_model_scoring_runs=0,test_used_for_selection=False,
               results_replaced=False,seed_is_selector_feature=False,rank_cache_required_for_inference=False,
               checkpoint_generalization_tested=False,training_information_matched_between_adc_and_dyna=False,
               deployment='One fixed checkpoint pair and one applicable selector/rule. Reported averages are not an ensemble of selectors or base seeds.')
    path=OUT/'audit.json';path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    sources[path.relative_to(ROOT).as_posix()]=sha(path)
    (ROOT/'paper_a_draft/inference_scope_source_manifest.json').write_text(json.dumps(dict(version='inference_training_scope_v1',sources=sources),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources','version_warnings','matched_shared_models')},indent=2))


if __name__=='__main__':main()
