"""B10: verify returned rank differences and replay fixed endpoint conventions."""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_paper_a_endpoint_contract import CODE as SERVER_CODE
from scripts.analyze_paper_a_conservative_radius import KEY, GRID_COLUMNS, sha, require, metrics
from scripts.crossfit_heterogeneous_dev_policies import best_alpha
from scripts.review_paper_a_raw_score_contract import match_source_bytes

RETURN=ROOT/'outputs/paper_a_safe_correction/endpoint_contract_audit_v1_return'
OUT=ROOT/'outputs/paper_a_safe_correction/endpoint_contract_return_review_v1'
DATA='outputs/paper_a_safe_correction/data_checkpoint_review_v1'
PAIRS={'mkgw_mhyper_native':'W-N','mkgw_mhyper_adamf':'W-A','db15k_mhyper_native':'D-N',
       'db15k_mhyper_adamf':'D-A','mkgw_native_adamf':'W-NA','db15k_native_adamf':'D-NA'}
COUNTS=('rows','export_vs_shared_mismatches','alpha_one_vs_shared_mismatches','alpha_zero_vs_shared_mismatches',
        'normalized_vs_raw_mismatches','raw_nonfinite_rows','normalized_nonfinite_gold_rows','raw_contract_exception_rows')
RR_SUMS=('raw_rr_sum','normalized_rr_sum')
CODE=('scripts/review_paper_a_endpoint_contract.py','scripts/review_paper_a_raw_score_contract.py',
      'scripts/analyze_paper_a_conservative_radius.py','scripts/crossfit_heterogeneous_dev_policies.py',
      'tests/test_endpoint_contract_return.py','docs/protocols/paper_a_endpoint_return_review.md')


def check_differences(rows,cell,canonical,num_entities):
    totals=cell['totals']
    require(set(totals)==set(COUNTS+RR_SUMS),'Unexpected counters')
    for k in COUNTS:
        require(type(totals[k]) is int and 0<=totals[k]<=len(canonical),'Invalid count: '+k)
    require(totals['rows']==len(canonical),'Wrong row count')
    require(len(rows)==totals['normalized_vs_raw_mismatches'],'Difference count mismatch')
    indices=[]; delta=[]
    for row in rows:
        i=row['split_index']; a=row['raw_rank']; b=row['normalized_rank']
        require(type(i) is int and 0<=i<len(canonical),'Invalid difference index')
        require(row['triple']==canonical[i],'Difference triple mismatch')
        require(type(a) is int and type(b) is int and 1<=a<=num_entities and 1<=b<=num_entities and a!=b,
                'Invalid difference rank')
        indices.append(i);delta.append(1/b-1/a)
    require(indices==sorted(set(indices)),'Duplicate/unordered difference rows')
    require(cell['examples']==rows[:10],'Difference examples mismatch')
    for k in RR_SUMS:
        require(math.isfinite(totals[k]) and len(canonical)/num_entities<=totals[k]<=len(canonical),'Invalid RR sum')
    require(math.isclose(math.fsum(delta),totals['normalized_rr_sum']-totals['raw_rr_sum'],rel_tol=0,abs_tol=1e-8),
            'Difference RR conservation failed')


def verify(folder=RETURN,report=None):
    sources={}; byte_checks=[]
    def bind(path,expected=None):
        value=sha(path)
        if expected is not None:require(value==expected,'Hash mismatch: '+str(path))
        sources[path.relative_to(ROOT).as_posix()]=value
        return path
    def obj(path,expected=None):return json.loads(bind(path,expected).read_text(encoding='utf-8'))
    plan=obj(folder/'plan.json'); plan_sha=sha(folder/'plan.json')
    original=obj(folder/'audit.json'); report=original if report is None else report
    baseline=obj(ROOT/'paper_a_draft/data_checkpoint_source_manifest.json')['sources']
    def frame(rel):return pd.read_csv(bind(ROOT/rel,baseline[rel]),float_precision='round_trip')
    ledger=frame(DATA+'/checkpoint_diagnostics.csv')
    sizes=frame(DATA+'/split_counts.csv')
    require(len(ledger)==18 and ledger.run.nunique()==18,'Checkpoint coverage mismatch')
    roles={(d,m,s) for d in ('mkg_w','db15k') for m in ('M-Hyper','NativE','AdaMF-MAT') for s in (1,2,3)}
    require(set(zip(ledger.dataset,ledger.model,ledger.seed))==roles,'Checkpoint roles mismatch')
    expected_sources=set(SERVER_CODE)|{'paper_a_draft/data_checkpoint_source_manifest.json',DATA+'/checkpoint_diagnostics.csv',DATA+'/split_counts.csv'}
    expected_sources|={p.relative_to(ROOT).as_posix() for p in (ROOT/'ml/training/src').rglob('*.py')}
    expected_sources|={r+'/config_merged.json' for r in ledger.run}
    require(set(plan['sources'])==expected_sources,'Server source inventory mismatch')
    require(plan['git_commit']=='19b4d39e704c88d6d716ecbd3df4c23282ad1beb','Unexpected server revision')
    require(plan['cells']==72 and plan['expected_rows']==474732 and plan['filter_scope']=='train_dev_test','Changed audit scope')
    require(plan['training_runs']==0 and plan['policy_selection'] is False,'Unplanned training or selection')
    for rel,value in plan['sources'].items():
        path=bind(ROOT/rel)
        mode=match_source_bytes(path.read_bytes(),value,allow_newlines=rel.endswith('.py'))
        byte_checks.append(dict(path=rel,match=mode,local_sha256=sha(path),server_sha256=value))
    require(plan['checkpoint_hashes']==dict(zip(ledger.run,ledger.checkpoint_sha256)),'Checkpoint hashes differ')
    configs={}
    for _,r in ledger.iterrows():
        require(r.checkpoint_sha256==baseline[r.run+'/best.ckpt'],'Checkpoint differs from original manifest')
        configs[r.run]=obj(ROOT/r.run/'config_merged.json',baseline[r.run+'/config_merged.json'])
    canonical={}
    for dataset in ('mkg_w','db15k'):
        rows=frame(DATA+'/'+dataset+'_split_rows.csv.gz')
        for split,name in [('dev','valid'),('test','test')]:
            part=rows[rows.canonical_split==name].sort_values('canonical_row_index')
            n=int(sizes[(sizes.dataset==dataset)&(sizes.stage=='canonical')&(sizes.split==name)].triples.iloc[0])
            require(part.canonical_row_index.tolist()==list(range(n)),'Canonical coverage mismatch')
            canonical[dataset,split]=part[['head','relation','tail']].astype(int).values.tolist()
    expected={(r.run,s,d):r for _,r in ledger.iterrows() for s in ('dev','test') for d in ('head','tail')}
    cells=report['cells']; keys=[(c['run'],c['split'],c['direction']) for c in cells]
    require(len(keys)==72 and set(keys)==set(expected),'Incomplete or duplicate cells')
    require(report['plan_sha256']==plan_sha,'Wrong plan hash')
    require(report['training_runs']==0 and report['policy_selection'] is False and report['historical_results_replaced'] is False,
            'Unplanned training, selection or replacement')
    diff_rows=[]; inventory={'audit.json','plan.json'}; summaries=[]
    for c,key in zip(cells,keys):
        r=expected[key]; cfg=configs[r.run]
        name=f'{r.dataset}_{r.model}_s{r.seed}_{c["split"]}_{c["direction"]}'
        inventory|={'cells/'+name+'.json','cells/'+name+'.differences.json'}
        require(obj(folder/'cells'/(name+'.json'))==c,'Receipt differs from audit')
        require((c['dataset'],c['model'],c['seed'])==(r.dataset,r.model,int(r.seed)),'Cell role mismatch')
        require(c['checkpoint_sha256']==r.checkpoint_sha256 and c['plan_sha256']==plan_sha,'Cell identity mismatch')
        require(c['query_batch_size']==int(cfg['evaluation'].get('query_batch_size',8)) and c['chunk_size']==int(cfg['evaluation'].get('chunk_size',4096)),
                'Scoring batch changed')
        require(datetime.fromisoformat(c['completed_at_utc'])>=datetime.fromisoformat(plan['prepared_at_utc']),'Cell predates plan')
        triples=canonical[c['dataset'],c['split']]
        require(c['triple_order_sha256']==hashlib.sha256(json.dumps(triples,separators=(',',':')).encode()).hexdigest(),'Wrong canonical order')
        require(c['differences_file']==name+'.differences.json','Wrong difference filename')
        diffs=obj(folder/'cells'/c['differences_file'],c['differences_sha256'])
        check_differences(diffs,c,triples,cfg['_dataset_manifest']['counts']['entities'])
        for row in diffs:
            diff_rows.append(dict(dataset=c['dataset'],model=c['model'],seed=c['seed'],split=c['split'],direction=c['direction'],
                                  split_index=row['split_index'],head_id=row['triple'][0],relation_id=row['triple'][1],tail_id=row['triple'][2],
                                  raw_rank=row['raw_rank'],normalized_rank=row['normalized_rank']))
        summaries.append({**{k:c[k] for k in ('dataset','model','seed','split','direction')},**c['totals']})
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==inventory,'Unexpected or missing return files')
    totals={k:sum(c['totals'][k] for c in cells) for k in COUNTS+RR_SUMS}
    for k in totals:
        require(math.isclose(totals[k],report['totals'][k],rel_tol=0,abs_tol=1e-8 if k in RR_SUMS else 0),'Total counter mismatch: '+k)
    require(set(report['totals'])==set(totals),'Wrong summary keys')
    zero_keys=[k for k in COUNTS if k not in ('rows','normalized_vs_raw_mismatches')]
    require(all(totals[k]==0 for k in zero_keys),'Shared/deployed endpoint or finite-score contract failed')
    status='endpoint_difference_review_required' if totals['normalized_vs_raw_mismatches'] else 'endpoint_equivalence_checks_passed'
    require(report['status']==status,'False endpoint equivalence status')
    review=dict(status='endpoint_return_checks_passed',sources=sources,source_byte_checks=byte_checks,checkpoints=18,cells=72,
                returned_files=len(inventory),totals=totals,runtimes=[json.loads(v) for v in sorted({json.dumps(c['runtime'],sort_keys=True) for c in cells})],
                server_git_commit=plan['git_commit'],training_runs=0,test_used_for_selection=False,historical_results_replaced=False,
                historical_raw_score_equality_established=False,normalized_endpoint_equivalence=False,
                interpretation='Deployed endpoints match shared raw ranks; normalized ranks require the reported separate convention.')
    return review,pd.DataFrame(summaries),pd.DataFrame(diff_rows),canonical


def endpoint_substitution(old,alpha,delta_a,delta_b):
    """Change only exact endpoints; no tolerance or policy re-selection."""
    old,alpha=np.asarray(old,dtype=float),np.asarray(alpha,dtype=float)
    return old+np.where(alpha==1,delta_a,np.where(alpha==0,delta_b,0.))


def sensitivity(review,cells,differences,canonical):
    expected=json.loads((ROOT/'paper_a_draft/rerun_source_manifest.json').read_text())['sources']
    sources=review['sources']
    sources['paper_a_draft/rerun_source_manifest.json']=sha(ROOT/'paper_a_draft/rerun_source_manifest.json')
    def bound(rel):
        require(sha(ROOT/rel)==expected[rel],'Changed historical export: '+rel)
        sources[rel]=expected[rel];return ROOT/rel
    results=[]; by_cell=[]; anchor_checks=[]; joins=[]; effects=[]
    cell_index=cells.set_index(['dataset','model','seed','split','direction'])
    for pair,label in PAIRS.items():
        base='outputs/paper_a_safe_correction/information_boundary_v2/'+pair
        lock=json.loads(bound(base+'/dev_lock/anchored_dev_lock.json').read_text())
        for split,suffix in [('dev','dev_lock/dev_locked_query_rows.csv'),('test','test_anchored/test_locked_query_rows.csv')]:
            f=pd.read_csv(bound(base+'/'+suffix),float_precision='round_trip')
            dataset=f.dataset.iloc[0]; require(f.dataset.nunique()==1 and set(f.split)=={split},'Wrong export split')
            require(not f.duplicated(KEY).any() and len(f)==6*len(canonical[dataset,split]),'Wrong export coverage')
            deltas=[]; ranks=[]
            for role in ('a','b'):
                model=f['expert_'+role+'_name'].iloc[0]
                require(f['expert_'+role+'_name'].nunique()==1,'Inconsistent model role')
                diff=differences[(differences.dataset==dataset)&(differences.model==model)&(differences.split==split)]
                idx=pd.MultiIndex.from_frame(f[KEY]); change=diff.set_index(KEY)
                require(change.index.isin(idx).all(),'Difference missing from export')
                raw=f['rank_'+role].to_numpy(); updated=raw.copy()
                loc=idx.get_indexer(change.index)
                require(np.array_equal(raw[loc],change.raw_rank),'Historical difference rank mismatch')
                updated[loc]=change.normalized_rank
                old_rr=1/raw;new_rr=1/updated
                np.testing.assert_allclose(old_rr,f['rr_'+role],rtol=0,atol=1e-12)
                for (seed,direction),part in f.groupby(['seed','direction']):
                    expected_triples={tuple(t) for t in canonical[dataset,split]}
                    require(set(map(tuple,part[['head_id','relation_id','tail_id']].values.tolist()))==expected_triples,'Export canonical triples mismatch')
                    c=cell_index.loc[dataset,model,seed,split,direction]
                    require(len(part)==c.rows,'Export cell coverage mismatch')
                    require(math.isclose(math.fsum(old_rr[part.index]),c.raw_rr_sum,rel_tol=0,abs_tol=1e-8),'Historical raw RR sum mismatch')
                    require(math.isclose(math.fsum(new_rr[part.index]),c.normalized_rr_sum,rel_tol=0,abs_tol=1e-8),'Reconstructed normalized RR sum mismatch')
                deltas.append(new_rr-old_rr); ranks.append(updated)
                joins.append(dict(pair=pair,split=split,model=model,role=role,rows=len(f),difference_rows=len(diff),raw_difference_ranks_exact=True,cell_rr_sums_matched=True))
            a,b=deltas;grid=f[GRID_COLUMNS].to_numpy()
            np.testing.assert_allclose(grid[:,0],f.rr_b,rtol=0,atol=1e-12)
            np.testing.assert_allclose(grid[:,-1],f.rr_a,rtol=0,atol=1e-12)
            actions={'Global':(f.alpha_global,f.rr_global),'ADC':(f.alpha_anchored_locked,f.rr_anchored_locked),
                     'Query-soft':(f.alpha_query_soft_locked,f.rr_query_soft_locked),'Relation':(f.alpha_relation,f.rr_relation),
                     'Equal':(np.full(len(f),.5),f.rr_equal)}
            np.testing.assert_array_equal(f.alpha_global,np.full(len(f),lock['alpha0']))
            converted={}
            for method,(alpha,rr) in actions.items():
                np.testing.assert_allclose(grid[np.arange(len(f)),np.rint(20*np.asarray(alpha)).astype(int)],rr,rtol=0,atol=1e-12)
                converted[method]=endpoint_substitution(rr,alpha,a,b)
            actions['RRF']=(np.full(len(f),np.nan),f.rr_rrf);converted['RRF']=f.rr_rrf.to_numpy()
            old_effect=f.rr_anchored_locked.to_numpy()-f.rr_global.to_numpy()
            new_effect=converted['ADC']-converted['Global']
            effects.append(dict(pair=pair,split=split,rows=len(f),changed_effect_rows_at_1e_12=int(np.count_nonzero(np.abs(new_effect-old_effect)>1e-12)),
                                max_abs_row_effect_change=float(np.abs(new_effect-old_effect).max()),
                                raw_mean_effect=float(old_effect.mean()),normalized_endpoint_mean_effect=float(new_effect.mean())))
            for method,(alpha,rr) in actions.items():
                alpha=np.asarray(alpha);new=converted[method];old=np.asarray(rr)
                for convention,values,reference in [('raw_endpoint',old,f.rr_global.to_numpy()),('normalized_endpoint',new,converted['Global'])]:
                    row=dict(pair=pair,label=label,split=split,method=method,convention=convention,
                             changed_rows=int(np.count_nonzero(new!=old)),endpoint_rows=int(np.count_nonzero((alpha==0)|(alpha==1))))
                    results.append({**row,**metrics(values,reference,alpha if method!='RRF' else None,f.alpha_global.to_numpy())})
                    for (seed,direction),part in f.groupby(['seed','direction']):
                        i=part.index
                        by_cell.append({**row,'seed':int(seed),'direction':direction,'changed_rows':int(np.count_nonzero(new[i]!=old[i])),
                                        'endpoint_rows':int(np.count_nonzero((alpha[i]==0)|(alpha[i]==1))),**metrics(values[i],reference[i])})
            if split=='dev':
                original,_=best_alpha(f.to_dict('records'),tuple(lock['alpha_grid']))
                changed=f[GRID_COLUMNS].copy();changed.iloc[:,0]+=b;changed.iloc[:,-1]+=a
                alternative,_=best_alpha(changed.to_dict('records'),tuple(lock['alpha_grid']))
                require(original==lock['alpha0'],'Original DEV anchor not reproduced')
                labels=np.sign(f.rr_a.to_numpy()-f.rr_b.to_numpy())
                new_labels=np.sign(1/ranks[0]-1/ranks[1])
                anchor_checks.append(dict(pair=pair,label=label,original_anchor=original,normalized_endpoint_grid_anchor=alternative,
                                          changed=original!=alternative,supervision_category_changes=int(np.count_nonzero(labels!=new_labels)),
                                          dev_rows=len(f),alternative_applied_to_test=False))
            print('[VERIFIED] '+pair+'/'+split,flush=True)
    review.update(historical_join_checks=joins,full_dev_anchor_checks=anchor_checks,paired_effect_checks=effects,
                  primary_test_effects_unchanged_at_1e_12=all(r['changed_effect_rows_at_1e_12']==0 for r in effects if r['split']=='test' and r['pair'] in list(PAIRS)[:4]),
                  policy_rows=len(results),policy_cell_rows=len(by_cell),
                  selector_fits=0,local_scorer_runs=0,new_confidence_intervals=False,
                  sensitivity_scope='Exact endpoint substitutions with original full-DEV actions fixed; interior cache ranks unchanged; DEV is in-sample replay.')
    return pd.DataFrame(results),pd.DataFrame(by_cell),pd.DataFrame(anchor_checks)


def main():
    review,cells,differences,canonical=verify()
    policies,policy_cells,anchors=sensitivity(review,cells,differences,canonical)
    OUT.mkdir(parents=True,exist_ok=True)
    summary=cells.groupby(['dataset','model','split'],sort=False)[list(COUNTS+RR_SUMS)].sum().reset_index()
    summary['delta_mrr']=(summary.normalized_rr_sum-summary.raw_rr_sum)/summary.rows
    for name,frame in [('expert_cells',cells),('expert_summary',summary),('differences',differences),
                       ('policy_sensitivity',policies),('policy_cells',policy_cells),('dev_anchor_sensitivity',anchors)]:
        path=OUT/(name+'.csv');frame.to_csv(path,index=False)
        review['sources'][path.relative_to(ROOT).as_posix()]=sha(path)
    for rel in CODE:review['sources'][rel]=sha(ROOT/rel)
    review['normalized_better_rows']=int((differences.normalized_rank<differences.raw_rank).sum())
    review['normalized_worse_rows']=int((differences.normalized_rank>differences.raw_rank).sum())
    review['b10_status']='explicit_endpoint_convention_verified_normalized_inequivalence_quantified'
    path=OUT/'audit.json';path.write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    manifest=dict(version='endpoint_contract_return_review_v1',sources={**review['sources'],path.relative_to(ROOT).as_posix():sha(path)})
    (ROOT/'paper_a_draft/endpoint_contract_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('sources','source_byte_checks','historical_join_checks')},indent=2))


if __name__=='__main__':main()
