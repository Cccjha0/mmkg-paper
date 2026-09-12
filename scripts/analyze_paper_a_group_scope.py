"""E05/E06: train-defined group reports; no scorer execution or new selection."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.group_scope_diagnostics import FREQUENCY,TOP,METRICS,relation_map,aggregate,top_concentration
from scripts.analyze_paper_a_conservative_radius import sha,require,close
from scripts.analyze_paper_a_winner_signal import PAIRS

OUT=ROOT/'outputs/paper_a_safe_correction/group_scope_v1'
CODE=('scripts/analyze_paper_a_group_scope.py','router/group_scope_diagnostics.py',
    'tests/test_group_scope_diagnostics.py','docs/protocols/paper_a_group_scope_review.md',
    'scripts/analyze_paper_a_conservative_radius.py','scripts/analyze_paper_a_winner_signal.py')


def main():
    require(not (OUT/'audit.json').exists(),'Completed group review is immutable')
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    sources={r:sha(ROOT/r) for r in CODE};expected={};outputs={}
    for name in ('data_checkpoint','alignment_training','complementarity','scope_boundary'):
        p=ROOT/f'paper_a_draft/{name}_source_manifest.json';sources[p.relative_to(ROOT).as_posix()]=sha(p)
        expected.update(json.loads(p.read_text())['sources'])
    def old(rel):
        p=ROOT/rel;require(sha(p)==expected.get(rel),'Changed original evidence '+rel);sources[rel]=sha(p);return p
    def save(name,frame):
        p=OUT/name
        frame.to_csv(p,index=False,lineterminator='\n',compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
        outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    base='outputs/paper_a_safe_correction/'
    maps={};training_checks=[]
    for dataset,expected_n in [('mkg_w',34196),('db15k',71300)]:
        manifest=json.loads(old(base+f'data_checkpoint_review_v1/{dataset}_split_manifest.json').read_text())
        m=manifest['training_manifest']
        ledger=pd.read_csv(old(base+f'data_checkpoint_review_v1/{dataset}_split_rows.csv.gz'))
        train=ledger[ledger.canonical_split=='train'].sort_values('canonical_row_index')
        require(train.source_split.eq('train').all(),'TRAIN ledger includes another source split')
        require(len(train)==expected_n==m['counts']['train'],'Effective TRAIN count differs')
        require(np.array_equal(train.canonical_row_index,np.arange(expected_n)),'TRAIN row indices incomplete')
        triples=train[['head','relation','tail']].to_numpy(dtype=np.int64)
        ending={'LF':'\n','CRLF':'\r\n'}[manifest['reconstructed_tsv_line_endings']['train']]
        digest=hashlib.sha256(''.join(f'{h}\t{r}\t{t}{ending}' for h,r,t in triples).encode()).hexdigest()
        require(digest==m['hashes']['splits']['train'],'Reconstructed effective TRAIN hash differs')
        names=json.loads(old(base+f'alignment_training_v1/{dataset}_relation_by_id.json').read_text())
        require(len(names)==m['counts']['relations'],'Relation universe differs')
        mapping=relation_map(triples,names);mapping.insert(0,'dataset',dataset);maps[dataset]=mapping
        training_checks.append(dict(dataset=dataset,effective_train_triples=expected_n,relations=len(mapping),
            canonical_train_sha256=digest,train_only_map=True,duplicate_train_triples=0,
            top_ids=mapping[mapping.named_group!='Other'].sort_values('train_rank').relation_id.tolist()))
    # Freeze both train-only maps before reading any outcome summary.
    save('relation_map.csv',pd.concat(maps.values(),ignore_index=True))
    summary=pd.read_csv(old(base+'complementarity_review_v1/pair_summary.csv'),float_precision='round_trip').set_index(['pair','split'])
    conditional=pd.read_csv(old(base+'complementarity_review_v1/conditional_summary.csv'),float_precision='round_trip')
    relation=pd.read_csv(old(base+'complementarity_review_v1/relation_summary.csv.gz'),float_precision='round_trip')
    previous=pd.read_csv(old(base+'scope_boundary_v1/summary.csv'),float_precision='round_trip').set_index(['pair','split'])
    direction_rows=[];group_rows=[];group_direction=[];concentration=[];cells=[];checks=[]
    for pair,label in PAIRS.items():
        dataset='mkg_w' if pair.startswith('mkgw') else 'db15k';mapping=maps[dataset]
        for split in ('dev_oof','test'):
            s=summary.loc[pair,split];n=int(s.n_observations)
            f=relation[(relation.pair==pair)&(relation.split==split)].copy()
            f=f.merge(mapping,on='relation_id',validate='many_to_one',how='left')
            require(f.frequency.notna().all(),'Outcome relation missing from TRAIN map')
            pooled=f[f.scope=='relation'];directed=f[f.scope=='relation_direction']
            require(not pooled.relation_id.duplicated().any(),'Repeated relation cell')
            require(not directed[['relation_id','direction']].duplicated().any(),'Repeated direction cell')
            overall=aggregate(pooled,n,6)
            for k in METRICS:close(overall[k],s[k],'Prior pair summary differs: '+k)
            close(overall['delta_mrr'],previous.loc[pair,split].utility,'Later full utility differs')
            for direction in ('head','tail'):
                part=directed[directed.direction==direction];r=aggregate(part,n,3)
                old_direction=conditional[(conditional.pair==pair)&(conditional.split==split)&
                    (conditional.stratum=='direction')&(conditional.value==direction)]
                require(len(old_direction)==1,'Missing prior direction cell')
                for k in METRICS:close(r[k],old_direction.iloc[0][k],'Prior direction point differs: '+k)
                direction_rows.append(dict(pair=pair,label=label,dataset=dataset,split=split,direction=direction,**r))
            for partition,names in [('frequency',FREQUENCY),('named_group',TOP)]:
                records=[]
                for name in names:
                    r=aggregate(pooled[pooled[partition]==name],n,6)
                    context=dict(pair=pair,label=label,dataset=dataset,split=split,partition=partition,group=name,
                        defined_relations=int((mapping[partition]==name).sum()))
                    records.append(dict(**context,**r));group_rows.append(records[-1])
                    for direction in ('head','tail'):
                        part=directed[(directed[partition]==name)&(directed.direction==direction)]
                        group_direction.append(dict(**context,direction=direction,**aggregate(part,n,3)))
                require(sum(r['n_observations'] for r in records)==n,'Group partition incomplete')
                close(sum(r['delta_contribution'] for r in records),overall['delta_mrr'],'Group contributions do not close')
            top=pooled[pooled.named_group!='Other']
            c=top_concentration(top,pooled);close(c['top_contribution']+c['other_contribution'],overall['delta_mrr'],'Top/rest contributions differ')
            concentration.append(dict(pair=pair,label=label,dataset=dataset,split=split,**c))
            cells.append(f[['pair','label','dataset','split','scope','relation_id','relation_iri','direction','train_triples',
                'frequency','named_group','train_rank','n_triples','n_observations',*METRICS,'mass','delta_contribution']])
            checks.append(dict(pair=pair,label=label,split=split,observations=n,original_absolute_mrr_reconciled=True,
                original_direction_mrr_reconciled=True,later_utility_reconciled=True,group_contributions_reconciled=True,
                negative_frequency_groups=int(sum(r['delta_mrr']<0 for r in group_rows if r['pair']==pair and r['split']==split and r['partition']=='frequency')),
                negative_named_groups=int(sum(r['delta_mrr']<0 for r in group_rows if r['pair']==pair and r['split']==split and r['partition']=='named_group'))))
            print(f'[GROUP] {label} {split}: {len(pooled)} observed relations; all partitions reconciled',flush=True)
    require(len(direction_rows)==24 and len(group_rows)==120 and len(group_direction)==240 and len(concentration)==12,'Incomplete report inventory')
    require(sum(c['observations'] for c in checks)==474732,'Incomplete six-pair inventory')
    for name,rows in [('directions.csv',direction_rows),('groups.csv',group_rows),('group_directions.csv',group_direction),
            ('concentration.csv',concentration)]:save(name,pd.DataFrame(rows))
    save('relation_cells.csv.gz',pd.concat(cells,ignore_index=True))
    decision=dict(choice='narrow_two_benchmark_retrospective',regime='reliable-primary',primary_pairs=4,additional_pairs=2,
        datasets={'MKG-W':'primary retrospective evaluation','DB15K':'secondary external retrospective evaluation',
                  'OpenBG-IMG':'historical discovery; excluded from corrected numerical evaluation'},
        evaluated_benchmarks=2,independent_confirmation_sets=0,new_unexposed_datasets=0,new_heldout_pairs=0,
        broad_benchmark_robustness_established=False,requires_server_run=False)
    p=OUT/'scope_decision.json';p.write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8');outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    audit=dict(status='group_scope_checks_passed',failures=[],sources=sources,outputs=outputs,checks=checks,training_checks=training_checks,
        direction_rows=24,group_rows=120,group_direction_rows=240,concentration_rows=12,relation_rows=sum(len(f) for f in cells),
        relation_map_rows=sum(len(m) for m in maps.values()),observations=474732,selector_fits=0,scorer_runs=0,
        test_used_for_group_selection=False,new_policy_selection=False,historical_results_replaced=False,
        group_map_written_before_outcomes=True,new_confidence_intervals=False,group_metrics_descriptive=True,
        scope_choice=decision['choice'],independent_confirmation_sets=0,elapsed_seconds=time.monotonic()-started)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
