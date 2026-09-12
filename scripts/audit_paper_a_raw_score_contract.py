"""Server-only full-score audit for B01; no fitting, policy choice, or result replacement.

Scores 18 existing checkpoints on DEV/TEST in both directions and returns one
small JSON. --plan-only lists the scope without loading models or scoring.
"""
import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.eval_heterogeneous_complementarity import load_expert, score_expert_block, query_zscore_with_reference
from scripts.analyze_paper_a_conservative_radius import sha, require

DATA=ROOT/'outputs/paper_a_safe_correction/data_checkpoint_review_v1'
OUTPUT=ROOT/'outputs/paper_a_safe_correction/raw_score_contract_audit.json'


def exception_counts(raw,reference):
    reference=reference.reshape(-1)
    require(raw.ndim==2 and len(reference)==len(raw),'Expected a candidate matrix and one gold score per row')
    finite=torch.isfinite(raw)
    count=finite.sum(1)
    values=torch.where(finite,raw,torch.zeros_like(raw))
    mean=values.sum(1)/count.clamp_min(1)
    centered=torch.where(finite,raw-mean[:,None],torch.zeros_like(raw))
    variance=centered.square().sum(1)/count.clamp_min(1)
    z,gold=query_zscore_with_reference(raw,reference)
    failures=(~finite).any(1)|~torch.isfinite(reference)|~torch.isfinite(mean)|~torch.isfinite(variance)|~torch.isfinite(gold)|(finite&~torch.isfinite(z)).any(1)
    return dict(rows=len(raw),nonfinite_candidate_values=int((~finite).sum()),
                partial_nonfinite_rows=int(((count>0)&(count<raw.shape[1])).sum()),
                empty_finite_rows=int((count==0).sum()),constant_finite_rows=int(((count>0)&(variance==0)).sum()),
                nonfinite_gold_rows=int((~torch.isfinite(reference)).sum()),
                nonfinite_moment_rows=int((~torch.isfinite(mean)|~torch.isfinite(variance)).sum()),
                invalid_normalized_finite_values=int((finite&~torch.isfinite(z)).sum()),
                nonfinite_normalized_gold_rows=int((~torch.isfinite(gold)).sum()),
                exceptional_rows=int(failures.sum())), failures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    expected=json.loads((ROOT/'paper_a_draft/data_checkpoint_source_manifest.json').read_text())['sources']
    sources={}
    for name in ('checkpoint_diagnostics.csv','split_counts.csv'):
        path=DATA/name;rel=path.relative_to(ROOT).as_posix()
        require(expected.get(rel)==sha(path),'Unverified checkpoint/split ledger')
        sources[rel]=sha(path)
    with (DATA/'checkpoint_diagnostics.csv').open() as f:runs=list(csv.DictReader(f))
    require(len(runs)==18 and len({r['run'] for r in runs})==18,'Expected 18 unique checkpoints')
    with (DATA/'split_counts.csv').open() as f:
        counts={(r['dataset'],r['split']):int(r['triples']) for r in csv.DictReader(f) if r['stage']=='canonical'}
    if args.plan_only:
        print(json.dumps(dict(checkpoints=18,cells=72,device=args.device,output=str(OUTPUT),
                              split_counts={str(k):v for k,v in counts.items() if k[1] in ('valid','test')}),indent=2));return
    require(not OUTPUT.exists(),'Audit output exists; archive it before a deliberate new run')
    if args.device=='cuda':require(torch.cuda.is_available(),'CUDA requested but unavailable')
    sources[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    for rel in ('router/score_combination.py','scripts/eval_heterogeneous_complementarity.py'):
        sources[rel]=sha(ROOT/rel)
    results=[]
    for run in runs:
        folder=ROOT/run['run'];checkpoint=folder/'best.ckpt'
        require(sha(checkpoint)==run['checkpoint_sha256'],'Checkpoint differs from audited baseline')
        sources[checkpoint.relative_to(ROOT).as_posix()]=run['checkpoint_sha256']
        config_rel=(folder/'config_merged.json').relative_to(ROOT).as_posix()
        config_hash=sha(folder/'config_merged.json')
        require(expected.get(config_rel)==config_hash,'Config differs from audited baseline')
        sources[config_rel]=config_hash
        expert=load_expert(run['model'],folder,args.device)
        for split,triples in [('dev',expert.bundle.valid_triples),('test',expert.bundle.test_triples)]:
            require(len(triples)==counts[(run['dataset'],'valid' if split=='dev' else split)],'Effective split count differs')
            triples_sha=hashlib.sha256(json.dumps([list(t) for t in triples],separators=(',',':')).encode()).hexdigest()
            for direction in ('head','tail'):
                total={};examples=[];batch=max(1,expert.query_batch_size)
                for start in range(0,len(triples),batch):
                    q=torch.tensor(triples[start:start+batch],dtype=torch.long)
                    _,reference,raw=score_expert_block(expert,q,direction,{},args.device,retain_unfiltered=True)
                    row,bad=exception_counts(raw,reference)
                    for key,value in row.items():total[key]=total.get(key,0)+value
                    for index in torch.where(bad)[0].tolist():
                        if len(examples)<20:examples.append(dict(split_index=start+index,triple=list(triples[start+index])))
                results.append(dict(run=run['run'],dataset=run['dataset'],model=run['model'],seed=int(run['seed']),
                                    split=split,direction=direction,triple_order_sha256=triples_sha,examples=examples,**total))
                print(f"[AUDITED] {run['dataset']} {run['model']} seed={run['seed']} {split}/{direction}: {total['exceptional_rows']} exceptional",flush=True)
        del expert;gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    exceptional=sum(r['exceptional_rows'] for r in results)
    report=dict(status='raw_score_contract_passed' if exceptional==0 else 'raw_score_exceptions_found',
                sources=sources,cells=results,expected_cells=72,exceptional_rows=exceptional,
                training_runs=0,policy_selection=False,historical_rows_replaced=False,
                note='Re-execution of frozen checkpoints; does not prove bitwise equality to historical raw scores that were not retained.')
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(OUTPUT)
    if exceptional:raise SystemExit(2)


if __name__=='__main__':main()
