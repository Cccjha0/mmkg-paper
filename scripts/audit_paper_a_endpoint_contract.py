"""B10 server-only raw/shared/normalized endpoint audit; no fitting or selection."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from router.endpoint_audit import compare_endpoints
from scripts.analyze_paper_a_conservative_radius import sha, require
from scripts.audit_paper_a_raw_score_contract import exception_counts
from scripts.eval_heterogeneous_complementarity import load_expert, evaluation_fact_indexes, score_expert_block

OUT=ROOT/'outputs/paper_a_safe_correction/endpoint_contract_audit_v1'
CODE=('scripts/audit_paper_a_endpoint_contract.py','router/endpoint_audit.py','scripts/eval_heterogeneous_complementarity.py',
      'scripts/audit_paper_a_raw_score_contract.py','scripts/analyze_paper_a_conservative_radius.py',
      'router/query_geometry.py','router/score_combination.py','router/information_boundary.py','tests/test_guarded_inference_endpoints.py',
      'docs/protocols/paper_a_inference_endpoint_contract_review.md','tests/test_endpoint_audit_runner.py')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    partial.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    manifest_path=ROOT/'paper_a_draft/data_checkpoint_source_manifest.json'
    expected=json.loads(manifest_path.read_text())['sources']
    ledger_path=ROOT/'outputs/paper_a_safe_correction/data_checkpoint_review_v1/checkpoint_diagnostics.csv'
    require(sha(ledger_path)==expected[ledger_path.relative_to(ROOT).as_posix()],'Changed checkpoint ledger')
    ledger=pd.read_csv(ledger_path)
    require(len(ledger)==18 and ledger.run.nunique()==18,'Expected 18 checkpoints')
    split_path=ledger_path.with_name('split_counts.csv')
    require(sha(split_path)==expected[split_path.relative_to(ROOT).as_posix()],'Changed split ledger')
    split_frame=pd.read_csv(split_path)
    sizes={(r.dataset,r.split):int(r.triples) for _,r in split_frame.iterrows() if r.stage=='canonical'}
    expected_rows=sum(2*(sizes[(r.dataset,'valid')]+sizes[(r.dataset,'test')]) for _,r in ledger.iterrows())
    if args.plan_only:
        print(json.dumps(dict(checkpoints=18,cells=72,rows=expected_rows,device='cuda',training_runs=0,policy_selection=False,
                              compares=['shared versus raw export','deployed endpoint versus shared','normalized versus raw'],
                              output=str(OUT/'audit.json')),indent=2));return
    require(torch.cuda.is_available(),'Run full scoring on the authorized CUDA server')
    torch.set_num_threads(1)
    sources={p:sha(ROOT/p) for p in CODE}
    for path in (manifest_path,ledger_path,split_path,*list((ROOT/'ml/training/src').rglob('*.py'))):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for run in ledger.run:
        rel=run+'/config_merged.json'
        require(sha(ROOT/rel)==expected[rel],'Changed checkpoint config')
        sources[rel]=expected[rel]
    plan=dict(sources=sources,checkpoint_hashes={r.run:r.checkpoint_sha256 for _,r in ledger.iterrows()},
              cells=72,expected_rows=expected_rows,training_runs=0,policy_selection=False,filter_scope='train_dev_test')
    plan_path=OUT/'plan.json'
    if plan_path.exists():
        previous=json.loads(plan_path.read_text())
        require(all(previous[k]==v for k,v in plan.items()),'Plan changed; do not overwrite completed cells')
    else:
        write(plan_path,dict(plan,prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                             git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))
    plan_sha=sha(plan_path)
    def receipt(path,run,split,direction):
        c=json.loads(path.read_text())
        require(c['plan_sha256']==plan_sha and c['run']==run.run and c['split']==split and c['direction']==direction,
                'Cell identity/plan mismatch')
        require(c['checkpoint_sha256']==run.checkpoint_sha256,'Cell checkpoint mismatch')
        require(c['totals']['rows']==sizes[(run.dataset,'valid' if split=='dev' else split)],'Cell coverage mismatch')
        require(c['differences_file']==path.with_suffix('.differences.json').name,'Cell difference path mismatch')
        require(sha(path.parent/c['differences_file'])==c['differences_sha256'],'Changed difference rows')
        return c
    cells=[]
    for _,run in ledger.iterrows():
        checkpoint=ROOT/run.run/'best.ckpt'
        require(sha(checkpoint)==run.checkpoint_sha256,'Changed checkpoint')
        cell_paths={(split,d):OUT/'cells'/f'{run.dataset}_{run.model}_s{run.seed}_{split}_{d}.json'
                    for split in ('dev','test') for d in ('head','tail')}
        if all(p.exists() for p in cell_paths.values()):
            for (split,direction),path in cell_paths.items():
                cells.append(receipt(path,run,split,direction))
            print(f'[RESUME] {run.dataset} {run.model} {run.seed}',flush=True);continue
        expert=load_expert(run.model,ROOT/run.run,args.device)
        require(expert.bundle.manifest['hashes']==expert.cfg['_dataset_manifest']['hashes'],'Changed canonical data/features')
        facts=evaluation_fact_indexes(expert.bundle,include_test=True)
        for (split,direction),path in cell_paths.items():
            if path.exists():
                cells.append(receipt(path,run,split,direction));continue
            triples=expert.bundle.valid_triples if split=='dev' else expert.bundle.test_triples
            require(len(triples)==sizes[(run.dataset,'valid' if split=='dev' else split)],'Effective split count differs')
            totals={}; examples=[]; differences=[]
            for start in range(0,len(triples),expert.query_batch_size):
                q=torch.tensor(triples[start:start+expert.query_batch_size],dtype=torch.long)
                _,gold,raw=score_expert_block(expert,q,direction,{},args.device,retain_unfiltered=True)
                counts,shared,normalized=compare_endpoints(raw,gold,q,direction,facts[direction])
                health,_=exception_counts(raw,gold)
                counts['raw_contract_exception_rows']=health['exceptional_rows']
                for k,v in counts.items(): totals[k]=totals.get(k,0)+v
                for i in torch.where(shared!=normalized)[0].tolist():
                    row=dict(split_index=start+i,triple=q[i].tolist(),raw_rank=int(shared[i]),normalized_rank=int(normalized[i]))
                    differences.append(row)
                    if len(examples)<10:examples.append(row)
            difference_path=path.with_suffix('.differences.json')
            write(difference_path,differences)
            c=dict(run=run.run,dataset=run.dataset,model=run.model,seed=int(run.seed),split=split,direction=direction,
                   checkpoint_sha256=run.checkpoint_sha256,plan_sha256=plan_sha,totals=totals,examples=examples,
                   differences_file=difference_path.name,differences_sha256=sha(difference_path),
                   triple_order_sha256=hashlib.sha256(json.dumps([list(t) for t in triples],separators=(',',':')).encode()).hexdigest(),
                   chunk_size=expert.chunk_size,query_batch_size=expert.query_batch_size,
                   runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()),
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
            write(path,c);cells.append(c)
            print(f'[AUDITED] {run.dataset} {run.model} s{run.seed} {split}/{direction}: normalized differences={totals["normalized_vs_raw_mismatches"]}',flush=True)
        del expert;gc.collect();torch.cuda.empty_cache()
    require(len(cells)==72,'Incomplete audit')
    totals={k:sum(c['totals'][k] for c in cells) for k in cells[0]['totals']}
    require(totals['rows']==expected_rows,'Incomplete row coverage')
    for c in cells:
        path=OUT/'cells'/c['differences_file']
        require(sha(path)==c['differences_sha256'],'Changed difference rows')
    passed=all(totals[k]==0 for k in ('export_vs_shared_mismatches','alpha_one_vs_shared_mismatches','alpha_zero_vs_shared_mismatches',
                                      'normalized_vs_raw_mismatches','raw_contract_exception_rows'))
    write(OUT/'audit.json',dict(status='endpoint_equivalence_checks_passed' if passed else 'endpoint_difference_review_required',
                               plan_sha256=plan_sha,cells=cells,totals=totals,training_runs=0,policy_selection=False,
                               historical_results_replaced=False,note='Exact deployed endpoints use raw standalone ranks. Normalized equivalence is checked empirically, not assumed.'))
    import zipfile
    target=OUT.parent/'endpoint_contract_audit_v1_return.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*.json')):z.write(path,path.relative_to(OUT).as_posix())
    print(f'Return {target}',flush=True)


if __name__=='__main__':main()
