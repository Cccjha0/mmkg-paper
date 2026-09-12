"""Exercise the server audit's receipts/resume with tiny CPU-only fake scorers."""
import json
import sys
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts import audit_paper_a_endpoint_contract as audit


@contextmanager
def workspace_temp(scratch):
    # Python 3.13's Windows mode-700 temp ACL is incompatible with this sandbox.
    path=scratch/('endpoint-test-'+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().is_relative_to(scratch.resolve()) and path.resolve()!=scratch.resolve()
        shutil.rmtree(path)


def test_runner_reports_differences_and_verifies_resume(monkeypatch):
    scratch=Path(__file__).resolve().parents[1]/'tmp'
    scratch.mkdir(exist_ok=True)
    with workspace_temp(scratch) as temp:
        root=Path(temp); data=root/'outputs/paper_a_safe_correction/data_checkpoint_review_v1'
        data.mkdir(parents=True)
        (root/'paper_a_draft').mkdir()
        (root/'toy.py').write_text('# synthetic source\n')
        rows=[]; sources={}
        for dataset in ('w','d'):
            for model in ('a','b','c'):
                for seed in (1,2,3):
                    run=f'runs/{dataset}_{model}_{seed}'; folder=root/run; folder.mkdir(parents=True)
                    (folder/'best.ckpt').write_bytes(b'synthetic checkpoint')
                    (folder/'config_merged.json').write_text('{}')
                    sources[run+'/config_merged.json']=audit.sha(folder/'config_merged.json')
                    rows.append(dict(run=run,dataset=dataset,model=model,seed=seed,checkpoint_sha256=audit.sha(folder/'best.ckpt')))
        pd.DataFrame(rows).to_csv(data/'checkpoint_diagnostics.csv',index=False)
        pd.DataFrame([dict(dataset=d,stage='canonical',split=s,triples=2) for d in ('w','d') for s in ('valid','test')]).to_csv(data/'split_counts.csv',index=False)
        for path in data.iterdir():sources[path.relative_to(root).as_posix()]=audit.sha(path)
        (root/'paper_a_draft/data_checkpoint_source_manifest.json').write_text(json.dumps(dict(sources=sources)))
        monkeypatch.setattr(audit,'ROOT',root)
        monkeypatch.setattr(audit,'OUT',root/'outputs/endpoint')
        monkeypatch.setattr(audit,'CODE',('toy.py',))
        monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**k:b'synthetic-test-commit')
        monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
        monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'synthetic CPU-only test')
        monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
        bundle=SimpleNamespace(valid_triples=[(0,0,0),(0,0,0)],test_triples=[(0,0,0),(0,0,0)],manifest={'hashes':{}})
        expert=SimpleNamespace(bundle=bundle,cfg={'_dataset_manifest':{'hashes':{}}},query_batch_size=1,chunk_size=3)
        calls=[]
        monkeypatch.setattr(audit,'load_expert',lambda *a:expert)
        monkeypatch.setattr(audit,'evaluation_fact_indexes',lambda *a,**k:{'head':{},'tail':{}})
        def score(*args,**kwargs):
            calls.append(1)
            raw=torch.tensor([[0.,1.,1e8]])
            return raw,raw[:,0],raw
        monkeypatch.setattr(audit,'score_expert_block',score)
        monkeypatch.setattr(sys,'argv',['audit','--device','cuda'])
        audit.main()
        result=json.loads((audit.OUT/'audit.json').read_text())
        assert result['status']=='endpoint_difference_review_required'
        assert result['totals']['rows']==result['totals']['normalized_vs_raw_mismatches']==144
        assert result['totals']['export_vs_shared_mismatches']==0
        assert len(calls)==144 and len(result['cells'])==72
        audit.main()
        assert len(calls)==144  # Every completed cell resumes without scoring.
        first=next((audit.OUT/'cells').glob('*.differences.json'))
        first.write_text('[]')
        with pytest.raises(ValueError,match='Changed difference rows'):
            audit.main()
