"""C05/C06: mapping, checkpoint-shape and source/config inventory; no scoring/training."""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_paper_a_data_checkpoint import sha, canonical_json_sha, read_openke_mapping

OUT = ROOT / 'outputs/paper_a_safe_correction/alignment_training_v1'
PRIOR = ROOT / 'outputs/paper_a_safe_correction/data_checkpoint_review_v1'
# Audited reconstruction source, NOT an assertion about the historical execution.
SOURCE_REV = 'f2e8198d24d5dde56476fc768a56436adf72e2ba'
ARCH = {'M-Hyper': 'mhyper', 'NativE': 'native', 'AdaMF-MAT': 'adamf_mat'}


def ordered_mapping(mapping):
    ids = list(mapping.values())
    if len(set(ids)) != len(ids) or set(ids) != set(range(len(ids))):
        raise ValueError('Mapping must be bijective onto contiguous candidate IDs')
    return [name for name, _ in sorted(mapping.items(), key=lambda row: row[1])]


def check_state_support(state, n, r, model):
    for key in ('text_feat', 'img_feat', 'has_text', 'has_img'):
        if state[key].shape[0] != n:
            raise ValueError('Partial entity support: ' + key)
    if model == 'M-Hyper':
        for key in ('all.weight', 'structure.weight', 'stru.weight', 'img.weight', 'text.weight'):
            if state[key].shape[0] != n:
                raise ValueError('Partial entity support: ' + key)
        if state['rel_embedding.weight'].shape[0] != 2*r:
            raise ValueError('Reciprocal relation count mismatch')
        expected = torch.cat((torch.arange(r) + r, torch.arange(r)))
        if not torch.equal(state['inverse_relation_ids'], expected):
            raise ValueError('Inverse relation ID mismatch')
    else:
        if state['ent_embeddings.weight'].shape[0] != n:
            raise ValueError('Partial structural entity support')
        if state['rel_embeddings.weight'].shape[0] != r or 'inverse_relation_ids' in state:
            raise ValueError('Unexpected base-relation convention')


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    expected = json.loads((ROOT/'paper_a_draft/data_checkpoint_source_manifest.json').read_text())['sources']
    def bind(path, previous=False):
        rel = path.relative_to(ROOT).as_posix(); digest = sha(path)
        if previous and expected.get(rel) != digest:
            raise ValueError('Changed prior evidence: ' + rel)
        sources[rel] = digest
        return path
    def save(name, value):
        p = OUT/name
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        bind(p)
    rows = list(csv.DictReader(bind(PRIOR/'checkpoint_diagnostics.csv', True).open()))
    assert len(rows) == 18
    lock = json.loads(bind(ROOT/'docs/EXTERNAL_SOURCES_LOCK.json', True).read_text())
    mappings = {}
    for dataset, source_name in [('mkg_w', 'MKG-W'), ('db15k', 'DB15K')]:
        manifest = json.loads(bind(PRIOR/(dataset+'_split_manifest.json'), True).read_text())['training_manifest']
        record = dict(dataset=dataset, counts=manifest['counts'], mappings={})
        for kind in ('entity', 'relation'):
            p = ROOT/f'external/NATIVE/benchmarks/{source_name}/{kind}2id.txt'
            mapping, count = read_openke_mapping(p, kind)
            assert count == len(mapping)
            digest = canonical_json_sha(mapping)
            assert digest == manifest['hashes'][kind+'_mapping']
            assert sha(p) == lock['datasets'][dataset]['files'][kind+'2id']['sha256']
            ordered = ordered_mapping(mapping)
            name = f'{dataset}_{kind}_by_id.json'
            save(name, ordered)
            record['mappings'][kind] = dict(canonical_mapping_sha256=digest,
                source_file_sha256=sha(p), ordered_names_sha256=canonical_json_sha(ordered),
                ordered_names_file=name, serialization='UTF-8 JSON; ensure_ascii=False; compact separators; no trailing newline',
                source_path=p.relative_to(ROOT).as_posix())
        mappings[dataset] = record
    save('mapping_inventory.json', mappings)
    checkpoints = []
    for row in rows:
        folder = ROOT/row['run']; cfg_path = bind(folder/'config_merged.json', True)
        cfg = json.loads(cfg_path.read_text()); dataset = row['dataset']; model = row['model']
        assert cfg['system']['seed'] == int(row['seed'])
        assert cfg['evaluation']['run_test'] is False
        for kind in ('entity', 'relation'):
            assert cfg['_dataset_manifest']['hashes'][kind+'_mapping'] == mappings[dataset]['mappings'][kind]['canonical_mapping_sha256']
        ckpt = folder/'best.ckpt'
        assert sha(ckpt) == row['checkpoint_sha256']
        sources[ckpt.relative_to(ROOT).as_posix()] = row['checkpoint_sha256']
        state = torch.load(ckpt, map_location='cpu', weights_only=True, mmap=True)
        n, r = [mappings[dataset]['counts'][k] for k in ('entities', 'relations')]
        check_state_support(state, n, r, model)
        del state
        bind(folder/row['metrics_file'], True)
        checkpoints.append(dict(dataset=dataset, architecture=ARCH[model], seed=int(row['seed']),
            run=row['run'], checkpoint_sha256=row['checkpoint_sha256'], config_sha256=sha(cfg_path),
            best_epoch=int(row['best_epoch']), last_epoch=int(row['last_epoch']),
            best_dev_mrr=float(row['best_dev_mrr']), exact_training_execution_commit=None,
            checkpoint_has_epoch_or_optimizer_state=False, support_checked=True,
            full_config=cfg))
        print(f"[CHECKED] {dataset}/{model}/seed{row['seed']}", flush=True)
    save('checkpoint_inventory.json', checkpoints)
    # Check every tracked Python source used by the training package against a retrievable Git tree.
    # Store both byte and LF-normalized identities; don't rewrite historical bound files.
    paths = git('ls-tree', '-r', '--name-only', SOURCE_REV, 'ml/training').decode().splitlines()
    versions = []
    for rel in paths:
        if not rel.endswith('.py') or '/tests/' in rel:
            continue
        blob = git('show', SOURCE_REV+':'+rel)
        current = (ROOT/rel).read_bytes()
        assert blob.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n'), rel
        versions.append(dict(path=rel, git_blob_sha256=hashlib.sha256(blob).hexdigest(),
            working_bytes_sha256=hashlib.sha256(current).hexdigest(),
            lf_normalized_sha256=hashlib.sha256(current.replace(b'\r\n', b'\n')).hexdigest()))
    upstream = {}
    for key, folder in [('native','NATIVE'), ('adamf_mat','AdaMF-MAT'), ('mhyper','M-Hyper')]:
        repo = ROOT/'external'/folder
        assert git('rev-parse', 'HEAD', cwd=repo).decode().strip() == lock['repositories'][key]['commit']
        upstream[key] = lock['repositories'][key]
        names = ['README.md'] if key == 'mhyper' else ['scripts/run_mkgw.sh','scripts/run_db15k.sh','args.py']
        for name in names:
            data = (repo/name).read_bytes()
            assert data.replace(b'\r\n',b'\n') == git('show','HEAD:'+name,cwd=repo).replace(b'\r\n',b'\n')
            target = OUT/'upstream'/key/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data); bind(target)
    save('source_versions.json', dict(reconstruction_commit=SOURCE_REV,
        historical_execution_commit_established=False, upstream=upstream, local_sources=versions))
    raw = ROOT/'outputs/paper_a_safe_correction/raw_score_contract_audit.json'
    raw_manifest = json.loads((ROOT/'paper_a_draft/raw_score_source_manifest.json').read_text())
    assert raw_manifest['sources'][raw.relative_to(ROOT).as_posix()] == sha(raw)
    raw_report = json.loads(bind(raw).read_text())
    assert raw_report['exceptional_rows'] == 0 and len(raw_report['cells']) == 72
    for rel in ('scripts/audit_paper_a_alignment_training.py','scripts/rebuild_paper_a_base_model.py',
                'tests/test_alignment_training_contract.py','docs/protocols/paper_a_alignment_training.md'):
        bind(ROOT/rel)
    save('audit.json', dict(status='alignment_training_checks_passed', checkpoints=18,
        datasets=2, mapping_checks=36, support_checks=18, reciprocal_checkpoint_checks=6,
        raw_score_cells=72, raw_score_exceptions=0, historical_execution_commit_established=False,
        training_runs=0, full_candidate_scorer_runs=0, historical_results_replaced=False,
        reconstruction_commit=SOURCE_REV, local_source_files=len(versions)))
    (ROOT/'paper_a_draft/alignment_training_source_manifest.json').write_text(
        json.dumps(dict(version='alignment_training_v1', sources=sources, tables={}),indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
