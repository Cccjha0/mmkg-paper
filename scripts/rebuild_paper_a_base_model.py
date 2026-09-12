"""Print a frozen base-model rebuild plan; --train explicitly runs on a CUDA server."""
import argparse
import copy
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_paper_a_alignment_training import OUT, sha


def rebuild_config(record, output_root):
    cfg = copy.deepcopy(record['full_config'])
    cfg.pop('_config_paths', None)
    cfg['output']['root_dir'] = str(output_root)
    cfg['output']['exp_name'] = 'rebuild_' + cfg['output']['exp_name']
    cfg['evaluation']['run_test'] = False
    cfg['system']['device'] = 'cuda'
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('mkg_w', 'db15k'), required=True)
    parser.add_argument('--architecture', choices=('mhyper', 'native', 'adamf_mat'), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'paper_a_draft/alignment_training_source_manifest.json').read_text())
    def bound(name):
        path = OUT/name
        assert sha(path) == manifest['sources'][path.relative_to(ROOT).as_posix()], name
        return json.loads(path.read_text(encoding='utf-8'))
    records = bound('checkpoint_inventory.json')
    record = next(r for r in records if (r['dataset'], r['architecture'], r['seed']) ==
                  (args.dataset, args.architecture, args.seed))
    version = bound('source_versions.json')
    for source in version['local_sources']:
        data = (ROOT/source['path']).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == source['lf_normalized_sha256'], source['path']
    folder = ROOT/'outputs/base_model_rebuilds'
    cfg = rebuild_config(record, folder)
    print(json.dumps(dict(dataset=args.dataset, architecture=args.architecture, seed=args.seed,
        config=cfg, original_checkpoint_sha256=record['checkpoint_sha256'],
        source_commit=version['reconstruction_commit'], train_requested=args.train,
        note='Rebuild under audited source; historical execution revision/environment and bitwise identity are not established.'), indent=2))
    if not args.train:
        return
    import torch
    from ml.training.src.data.dataset_loader import load_dataset_bundle
    assert torch.cuda.is_available(), 'Server CUDA is required; no automatic CPU fallback'
    bundle = load_dataset_bundle(cfg)
    assert bundle.manifest == record['full_config']['_dataset_manifest'], 'Processed input differs from frozen training manifest'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_dir = folder/('receipt_'+stamp)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    config_path = receipt_dir/'config.json'
    config_path.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    common = receipt_dir/'empty_common.json'; common.write_text('{}\n', encoding='utf-8')
    receipt = dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        reconstruction_source_commit=version['reconstruction_commit'], python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        original_checkpoint_sha256=record['checkpoint_sha256'], launch_config_sha256=sha(config_path),
        source_inventory_sha256=sha(OUT/'source_versions.json'), no_test_selection=True)
    (receipt_dir/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-m','ml.training.scripts.run_train','--config',str(config_path),
                    '--common',str(common)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
