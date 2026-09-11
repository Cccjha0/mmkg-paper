"""Export small existing cache evidence; no scoring, training, CUDA or NumPy needed."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def export(root):
    pairs = ('mkgw_mhyper_native', 'mkgw_mhyper_adamf', 'db15k_mhyper_native', 'db15k_mhyper_adamf')
    files = []
    for pair in pairs:
        lock = json.loads((root/pair/'dev_lock.json').read_text(encoding='utf-8'))
        for split in ('dev', 'test'):
            for seed in (1, 2, 3):
                directory = root/pair/'cache'/f'{split}_seed{seed}'
                manifest_path = directory/'manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                expected = lock['cache_manifests'][str(seed)] if split == 'dev' else json.loads(
                    (root/pair/'test'/f'R3_softplus_b{seed}_s11.json').read_text(encoding='utf-8'))['identity']['cache_manifest_sha256']
                if sha(manifest_path) != expected:
                    raise ValueError('Cache manifest changed: '+str(directory))
                files.append(manifest_path)
                for name in ('queries', 'features'):
                    path = directory/(name+'.npy')
                    if sha(path) != manifest['array_sha256'][name]:
                        raise ValueError('Cache array changed: '+str(path))
                    files.append(path)
    target = root.parent/(root.name+'_cache_evidence.zip')
    with zipfile.ZipFile(target.with_suffix('.zip.partial'), 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(root).as_posix())
    target.with_suffix('.zip.partial').replace(target)
    print(f'{target} ({target.stat().st_size / 1024**2:.2f} MiB; {len(files)} files)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/paper_a_safe_correction/dynasemble_controls_v1'))
    export(parser.parse_args().root.resolve())
