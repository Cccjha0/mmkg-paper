"""Publish C05/C06 mapping identities, setting origins and all checkpoint digests."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_paper_a_alignment_training import OUT, sha


def digest_cell(value):
    return r'\shortstack[l]{\texttt{'+value[:32]+r'}\\\texttt{'+value[32:]+'}}'


def table(rows,caption,label,cols,head):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\footnotesize',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',head+r'\\',r'\midrule',
        *rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    path=ROOT/'paper_a_draft/alignment_training_source_manifest.json'
    manifest=json.loads(path.read_text())
    def read(name):
        p=OUT/name
        assert sha(p)==manifest['sources'][p.relative_to(ROOT).as_posix()],name
        return json.loads(p.read_text(encoding='utf-8'))
    audit=read('audit.json'); mappings=read('mapping_inventory.json'); runs=read('checkpoint_inventory.json')
    assert audit['status']=='alignment_training_checks_passed' and len(runs)==18
    assets={}; rows=[]
    for dataset, display in [('mkg_w','MKG-W'),('db15k','DB15K')]:
        for kind in ('entity','relation'):
            r=mappings[dataset]
            count=r['counts']['entities' if kind=='entity' else 'relations']
            rows.append(f'{display} & {kind} & {count:,} & '+digest_cell(r['mappings'][kind]['canonical_mapping_sha256'])+r'\\[3pt]')
    assets['mapping.tex']=table(rows,
        r'Complete canonical mapping SHA-256 digests, each printed as two consecutive 32-character lines. All nine runs per dataset agree. The supplement also publishes the full entity/relation names in ID order, their ordered-array hashes and the original OpenKE file hashes. Counts do not replace identity checks.',
        'tab:alignment-mapping','llrl',r'Dataset & Map & IDs & Canonical mapping SHA-256')
    rows=[
        r'M-Hyper & README: rank 128, Adagrad 0.1, batch 1,000, N3 0.005, 200 epochs, evaluation every 5 & Shared HDF5/masks; TRAIN-available PCA; canonical DEV and strict-$>$ evaluator\\',
        r'NativE & Dataset scripts: dimension 250, margin 4/12, LR and generator LR $10^{-4}$, 128 negatives, $\mu=10^{-4}$, cap 1,000 & Fixed raw feature buffers; availability masks; shared sampler/evaluator; patience 10 DEV checks\\',
        r'AdaMF & Dataset scripts: dimension 200/250, margin 12, LR $2\!\times\!10^{-5}$, 128 negatives, $\mu=0$, cap 1,000 & Fixed raw feature buffers; availability masks; shared sampler/evaluator; patience 10 DEV checks\\']
    assets['origins.tex']=table(rows,
        r'Setting provenance for MKG-W/DB15K (slash-separated values in that order), complementing Table~\ref{tab:data-training}. Numerical agreement with an upstream command is distinguished from local protocol changes and DEV checkpoint selection. A historical base-hyperparameter search ledger is not established. AdaMF names the retained AdaMF-MAT adapter with $\mu=0$; its generator does not contribute to the base-model gradient. Its dataset scripts specify LR $2\times10^{-5}$, although its README example gives $10^{-4}$.',
        'tab:training-origins',r'p{.13\linewidth}p{.40\linewidth}p{.37\linewidth}',r'Model & Released numerical anchor & Local protocol adaptations')
    for dataset,display in [('mkg_w','MKG-W'),('db15k','DB15K')]:
        rows=[]
        for arch,label in [('mhyper','M-Hyper'),('native','NativE'),('adamf_mat','AdaMF')]:
            for r in sorted([r for r in runs if r['dataset']==dataset and r['architecture']==arch],key=lambda r:r['seed']):
                rows.append(f"{label} & {r['seed']} & {r['best_epoch']}/{r['last_epoch']} & "+digest_cell(r['checkpoint_sha256'])+r'\\[3pt]')
        assets['checkpoints_'+dataset+'.tex']=table(rows,
            display+r' frozen checkpoints. Epochs are first maximum DEV / final logged epoch. Concatenate the two SHA-256 lines in each cell. The machine-readable inventory supplies the exact run path, full merged configuration and its independent digest, feature/split manifest and best DEV MRR. These are identifiers of the used artifacts, not a claim of a public checkpoint download or bitwise reproducible retraining.',
            'tab:training-hashes-'+dataset,'llrl',r'Model & Seed & Epochs & Checkpoint SHA-256')
    directory=ROOT/'paper_a_draft/tables/alignment_training';directory.mkdir(parents=True,exist_ok=True)
    manifest['tables']={}
    for name,value in assets.items():
        p=directory/name;p.write_text(value,encoding='utf-8')
        manifest['tables'][p.relative_to(ROOT/'paper_a_draft').as_posix()]=sha(p)
    for p in (Path(__file__), ROOT/'docs/reports/paper_a_alignment_training_review_2026-09-12.md'):
        if p.exists(): manifest['sources'][p.relative_to(ROOT).as_posix()]=sha(p)
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=len(assets),mapping_rows=4,checkpoint_rows=18),indent=2))


if __name__=='__main__':main()
