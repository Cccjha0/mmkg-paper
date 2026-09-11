"""Build C01--C03 tables only from a passing audit and labeled public references."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_a_safe_correction/data_checkpoint_review_v1"
PAPER = ROOT / "paper_a_draft"
TABLES = PAPER / "tables/data_checkpoint"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "data_checkpoint_checks_passed" and not audit["failures"]
    assert audit["runs_checked"] == 18 and audit["exports_checked"] == 12
    reference_path = ROOT / "docs/protocols/paper_a_data_checkpoint_reference.json"
    public = json.loads(reference_path.read_text())["mhyper_paper"]["statistics"]
    manifests = {ds: json.loads((OUT / f"{ds}_split_manifest.json").read_text(encoding="utf-8"))["training_manifest"]
                 for ds in public}
    tables = {}

    def table(name, caption, label, columns, header, rows, size=r"\small"):
        text = (r"\begin{table}[tbp]" + "\n" + r"\centering" + size + "\n" +
                r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n" +
                r"\begin{tabular}{" + columns + "}\n" + r"\toprule" + "\n" +
                header + r"\\" + "\n" + r"\midrule" + "\n" +
                "\n".join(" & ".join(str(c) for c in row) + r"\\" for row in rows) + "\n" +
                r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
        path = TABLES / name
        path.write_text(text, encoding="utf-8", newline="\n")
        tables[path.relative_to(PAPER).as_posix()] = sha(path)

    rows, modal = [], []
    for ds, title in (("mkg_w", "MKG-W"), ("db15k", "DB15K")):
        ref, own = public[ds], manifests[ds]
        for stage, stats in (("Published", ref), ("Pinned files", own["counts"] | own["source_counts"]),
                             ("Effective", own["counts"])):
            rows.append([title, stage] + [f"{stats[k]:,}" for k in ("entities", "relations", "train", "valid", "test")])
        modal.append([title, "Published", f"{ref['image_count']:,}", ref["image_dim"],
                      f"{ref['text_count']:,}", ref["text_dim"]])
        modal.append([title, "Effective", f"{own['features']['image']['aligned_entities']:,}", own['features']['image']['dimension'],
                      f"{own['features']['text']['aligned_entities']:,}", own['features']['text']['dimension']])
    table("datasets.tex", r"Dataset ledger. Published denotes M-Hyper Table 7~\citep{mhyper}; pinned files are the exact NativE mirror used here. Effective counts are reconstructed and hash-verified against all training manifests. DEV is the effective valid split. Each evaluated triple yields six rows per pair (three seeds, two directions).",
          "tab:data", "llrrrrr", "Dataset & Stage & Entities & Relations & TRAIN & valid/DEV & TEST", rows)
    table("modalities.tex", r"Published and effective modality inputs. Counts are entities with aligned features; dimensions are before model projection. Published rows come from M-Hyper Table 7~\citep{mhyper}. Effective rows are checked against the feature buffers in every frozen checkpoint; missing modalities do not remove entities or triples.",
          "tab:data-modalities", "llrrrr", "Dataset & Source & Image count & Image dim. & Text count & Text dim.", modal)
    diag = pd.read_csv(OUT / "checkpoint_diagnostics.csv")
    models = {"M-Hyper": 0, "NativE": 1, "AdaMF-MAT": 2}
    diag["model_order"] = diag.model.map(models)
    diag = diag.sort_values(["dataset", "model_order", "seed"], ascending=[False, True, True])
    rows = []
    for r in diag.itertuples():
        rows.append(["MKG-W" if r.dataset == "mkg_w" else "DB15K", r.model, r.seed, r.best_epoch,
                     f"{r.best_dev_mrr:.6f}", r.last_epoch, f"{r.last_dev_mrr:.6f}",
                     f"{r.first_loss:.3f}/{r.last_loss:.3f}"])
    table("checkpoints.tex", r"All base-model training histories. Best is the first maximum of full DEV MRR; last is the actual final evaluated epoch. Loss shows the first and last logged averages and is not comparable across engines. Exported DEV MRR agrees with each log maximum within $4.20\times10^{-8}$. Checkpoint hashes also match the later DynaSemble cache records. No TEST value selects an epoch.",
          "tab:data-checkpoints", "llrrrrrl", "Dataset & Model & Seed & Best ep. & Best DEV & Last ep. & Last DEV & Loss", rows, r"\footnotesize")
    settings = json.loads((OUT / "training_settings.json").read_text())
    rows, seen = [], set()
    for r in settings:
        group = r["run"].split("/")[3]
        if group in seen:
            continue
        seen.add(group)
        t, m = r["training"], r["model_settings"]
        rows.append(["MKG-W" if group.startswith("mkg_w") else "DB15K", r["model"],
                     m.get("rank", m.get("dim")), f"{t['lr']:g}", t["batch_size"],
                     "all entities" if t["sampler"] == "none" else t["neg_ratio"],
                     "200 fixed" if t["epochs"] == 200 else "1,000 / 10 checks"])
    table("training.tex", r"Settings actually used by the frozen base models, shared across three seeds. Size is the model-specific rank (M-Hyper) or dimension (others), not raw feature dimension. All evaluate full DEV every five epochs. The last column is the fixed budget, or epoch cap / patience in DEV evaluations.",
          "tab:data-training", "llrrrrl", "Dataset & Model & Size & LR & Batch & Negatives & Epoch / stopping", rows, r"\footnotesize")
    rows = []
    for ds in ("mkg_w", "db15k"):
        own_mrr = float(diag[(diag.dataset == ds) & (diag.model == "M-Hyper")].test_mrr.mean())
        rows.append(["MKG-W" if ds == "mkg_w" else "DB15K", f"{public[ds]['mrr']:.4f}",
                     f"{own_mrr:.6f}", f"{own_mrr-public[ds]['mrr']:+.6f}", "Not protocol-equivalent"])
    table("public_comparison.tex", r"M-Hyper published MRR (Table 1~\citep{mhyper}) and this study's three-seed mean. Differences are descriptive: neither row is an identical-protocol reproduction, and no causal share is assigned to any implementation difference.",
          "tab:data-public", "lrrrl", "Dataset & Published & This study & Difference & Interpretation", rows)
    sources = dict(audit["sources"])
    feature_audit_path = OUT / "feature_provenance_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))
    assert feature_audit["status"] == "source_identity_evidence_checks_passed"
    assert not feature_audit["failures"] and feature_audit["source_files_checked"] == 6
    for rel, digest in feature_audit["sources"].items():
        assert sha(ROOT / rel) == digest, rel
    sources.update(feature_audit["sources"])
    for path in list(OUT.iterdir()) + [reference_path, Path(__file__)]:
        if path.is_file():
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    (PAPER / "data_checkpoint_source_manifest.json").write_text(json.dumps({
        "version": "data_checkpoint_review_v1", "sources": sources, "tables": tables}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"Built {len(tables)} data/checkpoint tables; bound {len(sources)} sources")


if __name__ == "__main__":
    main()
