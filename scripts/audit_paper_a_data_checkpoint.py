"""Read-only C01--C03 audit. No training, scorer execution, or TEST selection.

Reconstruct canonical splits from pinned OpenKE files; join every export row;
check all frozen checkpoints, their feature buffers and DEV selection histories.
Only compact evidence is written. Original data, weights and results stay intact.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.training.scripts.preprocess_external_mmkg import (
    construct_canonical_splits, read_openke_mapping, read_openke_triples,
    split_integrity_report,
)
from ml.training.src.data.feature_bundle import _sha256_tensor

OUT = ROOT / "outputs/paper_a_safe_correction/data_checkpoint_review_v1"
DEV_MRR_TOL = 1e-7  # training logs reduce float32 RR; exports reduce float64 RR


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def split_bytes(rows, newline: str) -> bytes:
    return "".join(f"{h}\t{r}\t{t}{newline}" for h, r, t in rows).encode()


def matching_split_encoding(rows, expected_sha):
    """Identify actual saved TSV encoding, without normalizing its provenance hash."""
    matches = [name for name, nl in (("LF", "\n"), ("CRLF", "\r\n"))
               if hashlib.sha256(split_bytes(rows, nl)).hexdigest() == expected_sha]
    assert len(matches) == 1, "Reconstructed canonical split differs from training manifest"
    return matches[0]


def check_export(frame: pd.DataFrame, triples, seeds=(1, 2, 3)):
    keys = ["seed", "direction", "query_id"]
    assert len(frame) == len(triples) * len(seeds) * 2, "Export row count mismatch"
    assert not frame.duplicated(keys).any(), "Duplicate query/seed/direction in export"
    assert set(frame.seed) == set(seeds)
    assert set(frame.direction) == {"head", "tail"}
    for (_, direction), group in frame.groupby(["seed", "direction"]):
        triple_columns = ["head_id", "relation_id", "tail_id"]
        assert not group.duplicated(triple_columns).any(), "Repeated triple in export cell"
        expected = np.asarray(sorted(tuple(t) for t in triples))
        np.testing.assert_array_equal(group.sort_values(triple_columns)[triple_columns], expected)
        gold_column = "head_id" if direction == "head" else "tail_id"
        np.testing.assert_array_equal(group.target_entity_id, group[gold_column])
    for role in "ab":
        rank = frame[f"rank_{role}"].to_numpy()
        assert np.isfinite(rank).all() and (rank >= 1).all()
        np.testing.assert_allclose(frame[f"rr_{role}"], 1.0 / rank, atol=1e-15, rtol=0)


def check_history(frame, config, exported_mrr):
    assert frame.epoch.is_unique and frame.epoch.is_monotonic_increasing
    assert np.isfinite(frame[["mrr", "avg_loss"]].to_numpy()).all()
    best = frame.loc[frame.mrr.idxmax()]  # trainer saves on strict improvement
    last = frame.iloc[-1]
    training = config["training"]
    assert training["eval_every"] == 5
    np.testing.assert_array_equal(frame.epoch, np.arange(5, int(last.epoch) + 1, 5))
    error = abs(float(best.mrr) - exported_mrr)
    assert error <= DEV_MRR_TOL, f"Export does not match best DEV checkpoint: {error}"
    assert config["evaluation"]["run_test"] is False
    if training["termination_policy"] == "fixed_budget":
        assert last.epoch == training["epochs"]
    else:
        assert (last.epoch == training["epochs"] or
                last.epoch - best.epoch == 5 * training["early_stop_patience"])
    return {"best_epoch": int(best.epoch), "best_dev_mrr": float(best.mrr),
            "last_epoch": int(last.epoch), "last_dev_mrr": float(last.mrr),
            "evaluations_after_best": int((frame.epoch > best.epoch).sum()),
            "first_loss": float(frame.iloc[0].avg_loss), "last_loss": float(last.avg_loss),
            "export_dev_mrr": exported_mrr, "dev_mrr_abs_error": error}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def bind(path):
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        if key not in sources:
            sources[key] = sha(path)
        return sources[key]

    def read(path):
        bind(path)
        return json.loads(path.read_text(encoding="utf-8"))

    lock = read(ROOT / "docs/EXTERNAL_SOURCES_LOCK.json")
    design = read(ROOT / "configs/paper_a_dynasemble_controls.json")
    datasets, canonical, runs = {}, {}, {}
    for pair in design["pairs"]:
        for item in pair["runs"]:
            for role in "ab":
                rel = item[f"expert_{role}_run"]
                if rel in runs:
                    continue
                cfg = read(ROOT / rel / "config_merged.json")
                assert cfg["system"]["seed"] == item["seed"]
                manifest = cfg["_dataset_manifest"]
                ds = pair["dataset"]
                if ds in datasets:
                    assert datasets[ds] == manifest, "Experts do not share dataset manifest"
                datasets[ds] = manifest
                runs[rel] = {"dataset": ds, "model": pair[f"expert_{role}_name"],
                             "seed": item["seed"], "config": cfg}

    for repo in ("native", "mhyper"):
        folder = ROOT / "external" / ("NATIVE" if repo == "native" else "M-Hyper")
        commit = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        assert commit == lock["repositories"][repo]["commit"]

    ledger, stats = [], []
    for ds, manifest in datasets.items():
        folder = ROOT / "external/NATIVE/benchmarks" / ("MKG-W" if ds == "mkg_w" else "DB15K")
        raw = {}
        for key, filename in {"entity2id": "entity2id.txt", "relation2id": "relation2id.txt",
                              "train": "train2id.txt", "valid": "valid2id.txt", "test": "test2id.txt"}.items():
            path = folder / filename
            assert bind(path) == lock["datasets"][ds]["files"][key]["sha256"]
            assert path.stat().st_size == lock["datasets"][ds]["files"][key]["bytes"]
        entities, ne = read_openke_mapping(folder / "entity2id.txt", "entity")
        relations, nr = read_openke_mapping(folder / "relation2id.txt", "relation")
        assert canonical_json_sha(entities) == manifest["hashes"]["entity_mapping"]
        assert canonical_json_sha(relations) == manifest["hashes"]["relation_mapping"]
        for split in ("train", "valid", "test"):
            raw[split] = read_openke_triples(folder / f"{split}2id.txt", ne, nr)
        assert {k: len(v) for k, v in raw.items()} == manifest["source_counts"]
        integrity = split_integrity_report(raw)
        assert integrity == manifest["source_split_integrity"]
        rebuilt, construction = construct_canonical_splits(
            ds, raw, split_policy=manifest["split"], db15k_valid_fraction=0.1, split_seed=2025)
        assert construction == manifest["split_construction"]
        assert ne == manifest["counts"]["entities"] and nr == manifest["counts"]["relations"]
        encodings = {}
        for split, rows in rebuilt.items():
            assert len(rows) == manifest["counts"][split]
            encodings[split] = matching_split_encoding(rows, manifest["hashes"]["splits"][split])
            canonical[ds, split] = rows
            ledger.append({"dataset": ds, "stage": "canonical", "split": split,
                           "triples": len(rows), "observations_per_pair": len(rows) * 6 if split != "train" else None,
                           "sha256": manifest["hashes"]["splits"][split]})
        for split, rows in raw.items():
            ledger.append({"dataset": ds, "stage": "pinned_openke", "split": split, "triples": len(rows),
                           "observations_per_pair": None, "sha256": lock["datasets"][ds]["files"][split]["sha256"]})
        # An explicit original-file-row -> canonical-row ledger, including excluded valid rows.
        lookup = {t: (s, i) for s, rows in rebuilt.items() for i, t in enumerate(rows)}
        buf = io.StringIO(newline="")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["source_split", "source_row_index", "head", "relation", "tail",
                         "canonical_split", "canonical_row_index", "disposition"])
        for split, rows in raw.items():
            for i, triple in enumerate(rows):
                excluded = ds == "db15k" and split == "valid"
                dest, index = ("", "") if excluded else lookup[triple]
                writer.writerow([split, i, *triple, dest, index,
                                 "exclude_entire_source_valid" if excluded else "preserved" if split == dest else "train_to_dev_holdout"])
        (out / f"{ds}_split_rows.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
        write_json(out / f"{ds}_split_manifest.json", {
            "training_manifest": manifest, "reconstructed_tsv_line_endings": encodings,
            "source_integrity_recomputed": integrity,
            "row_index_convention": "zero-based data row (OpenKE header excluded); exports join by head/relation/tail within seed and direction, not textual query_id order",
            "deduplication_policy": "reject canonical duplicates/overlaps; exclude entire DB15K source valid; no silent deduplication",
            "raw_hdf5_available_locally": {k: (ROOT / v["path"].replace("\\", "/")).is_file()
                                          for k, v in manifest["feature_sources"].items()}})
        for modality, spec in manifest["features"].items():
            stats.append({"dataset": ds, "modality": modality, "entities": ne, "relations": nr,
                          **{k: spec[k] for k in ("aligned_entities", "missing_entities", "coverage", "dimension")},
                          "encoder_provenance": manifest["feature_sources"][modality]["encoder"]})
        print(f"{ds}: pinned raw hashes and reconstructed canonical split hashes PASS", flush=True)

    exports, dev_mrr, test_mrr = [], {}, {}
    base = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2"
    # Include both additional NativE--AdaMF pairs, not only the four main pairs.
    pair_datasets = {p["pair"]: p["dataset"] for p in design["pairs"]}
    pair_datasets.update(mkgw_native_adamf="mkg_w", db15k_native_adamf="db15k")
    columns = ["seed", "direction", "query_id", "head_id", "relation_id", "tail_id",
               "target_entity_id", "rank_a", "rr_a", "rank_b", "rr_b", "expert_a_name", "expert_b_name"]
    for pair, ds in pair_datasets.items():
        for split, suffix in (("valid", "full_ranking/dev_query_rows.csv"),
                              ("test", "test_anchored/test_locked_query_rows.csv")):
            path = base / pair / suffix
            bind(path)
            df = pd.read_csv(path, usecols=columns)
            check_export(df, canonical[ds, split])
            exports.append({"pair": pair, "split": split, "triples": len(canonical[ds, split]),
                            "rows": len(df), "missing": 0, "extra": 0, "duplicate_keys": 0})
            dest = dev_mrr if split == "valid" else test_mrr
            for role in "ab":
                names = df[f"expert_{role}_name"].unique()
                assert len(names) == 1
                for seed, score in df.groupby("seed")[f"rr_{role}"].mean().items():
                    key = ds, str(names[0]), int(seed)
                    if key in dest:
                        assert abs(dest[key] - score) < 1e-15, "Standalone endpoints differ by pair"
                    dest[key] = float(score)

    checkpoints, training_settings = [], []
    torch.set_num_threads(2)
    for rel, record in runs.items():
        folder, cfg = ROOT / rel, record["config"]
        histories = list(folder.glob("metrics*.csv"))
        assert len(histories) == 1
        bind(histories[0])
        key = record["dataset"], record["model"], record["seed"]
        hist = check_history(pd.read_csv(histories[0]), cfg, dev_mrr[key])
        ckpt_sha = bind(folder / "best.ckpt")
        weights = torch.load(folder / "best.ckpt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(t).all().item() for t in weights.values() if t.is_floating_point())
        manifest = datasets[record["dataset"]]
        actual_features = {k: _sha256_tensor(weights[k]) for k in ("text_feat", "img_feat", "has_text", "has_img")}
        assert actual_features == manifest["hashes"]["canonical_features"]
        for modality, feature, mask in (("image", "img_feat", "has_img"), ("text", "text_feat", "has_text")):
            spec = manifest["features"][modality]
            assert list(weights[feature].shape) == [manifest["counts"]["entities"], spec["dimension"]]
            assert int(weights[mask].sum()) == spec["aligned_entities"]
            assert torch.count_nonzero(weights[feature][~weights[mask]]).item() == 0
        checkpoints.append({k: record[k] for k in ("dataset", "model", "seed")} | hist | {
            "run": rel, "checkpoint_sha256": ckpt_sha, "metrics_file": histories[0].name,
            "test_mrr": test_mrr[key], "feature_buffers_verified": True,
            "all_parameters_finite": True})
        training_settings.append({"run": rel, "model": record["model"], "seed": record["seed"],
                                  "training": cfg["training"], "model_settings": cfg["model"],
                                  "embedding": cfg["embedding"], "evaluation": cfg["evaluation"]})
        del weights
        print(f"{key}: checkpoint features and best-DEV export PASS", flush=True)

    cache_checks = 0
    cache_root = ROOT / "outputs/paper_a_safe_correction/dynasemble_controls_v1_cache_evidence"
    for pair in design["pairs"]:
        for item in pair["runs"]:
            for split in ("dev", "test"):
                cache = read(cache_root / pair["pair"] / "cache" / f"{split}_seed{item['seed']}" / "manifest.json")
                assert cache["run"] == item
                fingerprints = cache["inputs"]["experts"][0]
                for role in "ab":
                    run = ROOT / item[f"expert_{role}_run"]
                    assert fingerprints[f"{role}_checkpoint"] == bind(run / "best.ckpt")
                    assert fingerprints[f"{role}_config"] == bind(run / "config_merged.json")
                manifest = datasets[pair["dataset"]]
                for name in ("train", "valid") + (("test",) if split == "test" else ()):
                    assert cache["inputs"]["dataset"][name + ".tsv"] == manifest["hashes"]["splits"][name]
                cache_checks += 1

    code_files = [Path(__file__), ROOT / "docs/EXTERNAL_DATASETS.md",
                  ROOT / "ml/training/scripts/preprocess_external_mmkg.py",
                  ROOT / "ml/training/src/data/feature_bundle.py", ROOT / "ml/training/src/data/readers/openke_mmkg.py",
                  ROOT / "ml/training/src/train/trainer_yaml.py", ROOT / "ml/training/src/train/trainer_recent.py",
                  ROOT / "ml/training/src/eval/filtered_ranking.py"]
    code_files += list((ROOT / "ml/training/src/models/recent_baselines").glob("*.py"))
    code_files += [ROOT / "external/M-Hyper" / name for name in
                   ("README.md", "src/datasets.py", "src/models.py", "src/process_datasets.py", "src/run.py", "src/optimizers.py")]
    for path in code_files:
        bind(path)
    for name, rows in (("split_counts", ledger), ("modality_statistics", stats),
                       ("export_coverage", exports), ("checkpoint_diagnostics", checkpoints)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    write_json(out / "training_settings.json", training_settings)
    write_json(out / "audit.json", {
        "version": "data_checkpoint_review_v1", "status": "data_checkpoint_checks_passed",
        "test_used_for_selection": False, "scorers_executed": False,
        "datasets": list(datasets), "runs_checked": len(runs), "cache_manifests_checked": cache_checks,
        "exports_checked": len(exports), "dev_mrr_tolerance": DEV_MRR_TOL,
        "max_dev_mrr_abs_error": max(r["dev_mrr_abs_error"] for r in checkpoints),
        "checkpoint_epoch_evidence": "log argmax plus export DEV metric agreement and cache checkpoint SHA256; state_dict has no epoch metadata",
        "limitations": ["Original feature HDF5 and crosswalk files are not re-encoded by this audit; canonical tensors are independently verified from every checkpoint.",
                        "Text encoder/version/download provenance is absent from the supplied feature manifests; image encoder name is filename-derived.",
                        "Current checkpoints are reasonable within their logged budgets; no claim of global hyperparameter optimality or causal attribution of the published MRR gap.",
                        "Pinned upstream code behavior is not proof of the execution path used for the published paper."],
        "sources": sources, "failures": []})
    print(json.dumps({"status": "data_checkpoint_checks_passed", "runs": len(runs),
                      "exports": len(exports), "max_dev_error": max(r["dev_mrr_abs_error"] for r in checkpoints)}))


if __name__ == "__main__":
    main()
