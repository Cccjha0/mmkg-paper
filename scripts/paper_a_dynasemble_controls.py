"""Server-only D01-D03 cache, grouped DEV selection, immutable TEST and reporting.

The original eval_openbg_dynasemble.py and its R0 exports are not modified.
Use --stage plan locally. CPU experiments require an explicit --allow-cpu flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.dynasemble_controls import (ControlSelector, VARIANTS, coefficients, fact_sets,
    grouped_folds, training_candidates, complete_order_equal, choose_configuration)
from router.information_boundary import SCORE_INFORMATION_CONTRACT, require_score_information_contract
from scripts.eval_openbg_dynasemble import normalize_and_features
from scripts.eval_heterogeneous_complementarity import load_expert, validate_pair, score_expert_block

SCHEMA = "dynasemble_controls_v1"
SELECTOR_SEEDS = (11, 23, 37)
LEARNING_RATES = (1e-5, 5e-5, 1e-4)
CHECKPOINT_EPOCHS = (1, 3, 5, 10)
NEGATIVES = 9999
BATCH_SIZE = 16
FOLDS = 3
INPUT_ROOT = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2"
DEFAULT_CONFIG = ROOT / "configs/paper_a_dynasemble_controls.json"
SOURCE_FILES = ("router/dynasemble_controls.py", "scripts/paper_a_dynasemble_controls.py",
                "scripts/eval_openbg_dynasemble.py", "scripts/eval_heterogeneous_complementarity.py",
                "router/information_boundary.py", "docs/protocols/paper_a_dynasemble_controls.md")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty export: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def provenance(config):
    return {"schema": SCHEMA, "config_sha256": digest(config),
            "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runtime": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "sources": {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in SOURCE_FILES}}


def check_provenance(saved, current):
    if saved != current:
        raise RuntimeError("Protocol/config/source changed. Use a new experiment directory; do not rebind locks.")


def pair_assets(pair):
    return [{"seed": run["seed"], **{f"{side}_{kind}": digest(ROOT / run[f"expert_{side}_run"] / file)
             for side in ("a", "b") for kind, file in (("config", "config_merged.json"), ("checkpoint", "best.ckpt"))}}
            for run in pair["runs"]]


def cache_assets(pair, run, split):
    dataset = pair["dataset"]
    base = ROOT / "data/datasets" / dataset / "processed"
    names = ["manifest.json", "train.tsv", "valid.tsv", "entity2id.json", "relation2id.json",
             "text_feat.pt", "img_feat.pt", "has_text.pt", "has_img.pt"]
    if split == "test":
        names.append("test.tsv")
    return {"experts": pair_assets({"runs": [run]}), "dataset": {name: digest(base / name) for name in names}}


def load_cache(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("score_information_contract") != SCORE_INFORMATION_CONTRACT:
        raise RuntimeError("Incompatible candidate cache")
    arrays = {}
    for name, sha in manifest["array_sha256"].items():
        path = directory / (name + ".npy")
        if digest(path) != sha:
            raise RuntimeError(f"Cache checksum mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"manifest": manifest, "directory": directory, **arrays}


def build_cache(pair, run, split, directory, device, source):
    inputs = cache_assets(pair, run, split)
    if (directory / "manifest.json").exists():
        data = load_cache(directory)
        check_provenance(data["manifest"]["provenance"], source)
        if data["manifest"]["run"] != run or data["manifest"]["split"] != split:
            raise RuntimeError("Cache identity mismatch")
        if data["manifest"]["inputs"] != inputs:
            raise RuntimeError("Cache inputs changed; use a new experiment directory")
        return data
    directory.mkdir(parents=True, exist_ok=True)
    experts = [load_expert(pair[f"expert_{s}_name"], ROOT / run[f"expert_{s}_run"], device,
                          dev_only_no_test_access=(split == "dev")) for s in ("a", "b")]
    a, b = experts
    validate_pair(a, b)
    if a.seed != run["seed"]:
        raise RuntimeError("Base seed differs from protocol")
    triples = a.bundle.valid_triples if split == "dev" else a.bundle.test_triples
    triples = np.asarray(triples, dtype=np.int64)
    n, entities = len(triples), a.num_entities
    queries = np.concatenate([np.column_stack((triples, np.full(n, d))) for d in (1, 0)])
    facts = list(a.bundle.train_triples) + list(a.bundle.valid_triples)
    if split == "test":
        facts += list(a.bundle.test_triples)
    eval_facts = fact_sets(facts)
    arrays = {}
    for name, shape, dtype in (("a", (2*n, entities), np.float32), ("b", (2*n, entities), np.float32),
                               ("features", (2*n, 4), np.float32), ("references", (2*n, 2), np.float32),
                               ("endpoint_ranks", (2*n, 2), np.int64)):
        arrays[name] = np.lib.format.open_memmap(directory / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
    mask_offsets, mask_ids = [0], []
    block = max(a.query_batch_size, b.query_batch_size)
    for direction_index, d in enumerate((1, 0)):
        direction = "tail" if d else "head"
        for start in range(0, n, block):
            end = min(n, start + block)
            indices = np.arange(direction_index*n+start, direction_index*n+end)
            q = torch.tensor(triples[start:end])
            normalized, refs, features, ranks = [], [], [], []
            for expert in experts:
                # Empty filter: no masked array can reach normalization or the selector.
                _, reference, raw = score_expert_block(expert, q, direction, {}, device, retain_unfiltered=True)
                z, ref, feat = normalize_and_features(raw, reference)
                raw_eval = raw.clone()
                for j, (h, r, t) in enumerate(triples[start:end]):
                    gold = int(t if d else h)
                    key = (int(h), int(r)) if d else (int(r), int(t))
                    hidden = sorted(eval_facts[d].get(key, set()) - {gold})
                    raw_eval[j, hidden] = -torch.inf
                ranks.append(((raw_eval > reference[:, None]).sum(1) + 1).numpy())
                normalized.append(z.numpy())
                refs.append(ref.numpy())
                features.append(feat.numpy())
            for name, value in (("a", normalized[0]), ("b", normalized[1]),
                                ("references", np.column_stack(refs)), ("features", np.concatenate(features, axis=1)),
                                ("endpoint_ranks", np.column_stack(ranks))):
                arrays[name][indices] = value
            for h, r, t in triples[start:end]:
                key = (int(h), int(r)) if d else (int(r), int(t))
                mask_ids.extend(sorted(eval_facts[d].get(key, set()) - {int(t if d else h)}))
                mask_offsets.append(len(mask_ids))
            if start == 0 or (start // block) % 100 == 0:
                print(f"[CACHE] {pair['pair']} seed={a.seed} {split}/{direction} {end}/{n}", flush=True)
    for value in arrays.values():
        value.flush()
    for name, value in (("queries", queries), ("train_facts", np.asarray(a.bundle.train_triples, dtype=np.int64)),
                        ("mask_offsets", np.asarray(mask_offsets, dtype=np.int64)), ("mask_ids", np.asarray(mask_ids, dtype=np.int64))):
        np.save(directory / (name + ".npy"), value, allow_pickle=False)
    manifest = {"schema": SCHEMA, "score_information_contract": SCORE_INFORMATION_CONTRACT,
                "pair": pair["pair"], "run": run, "inputs": inputs, "split": split, "n_triples": n, "entities": entities,
                "filter_scope": "train_dev" if split == "dev" else "train_dev_test", "provenance": source,
                "array_sha256": {p.stem: digest(p) for p in sorted(directory.glob("*.npy"))}}
    write_json(directory / "manifest.json", manifest)
    del arrays, a, b, experts
    if device == "cuda":
        torch.cuda.empty_cache()
    return load_cache(directory)


@torch.no_grad()
def diagnostics(model, features, device):
    x = torch.tensor(np.asarray(features).copy(), device=device)
    z = model.preactivation(x)
    w = model(x)
    alpha = coefficients(w, model.variant)[0] / (1 + w)
    def stats(v):
        v = v.cpu().numpy()
        return {"min": float(v.min()), "p05": float(np.quantile(v, .05)), "median": float(np.median(v)),
                "p95": float(np.quantile(v, .95)), "max": float(v.max()), "std": float(v.std())}
    return {"preactivation": stats(z), "weight": stats(w), "primary_ratio": stats(alpha),
            "zero_weight_fraction": float((w == 0).float().mean()),
            "negative_preactivation_fraction": float((z < 0).float().mean())}


@torch.no_grad()
def evaluate(model, cache, indices, device, *, export=False, probe_indices=()):
    model.eval()
    rr, rows = [], []
    probe_indices = set(probe_indices)
    for start in range(0, len(indices), 64):
        ids = np.asarray(indices[start:start+64])
        a = torch.tensor(np.asarray(cache["a"][ids]), device=device)
        b = torch.tensor(np.asarray(cache["b"][ids]), device=device)
        refs = torch.tensor(np.asarray(cache["references"][ids]), device=device)
        features = torch.tensor(np.asarray(cache["features"][ids]), device=device)
        w = model(features)
        ca, cb = coefficients(w, model.variant)
        mixed = ca[:, None]*a + cb[:, None]*b
        gold = ca*refs[:, 0] + cb*refs[:, 1]
        order_equal = {}
        for j, index in enumerate(ids):
            if int(index) in probe_indices:
                order_equal[int(index)] = complete_order_equal(a[j].cpu().numpy(), mixed[j].cpu().numpy())
            lo, hi = cache["mask_offsets"][index:index+2]
            hidden = np.asarray(cache["mask_ids"][lo:hi])
            mixed[j, torch.tensor(hidden, device=device)] = -torch.inf
        ranks = (mixed > gold[:, None]).sum(1).cpu().numpy() + 1
        ca_np, cb_np = ca.cpu().numpy(), cb.cpu().numpy()
        ranks = np.where(ca_np == 0, cache["endpoint_ranks"][ids, 1], ranks)
        ranks = np.where(cb_np == 0, cache["endpoint_ranks"][ids, 0], ranks)
        rr.extend((1. / ranks).tolist())
        if export:
            z = model.preactivation(features).cpu().numpy()
            for j, index in enumerate(ids):
                h, r, t, d = map(int, cache["queries"][index])
                ra, rb = map(int, cache["endpoint_ranks"][index])
                rows.append({"head_id": h, "relation_id": r, "tail_id": t, "direction": "tail" if d else "head",
                             "target_entity_id": t if d else h, "rank_a": ra, "rank_b": rb,
                             "rr_a": 1/ra, "rr_b": 1/rb, "rank_method": int(ranks[j]), "rr_method": 1/float(ranks[j]),
                             "learned_weight": float(w[j]), "primary_ratio": float(ca_np[j]/(ca_np[j]+cb_np[j])),
                             "preactivation": float(z[j]), "gold_rank_matches_primary": bool(ranks[j] == ra),
                             "complete_unfiltered_order_matches_normalized_primary": order_equal.get(int(index), "")})
    return {"mrr": float(np.mean(rr)), "n": len(rr)}, rows


def fit(cache, variant, selector_seed, lr, epochs, fit_triples, validation, device, checkpoints, output):
    """No access to TEST cache/files; held-out DEV labels excluded from negative filtering."""
    torch.manual_seed(selector_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(selector_seed)
    model = ControlSelector(variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MultiMarginLoss(margin=2.)
    n = cache["manifest"]["n_triples"]
    fit_triples = np.asarray(fit_triples, dtype=np.int64)
    facts = fact_sets(np.concatenate((cache["train_facts"], cache["queries"][fit_triples, :3])))
    generator = np.random.default_rng(selector_seed + 104729)
    history, metrics = [], []
    train_queries = np.concatenate((fit_triples, fit_triples+n))
    initial = diagnostics(model, cache["features"][train_queries], device)
    for epoch in range(1, epochs+1):
        model.train()
        ordered = fit_triples.tolist()
        random.Random(selector_seed + epoch - 1).shuffle(ordered)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        grads = {name: [] for name, p in model.named_parameters()}
        losses = []
        for start in range(0, len(ordered), BATCH_SIZE):
            batch = ordered[start:start+BATCH_SIZE]
            midpoint = len(batch)//2
            ids = np.asarray(batch[:midpoint] + [i+n for i in batch[midpoint:]], dtype=np.int64)
            # Both features AND score normalization were computed on all entities.
            sa, sb = training_candidates(cache["a"][ids], cache["b"][ids], cache["references"][ids, 0],
                                         cache["references"][ids, 1], cache["queries"][ids], facts, NEGATIVES, generator)
            x = torch.tensor(np.asarray(cache["features"][ids]), device=device)
            w = model(x)
            ca, cb = coefficients(w, variant)
            scores = ca[:, None]*torch.tensor(sa, device=device) + cb[:, None]*torch.tensor(sb, device=device)
            loss = loss_fn(scores, torch.zeros(len(ids), dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; failed run retained, never omitted")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError("Missing/non-finite gradient")
                grads[name].append(float(p.grad.norm()))
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss_mean": float(np.mean(losses)), "loss_first": losses[0], "loss_last": losses[-1],
                  "gradients": {name: {"mean_norm": float(np.mean(v)), "max_norm": max(v),
                                      "zero_batch_fraction": float(np.mean(np.asarray(v) == 0))} for name, v in grads.items()},
                  "parameter_update_norms": {name: float((p.detach()-before[name]).norm()) for name, p in model.named_parameters()},
                  "train_diagnostics": diagnostics(model, cache["features"][train_queries], device)}
        if epoch in checkpoints:
            score, _ = evaluate(model, cache, validation, device)
            metrics.append({"epoch": epoch, **score})
            record["heldout"] = score
            record["heldout_diagnostics"] = diagnostics(model, cache["features"][validation], device)
        history.append(record)
        write_json(output / "trace.json", {"variant": variant, "selector_seed": selector_seed, "learning_rate": lr,
                                           "initial": initial, "epochs": history, "status": "running"})
        print(f"[FIT] {output.name} epoch={epoch} loss={record['loss_mean']:.6f}", flush=True)
    trace = {"variant": variant, "selector_seed": selector_seed, "learning_rate": lr, "initial": initial, "epochs": history,
             "status": "complete", "permanent_zero_gradient": all(max(v["max_norm"] for v in e["gradients"].values()) == 0 for e in history),
             "final_all_zero_weight": history[-1]["train_diagnostics"]["zero_weight_fraction"] == 1.}
    write_json(output / "trace.json", trace)
    return model, metrics, trace


def fit_cached(cache, variant, ss, lr, epochs, fit_ids, val_ids, device, checkpoints, path, source):
    signature = {"provenance": source, "cache_manifest_sha256": digest(cache["directory"] / "manifest.json"),
                 "variant": variant, "selector_seed": ss, "learning_rate": lr, "epochs": epochs,
                 "fit_ids": np.asarray(fit_ids).tolist(), "validation_ids": np.asarray(val_ids).tolist(), "checkpoints": list(checkpoints)}
    if (path / "complete.json").exists():
        done = read_json(path / "complete.json")
        if done["signature"] != signature or digest(path / "model.pt") != done["model_sha256"] or digest(path / "trace.json") != done["trace_sha256"]:
            raise RuntimeError("Fit resume identity/hash mismatch")
        return done
    path.mkdir(parents=True, exist_ok=True)
    model, metrics, trace = fit(cache, variant, ss, lr, epochs, fit_ids, val_ids, device, checkpoints, path)
    temp = path / "model.pt.partial"
    torch.save(model.state_dict(), temp)
    temp.replace(path / "model.pt")
    done = {"signature": signature, "metrics": metrics, "model_sha256": digest(path / "model.pt"),
            "trace_sha256": digest(path / "trace.json"), "health": {k: trace[k] for k in ("permanent_zero_gradient", "final_all_zero_weight")}}
    write_json(path / "complete.json", done)
    return done


def run_dev(pair, root, device, source):
    folder = root / pair["pair"]
    if (folder / "test_started.json").exists():
        if not (folder / "dev_lock.json").exists():
            raise RuntimeError("TEST already started without a DEV lock")
    if (folder / "dev_lock.json").exists():
        lock = read_json(folder / "dev_lock.json")
        check_provenance(lock["provenance"], source)
        verify_lock(folder, lock, pair)
        print(f"[LOCK EXISTS] {pair['pair']}", flush=True)
        return
    assets = pair_assets(pair)
    cv_records = []
    caches = {}
    for run in pair["runs"]:
        bs = run["seed"]
        cache = build_cache(pair, run, "dev", folder / "cache" / f"dev_seed{bs}", device, source)
        caches[bs] = cache
        n = cache["manifest"]["n_triples"]
        folds = grouped_folds(cache["queries"][:n, :3])
        for ss in SELECTOR_SEEDS:
            for fold in range(FOLDS):
                fit_ids = np.flatnonzero(folds != fold)
                held = np.flatnonzero(folds == fold)
                val_ids = np.concatenate((held, held+n))
                for lr in LEARNING_RATES:
                    path = folder / "fits" / f"cv_b{bs}_s{ss}_f{fold}_lr{lr:g}"
                    done = fit_cached(cache, "R3_softplus", ss, lr, max(CHECKPOINT_EPOCHS), fit_ids, val_ids,
                                      device, CHECKPOINT_EPOCHS, path, source)
                    for record in done["metrics"]:
                        cv_records.append({"variant": "R3_softplus", "base_seed": bs, "selector_seed": ss,
                                           "fold": fold, "learning_rate": lr, **record})
    selection = choose_configuration(cv_records)
    write_csv(folder / "dev_cv.csv", cv_records)
    selectors = []
    for run in pair["runs"]:
        bs = run["seed"]
        cache = caches[bs]
        n = cache["manifest"]["n_triples"]
        for ss in SELECTOR_SEEDS:
            for variant in VARIANTS:
                lr, epochs = (5e-5, 1) if variant in ("R1_source", "R2_init", "R3_softplus_fixed") else (selection["learning_rate"], selection["epochs"])
                relative = f"fits/final_{variant}_b{bs}_s{ss}"
                done = fit_cached(cache, variant, ss, lr, epochs, np.arange(n), np.arange(2*n), device, (epochs,), folder / relative, source)
                selectors.append({"variant": variant, "base_seed": bs, "selector_seed": ss, "relative_dir": relative,
                                  "model_sha256": done["model_sha256"], "trace_sha256": done["trace_sha256"],
                                  "health": done["health"], "learning_rate": lr, "epochs": epochs,
                                  "full_dev_resubstitution_mrr": done["metrics"][-1]["mrr"]})
    if assets != pair_assets(pair):
        raise RuntimeError("Base assets changed during DEV run")
    write_json(folder / "dev_lock.json", {"schema": SCHEMA, "pair": pair["pair"], "provenance": source,
               "assets": assets, "selection": selection, "selectors": selectors,
               "dev_cv_sha256": digest(folder / "dev_cv.csv"),
               "cache_manifests": {str(k): digest(v["directory"] / "manifest.json") for k, v in caches.items()},
               "policy": "R1/R2 fixed one epoch; R3 DEV-selected; R4 inherits R3 hyperparameters without orientation selection",
               "dev_evaluation_filter": "train_dev", "training_negative_filter": "train_plus_fit_fold; full DEV on final refit"})


def verify_lock(folder, lock, pair):
    if lock["assets"] != pair_assets(pair):
        raise RuntimeError("Base asset hashes differ from DEV lock")
    if digest(folder / "dev_cv.csv") != lock["dev_cv_sha256"]:
        raise RuntimeError("DEV selection evidence changed")
    expected = {(v, bs, ss) for v in VARIANTS for bs in (1, 2, 3) for ss in SELECTOR_SEEDS}
    if len(lock["selectors"]) != len(expected) or {(s["variant"], s["base_seed"], s["selector_seed"]) for s in lock["selectors"]} != expected:
        raise RuntimeError("Incomplete final selector matrix")
    for selector in lock["selectors"]:
        path = folder / selector["relative_dir"]
        for file, key in (("model.pt", "model_sha256"), ("trace.json", "trace_sha256")):
            if digest(path / file) != selector[key]:
                raise RuntimeError(f"Locked file changed: {path / file}")
    for seed, sha in lock["cache_manifests"].items():
        cache_path = folder / "cache" / f"dev_seed{seed}" / "manifest.json"
        if digest(cache_path) != sha:
            raise RuntimeError("DEV cache manifest changed")
        run = next(r for r in pair["runs"] if r["seed"] == int(seed))
        if read_json(cache_path)["inputs"] != cache_assets(pair, run, "dev"):
            raise RuntimeError("DEV input data changed after locking")


def run_test(pair, root, device, source):
    folder = root / pair["pair"]
    lock = read_json(folder / "dev_lock.json")
    check_provenance(lock["provenance"], source)
    verify_lock(folder, lock, pair)
    lock_hash = digest(folder / "dev_lock.json")
    started = folder / "test_started.json"
    if started.exists() and read_json(started)["dev_lock_sha256"] != lock_hash:
        raise RuntimeError("DEV lock changed after TEST started")
    write_json(started, {"dev_lock_sha256": lock_hash})
    for run in pair["runs"]:
        bs = run["seed"]
        cache = build_cache(pair, run, "test", folder / "cache" / f"test_seed{bs}", device, source)
        # Deterministic direction-balanced complete-order probe; selection ignores scores/golds.
        n = cache["manifest"]["n_triples"]
        probes = []
        for d in (0, 1):
            candidates = np.flatnonzero(cache["queries"][:, 3] == d).tolist()
            candidates.sort(key=lambda i: hashlib.sha256(str((int(cache["queries"][i, 1]), int(cache["queries"][i, 0 if d else 2]), d)).encode()).digest())
            probes.extend(candidates[:32])
        for selector in (s for s in lock["selectors"] if s["base_seed"] == bs):
            variant, ss = selector["variant"], selector["selector_seed"]
            output = folder / "test" / f"{variant}_b{bs}_s{ss}"
            identity = {"dev_lock_sha256": lock_hash, "cache_manifest_sha256": digest(cache["directory"] / "manifest.json")}
            if output.with_suffix(".json").exists():
                previous = read_json(output.with_suffix(".json"))
                if previous["identity"] != identity or previous["rows_sha256"] != digest(output.with_suffix(".csv")):
                    raise RuntimeError("TEST resume provenance mismatch")
                continue
            model = ControlSelector(variant).to(device)
            model.load_state_dict(torch.load(folder / selector["relative_dir"] / "model.pt", map_location=device, weights_only=True))
            metrics, rows = evaluate(model, cache, np.arange(2*n), device, export=True, probe_indices=probes)
            for row in rows:
                row.update(pair=pair["pair"], base_seed=bs, selector_seed=ss, variant=variant)
            write_csv(output.with_suffix(".csv"), rows)
            write_json(output.with_suffix(".json"), {"identity": identity, "rows_sha256": digest(output.with_suffix(".csv")),
                "metrics": metrics, "diagnostics": diagnostics(model, cache["features"], device), "selector": selector,
                "complete_order_probe_rows": len(probes), "complete_order_probe_scope": "unfiltered normalized primary; 32 deterministic queries per direction"})
            print(f"[TEST] {pair['pair']} {variant} b={bs} s={ss} MRR={metrics['mrr']:.6f}", flush=True)


def summarize(config, root):
    import pandas as pd
    from scripts.analyze_paper_a_negative_transfer import clustered_bootstrap, point_summary
    summaries, cells, health, manifest = [], [], [], {}
    key = ["base_seed", "direction", "head_id", "relation_id", "tail_id"]
    for pair in config["pairs"]:
        folder = root / pair["pair"]
        lock = read_json(folder / "dev_lock.json")
        for selector in lock["selectors"]:
            trace_path = folder / selector["relative_dir"] / "trace.json"
            if digest(trace_path) != selector["trace_sha256"]:
                raise RuntimeError("Training diagnostics changed after locking")
            trace = read_json(trace_path)
            last = trace["epochs"][-1]
            health.append({"pair": pair["pair"], "variant": selector["variant"], "base_seed": selector["base_seed"],
                           "selector_seed": selector["selector_seed"], "learning_rate": selector["learning_rate"], "epochs": selector["epochs"],
                           **selector["health"], "initial_zero_weight_fraction": trace["initial"]["zero_weight_fraction"],
                           "final_zero_weight_fraction": last["train_diagnostics"]["zero_weight_fraction"],
                           "initial_weight_std": trace["initial"]["weight"]["std"], "final_weight_std": last["train_diagnostics"]["weight"]["std"],
                           "first_epoch_mean_loss": trace["epochs"][0]["loss_mean"], "last_epoch_mean_loss": last["loss_mean"],
                           "last_epoch_max_gradient_norm": max(v["max_norm"] for v in last["gradients"].values()),
                           "full_dev_resubstitution_mrr": selector["full_dev_resubstitution_mrr"]})
        manifest[pair["pair"]] = {"lock_sha256": digest(folder / "dev_lock.json"), "test_files": {}}
        reference_path = INPUT_ROOT / pair["pair"] / "test_anchored/test_locked_query_rows.csv"
        reference = pd.read_csv(reference_path).rename(columns={"seed": "base_seed"})
        require_score_information_contract({"score_information_contract": reference.score_information_contract.unique().item()})
        manifest[pair["pair"]]["reference_sha256"] = digest(reference_path)
        method_frames = {}
        for variant in VARIANTS:
            parts = []
            for bs in (1, 2, 3):
                for ss in SELECTOR_SEEDS:
                    path = folder / "test" / f"{variant}_b{bs}_s{ss}.csv"
                    record = read_json(path.with_suffix(".json"))
                    if digest(path) != record["rows_sha256"] or record["identity"]["dev_lock_sha256"] != digest(folder / "dev_lock.json"):
                        raise RuntimeError("Unverified TEST output")
                    frame = pd.read_csv(path)
                    if len(frame) != len(reference[reference.base_seed == bs]):
                        raise RuntimeError("Incomplete TEST observations")
                    manifest[pair["pair"]]["test_files"][path.name] = digest(path)
                    parts.append(frame)
            frame = pd.concat(parts, ignore_index=True)
            if frame.duplicated(key + ["selector_seed"]).any():
                raise RuntimeError("Duplicate selector observations")
            joined = frame.merge(reference[key + ["rr_a", "rr_b", "rr_global", "rr_anchored_locked", "rr_query_soft_locked"]],
                                 on=key, validate="many_to_one", suffixes=("", "_reference"))
            if len(joined) != len(frame) or not np.array_equal(joined.rr_a, joined.rr_a_reference) or not np.array_equal(joined.rr_b, joined.rr_b_reference):
                raise RuntimeError("Shared evaluator endpoints differ")
            method_frames[variant] = joined
        r0_path = INPUT_ROOT / pair["pair"] / "dynasemble/test_query_rows.csv"
        r0 = pd.read_csv(r0_path).rename(columns={"seed": "base_seed", "rr_dynasemble": "rr_method"})
        r0["selector_seed"] = r0.base_seed
        r0 = r0.merge(reference[key + ["rr_anchored_locked", "rr_query_soft_locked"]], on=key, validate="one_to_one")
        if len(r0) != len(reference):
            raise RuntimeError("Incomplete historical R0")
        manifest[pair["pair"]]["historical_r0_sha256"] = digest(r0_path)
        method_frames["R0_historical"] = r0
        # Include static and ADC comparisons without inventing selector replications.
        for name, col in (("Global", "rr_global"), ("ADC", "rr_anchored_locked"), ("Query-soft", "rr_query_soft_locked")):
            f = reference.copy()
            f["rr_method"] = f[col]
            f["selector_seed"] = -1
            method_frames[name] = f
        for method, frame in method_frames.items():
            for (bs, ss, direction), group in frame.groupby(["base_seed", "selector_seed", "direction"]):
                cells.append({"pair": pair["pair"], "method": method, "base_seed": int(bs), "selector_seed": int(ss), "direction": direction,
                              "mrr": group.rr_method.mean(), "delta_vs_global": (group.rr_method-group.rr_global).mean(), "n": len(group)})
            # Average selector repetitions first so the sampling unit matches the original six observations.
            grouped = frame.groupby(key)[["rr_method", "rr_global", "rr_anchored_locked"]].mean().reset_index()
            grouped["seed"] = grouped.base_seed
            grouped["raw_triple_id"] = grouped.head_id.astype(str)+"|"+grouped.relation_id.astype(str)+"|"+grouped.tail_id.astype(str)
            grouped["delta_rr"] = grouped.rr_method-grouped.rr_global
            # Harm must be calculated on realized selectors, not their averaged reciprocal ranks.
            realized = frame.copy()
            realized["delta_rr"] = realized.rr_method-realized.rr_global
            realized["raw_triple_id"] = realized.head_id.astype(str)+"|"+realized.relation_id.astype(str)+"|"+realized.tail_id.astype(str)
            point = point_summary(realized)
            intervals = clustered_bootstrap(grouped, samples=10000, seed=20260910)
            adc_comparison = grouped.copy()
            adc_comparison["delta_rr"] = grouped.rr_method-grouped.rr_anchored_locked
            adc_interval = clustered_bootstrap(adc_comparison, samples=10000, seed=20260910)
            summaries.append({"pair": pair["pair"], "method": method, **point,
                "delta_mrr_ci95": intervals["delta_mrr_ci95"], "delta_mrr_vs_adc": float((grouped.rr_method-grouped.rr_anchored_locked).mean()),
                "delta_mrr_vs_adc_ci95": adc_interval["delta_mrr_ci95"],
                "n_selector_seeds_per_base": int(frame.groupby("base_seed").selector_seed.nunique().max()),
                "interval_scope": "original triples, averaged selector repeats; conditional on fitted base/selector seeds"})
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "seed_direction.csv", cells)
    write_csv(root / "health_summary.csv", health)
    write_json(root / "summary_manifest.json", manifest)
    write_json(root / "completion.json", {"status": "experiments_complete_requires_health_and_scientific_review", "pairs": len(config["pairs"]),
        "summary_sha256": digest(root / "summary.csv"), "seed_direction_sha256": digest(root / "seed_direction.csv"),
        "note": "No automatic D01 closure: inspect all zero-gradient/weight flags and held-out diagnostics; retain failures."})


def bundle(root):
    root = Path(root).resolve()
    if not (root / "completion.json").exists():
        raise RuntimeError("Run Summarize before bundling completed evidence")
    target = root.parent / (root.name + "_review.zip")
    temporary = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root)
                if "cache" not in relative.parts and not path.name.endswith(".partial"):
                    archive.write(path, relative.as_posix())
    temporary.replace(target)
    print(f"[REVIEW BUNDLE] {target} ({target.stat().st_size/1024**2:.1f} MiB); candidate caches excluded", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "dev", "test", "summarize", "bundle"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/paper_a_safe_correction/dynasemble_controls_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    source = provenance(args.config)
    if args.stage == "plan":
        missing = []
        for pair in config["pairs"]:
            for run in pair["runs"]:
                for side in ("a", "b"):
                    for filename in ("config_merged.json", "best.ckpt"):
                        p = ROOT / run[f"expert_{side}_run"] / filename
                        if not p.exists():
                            missing.append(p.relative_to(ROOT).as_posix())
        for ds in ("mkg_w", "db15k"):
            for name in ("manifest.json", "train.tsv", "valid.tsv", "test.tsv", "entity2id.json", "relation2id.json",
                         "img_feat.pt", "text_feat.pt", "has_img.pt", "has_text.pt"):
                p = ROOT / "data/datasets" / ds / "processed" / name
                if not p.exists():
                    missing.append(p.relative_to(ROOT).as_posix())
        plan = {"provenance": source, "pairs": [p["pair"] for p in config["pairs"]], "missing_local_assets": sorted(set(missing)),
                "cv_trajectories_per_pair": 3*3*3*3, "final_selectors_per_pair": len(VARIANTS)*3*3,
                "max_epochs_per_cv_trajectory": 10, "selector_seeds": SELECTOR_SEEDS,
                "cache_note": "Float32 full normalized candidate matrices; approximately 32 GiB for four pairs/all DEV+TEST base seeds, plus exports. Reserve >=60 GiB.",
                "test_after_all_dev_locks": True, "cpu_training_started": False}
        write_json(args.output_root / "plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.stage == "summarize":
        for pair in config["pairs"]:
            check_provenance(read_json(args.output_root / pair["pair"] / "dev_lock.json")["provenance"], source)
        summarize(config, args.output_root)
        return
    if args.stage == "bundle":
        bundle(args.output_root)
        return
    if args.device == "cpu" and not args.allow_cpu:
        raise SystemExit("Heavy work belongs on the server. Explicit --allow-cpu required for CPU experiments.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing silent CPU fallback.")
    if args.stage == "test":
        # Global barrier: lock all four pairs before opening any TEST candidates.
        for pair in config["pairs"]:
            folder = args.output_root / pair["pair"]
            lock = read_json(folder / "dev_lock.json")
            check_provenance(lock["provenance"], source)
            verify_lock(folder, lock, pair)
    for pair in config["pairs"]:
        (run_dev if args.stage == "dev" else run_test)(pair, args.output_root, args.device, source)


if __name__ == "__main__":
    main()
