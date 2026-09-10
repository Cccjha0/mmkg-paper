"""Verify B07/C04 rerun provenance, observable-query invariance and frozen metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.information_boundary import SCORE_INFORMATION_CONTRACT
from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.analyze_paper_a_negative_transfer import clustered_bootstrap, point_summary

KEY = ["seed", "direction", "head_id", "relation_id", "tail_id"]
EXPECTED_PAIRS = {f"{ds}_{pair}" for ds in ("mkgw", "db15k")
                  for pair in ("mhyper_native", "mhyper_adamf", "native_adamf")}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def invariance(frame, fields):
    if not np.isfinite(frame[fields].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values cannot pass an invariance audit")
    work = frame.copy()
    work["observed_entity"] = np.where(work.direction == "head", work.tail_id, work.head_id)
    grouped = work.groupby(["seed", "direction", "relation_id", "observed_entity"], sort=True)
    multi = grouped.target_entity_id.nunique() > 1
    spans = (grouped[fields].max() - grouped[fields].min()).loc[multi]
    return {
        "multi_gold_groups": int(multi.sum()),
        "fields": {f: {"nonidentical_groups": int((spans[f] != 0).sum()),
                       "max_abs_span": float(spans[f].max()) if len(spans) else 0.0}
                   for f in fields},
    }


def compare_rows(a, b, fields):
    joined = a[KEY + fields].merge(b[KEY + fields], on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b) or not len(joined):
        raise ValueError("Query exports are not a complete one-to-one match")
    if not np.isfinite(joined.drop(columns=KEY).to_numpy(dtype=float)).all():
        raise ValueError("Non-finite comparison values")
    return {"left_count": len(a), "right_count": len(b), "matched_count": len(joined),
            "max_errors": {f: float(np.max(np.abs(joined[f + "_a"] - joined[f + "_b"])))
                           for f in fields}}


def summarize(frame, column, pair, method, split, samples):
    work = frame[KEY].copy()
    work["raw_triple_id"] = frame.head_id.astype(str) + "|" + frame.relation_id.astype(str) + "|" + frame.tail_id.astype(str)
    work["rr_method"] = frame[column]
    work["rr_global"] = frame.rr_global
    work["delta_rr"] = work.rr_method - work.rr_global
    result = {"pair": pair, "split": split, "method": method, **point_summary(work)}
    if split == "test":
        result.update(clustered_bootstrap(work, samples=samples, seed=20260910))
    return result


def finalize(input_root, output_dir):
    """Recheck files and aggregate all checks; never treat epsilon drift as exact equality."""
    audit = read_json(output_dir / "audit.json")
    failures = list(audit["failures"])
    warnings = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    pairs = {p["pair"] for p in audit["pairs"]}
    require(pairs == EXPECTED_PAIRS and len(audit["pairs"]) == 6, "Expected exactly six audited pairs")
    methods = pd.read_csv(output_dir / "method_summary.csv")
    variants = pd.read_csv(output_dir / "frozen_ablation.csv")
    require(len(variants) == 18, "Expected 18 frozen ablations")
    require((variants.loc[variants.pair.isin(EXPECTED_PAIRS) & (variants.method == "full"),
                          "reproduction_error"] == 0).all(), "Frozen full policy does not reproduce")
    for item in audit["pairs"]:
        pair = item["pair"]
        base = input_root / pair
        require(all(item["hash_checks"].values()) and item["contracts_valid"], pair + ": ADC lock")
        lock = read_json(base / "dev_lock/anchored_dev_lock.json")
        for key, rel in (("source_dev_query_rows_sha256", "full_ranking/dev_query_rows.csv"),
                         ("source_selection_json_sha256", "full_ranking/selection.json"),
                         ("source_crossfit_summary_sha256", "p3_ablation/dev_p3_summary.json"),
                         ("model_sha256", "dev_lock/anchored_model.pkl")):
            require(digest(base / rel) == lock[key], pair + ": current " + key)
        triples = {}
        item["current_artifact_hashes"] = {}
        for split, export_dir, policy_dir in (("dev", "full_ranking", "dev_lock"),
                                               ("test", "test_full_ranking", "test_anchored")):
            context = f"{pair}/{split}"
            entry = item["exports"][split]
            raw_path = base / export_dir / f"{split}_query_rows.csv"
            policy_path = base / policy_dir / f"{split}_locked_query_rows.csv"
            require(digest(raw_path) == entry["sha256"], context + ": changed export")
            frame = pd.read_csv(policy_path)
            raw = pd.read_csv(raw_path)
            item["current_artifact_hashes"][str(policy_path.relative_to(base))] = digest(policy_path)
            triples[split] = set(map(tuple, raw[["head_id", "relation_id", "tail_id"]].to_numpy()))
            expected_n = ({"dev": 4276, "test": 4274} if pair.startswith("mkgw")
                          else {"dev": 7922, "test": 9902})[split]
            require(entry["n_original_triples"] == expected_n and len(raw) == expected_n * 6,
                    context + ": incomplete split")
            require(entry["seeds"] == [1, 2, 3] and entry["directions"] == ["head", "tail"]
                    and entry["group_sizes"] == [6] and entry["duplicate_observations"] == 0,
                    context + ": identity coverage")
            require(entry["contracts"] == [SCORE_INFORMATION_CONTRACT]
                    and entry["filter_scopes"] == ["train_dev_test"], context + ": boundary contract")
            # Repeat against current files, including ranks used in summaries.
            entry["policy_export_match"] = compare_rows(raw, frame, ["rr_a", "rr_b", "rr_global", *QUERY_GEOMETRY_FIELDS])
            entry["geometry_invariance"] = invariance(raw, list(QUERY_GEOMETRY_FIELDS))
            entry["policy_invariance"] = invariance(frame, ["alpha_anchored_continuous", "alpha_anchored_locked", "alpha_query_soft_locked"])
            for field in ("alpha_anchored_locked", "alpha_query_soft_locked"):
                require(entry["policy_invariance"]["fields"][field]["nonidentical_groups"] == 0,
                        context + ": deployed weight differs: " + field)
            for name in ("policy_export_match", "old_endpoint_match"):
                if name in entry:
                    check = entry[name]
                    require(check["left_count"] == check["right_count"] == check["matched_count"]
                            and all(v == 0 for v in check["max_errors"].values()), context + ": " + name)
            for method, column in (("Global", "rr_global"), ("ADC", "rr_anchored_locked"),
                                   ("Query-soft", "rr_query_soft_locked"), ("Relation", "rr_relation")):
                saved = methods[(methods.pair == pair) & (methods.split == split) & (methods.method == method)]
                require(len(saved) == 1 and abs(saved.iloc[0].mrr - frame[column].mean()) < 1e-14,
                        context + ": metric " + method)
            drift = {f: v for group in ("geometry_invariance", "policy_invariance")
                     for f, v in entry[group]["fields"].items() if v["nonidentical_groups"]}
            if drift:
                warnings.append({"context": context, "continuous_or_geometry_nonidentity": drift})
            if "mhyper" in pair:
                ddir = base / "dynasemble"
                dlock = read_json(ddir / "dev_lock.json")
                dsumm = read_json(ddir / f"{split}_summary.json")
                dyna = pd.read_csv(ddir / f"{split}_query_rows.csv")
                d = entry["dynasemble"]
                require(digest(ddir / f"{split}_query_rows.csv") == d["sha256"], context + ": changed DynaSemble export")
                d["lock_hash_match"] = digest(ddir / "dev_lock.json") == dsumm["dev_lock_sha256"]
                d["reference_hash_match"] = digest(raw_path) == dsumm["reference_audit"]["reference_sha256"]
                d["selection_hash_match"] = digest(base / "full_ranking/selection.json") == dlock["baseline_selection_sha256"]
                require(d["lock_hash_match"] and d["reference_hash_match"] and d["selection_hash_match"]
                        and d["contract_valid"], context + ": DynaSemble provenance")
                d["base_match"] = compare_rows(raw, dyna, ["rr_a", "rr_b", "rr_global", "rr_equal"])
                require(all(v == 0 for v in d["base_match"]["max_errors"].values()), context + ": DynaSemble base ranks")
                d["selector_hash_checks"] = {seed: digest(ddir / "selectors" / f"seed{seed}.pt") == record["sha256"]
                                              for seed, record in dlock["selectors"].items()}
                require(set(d["selector_hash_checks"]) == {"1", "2", "3"}
                        and all(d["selector_hash_checks"].values()), context + ": selector hashes")
                d["source_matches"] = {}
                for key, filename in (("evaluator_sha256", "eval_openbg_dynasemble.py"),
                                      ("shared_full_ranking_evaluator_sha256", "eval_heterogeneous_complementarity.py")):
                    content = (ROOT / "scripts" / filename).read_bytes().replace(b"\r\n", b"\n")
                    choices = {"LF": content, "CRLF": content.replace(b"\n", b"\r\n")}
                    matches = [mode for mode, data in choices.items()
                               if hashlib.sha256(data).hexdigest() == dlock["source_provenance"][key]]
                    d["source_matches"][filename] = matches
                    require(bool(matches), context + ": source differs beyond line endings: " + filename)
                saved = methods[(methods.pair == pair) & (methods.split == split) & (methods.method == "DynaSemble")]
                require(len(saved) == 1 and abs(saved.iloc[0].mrr - dyna.rr_dynasemble.mean()) < 1e-14,
                        context + ": DynaSemble metric")
                drift = {f: v for f, v in d["invariance"]["fields"].items() if v["nonidentical_groups"]}
                if drift:
                    warnings.append({"context": context + "/DynaSemble", "continuous_or_geometry_nonidentity": drift})
        item["dev_test_overlap"] = len(triples["dev"] & triples["test"])
        require(item["dev_test_overlap"] == 0, pair + ": DEV/TEST overlap")
    audit["failures"] = sorted(set(failures))
    audit["numerical_repeatability_observations"] = warnings
    audit["status"] = "failed" if failures else "artifact_checks_passed_with_continuous_repeatability_caveat"
    audit["strict_bitwise_invariance_for_all_continuous_weights"] = not warnings
    audit["scope"] = "Six MKG-W/DB15K pairs; four retrained DynaSemble transfers. No OpenBG or efficiency validation."
    audit["summary_hashes"] = {name: digest(output_dir / name) for name in
                               ("method_summary.csv", "seed_direction.csv", "frozen_ablation.csv")}
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "failures": audit["failures"]}), flush=True)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/paper_a_safe_correction/information_boundary_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_a_safe_correction/information_boundary_rerun_audit")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--finalize-only", action="store_true", help="Revalidate completed audit files without resampling intervals")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        finalize(args.input_root, args.output_dir)
        return
    audit = {"contract": SCORE_INFORMATION_CONTRACT, "pairs": [], "failures": [], "bootstrap_samples": args.bootstrap_samples}
    summaries, directions, variants = [], [], []

    for base in sorted(args.input_root.iterdir()):
        if not base.is_dir() or "_" not in base.name:
            continue
        pair = base.name
        print(f"Auditing {pair}", flush=True)
        item = {"pair": pair, "hash_checks": {}, "exports": {}}
        lock = read_json(base / "dev_lock/anchored_dev_lock.json")
        selection = read_json(base / "full_ranking/selection.json")
        item["parameters"] = {f: lock[f] for f in ("alpha0", "beta", "confidence_threshold", "dev_filter_fact_scope")}
        for field, path in (("source_dev_query_rows_sha256", "full_ranking/dev_query_rows.csv"),
                            ("source_selection_json_sha256", "full_ranking/selection.json"),
                            ("source_crossfit_summary_sha256", "p3_ablation/dev_p3_summary.json"),
                            ("model_sha256", "dev_lock/anchored_model.pkl")):
            item["hash_checks"][field] = digest(base / path) == lock[field]
        item["contracts_valid"] = all(p.get("score_information_contract") == SCORE_INFORMATION_CONTRACT for p in (lock, selection))
        if not item["contracts_valid"] or not all(item["hash_checks"].values()):
            audit["failures"].append(pair + ": lock contract/hash failure")
        for split, export_dir, policy_dir in (("dev", "full_ranking", "dev_lock"), ("test", "test_full_ranking", "test_anchored")):
            path = base / export_dir / f"{split}_query_rows.csv"
            raw = pd.read_csv(path)
            frame = pd.read_csv(base / policy_dir / f"{split}_locked_query_rows.csv")
            geometry = list(QUERY_GEOMETRY_FIELDS)
            result = {"sha256": digest(path), "n_rows": len(raw),
                      "seeds": sorted(raw.seed.unique().tolist()), "directions": sorted(raw.direction.unique().tolist()),
                      "n_original_triples": len(raw[["head_id", "relation_id", "tail_id"]].drop_duplicates()),
                      "duplicate_observations": int(raw.duplicated(KEY).sum()),
                      "group_sizes": sorted(raw.groupby(["head_id", "relation_id", "tail_id"]).size().unique().tolist()),
                      "contracts": sorted(raw.score_information_contract.unique().tolist()),
                      "filter_scopes": sorted(raw.filter_fact_scope.unique().tolist()),
                      "geometry_invariance": invariance(raw, geometry),
                      "policy_invariance": invariance(frame, ["alpha_anchored_continuous", "alpha_anchored_locked", "alpha_query_soft_locked"]),
                      "policy_export_match": compare_rows(raw, frame, ["rr_a", "rr_b", "rr_global", *geometry])}
            if result["duplicate_observations"] or result["group_sizes"] != [6] or result["contracts"] != [SCORE_INFORMATION_CONTRACT]:
                audit["failures"].append(f"{pair}/{split}: export identity/contract failure")
            for f in ("alpha_anchored_locked", "alpha_query_soft_locked"):
                if result["policy_invariance"]["fields"][f]["nonidentical_groups"]:
                    audit["failures"].append(f"{pair}/{split}: {f} differs across golds")
            ds = "mkg_w" if pair.startswith("mkgw_") else "db15k"
            old_pair = pair.split("_", 1)[1] + "_seed123"
            old = ROOT / "outputs" / ds / "anchored_dynamic" / old_pair / export_dir / f"{split}_query_rows.csv"
            if old.exists():
                result["old_endpoint_match"] = compare_rows(raw, pd.read_csv(old, usecols=KEY + ["rr_a", "rr_b"]), ["rr_a", "rr_b"])
            for method, col in (("Global", "rr_global"), ("ADC", "rr_anchored_locked"), ("Query-soft", "rr_query_soft_locked"), ("Relation", "rr_relation")):
                summaries.append(summarize(frame, col, pair, method, split, args.bootstrap_samples))
            if split == "test":
                for (seed, direction), group in frame.groupby(["seed", "direction"]):
                    directions.append({"pair": pair, "seed": int(seed), "direction": direction,
                                       "n": len(group), "adc_mrr": float(group.rr_anchored_locked.mean()),
                                       "delta_vs_global": float((group.rr_anchored_locked-group.rr_global).mean())})
                # Fixed counterfactual maps, with no fitting or TEST selection.
                grid = np.asarray(lock["alpha_grid"])
                for variant, beta, threshold in (("full", lock["beta"], lock["confidence_threshold"]),
                                                 ("no_bound", 1., lock["confidence_threshold"]),
                                                 ("no_fallback", lock["beta"], 0.)):
                    weight = np.clip(lock["alpha0"] + beta*np.tanh(frame.anchored_decision.to_numpy()), 0, 1)
                    weight = np.where(frame.anchored_confidence.to_numpy() < threshold, lock["alpha0"], weight)
                    applied = np.asarray([min(grid, key=lambda a: (abs(a-w), abs(a-lock["alpha0"]), a)) for w in weight])
                    rr_matrix = frame[[f"rr_alpha_{a:.2f}".replace(".", "_") for a in grid]].to_numpy()
                    frame["variant_rr"] = rr_matrix[np.arange(len(frame)), np.searchsorted(grid, applied)]
                    record = summarize(frame, "variant_rr", pair, variant, "test", args.bootstrap_samples)
                    if variant == "full":
                        record["reproduction_error"] = float(np.abs(frame.variant_rr-frame.rr_anchored_locked).max())
                    variants.append(record)
            if (base / "dynasemble").exists():
                dp = base / "dynasemble" / f"{split}_query_rows.csv"
                dyna = pd.read_csv(dp)
                dsumm = read_json(base / "dynasemble" / f"{split}_summary.json")
                dlock = read_json(base / "dynasemble/dev_lock.json")
                config = dsumm["method_config"]
                result["dynasemble"] = {"sha256": digest(dp), "n_rows": len(dyna),
                    "invariance": invariance(dyna, ["weight_expert_a", "effective_alpha_expert_a", "feature_a_one_minus_mean", "feature_a_variance", "feature_b_one_minus_mean", "feature_b_variance"]),
                    "lock_hash_match": digest(base / "dynasemble/dev_lock.json") == dsumm["dev_lock_sha256"],
                    "reference_hash_match": digest(path) == dsumm["reference_audit"]["reference_sha256"],
                    "contract_valid": config.get("score_information_contract") == SCORE_INFORMATION_CONTRACT and config.get("negative_filter_scope") == "train_dev",
                    "base_match": compare_rows(raw, dyna, ["rr_a", "rr_b", "rr_global", "rr_equal"])}
                summaries.append(summarize(dyna, "rr_dynasemble", pair, "DynaSemble", split, args.bootstrap_samples))
            item["exports"][split] = result
        audit["pairs"].append(item)
        # Persist progress without marking the audit complete.
        (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.output_dir / "method_summary.csv", index=False)
    pd.DataFrame(directions).to_csv(args.output_dir / "seed_direction.csv", index=False)
    pd.DataFrame(variants).to_csv(args.output_dir / "frozen_ablation.csv", index=False)
    finalize(args.input_root, args.output_dir)
    print(json.dumps({"pairs": len(audit["pairs"]), "failures": audit["failures"]}), flush=True)


if __name__ == "__main__":
    main()
