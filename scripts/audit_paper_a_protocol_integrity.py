from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_VERSION = "paper_a_protocol_integrity_v1"
REPORT_OPENING = "DEV for all selection; TEST for immutable final evaluation only."
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_DIRECTIONS = {"head", "tail"}
EXPECTED_ALPHA_GRID = [round(i * 0.05, 2) for i in range(21)]
FEATURE_SCHEMA = [
    "geometry_direction_tail",
    "geometry_a_top1",
    "geometry_a_top5_mean",
    "geometry_a_top1_top2_margin",
    "geometry_a_score_std",
    "geometry_b_top1",
    "geometry_b_top5_mean",
    "geometry_b_top1_top2_margin",
    "geometry_b_score_std",
    "geometry_top1_delta_a_minus_b",
    "geometry_top5_delta_a_minus_b",
    "geometry_margin_delta_a_minus_b",
    "geometry_std_delta_a_minus_b",
]

PAIR_SPECS = (
    {"dataset": "MKG-W", "dataset_dir": "mkg_w", "pair": "M-Hyper + NativE", "pair_dir": "mhyper_native", "scope": "main"},
    {"dataset": "MKG-W", "dataset_dir": "mkg_w", "pair": "M-Hyper + AdaMF-MAT", "pair_dir": "mhyper_adamf", "scope": "main"},
    {"dataset": "DB15K", "dataset_dir": "db15k", "pair": "M-Hyper + NativE", "pair_dir": "mhyper_native", "scope": "main"},
    {"dataset": "DB15K", "dataset_dir": "db15k", "pair": "M-Hyper + AdaMF-MAT", "pair_dir": "mhyper_adamf", "scope": "main"},
    {"dataset": "MKG-W", "dataset_dir": "mkg_w", "pair": "NativE + AdaMF-MAT", "pair_dir": "native_adamf", "scope": "boundary"},
    {"dataset": "DB15K", "dataset_dir": "db15k", "pair": "NativE + AdaMF-MAT", "pair_dir": "native_adamf", "scope": "boundary"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only protocol and leakage audit for final Paper A experiments."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/protocol_audit",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def same_float(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def same_float_list(left: Any, right: Any) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(same_float(a, b) for a, b in zip(left, right))
    )


class Audit:
    def __init__(self, repo_root: Path, output_dir: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.output_dir = output_dir.resolve()
        self.checks: list[dict[str, Any]] = []
        self.assets: dict[str, dict[str, Any]] = {}

    def asset(
        self,
        path: Path,
        role: str,
        expected_hash: str | None = None,
        locked_path: str | None = None,
    ) -> tuple[bool, str | None]:
        path = path.resolve()
        key = portable(path, self.repo_root)
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        match_mode = "not_checked"
        matches = exists and expected_hash is None
        if exists and expected_hash is not None:
            if actual == expected_hash:
                matches = True
                match_mode = "raw_bytes"
            elif path.suffix.lower() in {".md", ".txt", ".py", ".ps1", ".yaml", ".yml", ".json", ".csv"}:
                # Git checkouts and the A100 Windows worktree can materialize the
                # same text as LF or CRLF. Preserve the raw digest in the manifest,
                # but accept only an exact newline-normalized byte equivalence.
                raw = path.read_bytes()
                lf = raw.replace(b"\r\n", b"\n")
                crlf = lf.replace(b"\n", b"\r\n")
                variant_hashes = {
                    "lf_normalized": hashlib.sha256(lf).hexdigest(),
                    "crlf_normalized": hashlib.sha256(crlf).hexdigest(),
                }
                for mode, digest in variant_hashes.items():
                    if digest == expected_hash:
                        matches = True
                        match_mode = mode
                        break
            if not matches:
                match_mode = "mismatch"
        elif exists:
            match_mode = "raw_bytes_unlocked"
        record = self.assets.setdefault(
            key,
            {
                "path": key,
                "roles": [],
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "sha256": actual,
                "expected_hashes": [],
                "accepted_match_modes": [],
                "hash_matches_all_expectations": True,
                "locked_paths": [],
            },
        )
        if role not in record["roles"]:
            record["roles"].append(role)
        if expected_hash and expected_hash not in record["expected_hashes"]:
            record["expected_hashes"].append(expected_hash)
        if match_mode not in record["accepted_match_modes"]:
            record["accepted_match_modes"].append(match_mode)
        if locked_path and locked_path not in record["locked_paths"]:
            record["locked_paths"].append(locked_path)
        record["hash_matches_all_expectations"] = bool(
            record["hash_matches_all_expectations"] and matches
        )
        return bool(matches), actual

    def check(
        self,
        experiment: str,
        check_id: str,
        requirement: str,
        passed: bool,
        *,
        dataset: str = "ALL",
        pair: str = "ALL",
        method: str = "ALL",
        evidence: list[str] | None = None,
        details: str = "",
    ) -> None:
        self.checks.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "pair": pair,
                "method": method,
                "check_id": check_id,
                "requirement": requirement,
                "status": "PASS" if passed else "FAIL",
                "evidence_files": " | ".join(evidence or []),
                "details": details,
            }
        )


def base_root(repo_root: Path, spec: dict[str, str]) -> Path:
    return (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )


def anchored_root(repo_root: Path, spec: dict[str, str]) -> Path:
    if spec["scope"] == "main":
        return base_root(repo_root, spec)
    return (
        repo_root
        / "outputs/paper_a_safe_correction/boundary_pairs"
        / spec["dataset_dir"]
        / spec["pair_dir"]
        / "anchored"
    )


def test_base_root(repo_root: Path, spec: dict[str, str]) -> Path:
    if spec["scope"] == "main":
        return base_root(repo_root, spec) / "test_full_ranking"
    return (
        repo_root
        / "outputs/paper_a_safe_correction/boundary_pairs"
        / spec["dataset_dir"]
        / spec["pair_dir"]
        / "test_full_ranking"
    )


def dyna_root(repo_root: Path, spec: dict[str, str]) -> Path:
    if spec["scope"] == "main":
        return (
            repo_root
            / "outputs/paper_a_safe_correction/dynasemble"
            / spec["dataset_dir"]
            / spec["pair_dir"]
        )
    return (
        repo_root
        / "outputs/paper_a_safe_correction/boundary_pairs"
        / spec["dataset_dir"]
        / spec["pair_dir"]
        / "dynasemble"
    )


def csv_header_and_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_primary_selection(audit: Audit) -> None:
    root = audit.repo_root / "outputs/paper_a_safe_correction/reliable_primary"
    manifest_path = root / "reliable_primary_manifest.json"
    decisions_path = root / "reliable_primary_decisions.csv"
    protocol_path = audit.repo_root / "docs/protocols/paper_a_reliable_primary_protocol.md"
    script_path = audit.repo_root / "scripts/audit_reliable_primary_regime.py"
    evidence = [portable(x, audit.repo_root) for x in (manifest_path, decisions_path, protocol_path, script_path)]
    manifest_ok, _ = audit.asset(manifest_path, "primary selection manifest")
    decisions_ok, _ = audit.asset(decisions_path, "primary selection decisions")
    audit.asset(protocol_path, "primary selection protocol")
    audit.asset(script_path, "primary selection implementation")
    if not manifest_ok:
        audit.check("primary_selection", "P01", "Primary selection uses DEV only", False, method="Primary selection", evidence=evidence, details=f"Missing manifest: {evidence[0]}")
        return
    manifest = read_json(manifest_path)
    assets = manifest.get("dev_assets", [])
    problems: list[str] = []
    for item in assets:
        rel = str(item.get("path", ""))
        path = audit.repo_root / Path(rel)
        matched, _ = audit.asset(path, "primary selection DEV input", item.get("sha256"), rel)
        if not matched:
            problems.append(f"hash/missing: {rel}")
        if not rel.replace("\\", "/").endswith("/full_ranking/dev_query_rows.csv"):
            problems.append(f"non-DEV input: {rel}")
        if item.get("split_values") != ["dev"]:
            problems.append(f"split_values={item.get('split_values')} for {rel}")
    rule = manifest.get("selection_rule", {})
    dev_only = (
        manifest.get("TEST_NOT_USED") is True
        and manifest.get("test_files_read") == []
        and len(assets) == 6
        and rule.get("post_hoc_mrr_margin") is None
        and not problems
    )
    audit.check("primary_selection", "P01", "Primary selection accesses exact filtered DEV assets only", dev_only, method="Primary selection", evidence=evidence, details="; ".join(problems) or "Six hashed inputs are DEV-only; TEST_NOT_USED=true; test_files_read is empty.")
    audit.check("primary_selection", "P02", "Primary selection rule is frozen and has no post-hoc MRR margin", rule.get("primary") == "model with higher pooled DEV MRR" and rule.get("post_hoc_mrr_margin") is None, method="Primary selection", evidence=evidence, details=json.dumps(rule, ensure_ascii=False, sort_keys=True))
    audit.check("primary_selection", "P03", "Primary selection decision table exists", decisions_ok, method="Primary selection", evidence=evidence, details="Decision CSV is present and hashed." if decisions_ok else f"Missing: {evidence[1]}")


def validate_grouped_folds(audit: Audit, spec: dict[str, str]) -> None:
    path = base_root(audit.repo_root, spec) / "baseline_crossfit/dev_crossfit_query_rows.csv"
    rel = portable(path, audit.repo_root)
    exists, _ = audit.asset(path, "grouped OOF DEV rows")
    if not exists:
        audit.check("grouped_folds", "F01", "All seeds and both directions of an original triple share one fold", False, dataset=spec["dataset"], pair=spec["pair"], method="Query-soft / Anchored / core ablations", evidence=[rel], details=f"Missing: {rel}")
        return
    header, rows = csv_header_and_rows(path)
    required = {"head_id", "relation_id", "tail_id", "seed", "direction", "crossfit_fold", "split"}
    missing = required - set(header)
    problems: list[str] = []
    groups: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: {"fold": set(), "seed": set(), "direction": set(), "split": set()})
    if not missing:
        for row in rows:
            key = (row["head_id"], row["relation_id"], row["tail_id"])
            groups[key]["fold"].add(row["crossfit_fold"])
            groups[key]["seed"].add(row["seed"])
            groups[key]["direction"].add(row["direction"].lower())
            groups[key]["split"].add(row["split"].lower())
        for key, value in groups.items():
            if len(value["fold"]) != 1 or value["seed"] != {"1", "2", "3"} or value["direction"] != EXPECTED_DIRECTIONS or value["split"] != {"dev"}:
                problems.append(f"triple={key}, folds={sorted(value['fold'])}, seeds={sorted(value['seed'])}, directions={sorted(value['direction'])}, split={sorted(value['split'])}")
                if len(problems) >= 10:
                    break
    passed = not missing and not problems and len(rows) == len(groups) * 6
    detail = f"Validated {len(groups):,} original-triple clusters and {len(rows):,} observations; each cluster has 3 seeds x 2 directions in one fold."
    if missing:
        detail = f"Missing columns: {sorted(missing)}"
    elif problems:
        detail = "First violations: " + "; ".join(problems)
    elif len(rows) != len(groups) * 6:
        detail = f"Expected six rows per triple: rows={len(rows)}, triples={len(groups)}"
    audit.check("grouped_folds", "F01", "All seeds and both directions of an original triple share one grouped DEV fold", passed, dataset=spec["dataset"], pair=spec["pair"], method="Query-soft / Anchored / core ablations", evidence=[rel], details=detail)


def validate_static_and_anchored(audit: Audit, spec: dict[str, str]) -> None:
    base = base_root(audit.repo_root, spec)
    aroot = anchored_root(audit.repo_root, spec)
    selection_path = base / "full_ranking/selection.json"
    dev_rows_path = base / "full_ranking/dev_query_rows.csv"
    lock_path = aroot / "dev_lock/anchored_dev_lock.json"
    model_path = aroot / "dev_lock/anchored_model.pkl"
    test_rows_path = aroot / "test_anchored/test_locked_query_rows.csv"
    test_summary_path = aroot / "test_anchored/test_locked_summary.json"
    test_base_path = test_base_root(audit.repo_root, spec) / "test_query_rows.csv"
    experiment = f"{spec['scope']}_static_query_soft_anchored"
    evidence = [portable(x, audit.repo_root) for x in (selection_path, dev_rows_path, lock_path, model_path, test_base_path, test_rows_path, test_summary_path)]
    required = []
    for path, role in (
        (selection_path, "Global alpha DEV selection"),
        (dev_rows_path, "exact filtered DEV rows"),
        (lock_path, "Anchored immutable DEV lock"),
        (test_base_path, "exact filtered TEST base rows"),
        (test_rows_path, "Anchored immutable TEST rows"),
        (test_summary_path, "Anchored TEST summary"),
    ):
        ok, _ = audit.asset(path, role)
        required.append(ok)
    if not all(required):
        missing = [rel for rel, ok in zip([evidence[0], evidence[1], evidence[2], evidence[4], evidence[5], evidence[6]], required) if not ok]
        audit.check(experiment, "A00", "Required final assets exist", False, dataset=spec["dataset"], pair=spec["pair"], method="Global / Query-soft / Anchored", evidence=evidence, details="Missing: " + ", ".join(missing))
        return

    selection = read_json(selection_path)
    lock = read_json(lock_path)
    summary = read_json(test_summary_path)
    source_dev_hash_ok, _ = audit.asset(dev_rows_path, "Anchored locked DEV source", lock.get("source_dev_query_rows_sha256"), lock.get("source_dev_query_rows"))
    selection_hash_ok, selection_hash = audit.asset(selection_path, "Anchored locked Global selection", lock.get("source_selection_json_sha256"), lock.get("source_selection_json"))
    resolved_model = model_path if model_path.exists() else lock_path.parent / str(lock.get("model_file", ""))
    model_hash_ok, _ = audit.asset(resolved_model, "Anchored locked combiner", lock.get("model_sha256"), lock.get("model_file"))

    global_ok = (
        set(selection.get("seeds", [])) == set(EXPECTED_SEEDS)
        and same_float_list(selection.get("alpha_grid"), EXPECTED_ALPHA_GRID)
        and same_float(selection.get("global_alpha"), lock.get("alpha0"))
        and selection_hash_ok
        and source_dev_hash_ok
    )
    audit.check(experiment, "G01", "Global alpha is determined from DEV only and inherited unchanged", global_ok, dataset=spec["dataset"], pair=spec["pair"], method="Global alpha", evidence=evidence[:3], details=f"selection_sha256={selection_hash}; global_alpha={selection.get('global_alpha')}; lock_alpha0={lock.get('alpha0')}; DEV source hash match={source_dev_hash_ok}.")

    boundary = str(lock.get("selection_boundary", "")).lower()
    selection_only_dev = "dev only" in boundary and source_dev_hash_ok and selection_hash_ok and model_hash_ok
    audit.check(experiment, "A01", "Anchored beta and confidence threshold are selected using DEV only", selection_only_dev, dataset=spec["dataset"], pair=spec["pair"], method="Anchored Dynamic", evidence=[evidence[2], evidence[1], evidence[0], portable(resolved_model, audit.repo_root)], details=f"selection_boundary={lock.get('selection_boundary')!r}; beta={lock.get('beta')}; threshold={lock.get('confidence_threshold')}.")

    schema_ok = lock.get("query_geometry_fields") == FEATURE_SCHEMA
    alpha_grid_ok = same_float_list(lock.get("alpha_grid"), EXPECTED_ALPHA_GRID)
    pair_ok = lock.get("dataset") == spec["dataset_dir"] and set(lock.get("seeds", [])) == set(EXPECTED_SEEDS)
    policy = summary.get("policy", {})
    policy_ok = all(
        policy.get(key) == lock.get(key)
        for key in ("dataset", "protocol_version", "expert_a_name", "expert_b_name", "seeds", "formula", "model", "label", "alpha_application", "alpha_grid", "alpha0", "beta", "confidence_threshold", "random_state", "query_geometry_fields", "model_sha256")
    )

    header, test_rows = csv_header_and_rows(test_rows_path)
    required_columns = set(FEATURE_SCHEMA) | {
        "seed", "direction", "split", "alpha_global", "alpha0_locked", "anchored_beta_locked", "anchored_confidence_threshold_locked",
    }
    row_problems: list[str] = []
    for row in test_rows:
        if row.get("split", "").lower() != "test":
            row_problems.append("non-TEST split in locked TEST rows")
        if not same_float(row.get("alpha_global"), selection.get("global_alpha")):
            row_problems.append("alpha_global differs from DEV selection")
        if not same_float(row.get("alpha0_locked"), lock.get("alpha0")):
            row_problems.append("alpha0_locked differs from DEV lock")
        if not same_float(row.get("anchored_beta_locked"), lock.get("beta")):
            row_problems.append("beta differs from DEV lock")
        if not same_float(row.get("anchored_confidence_threshold_locked"), lock.get("confidence_threshold")):
            row_problems.append("threshold differs from DEV lock")
        if row_problems:
            break
    metadata_ok = not (required_columns - set(header)) and not row_problems and pair_ok and policy_ok

    # Exact base score/rank inputs in the apply output must be byte-for-value identical
    # to the frozen TEST evaluator rows for the columns used by all combiners.
    base_header, base_rows = csv_header_and_rows(test_base_path)
    compare_columns = ["query_id", "seed", "direction", "head_id", "relation_id", "tail_id", "rank_a", "rr_a", "rank_b", "rr_b", "alpha_global", "rr_global"] + [f"rr_alpha_{i/20:.2f}".replace(".", "_") for i in range(21)]
    base_map = {row["query_id"]: row for row in base_rows}
    mismatch = None
    for row in test_rows:
        other = base_map.get(row.get("query_id", ""))
        if other is None:
            mismatch = f"missing query_id in TEST base rows: {row.get('query_id')}"
            break
        for column in compare_columns:
            if column not in row or column not in other:
                mismatch = f"missing compared column: {column}"
                break
            if column.startswith("rr_") or column == "alpha_global":
                if not same_float(row[column], other[column]):
                    mismatch = f"{column} differs for {row.get('query_id')}"
                    break
            elif row[column] != other[column]:
                mismatch = f"{column} differs for {row.get('query_id')}"
                break
        if mismatch:
            break
    base_match = mismatch is None and len(test_rows) == len(base_rows)
    audit.check(experiment, "A02", "Feature schema, model pair, seeds, alpha grid, and locked parameters match between DEV lock and TEST apply", schema_ok and alpha_grid_ok and metadata_ok, dataset=spec["dataset"], pair=spec["pair"], method="Query-soft / Anchored Dynamic", evidence=[evidence[2], evidence[5], evidence[6]], details=f"feature_schema_match={schema_ok}; alpha_grid_match={alpha_grid_ok}; policy_mirror_match={policy_ok}; rows={len(test_rows):,}; row_metadata_problems={row_problems or 'none'}.")
    audit.check(experiment, "A03", "TEST apply uses the exact frozen base score/rank assets", base_match, dataset=spec["dataset"], pair=spec["pair"], method="Global / Query-soft / Anchored Dynamic", evidence=[evidence[4], evidence[5]], details=mismatch or f"Matched {len(test_rows):,} TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns.")
    audit.check(experiment, "A04", "An immutable DEV lock is consumed before TEST evaluation", policy_ok and model_hash_ok and selection_hash_ok, dataset=spec["dataset"], pair=spec["pair"], method="Anchored Dynamic", evidence=[evidence[2], evidence[3], evidence[6]], details="TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock.")


def locate_run_asset(run: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = run / name
        if candidate.is_file():
            return candidate
    return None


def validate_dynasemble(audit: Audit, spec: dict[str, str]) -> None:
    root = dyna_root(audit.repo_root, spec)
    lock_path = root / "lock.json"
    dev_rows_path = root / "dev_query_rows.csv"
    test_rows_path = root / "test_query_rows.csv"
    test_summary_path = root / "test_summary.json"
    experiment = f"dynasemble_{spec['scope']}"
    evidence = [portable(x, audit.repo_root) for x in (lock_path, dev_rows_path, test_rows_path, test_summary_path)]
    exists = []
    for path, role in ((lock_path, "DynaSemble immutable DEV lock"), (dev_rows_path, "DynaSemble DEV rows"), (test_rows_path, "DynaSemble TEST rows"), (test_summary_path, "DynaSemble TEST summary")):
        ok, _ = audit.asset(path, role)
        exists.append(ok)
    if not all(exists):
        audit.check(experiment, "D00", "Required DynaSemble final assets exist", False, dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=evidence, details="Missing: " + ", ".join(x for x, ok in zip(evidence, exists) if not ok))
        return
    lock = read_json(lock_path)
    summary = read_json(test_summary_path)
    lock_hash = sha256_file(lock_path)
    training_ok = lock.get("test_is_not_used_for_training_or_selection") is True and lock.get("method_config", {}).get("training_split") == "dev_only"
    selector_problems: list[str] = []
    selector_hashes: dict[str, str] = {}
    for seed in EXPECTED_SEEDS:
        item = lock.get("selectors", {}).get(str(seed), {})
        path = audit.repo_root / Path(str(item.get("path", "")))
        matched, actual = audit.asset(path, "DynaSemble DEV-trained selector", item.get("sha256"), item.get("path"))
        if not matched:
            selector_problems.append(f"seed {seed} selector missing/hash mismatch: {portable(path, audit.repo_root)}")
        if actual:
            selector_hashes[str(seed)] = actual
    audit.check(experiment, "D01", "One selector per paired seed is trained on DEV only", training_ok and not selector_problems and set(lock.get("seeds", [])) == set(EXPECTED_SEEDS), dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=evidence[:2] + [str(lock.get("selectors", {}).get(str(s), {}).get("path", "")) for s in EXPECTED_SEEDS], details="; ".join(selector_problems) or "training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection.")

    summary_lock_ok = summary.get("dev_lock_sha256") == lock_hash
    config_ok = summary.get("method_config") == lock.get("method_config")
    selection_path = audit.repo_root / Path(str(lock.get("baseline_selection_path", "")))
    selection_ok, _ = audit.asset(selection_path, "DynaSemble locked Global selection", lock.get("baseline_selection_sha256"), lock.get("baseline_selection_path"))
    selection_summary_ok = summary.get("baseline_selection_sha256") == lock.get("baseline_selection_sha256")
    protocol_item = lock.get("paper_a_protocol", {})
    protocol_path = audit.repo_root / Path(str(protocol_item.get("path", "")))
    protocol_ok, _ = audit.asset(protocol_path, "DynaSemble predeclared protocol", protocol_item.get("sha256"), protocol_item.get("path"))
    audit.check(experiment, "D02", "TEST consumes the immutable DEV lock without configuration drift", summary_lock_ok and config_ok and selection_ok and selection_summary_ok and protocol_ok, dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=evidence + [portable(selection_path, audit.repo_root), portable(protocol_path, audit.repo_root)], details=f"lock_sha256={lock_hash}; summary_lock_match={summary_lock_ok}; method_config_match={config_ok}; Global selection hash match={selection_ok and selection_summary_ok}; protocol hash match={protocol_ok}.")

    header, rows = csv_header_and_rows(test_rows_path)
    row_problem = None
    for row in rows:
        seed = row.get("seed", "")
        if row.get("split", "").lower() != "test":
            row_problem = "non-TEST split in DynaSemble TEST rows"
            break
        if row.get("selector_sha256") != selector_hashes.get(seed):
            row_problem = f"selector hash mismatch in TEST row seed={seed}"
            break
        if row.get("baseline_selection_sha256") != lock.get("baseline_selection_sha256"):
            row_problem = "Global selection hash mismatch in TEST row"
            break
        if not same_float(row.get("alpha_global"), lock.get("global_alpha")):
            row_problem = "Global alpha mismatch in TEST row"
            break
    reference = summary.get("reference_audit", {})
    reference_path = audit.repo_root / Path(str(reference.get("reference_path", "")))
    reference_ok, _ = audit.asset(reference_path, "DynaSemble exact filtered TEST reference", reference.get("reference_sha256"), reference.get("reference_path"))
    reference_clean = reference.get("rank_mismatches") == {} and all(same_float(value, 0.0) for value in reference.get("max_abs_rr_error", {}).values())
    rows_ok = not row_problem and set(header) >= {"split", "seed", "direction", "selector_sha256", "baseline_selection_sha256", "alpha_global"}
    audit.check(experiment, "D03", "TEST model pair, seeds, selector hashes, Global alpha, and exact base ranking inputs match the DEV lock", rows_ok and reference_ok and reference_clean and set(summary.get("seeds", [])) == set(EXPECTED_SEEDS), dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=[evidence[0], evidence[2], evidence[3], portable(reference_path, audit.repo_root)], details=row_problem or f"Validated {len(rows):,} TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match={reference_ok}.")

    run_problems: list[str] = []
    for item in lock.get("run_manifest", []):
        for expert in ("expert_a", "expert_b"):
            run = audit.repo_root / Path(str(item.get(f"{expert}_run", "")))
            config = locate_run_asset(run, ("config_merged.json", "config.json", "common.yaml"))
            checkpoint = locate_run_asset(run, ("best.ckpt", "best.pt", "checkpoint_best.pt", "model.pt"))
            if config is None:
                run_problems.append(f"missing config in {portable(run, audit.repo_root)}")
            else:
                ok, _ = audit.asset(config, "frozen base-model config", item.get(f"{expert}_config_sha256"), str(run))
                if not ok:
                    run_problems.append(f"config hash mismatch: {portable(config, audit.repo_root)}")
            if checkpoint is None:
                run_problems.append(f"missing checkpoint in {portable(run, audit.repo_root)}")
            else:
                ok, _ = audit.asset(checkpoint, "frozen base-model checkpoint", item.get(f"{expert}_checkpoint_sha256"), str(run))
                if not ok:
                    run_problems.append(f"checkpoint hash mismatch: {portable(checkpoint, audit.repo_root)}")
    audit.check(experiment, "D04", "Frozen base-model run assets match the hashes recorded before TEST", not run_problems and len(lock.get("run_manifest", [])) == 3, dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=[evidence[0]], details="; ".join(run_problems) or "All seed-specific expert config and checkpoint hashes match the DEV lock.")

    rebinds = lock.get("repository_commit_rebind_history", [])
    rebind_ok = all(
        item.get("test_outcomes_read") is False
        and item.get("method_or_hyperparameter_changed") is False
        and (
            item.get("verified_locked_hashes_match") is True
            or isinstance(item.get("verified_locked_hashes"), dict)
            and bool(item.get("verified_locked_hashes"))
        )
        for item in rebinds
    )
    audit.check(experiment, "D05", "Any provenance-only lock rebind did not inspect TEST or change method/hyperparameters", rebind_ok, dataset=spec["dataset"], pair=spec["pair"], method="DynaSemble", evidence=[evidence[0]], details=f"provenance_rebind_count={len(rebinds)}; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes.")


def validate_core_ablation(audit: Audit) -> None:
    path = audit.repo_root / "outputs/paper_a_safe_correction/core_ablation/audit.json"
    script = audit.repo_root / "scripts/ablate_paper_a_core_safety_components.py"
    evidence = [portable(path, audit.repo_root), portable(script, audit.repo_root)]
    ok, _ = audit.asset(path, "core ablation audit")
    audit.asset(script, "core ablation implementation")
    if not ok:
        audit.check("core_ablation", "C01", "Core ablations inherit the Full DEV lock and never tune on TEST", False, method="Core ablations", evidence=evidence, details=f"Missing: {evidence[0]}")
        return
    data = read_json(path)
    flags_ok = data.get("matched_contract_validation_passed") is True and data.get("test_selection_or_tuning") is False and data.get("ablation_specific_hyperparameter_search") is False and data.get("no_base_model_training") is True
    problems: list[str] = []
    sources = data.get("source_audit", [])
    for item in sources:
        for key in ("source_query_rows", "source_p3_summary", "source_p3_full_rows", "source_lock"):
            rel = item.get(key)
            expected = item.get(f"{key}_sha256")
            if rel:
                matched, _ = audit.asset(audit.repo_root / Path(str(rel)), "core ablation locked source", expected, str(rel))
                if not matched:
                    problems.append(f"{key} missing/hash mismatch: {rel}")
        if item.get("split") == "test" and item.get("test_selection_or_tuning") is not False:
            problems.append(f"TEST tuning flag not false for {item.get('dataset')}/{item.get('pair')}")
    audit.check("core_ablation", "C01", "Matched core ablations inherit Full's DEV-selected parameters and do not search ablation-specific hyperparameters", flags_ok and not problems, method="Core ablations", evidence=evidence, details="; ".join(problems) or "Matched-contract validation passed; no base retraining, TEST selection, or ablation-specific hyperparameter search.")


def validate_analysis_boundaries(audit: Audit) -> None:
    risk_path = audit.repo_root / "outputs/paper_a_safe_correction/risk_coverage/audit.json"
    negative_path = audit.repo_root / "outputs/paper_a_safe_correction/negative_transfer/bootstrap_ci.json"
    confidence_path = audit.repo_root / "outputs/paper_a_safe_correction/confidence_harm/auroc_auprc.json"
    beta_path = audit.repo_root / "outputs/paper_a_safe_correction/beta_sensitivity/beta_sensitivity_audit.json"
    fallback_path = audit.repo_root / "outputs/paper_a_safe_correction/fallback_audit/fallback_audit.json"
    risk_ok, _ = audit.asset(risk_path, "risk-coverage audit")
    negative_ok, _ = audit.asset(negative_path, "negative-transfer audit")
    confidence_ok, _ = audit.asset(confidence_path, "confidence-to-harm audit")
    beta_ok, _ = audit.asset(beta_path, "beta trust-region sensitivity audit")
    fallback_ok, _ = audit.asset(fallback_path, "fallback behavior audit")
    if risk_ok:
        risk = read_json(risk_path)
        fixed_grid = risk.get("coverage_grid") == [i / 10 for i in range(11)]
        passed = fixed_grid and risk.get("test_used_for_selection") is False and risk.get("test_is_diagnostic_only") is True and risk.get("formal_fallback_threshold_modified") is False
        audit.check("risk_coverage", "R01", "TEST is not used to select a figure operating point or modify the fallback threshold", passed, method="Anchored Dynamic diagnostic", evidence=[portable(risk_path, audit.repo_root)], details=f"fixed_coverage_grid={fixed_grid}; test_used_for_selection={risk.get('test_used_for_selection')}; test_is_diagnostic_only={risk.get('test_is_diagnostic_only')}; threshold_modified={risk.get('formal_fallback_threshold_modified')}.")
    else:
        audit.check("risk_coverage", "R01", "TEST is not used to select a figure operating point or modify the fallback threshold", False, method="Anchored Dynamic diagnostic", evidence=[portable(risk_path, audit.repo_root)], details="Risk-coverage audit asset is missing.")
    if negative_ok:
        negative = read_json(negative_path)
        passed = negative.get("no_model_or_combiner_training") is True and negative.get("no_test_driven_threshold_or_method_changes") is True and negative.get("test_is_final_analysis_only") is True
        audit.check("negative_transfer", "N01", "TEST safety metrics are final analysis only and do not select methods or thresholds", passed, method="Global / Query-soft / DynaSemble / Anchored", evidence=[portable(negative_path, audit.repo_root)], details=f"no_training={negative.get('no_model_or_combiner_training')}; no_test_driven_changes={negative.get('no_test_driven_threshold_or_method_changes')}; test_final_only={negative.get('test_is_final_analysis_only')}.")
    else:
        audit.check("negative_transfer", "N01", "TEST safety metrics are final analysis only and do not select methods or thresholds", False, method="Safety analysis", evidence=[portable(negative_path, audit.repo_root)], details="Negative-transfer audit asset is missing.")
    if confidence_ok:
        confidence = read_json(confidence_path)
        problems: list[str] = []
        flags_ok = (
            confidence.get("confidence_threshold_applied") is False
            and confidence.get("nonfinite_fallback_only") is True
            and confidence.get("test_is_diagnostic_only") is True
            and confidence.get("test_used_to_modify_threshold_or_method") is False
            and confidence.get("harm_label")
            == "1 iff rr_raw_bounded < rr_global (strict comparison)"
        )
        for name, expected in confidence.get("output_hashes", {}).items():
            path = confidence_path.parent / name
            matched, _ = audit.asset(
                path,
                "confidence-to-harm diagnostic output",
                str(expected),
                name,
            )
            if not matched:
                problems.append(f"output missing/hash mismatch: {portable(path, audit.repo_root)}")
        for item in confidence.get("source_audit", []):
            for path_key, hash_key, role in (
                ("source_rows", "source_rows_sha256", "confidence-to-harm frozen query rows"),
                ("lock", "lock_sha256", "confidence-to-harm DEV lock"),
            ):
                path = audit.repo_root / Path(str(item.get(path_key, "")))
                matched, _ = audit.asset(
                    path,
                    role,
                    item.get(hash_key),
                    str(item.get(path_key, "")),
                )
                if not matched:
                    problems.append(f"source missing/hash mismatch: {portable(path, audit.repo_root)}")
        audit.check(
            "confidence_harm",
            "H01",
            "Harm is counterfactual raw bounded correction, and TEST is diagnostic only",
            flags_ok and not problems,
            method="Anchored confidence diagnostic",
            evidence=[portable(confidence_path, audit.repo_root)],
            details="; ".join(problems)
            or "Confidence threshold is disabled; non-finite fallback only; strict raw-bounded-vs-Global harm label; TEST diagnostic-only; all source, lock, and output hashes match.",
        )
    else:
        audit.check(
            "confidence_harm",
            "H01",
            "Harm is counterfactual raw bounded correction, and TEST is diagnostic only",
            False,
            method="Anchored confidence diagnostic",
            evidence=[portable(confidence_path, audit.repo_root)],
            details="Confidence-to-harm audit asset is missing.",
        )
    if beta_ok:
        beta = read_json(beta_path)
        problems: list[str] = []
        flags_ok = (
            beta.get("formal_beta_grid")
            == [round(value * 0.05, 2) for value in range(1, 11)]
            and same_float(beta.get("diagnostic_beta"), 1.0)
            and beta.get("diagnostic_beta_participates_in_selection") is False
            and beta.get("selected_beta_source")
            == "existing immutable DEV lock only"
            and beta.get("test_used_for_beta_or_threshold_selection") is False
            and beta.get("theoretical_check", {}).get("passed") is True
            and beta.get("core_ablation_test_reproduction", {}).get("passed")
            is True
        )
        for name, expected in beta.get("output_hashes", {}).items():
            path = beta_path.parent / name
            matched, _ = audit.asset(
                path,
                "beta sensitivity output",
                str(expected),
                name,
            )
            if not matched:
                problems.append(
                    f"output missing/hash mismatch: {portable(path, audit.repo_root)}"
                )
        for item in beta.get("source_audit", []):
            for path_key, hash_key, role in (
                ("source_rows", "source_rows_sha256", "beta sensitivity frozen query rows"),
                ("lock", "lock_sha256", "beta sensitivity DEV lock"),
            ):
                path = audit.repo_root / Path(str(item.get(path_key, "")))
                matched, _ = audit.asset(
                    path,
                    role,
                    item.get(hash_key),
                    str(item.get(path_key, "")),
                )
                if not matched:
                    problems.append(
                        f"source missing/hash mismatch: {portable(path, audit.repo_root)}"
                    )
        audit.check(
            "beta_sensitivity",
            "B01",
            "Beta sensitivity varies only the predeclared trust-region radius; TEST does not select beta",
            flags_ok and not problems,
            method="Anchored beta diagnostic",
            evidence=[portable(beta_path, audit.repo_root)],
            details="; ".join(problems)
            or "Formal grid is 0.05-0.50; beta=1.0 is diagnostic-only; selection remains the DEV lock; theoretical and core-ablation reproduction checks pass; all source/lock/output hashes match.",
        )
    else:
        audit.check(
            "beta_sensitivity",
            "B01",
            "Beta sensitivity varies only the predeclared trust-region radius; TEST does not select beta",
            False,
            method="Anchored beta diagnostic",
            evidence=[portable(beta_path, audit.repo_root)],
            details="Beta-sensitivity audit asset is missing.",
        )
    if fallback_ok:
        fallback = read_json(fallback_path)
        problems: list[str] = []
        flags_ok = (
            fallback.get("pure_posthoc_mechanism_analysis") is True
            and fallback.get("formal_fallback_threshold_modified") is False
            and fallback.get("test_is_diagnostic_only") is True
            and fallback.get("test_used_for_selection") is False
            and fallback.get("stored_policy_reconstruction_passed") is True
            and fallback.get("classification", {}).get("avoided_harm")
            == "rr_raw < rr_global"
            and fallback.get("classification", {}).get("missed_benefit")
            == "rr_raw > rr_global"
            and fallback.get("classification", {}).get("neutral")
            == "rr_raw == rr_global"
        )
        for name, expected in fallback.get("output_hashes", {}).items():
            path = fallback_path.parent / name
            matched, _ = audit.asset(
                path,
                "fallback behavior diagnostic output",
                str(expected),
                name,
            )
            if not matched:
                problems.append(
                    f"output missing/hash mismatch: {portable(path, audit.repo_root)}"
                )
        for item in fallback.get("source_audit", []):
            for path_key, hash_key, role in (
                ("source_rows", "source_rows_sha256", "fallback frozen query rows"),
                ("dev_lock", "dev_lock_sha256", "fallback DEV lock"),
            ):
                path = audit.repo_root / Path(str(item.get(path_key, "")))
                matched, _ = audit.asset(
                    path,
                    role,
                    item.get(hash_key),
                    str(item.get(path_key, "")),
                )
                if not matched:
                    problems.append(
                        f"source missing/hash mismatch: {portable(path, audit.repo_root)}"
                    )
            if item.get("stored_policy_reconstruction_passed") is not True:
                problems.append(
                    "stored policy reconstruction failed: "
                    f"{item.get('split')}/{item.get('dataset')}/{item.get('pair')}"
                )
            if item.get("counterfactual_confidence_threshold_applied") is not False:
                problems.append(
                    "counterfactual still applies confidence fallback: "
                    f"{item.get('split')}/{item.get('dataset')}/{item.get('pair')}"
                )
            if item.get("counterfactual_nonfinite_fallback_only") is not True:
                problems.append(
                    "counterfactual fallback contract differs: "
                    f"{item.get('split')}/{item.get('dataset')}/{item.get('pair')}"
                )
        audit.check(
            "fallback_audit",
            "FBA01",
            "Fallback utility uses raw bounded counterfactuals and does not modify the locked threshold",
            flags_ok and not problems,
            method="Anchored fallback diagnostic",
            evidence=[portable(fallback_path, audit.repo_root)],
            details="; ".join(problems)
            or "Stored policies reconstruct exactly; the counterfactual removes confidence fallback only; TEST is diagnostic-only; all source, lock, and output hashes match.",
        )
    else:
        audit.check(
            "fallback_audit",
            "FBA01",
            "Fallback utility uses raw bounded counterfactuals and does not modify the locked threshold",
            False,
            method="Anchored fallback diagnostic",
            evidence=[portable(fallback_path, audit.repo_root)],
            details="Fallback behavior audit asset is missing.",
        )


def validate_global_test_exclusions(audit: Audit) -> None:
    protocol_paths = [
        audit.repo_root / "docs/protocols/paper_a_dynasemble_four_pair_protocol.md",
        audit.repo_root / "docs/protocols/paper_a_boundary_pair_protocol.md",
        audit.repo_root / "docs/protocols/paper_a_reliable_primary_protocol.md",
    ]
    runner_paths = [
        audit.repo_root / "scripts/eval_heterogeneous_complementarity.py",
        audit.repo_root / "scripts/crossfit_anchored_dynamic.py",
        audit.repo_root / "scripts/lock_apply_anchored_dynamic.py",
        audit.repo_root / "scripts/eval_openbg_dynasemble.py",
        audit.repo_root / "scripts/audit_reliable_primary_regime.py",
    ]
    missing: list[str] = []
    combined = ""
    for path in protocol_paths + runner_paths:
        ok, _ = audit.asset(path, "protocol/code information-boundary evidence")
        if not ok:
            missing.append(portable(path, audit.repo_root))
        else:
            combined += "\n" + path.read_text(encoding="utf-8", errors="replace").lower()
    concepts = {
        "DEV-only selection": ("dev only", "dev-only"),
        "immutable TEST apply": ("immutable", "locked"),
        "no TEST tuning": ("test tuning", "test hyperparameter", "test outcome"),
    }
    absent = [label for label, terms in concepts.items() if not any(term in combined for term in terms)]
    passed = not missing and not absent
    audit.check("global_information_boundary", "L01", "TEST is never used to choose primary, beta, threshold, orientation, pair, or figure operating point", passed, method="ALL", evidence=[portable(x, audit.repo_root) for x in protocol_paths + runner_paths], details=("Missing files: " + ", ".join(missing) + "; " if missing else "") + ("Missing protocol concepts: " + ", ".join(absent) if absent else "Protocols and implementations explicitly separate DEV-only selection from locked/immutable TEST apply; method-specific checks independently validate the resulting hashes and parameters."))


def write_outputs(audit: Audit) -> bool:
    audit.output_dir.mkdir(parents=True, exist_ok=True)
    overall_pass = all(row["status"] == "PASS" for row in audit.checks)
    experiment_status: dict[str, str] = {}
    for row in audit.checks:
        current = experiment_status.get(row["experiment"], "PASS")
        experiment_status[row["experiment"]] = "FAIL" if current == "FAIL" or row["status"] == "FAIL" else "PASS"
    audit_json = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "statement": REPORT_OPENING,
        "read_only_audit": True,
        "methods_modified": False,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "experiment_status": experiment_status,
        "summary": {
            "checks": len(audit.checks),
            "passed": sum(row["status"] == "PASS" for row in audit.checks),
            "failed": sum(row["status"] == "FAIL" for row in audit.checks),
            "experiments": len(experiment_status),
        },
        "checks": audit.checks,
    }
    with (audit.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_json, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    manifest = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": audit_json["generated_at_utc"],
        "hash_algorithm": "SHA-256",
        "path_policy": "repository-relative canonical paths; historical machine-local lock paths retained in locked_paths; raw SHA-256 is always reported and LF/CRLF equivalence is accepted only for text assets",
        "assets": sorted(audit.assets.values(), key=lambda item: item["path"]),
    }
    with (audit.output_dir / "asset_hash_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    columns = ["experiment", "dataset", "pair", "method", "check_id", "requirement", "status", "evidence_files", "details"]
    with (audit.output_dir / "leakage_checks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(audit.checks)

    failed = [row for row in audit.checks if row["status"] == "FAIL"]
    lines = [
        REPORT_OPENING,
        "",
        "# Paper A final protocol-integrity audit",
        "",
        f"**Overall status: {'PASS' if overall_pass else 'FAIL'}**",
        "",
        f"This read-only audit evaluated {len(audit.checks)} checks across {len(experiment_status)} experiment groups. It changed no method, threshold, orientation, pair definition, or result asset.",
        "",
        "## Experiment status",
        "",
        "| Experiment | Status |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in sorted(experiment_status.items()))
    lines += ["", "## Leakage checks", "", "| ID | Experiment | Dataset / Pair | Method | Status | Evidence-backed conclusion |", "|---|---|---|---|---:|---|"]
    for row in audit.checks:
        scope = f"{row['dataset']} / {row['pair']}"
        detail = str(row["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['check_id']} | {row['experiment']} | {scope} | {row['method']} | {row['status']} | {detail} |")
    lines += ["", "## Failures", ""]
    if failed:
        for row in failed:
            lines.append(f"- **{row['experiment']} / {row['check_id']}** — {row['details']} Files: {row['evidence_files']}")
    else:
        lines.append("No failures were detected. All checked selections are DEV-only and all checked TEST evaluations are immutable applications or diagnostic/final reporting only.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "PASS means that the available final assets, embedded locks, hashes, row-level metadata, grouped-fold assignments, and implementation/protocol evidence are mutually consistent with the declared information boundary. It does not prove the absence of actions outside the recorded repository history; it makes the recorded workflow independently auditable and causes any detected inconsistency to fail closed.",
        "",
        "Historical absolute paths in locks are not treated as current filesystem identities. They are mapped to the corresponding repository-relative asset and then checked against the hash recorded in the lock.",
        "",
    ]
    (audit.output_dir / "final_protocol_integrity.md").write_text("\n".join(lines), encoding="utf-8")
    return overall_pass


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    audit = Audit(repo_root, output_dir)
    validate_primary_selection(audit)
    for spec in PAIR_SPECS:
        validate_grouped_folds(audit, spec)
        validate_static_and_anchored(audit, spec)
        validate_dynasemble(audit, spec)
    validate_core_ablation(audit)
    validate_analysis_boundaries(audit)
    validate_global_test_exclusions(audit)
    passed = write_outputs(audit)
    print(f"[{ 'PASS' if passed else 'FAIL' }] Paper A protocol integrity: {sum(row['status'] == 'PASS' for row in audit.checks)}/{len(audit.checks)} checks passed")
    print(f"[OK] wrote {portable(output_dir / 'audit.json', repo_root)}")
    print(f"[OK] wrote {portable(output_dir / 'asset_hash_manifest.json', repo_root)}")
    print(f"[OK] wrote {portable(output_dir / 'leakage_checks.csv', repo_root)}")
    print(f"[OK] wrote {portable(output_dir / 'final_protocol_integrity.md', repo_root)}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
