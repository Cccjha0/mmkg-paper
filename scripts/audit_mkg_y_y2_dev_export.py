from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or verify the frozen MKG-Y Y2 DEV full-ranking export."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=("preflight", "verify"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_normalized(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_hash(path: Path, expected: str, mode: str = "raw") -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if mode == "raw":
        actual = sha256_file(path)
    elif mode == "lf_normalized":
        actual = sha256_lf_normalized(path)
    else:
        raise ValueError(f"Unsupported hash mode: {mode}")
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {portable(path)}: {actual} != {expected}")
    return actual


def alpha_column(alpha: float) -> str:
    return f"rr_alpha_{alpha:.2f}".replace(".", "_")


def pair_output_paths(directory: Path) -> list[Path]:
    paths = [
        directory / "dev_query_rows.csv",
        directory / "selection.json",
        directory / "dev_summary.json",
        directory / "dev_results.csv",
        directory / "dev_results_by_seed.csv",
        directory / "dev_results.md",
    ]
    paths.extend(
        directory / "checkpoints" / f"dev_seed{seed}_{direction}.csv"
        for seed in (1, 2, 3)
        for direction in ("head", "tail")
    )
    return paths


def preflight(contract: dict) -> dict:
    if contract.get("status") != "frozen_before_y2_export":
        raise RuntimeError("Y2 contract is not frozen")
    if contract.get("split") != "dev":
        raise RuntimeError("Y2 permits DEV only")
    if contract.get("test_policy", {}).get("test_rows_opened") != 0:
        raise RuntimeError("Y2 TEST lock is not explicit")

    source_hashes = contract["frozen_source_hashes"]
    for name, specification in source_hashes.items():
        verify_hash(
            repo_path(name),
            specification["sha256"],
            specification.get("mode", "raw"),
        )

    acceptance = read_json(repo_path(contract["acceptance_decision"]))
    if acceptance.get("progression") != "Y2_DEV_EXPORT_ALLOWED":
        raise RuntimeError("Y0/Y1 acceptance does not allow Y2")
    if acceptance.get("test_status") != "LOCKED":
        raise RuntimeError("Acceptance decision did not retain TEST lock")

    expected_inventory = {}
    with repo_path(contract["checkpoint_inventory"]).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected_inventory[(row["model_key"], int(row["seed"]))] = row

    opened_sources = set(source_hashes)
    for model_key, model in contract["experts"].items():
        for seed_text, run in model["runs"].items():
            seed = int(seed_text)
            inventory = expected_inventory.get((model_key, seed))
            if inventory is None or inventory.get("accepted") != "True":
                raise RuntimeError(f"Checkpoint is not accepted: {model_key}/seed{seed}")
            if inventory["run_dir"].replace("\\", "/") != run["run_dir"]:
                raise RuntimeError(f"Run directory drift: {model_key}/seed{seed}")
            checkpoint = repo_path(run["run_dir"]) / "best.ckpt"
            config = repo_path(run["run_dir"]) / "config_merged.json"
            verify_hash(checkpoint, run["checkpoint_sha256"])
            cfg = read_json(config)
            if int(cfg.get("system", {}).get("seed", -1)) != seed:
                raise RuntimeError(f"Config seed mismatch: {model_key}/seed{seed}")
            if bool(cfg.get("evaluation", {}).get("run_test", True)):
                raise RuntimeError(f"run_test must remain false: {model_key}/seed{seed}")
            opened_sources.update((portable(checkpoint), portable(config)))

    forbidden = [value for value in opened_sources if Path(value).name.lower().startswith("test")]
    if forbidden:
        raise RuntimeError(f"Preflight source inventory contains TEST row paths: {forbidden}")
    return {
        "schema_version": 1,
        "study": contract["study"],
        "status": "PASS",
        "acceptance_progression": acceptance["progression"],
        "split": "dev",
        "filter_fact_scope": contract["ranking_protocol"]["filter_fact_scope"],
        "checkpoint_count": sum(len(value["runs"]) for value in contract["experts"].values()),
        "pair_count": len(contract["pairs"]),
        "test_rows_opened": 0,
        "checkpoint_retraining": 0,
        "checkpoint_reselection": 0,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_pair(pair: dict, contract: dict) -> tuple[dict, dict]:
    directory = repo_path(pair["output_dir"])
    query_path = directory / "dev_query_rows.csv"
    selection_path = directory / "selection.json"
    summary_path = directory / "dev_summary.json"
    output_paths = pair_output_paths(directory)
    for path in output_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    expected_alphas = [round(index * 0.05, 2) for index in range(21)]
    selection = read_json(selection_path)
    summary = read_json(summary_path)
    expected_rows = int(contract["dataset_counts"]["valid"]) * 2 * 3
    for payload, label in ((selection, "selection"), (summary, "summary")):
        if payload.get("pair_name") != pair["pair_name"]:
            raise RuntimeError(f"{pair['id']} {label} pair_name mismatch")
        if payload.get("dataset") != "mkg_y":
            raise RuntimeError(f"{pair['id']} {label} dataset mismatch")
        if payload.get("seeds") != [1, 2, 3]:
            raise RuntimeError(f"{pair['id']} {label} seed mismatch")
    if selection.get("alpha_grid") != expected_alphas:
        raise RuntimeError(f"{pair['id']} alpha grid drift")
    if summary.get("split") != "dev" or summary.get("n_rows") != expected_rows:
        raise RuntimeError(f"{pair['id']} DEV row declaration mismatch")
    if summary.get("dev_only_no_test_access") is not True:
        raise RuntimeError(f"{pair['id']} missing no-TEST-access declaration")
    if summary.get("test_rows_opened") != 0 or summary.get("filter_fact_scope") != "train_dev":
        raise RuntimeError(f"{pair['id']} TEST boundary/filter scope mismatch")

    counts: Counter[tuple[int, str]] = Counter()
    seen_ids = set()
    required_alpha_columns = {alpha_column(alpha) for alpha in expected_alphas}
    with query_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not required_alpha_columns.issubset(fields):
            raise RuntimeError(f"{pair['id']} is missing alpha-grid RR columns")
        for row in reader:
            if row["split"] != "dev" or row["filter_fact_scope"] != "train_dev":
                raise RuntimeError(f"{pair['id']} contains an invalid split/filter row")
            seed = int(row["seed"])
            direction = row["direction"]
            if seed not in (1, 2, 3) or direction not in ("head", "tail"):
                raise RuntimeError(f"{pair['id']} contains an invalid seed/direction")
            query_id = row["query_id"]
            if query_id in seen_ids:
                raise RuntimeError(f"{pair['id']} contains duplicate query_id: {query_id}")
            seen_ids.add(query_id)
            counts[(seed, direction)] += 1
            for column in required_alpha_columns:
                value = float(row[column])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise RuntimeError(f"{pair['id']} invalid reciprocal rank in {column}")
    expected_unit = int(contract["dataset_counts"]["valid"])
    expected_counts = {(seed, direction): expected_unit for seed in (1, 2, 3) for direction in ("head", "tail")}
    if counts != Counter(expected_counts):
        raise RuntimeError(f"{pair['id']} seed/direction coverage mismatch: {dict(counts)}")

    sources = {portable(path): sha256_file(path) for path in output_paths}
    pair_result = {
        "pair_id": pair["id"],
        "pair_name": pair["pair_name"],
        "status": "PASS",
        "rows": expected_rows,
        "queries_per_seed_direction": expected_unit,
        "seeds": [1, 2, 3],
        "directions": ["head", "tail"],
        "alpha_grid": expected_alphas,
        "score_normalization": "query_zscore",
        "filter_fact_scope": "train_dev",
        "test_rows_opened": 0,
        "global_alpha": selection["global_alpha"],
    }
    source_manifest = {
        "schema_version": 1,
        "pair": pair_result,
        "files": sources,
        "accepted_runs": {
            model_key: contract["experts"][model_key]["runs"]
            for model_key in pair["experts"]
        },
    }
    return pair_result, source_manifest


def verify_outputs(contract: dict, output_dir: Path, overwrite: bool) -> None:
    preflight_result = preflight(contract)
    pair_results = []
    generated = []
    for pair in contract["pairs"]:
        result, manifest = verify_pair(pair, contract)
        pair_results.append(result)
        path = output_dir / f"{pair['id']}_dev_source_manifest.json"
        write_json(path, manifest, overwrite)
        generated.append(path)

    verification_path = output_dir / "export_verification.json"
    verification = {
        "schema_version": 1,
        "study": contract["study"],
        "status": "Y2_DEV_EXPORT_VERIFIED",
        "pairs": pair_results,
        "test_access": 0,
        "checkpoint_retraining": 0,
        "checkpoint_reselection": 0,
        "checkpoint_modification": 0,
    }
    write_json(verification_path, verification, overwrite)
    generated.append(verification_path)

    contract_path = repo_path(contract["contract_path"])
    audit_path = output_dir / "audit_manifest.json"
    dataset_directory = repo_path("data/datasets/mkg_y/processed")
    dataset_sources = {
        dataset_directory / name
        for name in (
            "manifest.json",
            "train.tsv",
            "valid.tsv",
            "entity2id.json",
            "relation2id.json",
            "text_feat.pt",
            "img_feat.pt",
            "has_text.pt",
            "has_img.pt",
        )
    }
    source_paths = {
        contract_path,
        repo_path(contract["acceptance_decision"]),
        repo_path(contract["checkpoint_inventory"]),
        repo_path("scripts/eval_heterogeneous_complementarity.py"),
        repo_path("scripts/dev_only_dataset_loader.py"),
        repo_path("scripts/audit_mkg_y_y2_dev_export.py"),
        repo_path("scripts/run_mkg_y_y2_dev_full_ranking.ps1"),
        *[repo_path(run["run_dir"]) / "best.ckpt" for model in contract["experts"].values() for run in model["runs"].values()],
        *[repo_path(run["run_dir"]) / "config_merged.json" for model in contract["experts"].values() for run in model["runs"].values()],
        *dataset_sources,
        *[path for pair in contract["pairs"] for path in pair_output_paths(repo_path(pair["output_dir"]))],
        *generated,
    }
    audit = {
        "schema_version": 1,
        "study": contract["study"],
        "status": "PASS",
        "preflight": preflight_result,
        "source_and_output_hashes": {
            portable(path): sha256_file(path) for path in sorted(source_paths, key=lambda value: portable(value))
        },
        "audit_manifest_self_hash_excluded": True,
        "test_access": 0,
    }
    write_json(audit_path, audit, overwrite)
    print(f"[OK] verified {len(pair_results)} MKG-Y Y2 DEV pair exports")
    print(f"[OK] wrote {portable(audit_path)}")


def main() -> None:
    args = parse_args()
    contract_path = repo_path(args.contract)
    contract = read_json(contract_path)
    output_dir = repo_path(args.output_dir)
    if args.mode == "preflight":
        payload = preflight(contract)
        write_json(output_dir / "preflight.json", payload, args.overwrite)
        print("[OK] MKG-Y Y2 preflight passed; TEST remains locked")
    else:
        verify_outputs(contract, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
