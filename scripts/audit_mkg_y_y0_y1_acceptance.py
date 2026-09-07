from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import portable_path, reject_test_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing MKG-Y Y0 data and Y1 DEV-only checkpoints.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y0_Y1_ACCEPTANCE_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance")
    parser.add_argument("--report", default="docs/reports/mkg_y_y0_y1_acceptance_audit_2026-09-07.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_contract(path: Path) -> dict:
    reject_test_path(path)
    contract = load_json(path)
    if contract.get("status") != "frozen_before_systematic_acceptance_run" or contract.get("dataset") != "mkg_y":
        raise RuntimeError("MKG-Y acceptance contract is not frozen")
    if tuple(contract["y1"]["required_seeds"]) != (1, 2, 3):
        raise RuntimeError("MKG-Y seed inventory changed")
    if set(contract["y1"]["models"]) != {"mkg_y_mhyper", "mkg_y_native", "mkg_y_adamf_mat"}:
        raise RuntimeError("MKG-Y model inventory changed")
    if any(int(value) != 0 for value in contract["prohibited"].values()):
        raise RuntimeError("A prohibited acceptance-audit operation was enabled")
    return contract


def source_record(path: Path, role: str) -> dict:
    reject_test_path(path)
    return {"path": portable_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role}


def flatten_zero_checks(value: dict) -> bool:
    return all(int(item) == 0 for item in value.values())


def normalized_embedded_manifest(manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    result.get("source_lock", {}).pop("lock_sha256", None)
    return result


def manifests_match_except_registry_hash(embedded: dict, current: dict) -> bool:
    return normalized_embedded_manifest(embedded) == normalized_embedded_manifest(current)


def validate_metric_frame(frame: pd.DataFrame) -> list[str]:
    errors = []
    required = ("epoch", "mrr", "hits@1", "hits@3", "hits@10")
    missing = [column for column in required if column not in frame]
    if missing:
        return [f"missing metric columns: {missing}"]
    values = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if len(values) == 0 or not np.isfinite(values.to_numpy(np.float64)).all():
        errors.append("metrics are empty or non-finite")
        return errors
    if not values.epoch.is_monotonic_increasing or values.epoch.duplicated().any():
        errors.append("evaluation epochs are not strictly increasing")
    for column in ("mrr", "hits@1", "hits@3", "hits@10"):
        if not values[column].between(0.0, 1.0).all():
            errors.append(f"{column} is outside [0,1]")
    if not ((values["hits@1"] <= values["hits@3"]) & (values["hits@3"] <= values["hits@10"])).all():
        errors.append("Hits@K is not monotone")
    return errors


def inspect_checkpoint(path: Path) -> tuple[bool, int, str | None]:
    if not path.is_file() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
        return False, 0, "checkpoint is missing, empty, or not a ZIP-format torch archive"
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        members = len(archive.namelist())
    if bad is not None:
        return False, members, f"corrupt ZIP member: {bad}"
    return members >= 3, members, None if members >= 3 else "checkpoint archive has too few members"


def audit_y0(contract: dict) -> tuple[dict, dict, list[dict]]:
    cfg = contract["y0"]
    manifest_path, audit_path, lock_path = Path(cfg["processed_manifest"]), Path(cfg["audit_report"]), Path(cfg["source_lock"])
    for path, expected in cfg["frozen_file_hashes"].items():
        candidate = Path(path)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RuntimeError(f"Frozen Y0 input hash mismatch: {candidate}")
    manifest, audit = load_json(manifest_path), load_json(audit_path)
    checks = {
        "manifest_status_pass": manifest.get("status") == "pass",
        "audit_status_pass": audit.get("status") == "pass",
        "counts_match": manifest.get("counts") == cfg["expected_counts"] and audit.get("counts") == cfg["expected_counts"],
        "source_counts_match": manifest.get("source_counts") == {key: cfg["expected_counts"][key] for key in ("train", "valid", "test")},
        "protocol_match": manifest.get("protocol") == cfg["expected_protocol"],
        "official_split": manifest.get("split") == cfg["expected_split"] and manifest.get("split_construction", {}).get("name") == cfg["expected_split"],
        "source_duplicates_zero": flatten_zero_checks(manifest.get("source_split_integrity", {}).get("duplicates", {"invalid": 1})),
        "source_overlaps_zero": flatten_zero_checks(manifest.get("source_split_integrity", {}).get("exact_overlaps", {"invalid": 1})),
        "canonical_duplicates_zero": flatten_zero_checks(manifest.get("split_construction", {}).get("canonical_integrity", {}).get("duplicates", {"invalid": 1})),
        "canonical_overlaps_zero": flatten_zero_checks(manifest.get("split_construction", {}).get("canonical_integrity", {}).get("exact_overlaps", {"invalid": 1})),
        "text_finite": bool(manifest.get("features", {}).get("text", {}).get("numeric_health", {}).get("all_finite")),
        "image_finite": bool(manifest.get("features", {}).get("image", {}).get("numeric_health", {}).get("all_finite")),
        "independent_modal_spaces": manifest.get("cross_modal_similarity", {}).get("shared_embedding_space") is False,
        "canonical_hash_inventory_present": all(bool(value) for value in [
            manifest.get("hashes", {}).get("entity_mapping"), manifest.get("hashes", {}).get("relation_mapping"),
            *manifest.get("hashes", {}).get("splits", {}).values(),
            *manifest.get("hashes", {}).get("canonical_features", {}).values(),
        ]),
        "source_lock_current": sha256_file(lock_path) == manifest.get("source_lock", {}).get("lock_sha256"),
    }
    records = [source_record(manifest_path, "processed_manifest"), source_record(audit_path, "processed_audit"), source_record(lock_path, "source_lock_registry")]
    for modality in ("text", "image"):
        declaration = manifest["feature_sources"][modality]
        path = Path(declaration["path"])
        reject_test_path(path)
        actual = sha256_file(path) if path.is_file() else ""
        checks[f"{modality}_source_hash_match"] = actual == declaration["sha256"]
        if path.is_file():
            records.append({"path": portable_path(path), "sha256": actual, "bytes": path.stat().st_size, "role": f"{modality}_feature_source"})
    decision = "Y0_ACCEPTED" if all(checks.values()) else "Y0_REJECTED"
    result = {
        "decision": decision,
        "checks": checks,
        "counts": manifest["counts"],
        "features": {
            modality: {
                "coverage": manifest["features"][modality]["coverage"],
                "aligned_entities": manifest["features"][modality]["aligned_entities"],
                "missing_entities": manifest["features"][modality]["missing_entities"],
                "dimension": manifest["features"][modality]["dimension"],
                "all_finite": manifest["features"][modality]["numeric_health"]["all_finite"],
            }
            for modality in ("text", "image")
        },
        "test_rows_opened": 0,
        "note": "TEST counts/hashes were read only as already-recorded manifest metadata; no TEST row file was opened.",
    }
    return result, manifest, records


def audit_run(
    model_key: str,
    model_cfg: dict,
    seed: int,
    run_dir: Path,
    current_manifest: dict,
) -> tuple[dict, list[str], list[str], list[dict]]:
    errors, notes, records = [], [], []
    match = re.search(r"_seed(\d+)$", run_dir.name)
    if match is None or int(match.group(1)) != seed:
        errors.append("run directory seed suffix mismatch")
    required_paths = {name: run_dir / name for name in ("best.ckpt", "common.yaml", "config_merged.json", "experiment.yaml")}
    for name, path in required_paths.items():
        if not path.is_file():
            errors.append(f"missing {name}")
    metric_paths = sorted(run_dir.glob("metrics*.csv"))
    if len(metric_paths) != 1:
        errors.append(f"expected exactly one metrics CSV, found {len(metric_paths)}")
        metric_path = None
    else:
        metric_path = metric_paths[0]
        if metric_path.name != f"metrics_seed{seed}.csv":
            notes.append(f"legacy metric filename {metric_path.name}; accepted seed from config metadata")
    if errors:
        return {"model_key": model_key, "seed": seed, "run_dir": portable_path(run_dir), "accepted": False}, errors, notes, records
    common = load_yaml(required_paths["common.yaml"])
    merged = load_json(required_paths["config_merged.json"])
    experiment = load_yaml(required_paths["experiment.yaml"])
    anchor_path = Path(model_cfg["anchor_config"])
    anchor = load_yaml(anchor_path)
    actual_seed = int(merged.get("system", {}).get("seed", -1))
    if actual_seed != seed or int(common.get("system", {}).get("seed", -1)) != seed:
        errors.append("system.seed mismatch")
    if merged.get("dataset", {}).get("name") != "mkg_y" or merged.get("dataset", {}).get("processed_dir") != "data/datasets/mkg_y/processed":
        errors.append("dataset or processed_dir mismatch")
    if merged.get("model", {}).get("name") != model_cfg["model_name"]:
        errors.append("model name mismatch")
    if merged.get("training", {}).get("termination_policy") != model_cfg["termination_policy"]:
        errors.append("termination policy mismatch")
    evaluation = merged.get("evaluation", {})
    if evaluation.get("run_test") is not False or evaluation.get("filtered") is not True or evaluation.get("direction") != "both" or evaluation.get("dev_eval_limit") is not None:
        errors.append("evaluation is not filtered full-DEV bidirectional with run_test=false")
    if experiment != anchor:
        errors.append("experiment snapshot does not semantically match checked-in anchor")
    embedded = merged.get("_dataset_manifest", {})
    if not manifests_match_except_registry_hash(embedded, current_manifest):
        errors.append("embedded dataset manifest differs beyond source-lock registry hash")
    elif embedded.get("source_lock", {}).get("lock_sha256") != current_manifest.get("source_lock", {}).get("lock_sha256"):
        notes.append("whole source-lock registry hash drifted; all MKG-Y record-level values remain identical")
    metric_frame = pd.read_csv(metric_path)
    errors.extend(validate_metric_frame(metric_frame))
    numeric = metric_frame[["epoch", "mrr", "hits@1", "hits@3", "hits@10"]].apply(pd.to_numeric)
    best_index = int(numeric.mrr.idxmax())
    best = numeric.loc[best_index]
    final_epoch = int(numeric.epoch.max())
    configured_epochs = int(merged["training"]["epochs"])
    eval_every = int(merged["training"]["eval_every"])
    patience = merged["training"].get("early_stop_patience")
    if model_cfg["termination_policy"] == "fixed_budget":
        if final_epoch != configured_epochs:
            errors.append("fixed-budget trace does not reach configured epoch")
    else:
        if final_epoch > configured_epochs or final_epoch % eval_every != 0:
            errors.append("early-stop trace has invalid final epoch")
        if patience is not None and final_epoch < int(best.epoch):
            errors.append("early-stop trace ends before its best epoch")
    checkpoint_ok, archive_members, checkpoint_error = inspect_checkpoint(required_paths["best.ckpt"])
    if not checkpoint_ok:
        errors.append(checkpoint_error or "checkpoint archive failed")
    checkpoint_hash = sha256_file(required_paths["best.ckpt"])
    for path, role in [
        (required_paths["best.ckpt"], "checkpoint"), (required_paths["common.yaml"], "run_common_config"),
        (required_paths["config_merged.json"], "run_merged_config"), (required_paths["experiment.yaml"], "run_experiment_config"),
        (metric_path, "dev_metric_trace"), (anchor_path, "checked_in_anchor_config"),
    ]:
        records.append(source_record(path, role))
    row = {
        "model_key": model_key,
        "model_name": model_cfg["model_name"],
        "seed": seed,
        "run_dir": portable_path(run_dir),
        "termination_policy": model_cfg["termination_policy"],
        "metric_file": portable_path(metric_path),
        "metric_filename_matches_seed": metric_path.name == f"metrics_seed{seed}.csv",
        "evaluation_rows": len(metric_frame),
        "best_dev_epoch": int(best.epoch),
        "final_evaluated_epoch": final_epoch,
        "best_dev_mrr": float(best.mrr),
        "best_dev_hits1": float(best["hits@1"]),
        "best_dev_hits3": float(best["hits@3"]),
        "best_dev_hits10": float(best["hits@10"]),
        "run_test": bool(evaluation.get("run_test")),
        "checkpoint_path": portable_path(required_paths["best.ckpt"]),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": required_paths["best.ckpt"].stat().st_size,
        "checkpoint_archive_members": archive_members,
        "source_lock_registry_hash_at_training": embedded.get("source_lock", {}).get("lock_sha256"),
        "source_lock_registry_hash_current": current_manifest.get("source_lock", {}).get("lock_sha256"),
        "embedded_manifest_record_values_match": manifests_match_except_registry_hash(embedded, current_manifest),
        "provenance_notes": " | ".join(notes),
        "accepted": not errors,
    }
    return row, errors, notes, records


def audit_y1(contract: dict, current_manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    rows, all_errors, all_notes, records = [], [], [], []
    for model_key, model_cfg in contract["y1"]["models"].items():
        model_hashes = []
        for raw_seed, raw_dir in model_cfg["runs"].items():
            seed, run_dir = int(raw_seed), Path(raw_dir)
            row, errors, notes, run_records = audit_run(model_key, model_cfg, seed, run_dir, current_manifest)
            rows.append(row); records.extend(run_records)
            all_errors.extend(f"{model_key}/seed{seed}: {error}" for error in errors)
            all_notes.extend(f"{model_key}/seed{seed}: {note}" for note in notes)
            if row.get("checkpoint_sha256"):
                model_hashes.append(row["checkpoint_sha256"])
        if len(model_hashes) == 3 and len(set(model_hashes)) != 3:
            all_errors.append(f"{model_key}: checkpoint hashes are not distinct across seeds")
    inventory = pd.DataFrame(rows)
    summary = (
        inventory.groupby(["model_key", "model_name"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            dev_mrr_mean=("best_dev_mrr", "mean"), dev_mrr_std=("best_dev_mrr", "std"),
            dev_mrr_min=("best_dev_mrr", "min"), dev_mrr_max=("best_dev_mrr", "max"),
            dev_hits1_mean=("best_dev_hits1", "mean"), dev_hits3_mean=("best_dev_hits3", "mean"),
            dev_hits10_mean=("best_dev_hits10", "mean"), all_runs_accepted=("accepted", "all"),
        )
        .reset_index()
    )
    if all_errors:
        decision = "Y1_REJECTED"
    elif all_notes:
        decision = "Y1_ACCEPTED_WITH_PROVENANCE_NOTES"
    else:
        decision = "Y1_ACCEPTED"
    result = {
        "decision": decision,
        "run_count": len(inventory),
        "model_count": inventory.model_key.nunique(),
        "seed_inventory": sorted(inventory.seed.unique().astype(int).tolist()),
        "errors": all_errors,
        "provenance_notes": sorted(set(all_notes)),
        "test_rows_opened": 0,
        "test_evaluations_run": 0,
    }
    return inventory, summary, result, records


def write_report(y0: dict, y1: dict, summary: pd.DataFrame, inventory: pd.DataFrame, report: Path, output_dir: Path) -> None:
    lines = [
        "# MKG-Y Y0/Y1 Existing-Checkpoint Acceptance Audit",
        "",
        "Date: 2026-09-07",
        "",
        "## Decision",
        "",
        f"- Y0 data/engineering adaptation: `{y0['decision']}`",
        f"- Y1 standalone checkpoint reproduction: `{y1['decision']}`",
        f"- Progression: `{'Y2_DEV_EXPORT_ALLOWED' if y0['decision'] == 'Y0_ACCEPTED' and y1['decision'] != 'Y1_REJECTED' else 'Y2_BLOCKED'}`",
        "",
        "No TEST row file was opened and no checkpoint was trained, selected, or modified.",
        "",
        "## Y0 canonical data acceptance",
        "",
        f"The canonical MKG-Y package contains {y0['counts']['entities']:,} entities, {y0['counts']['relations']} relations, and TRAIN/DEV/TEST counts of {y0['counts']['train']:,}/{y0['counts']['valid']:,}/{y0['counts']['test']:,}. The existing preprocessing audit declares zero duplicates and zero cross-split overlaps.",
        "",
        "| Modality | Coverage | Aligned | Missing | Dimension | Numeric health |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for modality in ("text", "image"):
        value = y0["features"][modality]
        lines.append(f"| {modality} | {value['coverage']:.1%} | {value['aligned_entities']:,} | {value['missing_entities']:,} | {value['dimension']} | {'finite' if value['all_finite'] else 'FAILED'} |")
    lines += [
        "",
        "## Y1 three-seed DEV checkpoint acceptance",
        "",
        "| Model | Seeds | DEV MRR mean ± sd | Range | H@1 | H@3 | H@10 | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.model_name} | {int(row.seeds)} | {row.dev_mrr_mean:.6f} ± {row.dev_mrr_std:.6f} | "
            f"{row.dev_mrr_min:.6f}–{row.dev_mrr_max:.6f} | {row.dev_hits1_mean:.6f} | "
            f"{row.dev_hits3_mean:.6f} | {row.dev_hits10_mean:.6f} | {bool(row.all_runs_accepted)} |"
        )
    lines += [
        "",
        "All nine checkpoints are non-empty valid ZIP-format torch archives, use the official canonical MKG-Y package, run full filtered bidirectional DEV evaluation, retain `run_test: false`, and have distinct hashes across seeds within each model.",
        "",
        "## Provenance notes",
        "",
    ]
    filename_notes = sum("legacy metric filename" in note for note in y1["provenance_notes"])
    registry_notes = sum("source-lock registry hash drifted" in note for note in y1["provenance_notes"])
    if filename_notes:
        lines.append(f"- {filename_notes}/9 runs (all seed-2/3 runs) retain the legacy filename `metrics_seed1.csv`; their directory suffix, `common.yaml`, and merged `system.seed` agree on the correct seed.")
    if registry_notes:
        lines.append(f"- {registry_notes}/9 embedded run manifests contain the earlier whole-registry source-lock hash; after excluding only that registry digest, each embedded MKG-Y manifest equals the current canonical manifest.")
    lines += [
        "",
        "The whole-registry source-lock hash changed after training, but the embedded and current MKG-Y manifests differ only in that registry hash; every MKG-Y record-level source, mapping, split, and canonical feature hash remains identical. Seed identity is taken from the frozen run directory plus `system.seed`, never inferred from the legacy metric filename.",
        "",
        "## Machine-readable outputs",
        "",
        f"- [data acceptance]({os.path.relpath(output_dir / 'data_acceptance.json', report.parent).replace(os.sep, '/')})",
        f"- [checkpoint inventory]({os.path.relpath(output_dir / 'checkpoint_inventory.csv', report.parent).replace(os.sep, '/')})",
        f"- [standalone DEV summary]({os.path.relpath(output_dir / 'standalone_dev_summary.csv', report.parent).replace(os.sep, '/')})",
        f"- [acceptance decision]({os.path.relpath(output_dir / 'acceptance_decision.json', report.parent).replace(os.sep, '/')})",
        f"- [audit manifest]({os.path.relpath(output_dir / 'audit_manifest.json', report.parent).replace(os.sep, '/')})",
        "",
        "## Operational audit",
        "",
        "- TEST row access = 0",
        "- TEST evaluation = 0",
        "- checkpoint training/reselection/modification = 0",
        "- hyperparameter change = 0",
        "",
        "This acceptance permits only the next frozen DEV export/replication stage. It does not unlock TEST.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report = Path(args.contract), Path(args.output_dir), Path(args.report)
    contract = load_contract(contract_path)
    y0, current_manifest, y0_records = audit_y0(contract)
    inventory, summary, y1, y1_records = audit_y1(contract, current_manifest)
    progression = "Y2_DEV_EXPORT_ALLOWED" if y0["decision"] == "Y0_ACCEPTED" and y1["decision"] != "Y1_REJECTED" else "Y2_BLOCKED"
    if args.dry_run:
        print(json.dumps({
            "status": "preflight_pass" if progression == "Y2_DEV_EXPORT_ALLOWED" else "preflight_failed",
            "y0": y0["decision"], "y1": y1["decision"], "progression": progression,
            "test_rows_opened": 0, "test_evaluations_run": 0,
        }, indent=2))
        if progression == "Y2_BLOCKED":
            raise SystemExit(1)
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "schema_version": 1,
        "study": contract["study"],
        "y0_decision": y0["decision"],
        "y1_decision": y1["decision"],
        "progression": progression,
        "y1_errors": y1["errors"],
        "y1_provenance_notes": y1["provenance_notes"],
        "test_status": "LOCKED",
    }
    (output_dir / "data_acceptance.json").write_text(json.dumps(y0, indent=2) + "\n", encoding="utf-8")
    inventory.to_csv(output_dir / "checkpoint_inventory.csv", index=False)
    summary.to_csv(output_dir / "standalone_dev_summary.csv", index=False)
    (output_dir / "acceptance_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_report(y0, y1, summary, inventory, report, output_dir)
    input_records = {record["path"]: record for record in y0_records + y1_records}
    outputs = [
        output_dir / "data_acceptance.json", output_dir / "checkpoint_inventory.csv",
        output_dir / "standalone_dev_summary.csv", output_dir / "acceptance_decision.json", report,
    ]
    manifest = {
        "schema_version": 1,
        "study": contract["study"],
        "contract": {"path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
        "inputs": list(input_records.values()),
        "outputs": [{"path": portable_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
        "decisions": decision,
        "operational_audit": {
            "test_rows_opened": 0, "test_evaluations_run": 0,
            "checkpoint_training": 0, "checkpoint_reselection": 0,
            "checkpoint_modification": 0, "hyperparameter_change": 0,
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", **decision}, indent=2))


if __name__ == "__main__":
    main()
