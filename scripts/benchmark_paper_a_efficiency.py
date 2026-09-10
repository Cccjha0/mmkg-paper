#!/usr/bin/env python3
"""Benchmark Paper A base scoring separately from score-combination overhead."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


METHODS = (
    "Primary only",
    "Primary + Secondary + Global alpha",
    "Primary + Secondary + DynaSemble",
    "Primary + Secondary + Anchored Dynamic",
)
PAIR_SPECS = (
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
        "entity_count": 15000,
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
        "entity_count": 12842,
    },
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
        "entity_count": 15000,
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
        "entity_count": 12842,
    },
)
TIME_FIELDS = (
    "feature_computation_s",
    "combiner_inference_s",
    "score_combination_s",
    "combiner_only_overhead_s",
    "scoring_s",
    "total_s",
    "queries_per_second",
    "peak_gpu_memory_mb",
    "peak_cpu_memory_mb",
)
RAW_COLUMNS = (
    "dataset",
    "pair",
    "method",
    "seed",
    "repetition",
    "measurement_scope",
    "n_queries",
    "n_test_queries_available",
    "entity_count",
    "primary_query_batch_size",
    "secondary_query_batch_size",
    "outer_query_batch_size",
    "primary_params",
    "secondary_params",
    "base_params",
    "combiner_params",
    *TIME_FIELDS,
    "result_checksum",
)
SUMMARY_COLUMNS = (
    "dataset",
    "pair",
    "method",
    "measurement_scope",
    "n_seed_repetitions",
    "n_queries_per_repetition",
    "n_test_queries_available",
    "entity_count",
    "primary_query_batch_size",
    "secondary_query_batch_size",
    "outer_query_batch_size",
    "base_params",
    "combiner_params",
    "scoring_time_mean_s",
    "scoring_time_std_s",
    "feature_time_mean_s",
    "feature_time_std_s",
    "combiner_inference_time_mean_s",
    "combiner_inference_time_std_s",
    "score_combination_time_mean_s",
    "score_combination_time_std_s",
    "combiner_only_overhead_mean_s",
    "combiner_only_overhead_std_s",
    "total_time_mean_s",
    "total_time_std_s",
    "queries_per_second_mean",
    "queries_per_second_std",
    "peak_gpu_memory_mean_mb",
    "peak_gpu_memory_std_mb",
    "peak_cpu_memory_mean_mb",
    "peak_cpu_memory_std_mb",
    "relative_overhead_vs_primary",
    "combiner_only_overhead_vs_global_s",
    "combiner_only_overhead_vs_global_relative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/efficiency",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_efficiency_report.md",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--queries-per-direction",
        type=int,
        default=256,
        help="Deterministic TEST query sample per direction; 0 uses the full TEST split.",
    )
    parser.add_argument(
        "--include-adamf",
        action="store_true",
        help="Also benchmark the two M-Hyper + AdaMF-MAT main pairs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate assets and write the pending A100 protocol without importing torch.",
    )
    parser.add_argument("--worker-spec", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_repo_path(repo_root: Path, raw: str) -> Path:
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    return path if path.is_absolute() else repo_root / path


def pair_lock_path(repo_root: Path, spec: dict[str, Any]) -> Path:
    return (
        repo_root
        / "outputs/paper_a_safe_correction/dynasemble"
        / spec["dataset_dir"]
        / spec["pair_dir"]
        / "lock.json"
    )


def anchored_lock_path(repo_root: Path, spec: dict[str, Any]) -> Path:
    return (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
        / "dev_lock"
        / "anchored_dev_lock.json"
    )


def selected_specs(include_adamf: bool) -> tuple[dict[str, Any], ...]:
    return PAIR_SPECS if include_adamf else PAIR_SPECS[:2]


def validate_protocol_args(args: argparse.Namespace) -> None:
    if args.warmup < 3:
        raise ValueError("Protocol requires --warmup >= 3")
    if args.repetitions < 5:
        raise ValueError("Protocol requires --repetitions >= 5")
    if args.queries_per_direction < 0:
        raise ValueError("--queries-per-direction must be >= 0")


def preflight_assets(
    repo_root: Path, specs: tuple[dict[str, Any], ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workers: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for spec in specs:
        dyna_lock_path = pair_lock_path(repo_root, spec)
        anchor_lock_path = anchored_lock_path(repo_root, spec)
        for path, role in (
            (dyna_lock_path, "DynaSemble immutable DEV lock"),
            (anchor_lock_path, "Anchored immutable DEV lock"),
        ):
            if not path.exists():
                raise FileNotFoundError(path)
            assets.append(
                {
                    "path": portable(path, repo_root),
                    "role": role,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        dyna_lock = read_json(dyna_lock_path)
        anchor_lock = read_json(anchor_lock_path)
        if list(dyna_lock.get("seeds", [])) != [1, 2, 3]:
            raise RuntimeError(f"Unexpected seeds in {dyna_lock_path}")
        if dyna_lock.get("method_config", {}).get("evaluation") != (
            "exact filtered full-entity ranking, both directions"
        ):
            raise RuntimeError(f"DynaSemble ranking protocol drift in {dyna_lock_path}")
        if list(anchor_lock.get("seeds", [])) != [1, 2, 3]:
            raise RuntimeError(f"Unexpected seeds in {anchor_lock_path}")
        anchor_model = anchor_lock_path.parent / str(anchor_lock["model_file"])
        if not anchor_model.exists() or sha256_file(anchor_model) != anchor_lock["model_sha256"]:
            raise RuntimeError(f"Anchored model hash mismatch: {anchor_model}")
        assets.append(
            {
                "path": portable(anchor_model, repo_root),
                "role": "Anchored locked combiner",
                "sha256": sha256_file(anchor_model),
                "size_bytes": anchor_model.stat().st_size,
            }
        )
        run_by_seed = {int(item["seed"]): item for item in dyna_lock["run_manifest"]}
        for seed in (1, 2, 3):
            run = run_by_seed[seed]
            selector_info = dyna_lock["selectors"][str(seed)]
            selector_path = resolve_repo_path(repo_root, str(selector_info["path"]))
            if not selector_path.exists() or sha256_file(selector_path) != selector_info["sha256"]:
                raise RuntimeError(f"DynaSemble selector hash mismatch: {selector_path}")
            assets.append(
                {
                    "path": portable(selector_path, repo_root),
                    "role": f"DynaSemble selector seed {seed}",
                    "sha256": sha256_file(selector_path),
                    "size_bytes": selector_path.stat().st_size,
                }
            )
            for side in ("a", "b"):
                run_dir = resolve_repo_path(repo_root, str(run[f"expert_{side}_run"]))
                config_path = run_dir / "config_merged.json"
                checkpoint_path = run_dir / "best.ckpt"
                expected_config = run[f"expert_{side}_config_sha256"]
                expected_checkpoint = run[f"expert_{side}_checkpoint_sha256"]
                if not config_path.exists() or sha256_file(config_path) != expected_config:
                    raise RuntimeError(f"Frozen config hash mismatch: {config_path}")
                if not checkpoint_path.exists() or sha256_file(checkpoint_path) != expected_checkpoint:
                    raise RuntimeError(f"Frozen checkpoint hash mismatch: {checkpoint_path}")
                assets.extend(
                    (
                        {
                            "path": portable(config_path, repo_root),
                            "role": f"expert_{side} config seed {seed}",
                            "sha256": expected_config,
                            "size_bytes": config_path.stat().st_size,
                        },
                        {
                            "path": portable(checkpoint_path, repo_root),
                            "role": f"expert_{side} checkpoint seed {seed}",
                            "sha256": expected_checkpoint,
                            "size_bytes": checkpoint_path.stat().st_size,
                        },
                    )
                )
            for method in METHODS:
                workers.append(
                    {
                        "repo_root": str(repo_root),
                        "dataset": spec["dataset"],
                        "dataset_dir": spec["dataset_dir"],
                        "pair": spec["pair"],
                        "pair_dir": spec["pair_dir"],
                        "entity_count_expected": int(spec["entity_count"]),
                        "method": method,
                        "seed": seed,
                        "expert_a_name": run["expert_a_name"],
                        "expert_a_run": str(resolve_repo_path(repo_root, run["expert_a_run"])),
                        "expert_b_name": run["expert_b_name"],
                        "expert_b_run": str(resolve_repo_path(repo_root, run["expert_b_run"])),
                        "selector_path": str(selector_path),
                        "selector_sha256": selector_info["sha256"],
                        "dyna_method_config": dyna_lock["method_config"],
                        "global_alpha": float(dyna_lock["global_alpha"]),
                        "anchor_lock_path": str(anchor_lock_path),
                        "anchor_model_path": str(anchor_model),
                    }
                )
    unique_assets = {item["path"]: item for item in assets}
    return workers, [unique_assets[key] for key in sorted(unique_assets)]


def empty_csv(path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cpu_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.machine()


def hardware_info(torch: Any, device: str) -> dict[str, Any]:
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        ram_bytes = None
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if device == "cuda" else None
    gpu_total = (
        int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
        if device == "cuda"
        else None
    )
    return {
        "gpu_model": gpu_name,
        "gpu_total_memory_bytes": gpu_total,
        "cpu": cpu_name(),
        "ram_bytes": ram_bytes,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def process_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)
    except ImportError:
        return None


def synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed_stage(
    torch: Any,
    device: str,
    function: Callable[[], Any],
) -> tuple[Any, float]:
    synchronize(torch, device)
    started = time.perf_counter()
    value = function()
    synchronize(torch, device)
    return value, time.perf_counter() - started


def deterministic_query_sample(
    triples: list[tuple[int, int, int]], count: int
) -> list[tuple[int, int, int]]:
    if count == 0 or count >= len(triples):
        return list(triples)
    if count == 1:
        return [triples[len(triples) // 2]]
    indexes = [round(index * (len(triples) - 1) / (count - 1)) for index in range(count)]
    if len(set(indexes)) != count:
        raise RuntimeError("Deterministic query sample contains duplicate indexes")
    return [triples[index] for index in indexes]


def count_model_params(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_anchored_combiner_params(model: Any) -> int:
    classifier = model.steps[-1][1]
    return int(classifier.coef_.size + classifier.intercept_.size)


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    import pickle

    import numpy as np
    import torch

    repo_root = Path(spec["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from router.query_geometry import query_geometry_tensor
    from scripts.ablate_anchored_dynamic import model_outputs
    from scripts.eval_heterogeneous_complementarity import (
        endpoint_safe_mixed_ranks,
        load_expert,
        query_zscore_with_reference,
        ranks_against_reference,
        score_expert_block,
        validate_pair,
    )
    from scripts.eval_openbg_dynasemble import (
        FIXED_WEIGHT,
        load_selector,
        normalize_and_features,
        true_indexes_for_expert,
    )

    requested_device = str(spec["device"])
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    method = str(spec["method"])
    expert_a = load_expert(
        str(spec["expert_a_name"]), Path(spec["expert_a_run"]), device
    )
    expert_b = None
    if method != "Primary only":
        expert_b = load_expert(
            str(spec["expert_b_name"]), Path(spec["expert_b_run"]), device
        )
        validate_pair(expert_a, expert_b)
    if expert_a.num_entities != int(spec["entity_count_expected"]):
        raise RuntimeError("Entity count differs from the preflight manifest")

    selector = None
    anchored_model = None
    anchor_lock: dict[str, Any] | None = None
    if method.endswith("DynaSemble"):
        selector_path = Path(spec["selector_path"])
        if sha256_file(selector_path) != spec["selector_sha256"]:
            raise RuntimeError("DynaSemble selector hash drift in worker")
        selector = load_selector(
            selector_path,
            int(spec["seed"]),
            device,
            spec["dyna_method_config"],
        )
    if method.endswith("Anchored Dynamic"):
        anchor_lock = read_json(Path(spec["anchor_lock_path"]))
        if sha256_file(Path(spec["anchor_model_path"])) != anchor_lock["model_sha256"]:
            raise RuntimeError("Anchored model hash drift in worker")
        with Path(spec["anchor_model_path"]).open("rb") as handle:
            anchored_model = pickle.load(handle)

    true_indexes = true_indexes_for_expert(expert_a)
    selected_triples = deterministic_query_sample(
        expert_a.bundle.test_triples, int(spec["queries_per_direction"])
    )
    n_queries = 2 * len(selected_triples)
    n_available = 2 * len(expert_a.bundle.test_triples)
    outer_batch = expert_a.query_batch_size
    if expert_b is not None:
        outer_batch = max(outer_batch, expert_b.query_batch_size)
    primary_params = count_model_params(expert_a.model)
    secondary_params = count_model_params(expert_b.model) if expert_b is not None else 0
    combiner_params = 0
    if selector is not None:
        combiner_params = count_model_params(selector)
    if anchored_model is not None:
        combiner_params = count_anchored_combiner_params(anchored_model)

    def one_pass() -> dict[str, float]:
        scoring_s = 0.0
        feature_s = 0.0
        inference_s = 0.0
        combination_s = 0.0
        checksum = 0.0
        cpu_peak = process_rss_mb()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(torch, device)
        total_started = time.perf_counter()
        triples_tensor = torch.tensor(selected_triples, dtype=torch.long)
        for direction in ("head", "tail"):
            true_index = true_indexes[direction]
            for start in range(0, len(selected_triples), outer_batch):
                q_cpu = triples_tensor[start : start + outer_batch]

                def score_stage() -> tuple[Any, ...]:
                    raw_a, target_a, _ = score_expert_block(
                        expert_a, q_cpu, direction, true_index, device
                    )
                    rank_a = ranks_against_reference(raw_a, target_a)
                    if expert_b is None:
                        return raw_a, target_a, rank_a
                    raw_b, target_b, _ = score_expert_block(
                        expert_b, q_cpu, direction, true_index, device
                    )
                    rank_b = ranks_against_reference(raw_b, target_b)
                    return raw_a, target_a, rank_a, raw_b, target_b, rank_b

                scored, elapsed = timed_stage(torch, device, score_stage)
                scoring_s += elapsed
                if expert_b is None:
                    checksum += float(scored[2].sum().item())
                    current_rss = process_rss_mb()
                    if current_rss is not None:
                        cpu_peak = max(cpu_peak or current_rss, current_rss)
                    continue
                raw_a, target_a, rank_a, raw_b, target_b, rank_b = scored

                if method.endswith("Global alpha"):
                    def feature_stage() -> tuple[Any, ...]:
                        z_a, z_target_a = query_zscore_with_reference(raw_a, target_a)
                        z_b, z_target_b = query_zscore_with_reference(raw_b, target_b)
                        return z_a, z_target_a, z_b, z_target_b

                    features, elapsed = timed_stage(torch, device, feature_stage)
                    feature_s += elapsed
                    alpha, elapsed = timed_stage(
                        torch, device, lambda: float(spec["global_alpha"])
                    )
                    inference_s += elapsed

                    def combine_stage() -> Any:
                        return endpoint_safe_mixed_ranks(
                            features[0],
                            features[2],
                            features[1],
                            features[3],
                            alpha,
                            rank_a,
                            rank_b,
                        )

                    ranks, elapsed = timed_stage(torch, device, combine_stage)
                elif method.endswith("DynaSemble"):
                    def feature_stage() -> tuple[Any, ...]:
                        normalized_a, target_norm_a, features_a = normalize_and_features(
                            raw_a, target_a
                        )
                        normalized_b, target_norm_b, features_b = normalize_and_features(
                            raw_b, target_b
                        )
                        return (
                            normalized_a,
                            target_norm_a,
                            normalized_b,
                            target_norm_b,
                            torch.cat((features_a, features_b), dim=1),
                        )

                    features, elapsed = timed_stage(torch, device, feature_stage)
                    feature_s += elapsed

                    def infer_stage() -> Any:
                        return selector(features[4].to(device)).reshape(-1).cpu()

                    weights, elapsed = timed_stage(torch, device, infer_stage)
                    inference_s += elapsed

                    def combine_stage() -> Any:
                        ensemble = (
                            weights.unsqueeze(1) * features[0]
                            + FIXED_WEIGHT * features[2]
                        )
                        reference = weights * features[1] + FIXED_WEIGHT * features[3]
                        both_filtered = (~torch.isfinite(features[0])) & (
                            ~torch.isfinite(features[2])
                        )
                        ensemble = ensemble.masked_fill(both_filtered, float("-inf"))
                        result = ranks_against_reference(ensemble, reference)
                        return torch.where(weights == 0.0, rank_b, result)

                    ranks, elapsed = timed_stage(torch, device, combine_stage)
                elif method.endswith("Anchored Dynamic"):
                    def feature_stage() -> tuple[Any, ...]:
                        geometry = query_geometry_tensor(raw_a, raw_b, direction)
                        z_a, z_target_a = query_zscore_with_reference(raw_a, target_a)
                        z_b, z_target_b = query_zscore_with_reference(raw_b, target_b)
                        return geometry, z_a, z_target_a, z_b, z_target_b

                    features, elapsed = timed_stage(torch, device, feature_stage)
                    feature_s += elapsed

                    def infer_stage() -> Any:
                        matrix = features[0].numpy().astype(np.float64, copy=True)
                        nonfinite = ~np.isfinite(matrix).all(axis=1)
                        matrix[~np.isfinite(matrix)] = np.nan
                        decision, probability = model_outputs(anchored_model, matrix)
                        confidence = np.abs(2.0 * probability - 1.0)
                        fallback = nonfinite | (
                            confidence < float(anchor_lock["confidence_threshold"])
                        )
                        continuous = np.clip(
                            float(anchor_lock["alpha0"])
                            + float(anchor_lock["beta"]) * np.tanh(decision),
                            0.0,
                            1.0,
                        )
                        continuous = np.where(
                            fallback, float(anchor_lock["alpha0"]), continuous
                        )
                        alpha_grid = tuple(float(value) for value in anchor_lock["alpha_grid"])
                        alpha0 = float(anchor_lock["alpha0"])
                        applied = [
                            min(
                                alpha_grid,
                                key=lambda alpha: (
                                    abs(alpha - value),
                                    abs(alpha - alpha0),
                                    alpha,
                                ),
                            )
                            for value in continuous
                        ]
                        return torch.tensor(applied, dtype=features[1].dtype)

                    alpha, elapsed = timed_stage(torch, device, infer_stage)
                    inference_s += elapsed

                    def combine_stage() -> Any:
                        return endpoint_safe_mixed_ranks(
                            features[1],
                            features[3],
                            features[2],
                            features[4],
                            alpha,
                            rank_a,
                            rank_b,
                        )

                    ranks, elapsed = timed_stage(torch, device, combine_stage)
                else:
                    raise RuntimeError(f"Unknown method: {method}")
                combination_s += elapsed
                checksum += float(ranks.sum().item())
                current_rss = process_rss_mb()
                if current_rss is not None:
                    cpu_peak = max(cpu_peak or current_rss, current_rss)
        synchronize(torch, device)
        total_s = time.perf_counter() - total_started
        peak_gpu = (
            torch.cuda.max_memory_allocated() / (1024.0**2)
            if device == "cuda"
            else None
        )
        return {
            "feature_computation_s": feature_s,
            "combiner_inference_s": inference_s,
            "score_combination_s": combination_s,
            "combiner_only_overhead_s": feature_s + inference_s + combination_s,
            "scoring_s": scoring_s,
            "total_s": total_s,
            "queries_per_second": n_queries / total_s,
            "peak_gpu_memory_mb": peak_gpu,
            "peak_cpu_memory_mb": cpu_peak,
            "result_checksum": checksum,
        }

    for _ in range(int(spec["warmup"])):
        one_pass()
    measurements = [one_pass() for _ in range(int(spec["repetitions"]))]
    checksums = {measurement["result_checksum"] for measurement in measurements}
    if len(checksums) != 1:
        raise RuntimeError("Benchmark repetitions produced different exact-ranking checksums")
    rows = []
    for repetition, measurement in enumerate(measurements, start=1):
        rows.append(
            {
                "dataset": spec["dataset"],
                "pair": spec["pair"],
                "method": method,
                "seed": int(spec["seed"]),
                "repetition": repetition,
                "measurement_scope": "direct exact-filtered full-entity ranking on deterministic TEST query sample",
                "n_queries": n_queries,
                "n_test_queries_available": n_available,
                "entity_count": expert_a.num_entities,
                "primary_query_batch_size": expert_a.query_batch_size,
                "secondary_query_batch_size": expert_b.query_batch_size if expert_b else "",
                "outer_query_batch_size": outer_batch,
                "primary_params": primary_params,
                "secondary_params": secondary_params,
                "base_params": primary_params + secondary_params,
                "combiner_params": combiner_params,
                **measurement,
            }
        )
    return {
        "hardware": hardware_info(torch, device),
        "rows": rows,
        "protocol": {
            "warmup": int(spec["warmup"]),
            "repetitions": int(spec["repetitions"]),
            "queries_per_direction": int(spec["queries_per_direction"]),
            "query_sampling": "deterministic evenly spaced TEST triples",
            "ranking": "exact filtered full-entity ranking",
            "models_retrained": False,
        },
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(
    raw_rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        result = {field: value for field, value in zip(group_fields, key)}
        result.update(
            {
                "measurement_scope": rows[0]["measurement_scope"],
                "n_seed_repetitions": len(rows),
                "n_queries_per_repetition": rows[0]["n_queries"],
                "n_test_queries_available": rows[0]["n_test_queries_available"],
                "entity_count": rows[0]["entity_count"],
                "primary_query_batch_size": rows[0]["primary_query_batch_size"],
                "secondary_query_batch_size": rows[0]["secondary_query_batch_size"],
                "outer_query_batch_size": rows[0]["outer_query_batch_size"],
                "base_params": rows[0]["base_params"],
                "combiner_params": rows[0]["combiner_params"],
            }
        )
        field_map = {
            "scoring_s": "scoring_time",
            "feature_computation_s": "feature_time",
            "combiner_inference_s": "combiner_inference_time",
            "score_combination_s": "score_combination_time",
            "combiner_only_overhead_s": "combiner_only_overhead",
            "total_s": "total_time",
            "queries_per_second": "queries_per_second",
            "peak_gpu_memory_mb": "peak_gpu_memory",
            "peak_cpu_memory_mb": "peak_cpu_memory",
        }
        for source, target in field_map.items():
            values = [float(row[source]) for row in rows if row[source] not in (None, "")]
            if target.endswith("time") or target == "combiner_only_overhead":
                mean_key, std_key = f"{target}_mean_s", f"{target}_std_s"
            elif target.endswith("memory"):
                mean_key, std_key = f"{target}_mean_mb", f"{target}_std_mb"
            else:
                mean_key, std_key = f"{target}_mean", f"{target}_std"
            if values:
                result[mean_key], result[std_key] = mean_std(values)
            else:
                result[mean_key] = ""
                result[std_key] = ""
        output.append(result)
    return output


def add_relative_overheads(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> None:
    by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    for pair_rows in by_pair.values():
        by_method = {row["method"]: row for row in pair_rows}
        primary_total = float(by_method["Primary only"]["total_time_mean_s"])
        global_overhead = float(
            by_method["Primary + Secondary + Global alpha"][
                "combiner_only_overhead_mean_s"
            ]
        )
        for row in pair_rows:
            total = float(row["total_time_mean_s"])
            overhead = float(row["combiner_only_overhead_mean_s"])
            row["relative_overhead_vs_primary"] = total / primary_total - 1.0
            row["combiner_only_overhead_vs_global_s"] = overhead - global_overhead
            row["combiner_only_overhead_vs_global_relative"] = (
                overhead / global_overhead - 1.0 if global_overhead > 0.0 else ""
            )


def same_hardware(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "gpu_model",
        "gpu_total_memory_bytes",
        "cpu",
        "ram_bytes",
        "pytorch_version",
        "cuda_runtime_version",
        "device",
        "python_version",
        "platform",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def format_mean_std(row: dict[str, Any], prefix: str, scale: float = 1.0) -> str:
    mean = row.get(f"{prefix}_mean_s")
    std = row.get(f"{prefix}_std_s")
    if mean in (None, "") or std in (None, ""):
        return "not measured"
    return f"{scale * float(mean):.3f} +/- {scale * float(std):.3f}"


def write_report(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    specs: tuple[dict[str, Any], ...],
    summary: list[dict[str, Any]],
    hardware: dict[str, Any] | None,
) -> None:
    pairs = ", ".join(f"{spec['dataset']} / {spec['pair']}" for spec in specs)
    lines = [
        "# Paper A Efficiency and Complexity Benchmark",
        "",
        "## Status",
        "",
    ]
    if status == "prepared_not_measured":
        lines += [
            "The benchmark protocol and executable workers are prepared, but no timing "
            "numbers are reported from this local environment. PyTorch/CUDA is unavailable "
            "here, and the requested benchmark is intended for the A100 host. Empty result "
            "tables are deliberate and must not be interpreted as zero cost.",
            "",
            "Run on the A100 environment:",
            "",
            "```powershell",
            "python scripts/benchmark_paper_a_efficiency.py --device cuda --warmup 3 --repetitions 5 --queries-per-direction 256",
            "```",
            "",
            "Add `--include-adamf` for the two M-Hyper + AdaMF-MAT pairs. Set "
            "`--queries-per-direction 0` only when full-TEST timing is desired; candidate "
            "ranking remains exact and full-entity for either setting.",
        ]
    else:
        lines += [
            "Measured results keep base-model scoring separate from method overhead. "
            "Each method runs in a fresh process. Primary loads one frozen expert; Global, "
            "DynaSemble, and Anchored Dynamic load both frozen experts. No base model or "
            "combiner is retrained.",
            "",
            "## Hardware and workload",
            "",
            f"- GPU: {hardware.get('gpu_model') or 'CPU-only'}",
            f"- CPU: {hardware.get('cpu')}",
            f"- RAM: {float(hardware['ram_bytes']) / (1024.0**3):.2f} GiB"
            if hardware.get("ram_bytes")
            else "- RAM: unavailable",
            f"- PyTorch/CUDA: {hardware.get('pytorch_version')} / {hardware.get('cuda_runtime_version')}",
            f"- Warm-up/measured repetitions: {args.warmup} / {args.repetitions}",
            f"- Query sampling: {args.queries_per_direction or 'all'} TEST triples per direction, deterministic evenly spaced",
            f"- Pairs: {pairs}",
            "",
            "### Workload by method",
            "",
            "| Dataset/Pair | Method | Entities | Timed queries | Available TEST queries | Primary batch | Secondary batch | Outer batch |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            secondary_batch = row.get("secondary_query_batch_size") or "n/a"
            lines.append(
                f"| {row['dataset']} / {row['pair']} | {row['method']} | "
                f"{int(row['entity_count']):,} | {int(row['n_queries_per_repetition']):,} | "
                f"{int(row['n_test_queries_available']):,} | "
                f"{int(row['primary_query_batch_size']):,} | {secondary_batch} | "
                f"{int(row['outer_query_batch_size']):,} |"
            )
        lines += [
            "",
            "## Main table",
            "",
            "Times are mean +/- sample standard deviation across three paired seeds and "
            "all measured repetitions. Scoring Time is frozen base-model exact filtered "
            "full-entity scoring. Combiner Time is feature computation, selector/policy "
            "inference, and score combination/ranking. Total Time is directly measured "
            "wall time, not an arithmetic estimate.",
            "",
            "| Dataset/Pair | Method | Base Params | Combiner Params | Scoring Time (ms) | Combiner Time (ms) | Total Time (ms) | Queries/s | Peak GPU (MiB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            gpu = (
                f"{float(row['peak_gpu_memory_mean_mb']):.1f} +/- {float(row['peak_gpu_memory_std_mb']):.1f}"
                if row.get("peak_gpu_memory_mean_mb") not in (None, "")
                else "not measured"
            )
            lines.append(
                f"| {row['dataset']} / {row['pair']} | {row['method']} | "
                f"{int(row['base_params']):,} | {int(row['combiner_params']):,} | "
                f"{format_mean_std(row, 'scoring_time', 1000.0)} | "
                f"{format_mean_std(row, 'combiner_only_overhead', 1000.0)} | "
                f"{format_mean_std(row, 'total_time', 1000.0)} | "
                f"{float(row['queries_per_second_mean']):.2f} +/- {float(row['queries_per_second_std']):.2f} | "
                f"{gpu} |"
            )
        lines += [
            "",
            "## Overhead interpretation",
            "",
        ]
        for row in summary:
            if row["method"] == "Primary only":
                continue
            relative = 100.0 * float(row["relative_overhead_vs_primary"])
            versus_global = 1000.0 * float(row["combiner_only_overhead_vs_global_s"])
            lines.append(
                f"- {row['dataset']} / {row['pair']} / {row['method']}: "
                f"end-to-end overhead vs Primary is {relative:+.2f}%; combiner-only "
                f"overhead vs Global is {versus_global:+.3f} ms."
            )
    lines += [
        "",
        "## Complexity accounting",
        "",
        "Let Q be the number of evaluated directional queries, E the entity count, "
        "and C_p/C_s the per-candidate costs of the primary/secondary experts. Primary "
        "scoring is O(Q E C_p); every dual-expert method is O(Q E (C_p + C_s)). "
        "After frozen scores exist, Global normalization and exact mixing/ranking are "
        "O(Q E). DynaSemble adds O(Q E) score statistics plus a constant-size 4-16-16-1 "
        "selector. Anchored Dynamic adds O(Q E) query geometry and z-normalization plus "
        "a constant-size locked logistic combiner. Thus neither adaptive combiner changes "
        "the exact full-ranking asymptotic dependence on E; the benchmark measures their "
        "different constants separately.",
        "",
        "## Measurement boundary",
        "",
        "Base scoring includes the frozen evaluator's candidate construction, model "
        "forward pass, standard filtered masking, CPU transfer, and endpoint rank. "
        "Feature time begins only after both base score matrices exist. DynaSemble uses "
        "its released min-max statistics and selector. Anchored Dynamic uses the locked "
        "query-geometry schema, classifier, beta, threshold, fallback, and exact alpha "
        "grid. The ranking candidate set and filtering protocol are never reduced; only "
        "the number of TEST queries timed may be sampled, and that count is recorded.",
        "",
        "Peak GPU is the process allocator peak after the method's required models are "
        "loaded, so Primary is not charged for a secondary model. Peak CPU is sampled RSS "
        "at stage boundaries and should be treated as an approximate process peak.",
        "",
        "No timing result from TEST is used for model, pair, beta, threshold, orientation, "
        "or operating-point selection.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_driver(args: argparse.Namespace) -> None:
    validate_protocol_args(args)
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = selected_specs(args.include_adamf)
    workers, assets = preflight_assets(repo_root, specs)

    raw_path = output_dir / "efficiency_raw_repetitions.csv"
    summary_path = output_dir / "efficiency_summary.csv"
    by_seed_path = output_dir / "efficiency_by_seed.csv"
    if args.prepare_only:
        empty_csv(raw_path, RAW_COLUMNS)
        empty_csv(summary_path, SUMMARY_COLUMNS)
        empty_csv(by_seed_path, ("seed", *SUMMARY_COLUMNS))
        manifest = {
            "schema_version": 1,
            "status": "prepared_not_measured",
            "reason": "Local environment lacks the requested A100 PyTorch/CUDA runtime; no timing values fabricated.",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "queries_per_direction": args.queries_per_direction,
            "query_sampling": "deterministic evenly spaced TEST triples",
            "ranking": "exact filtered full-entity ranking",
            "methods": list(METHODS),
            "pairs": [{key: spec[key] for key in ("dataset", "pair", "entity_count")} for spec in specs],
            "fresh_process_per_method_seed": True,
            "base_and_combiner_timing_separated": True,
            "base_models_retrained": False,
            "test_used_for_selection": False,
            "assets": assets,
            "output_hashes": {
                path.name: sha256_file(path)
                for path in (raw_path, summary_path, by_seed_path)
            },
        }
        (output_dir / "benchmark_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_report(
            report_path,
            status="prepared_not_measured",
            args=args,
            specs=specs,
            summary=[],
            hardware=None,
        )
        print(f"[PREPARED] A100 benchmark protocol: {output_dir}")
        print(f"[NOT MEASURED] no timing values were generated: {report_path}")
        return

    raw_rows: list[dict[str, Any]] = []
    hardware: dict[str, Any] | None = None
    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="paper_a_efficiency_", dir=output_dir) as temp_dir:
        temp = Path(temp_dir)
        for index, worker in enumerate(workers):
            worker.update(
                {
                    "device": args.device,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "queries_per_direction": args.queries_per_direction,
                }
            )
            spec_path = temp / f"worker_{index:03d}.json"
            result_path = temp / f"result_{index:03d}.json"
            spec_path.write_text(
                json.dumps(worker, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(script_path),
                "--worker-spec",
                str(spec_path),
                "--worker-result",
                str(result_path),
            ]
            print(
                f"[WORKER {index + 1}/{len(workers)}] {worker['dataset']} / "
                f"{worker['pair']} / seed {worker['seed']} / {worker['method']}",
                flush=True,
            )
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not result_path.exists():
                raise RuntimeError(
                    f"Efficiency worker failed ({completed.returncode}).\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            result = read_json(result_path)
            if hardware is None:
                hardware = result["hardware"]
            elif not same_hardware(hardware, result["hardware"]):
                raise RuntimeError("Hardware/process environment changed across workers")
            raw_rows.extend(result["rows"])

    expected_rows = len(workers) * args.repetitions
    if len(raw_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} repetitions, received {len(raw_rows)}")
    summary = aggregate_rows(raw_rows, ("dataset", "pair", "method"))
    add_relative_overheads(summary, ("dataset", "pair"))
    by_seed = aggregate_rows(raw_rows, ("dataset", "pair", "method", "seed"))
    add_relative_overheads(by_seed, ("dataset", "pair", "seed"))
    summary.sort(key=lambda row: (row["dataset"], row["pair"], METHODS.index(row["method"])))
    by_seed.sort(
        key=lambda row: (
            row["dataset"],
            row["pair"],
            int(row["seed"]),
            METHODS.index(row["method"]),
        )
    )
    write_csv(raw_path, raw_rows, RAW_COLUMNS)
    write_csv(summary_path, summary, SUMMARY_COLUMNS)
    write_csv(by_seed_path, by_seed, ("seed", *SUMMARY_COLUMNS))
    hardware_path = output_dir / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths = (raw_path, summary_path, by_seed_path, hardware_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "queries_per_direction": args.queries_per_direction,
        "query_sampling": "deterministic evenly spaced TEST triples",
        "ranking": "exact filtered full-entity ranking",
        "methods": list(METHODS),
        "pairs": [{key: spec[key] for key in ("dataset", "pair", "entity_count")} for spec in specs],
        "fresh_process_per_method_seed": True,
        "base_and_combiner_timing_separated": True,
        "total_time_is_direct_wall_measurement": True,
        "base_models_retrained": False,
        "test_used_for_selection": False,
        "hardware": hardware,
        "assets": assets,
        "output_hashes": {path.name: sha256_file(path) for path in output_paths},
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        report_path,
        status="complete",
        args=args,
        specs=specs,
        summary=summary,
        hardware=hardware,
    )
    print(f"[OK] wrote efficiency benchmark to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


def main() -> None:
    args = parse_args()
    if args.worker_spec:
        if not args.worker_result:
            raise ValueError("--worker-result is required with --worker-spec")
        result = run_worker(read_json(Path(args.worker_spec)))
        Path(args.worker_result).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    run_driver(args)


if __name__ == "__main__":
    main()
