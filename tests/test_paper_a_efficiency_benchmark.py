from __future__ import annotations

from argparse import Namespace
import unittest

from scripts.benchmark_paper_a_efficiency import (
    add_relative_overheads,
    aggregate_rows,
    deterministic_query_sample,
    validate_protocol_args,
)


def test_protocol_minimums_are_enforced() -> None:
    validate_protocol_args(Namespace(warmup=3, repetitions=5, queries_per_direction=0))
    with unittest.TestCase().assertRaisesRegex(ValueError, "warmup"):
        validate_protocol_args(
            Namespace(warmup=2, repetitions=5, queries_per_direction=256)
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "repetitions"):
        validate_protocol_args(
            Namespace(warmup=3, repetitions=4, queries_per_direction=256)
        )


def test_query_sample_is_deterministic_and_spans_the_split() -> None:
    triples = [(index, 0, index + 1) for index in range(11)]
    sample = deterministic_query_sample(triples, 4)

    assert sample == [triples[index] for index in (0, 3, 7, 10)]
    assert deterministic_query_sample(triples, 0) == triples


def raw_row(method: str, seed: int, repetition: int, total: float) -> dict:
    overhead = 0.1 if method == "Primary only" else 0.2
    if method.endswith("Global alpha"):
        overhead = 0.4
    return {
        "dataset": "MKG-W",
        "pair": "M-Hyper + NativE",
        "method": method,
        "seed": seed,
        "repetition": repetition,
        "measurement_scope": "synthetic",
        "n_queries": 8,
        "n_test_queries_available": 80,
        "entity_count": 15000,
        "primary_query_batch_size": 2,
        "secondary_query_batch_size": "" if method == "Primary only" else 4,
        "outer_query_batch_size": 2 if method == "Primary only" else 4,
        "base_params": 10 if method == "Primary only" else 20,
        "combiner_params": 0,
        "scoring_s": total - overhead,
        "feature_computation_s": overhead / 2,
        "combiner_inference_s": 0.0,
        "score_combination_s": overhead / 2,
        "combiner_only_overhead_s": overhead,
        "total_s": total,
        "queries_per_second": 8 / total,
        "peak_gpu_memory_mb": 100,
        "peak_cpu_memory_mb": 200,
    }


def test_seed_relative_overheads_do_not_mix_seeds() -> None:
    methods = (
        "Primary only",
        "Primary + Secondary + Global alpha",
    )
    rows = []
    for repetition in (1, 2):
        rows.extend(
            (
                raw_row(methods[0], 1, repetition, 1.0),
                raw_row(methods[1], 1, repetition, 2.0),
                raw_row(methods[0], 2, repetition, 10.0),
                raw_row(methods[1], 2, repetition, 15.0),
            )
        )
    summary = aggregate_rows(rows, ("dataset", "pair", "method", "seed"))
    add_relative_overheads(summary, ("dataset", "pair", "seed"))
    relative = {
        (int(row["seed"]), row["method"]): row["relative_overhead_vs_primary"]
        for row in summary
    }

    assert abs(relative[(1, methods[1])] - 1.0) < 1e-12
    assert abs(relative[(2, methods[1])] - 0.5) < 1e-12
    assert {row["primary_query_batch_size"] for row in summary} == {2}
