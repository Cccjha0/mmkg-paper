from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_DIR = ROOT / "outputs" / "router" / "eval" / "clean"

RULE_ROWS = ROOT / "outputs" / "router" / "routing" / "clean" / "test_router_predictions_clean_rule_delta_0.01_tau_0.5_rule.csv"
NAIVE_ROWS = ROOT / "outputs" / "router" / "routing" / "clean" / "test_router_predictions_clean_logistic_delta_0.01_tau_0.9_C4.csv"
DIRECTION_ROWS = ROOT / "outputs" / "router" / "eval" / "clean" / "dual_threshold_scan_clean_logistic_delta_0.01_C4_query_rows.csv"
REGRESSION_ROWS = ROOT / "outputs" / "router" / "eval" / "clean" / "regression_router_scan_xgb_C4_query_rows.csv"

EXPECTED_MRR = {
    "Residual-only": 0.29304943039096304,
    "Clean rule": 0.29428194121180534,
    "Naive global clean router": 0.293863770631994,
    "Direction-specific threshold": 0.29743590499633776,
    "Regression-based clean router": 0.29818238528001295,
    "Oracle routing": 0.3337373406732906,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock current clean-routing baselines before candidate-aware router experiments."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_rows(path: Path, config_id: str | None = None) -> dict[str, dict[str, str]]:
    require_file(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if config_id is not None and row.get("config_id") != config_id:
                continue
            query_id = row["query_id"]
            if query_id in rows:
                raise RuntimeError(f"Duplicate query_id={query_id} in {path}")
            rows[query_id] = row
    if not rows:
        raise RuntimeError(f"No rows loaded from {path} with config_id={config_id!r}")
    return rows


def rr_to_hits(rr: float, k: int) -> int:
    return int(rr >= 1.0 / k)


def method_summary(
    name: str,
    rows: list[dict[str, Any]],
    rr_col: str,
    *,
    role: str,
    is_query_time_legal: bool,
    source: str,
    use_fusion_col: str | None = None,
) -> dict[str, Any]:
    rrs = [float(row[rr_col]) for row in rows]
    n = len(rrs)
    out: dict[str, Any] = {
        "method": name,
        "role": role,
        "is_query_time_legal": str(is_query_time_legal),
        "n_queries": n,
        "mrr": sum(rrs) / n,
        "hits1": sum(rr_to_hits(rr, 1) for rr in rrs) / n,
        "hits3": sum(rr_to_hits(rr, 3) for rr in rrs) / n,
        "hits10": sum(rr_to_hits(rr, 10) for rr in rrs) / n,
        "fusion_coverage": "",
        "source": source,
    }
    if use_fusion_col is not None:
        out["fusion_coverage"] = sum(int(row[use_fusion_col]) for row in rows) / n
    return out


def assert_expected(summary_rows: list[dict[str, Any]], tolerance: float) -> None:
    by_name = {row["method"]: float(row["mrr"]) for row in summary_rows}
    failures = []
    for method, expected in EXPECTED_MRR.items():
        actual = by_name[method]
        delta = abs(actual - expected)
        if delta > tolerance:
            failures.append((method, actual, expected, delta))
    if failures:
        formatted = "\n".join(
            f"{method}: actual={actual:.12f}, expected={expected:.12f}, abs_delta={delta:.12f}"
            for method, actual, expected, delta in failures
        )
        raise RuntimeError(f"Baseline lock failed tolerance={tolerance}:\n{formatted}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    rule_rows = read_rows(RULE_ROWS)
    naive_rows = read_rows(NAIVE_ROWS)
    direction_rows = read_rows(DIRECTION_ROWS, config_id="tau_head=0.3|tau_tail=0.9")
    regression_rows = read_rows(REGRESSION_ROWS, config_id="theta=0.00")

    query_ids = set(rule_rows)
    for name, table in {
        "naive": naive_rows,
        "direction": direction_rows,
        "regression": regression_rows,
    }.items():
        if set(table) != query_ids:
            raise RuntimeError(
                f"Query-id mismatch for {name}: base={len(query_ids)}, {name}={len(table)}"
            )

    locked_rows: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        base = rule_rows[query_id]
        naive = naive_rows[query_id]
        direction = direction_rows[query_id]
        regression = regression_rows[query_id]
        rr_gate = float(base["rr_gate"])
        rr_residual = float(base["rr_residual"])
        use_fusion_oracle = int(rr_gate > rr_residual)
        locked_rows.append(
            {
                "query_id": query_id,
                "split": base["split"],
                "seed": base["seed"],
                "direction": base["direction"],
                "target_regime": base["target_regime"],
                "relation_id": base["relation_id"],
                "rr_gate": rr_gate,
                "rr_residual": rr_residual,
                "rr_clean_rule": float(base["rr_final"]),
                "rr_naive_global_clean_router": float(naive["rr_final"]),
                "rr_direction_specific_threshold": float(direction["rr_final"]),
                "rr_regression_clean_router": float(regression["rr_final"]),
                "rr_oracle": max(rr_gate, rr_residual),
                "use_fusion_clean_rule": int(base["use_fusion"]),
                "use_fusion_naive_global_clean_router": int(naive["use_fusion"]),
                "use_fusion_direction_specific_threshold": int(direction["use_fusion"]),
                "use_fusion_regression_clean_router": int(regression["use_fusion"]),
                "use_fusion_oracle": use_fusion_oracle,
                "selected_expert_clean_rule": base["selected_expert"],
                "selected_expert_naive_global_clean_router": naive["selected_expert"],
                "selected_expert_direction_specific_threshold": direction["selected_expert"],
                "selected_expert_regression_clean_router": regression["selected_expert"],
                "selected_expert_oracle": "gate_only" if use_fusion_oracle else "residual_only",
            }
        )

    summary_rows = [
        method_summary(
            "Residual-only",
            locked_rows,
            "rr_residual",
            role="fixed structural expert",
            is_query_time_legal=True,
            source="baseline_locked_query_rows.csv",
        ),
        method_summary(
            "Clean rule",
            locked_rows,
            "rr_clean_rule",
            role="legal rule baseline",
            is_query_time_legal=True,
            source=str(RULE_ROWS.relative_to(ROOT)).replace("\\", "/"),
            use_fusion_col="use_fusion_clean_rule",
        ),
        method_summary(
            "Naive global clean router",
            locked_rows,
            "rr_naive_global_clean_router",
            role="global-threshold learned router",
            is_query_time_legal=True,
            source=str(NAIVE_ROWS.relative_to(ROOT)).replace("\\", "/"),
            use_fusion_col="use_fusion_naive_global_clean_router",
        ),
        method_summary(
            "Direction-specific threshold",
            locked_rows,
            "rr_direction_specific_threshold",
            role="structured clean policy",
            is_query_time_legal=True,
            source=f"{str(DIRECTION_ROWS.relative_to(ROOT)).replace('\\', '/')}#tau_head=0.3|tau_tail=0.9",
            use_fusion_col="use_fusion_direction_specific_threshold",
        ),
        method_summary(
            "Regression-based clean router",
            locked_rows,
            "rr_regression_clean_router",
            role="strongest current clean router",
            is_query_time_legal=True,
            source=f"{str(REGRESSION_ROWS.relative_to(ROOT)).replace('\\', '/')}#theta=0.00",
            use_fusion_col="use_fusion_regression_clean_router",
        ),
        method_summary(
            "Oracle routing",
            locked_rows,
            "rr_oracle",
            role="post-hoc upper bound",
            is_query_time_legal=False,
            source="max(rr_gate, rr_residual) recomputed from locked query rows",
            use_fusion_col="use_fusion_oracle",
        ),
    ]
    assert_expected(summary_rows, args.tolerance)

    summary_path = out_dir / "baseline_locked_summary.csv"
    query_rows_path = out_dir / "baseline_locked_query_rows.csv"
    manifest_path = out_dir / "baseline_locked_manifest.json"

    write_csv(
        query_rows_path,
        locked_rows,
        [
            "query_id",
            "split",
            "seed",
            "direction",
            "target_regime",
            "relation_id",
            "rr_gate",
            "rr_residual",
            "rr_clean_rule",
            "rr_naive_global_clean_router",
            "rr_direction_specific_threshold",
            "rr_regression_clean_router",
            "rr_oracle",
            "use_fusion_clean_rule",
            "use_fusion_naive_global_clean_router",
            "use_fusion_direction_specific_threshold",
            "use_fusion_regression_clean_router",
            "use_fusion_oracle",
            "selected_expert_clean_rule",
            "selected_expert_naive_global_clean_router",
            "selected_expert_direction_specific_threshold",
            "selected_expert_regression_clean_router",
            "selected_expert_oracle",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "method",
            "role",
            "is_query_time_legal",
            "n_queries",
            "mrr",
            "hits1",
            "hits3",
            "hits10",
            "fusion_coverage",
            "source",
        ],
    )

    manifest = {
        "dataset": "OpenBG-IMG paper_split",
        "evaluation": "filtered ranking",
        "direction": "both",
        "experts": ["Gate-only", "Residual-only"],
        "seeds": [1, 2, 3],
        "n_queries": len(locked_rows),
        "tolerance": args.tolerance,
        "residual_mrr": summary_rows[0]["mrr"],
        "clean_rule_mrr": summary_rows[1]["mrr"],
        "naive_global_clean_router_mrr": summary_rows[2]["mrr"],
        "direction_specific_mrr": summary_rows[3]["mrr"],
        "regression_clean_router_mrr": summary_rows[4]["mrr"],
        "oracle_mrr": summary_rows[5]["mrr"],
        "input_files": {
            "clean_rule": str(RULE_ROWS.relative_to(ROOT)).replace("\\", "/"),
            "naive_global_clean_router": str(NAIVE_ROWS.relative_to(ROOT)).replace("\\", "/"),
            "direction_specific_threshold": str(DIRECTION_ROWS.relative_to(ROOT)).replace("\\", "/"),
            "regression_clean_router": str(REGRESSION_ROWS.relative_to(ROOT)).replace("\\", "/"),
        },
        "selected_configs": {
            "clean_rule": "delta=0.01|tau=0.5|rule",
            "naive_global_clean_router": "logistic|delta=0.01|tau=0.9|C4",
            "direction_specific_threshold": "logistic|delta=0.01|C4|tau_head=0.3|tau_tail=0.9",
            "regression_clean_router": "xgb|C4|theta=0.00",
            "oracle": "per-query max(rr_gate, rr_residual); not query-time legal",
        },
        "outputs": {
            "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
            "query_rows": str(query_rows_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {query_rows_path}")
    print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()
