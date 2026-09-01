from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped DEV cross-fitting for global/relation alpha policies from "
            "precomputed heterogeneous-complementarity query rows."
        )
    )
    parser.add_argument("--query-rows", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def triple_key(row: dict[str, str]) -> str:
    return f"h={int(row['head_id'])}|r={int(row['relation_id'])}|t={int(row['tail_id'])}"


def alpha_column(alpha: float) -> str:
    return f"rr_alpha_{alpha:.2f}".replace(".", "_")


def best_alpha(rows: list[dict], alphas: tuple[float, ...]) -> tuple[float, float]:
    scored = []
    for alpha in alphas:
        column = alpha_column(alpha)
        mean = sum(float(row[column]) for row in rows) / len(rows)
        scored.append((mean, -abs(alpha - 0.5), -alpha, alpha))
    winner = max(scored)
    return float(winner[3]), float(winner[0])


def assign_grouped_folds(
    rows: list[dict[str, str]],
    folds: int,
    fold_seed: int,
) -> tuple[dict[str, int], dict]:
    """Assign original triples to relation-stratified deterministic folds."""
    by_relation: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_relation[int(row["relation_id"])].add(triple_key(row))

    assignment = {}
    for relation_id, keys in sorted(by_relation.items()):
        offset = int(
            hashlib.sha256(f"{fold_seed}|{relation_id}|offset".encode("utf-8")).hexdigest()[:16],
            16,
        ) % folds
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{fold_seed}|{relation_id}|{key}".encode("utf-8")
            ).hexdigest(),
        )
        for index, key in enumerate(ordered):
            assignment[key] = (index + offset) % folds

    group_sizes = Counter(assignment.values())
    observation_sizes = Counter(assignment[triple_key(row)] for row in rows)
    audit = {
        "assignment": (
            "relation-stratified SHA256 ordering, then round-robin by original triple "
            "with a deterministic per-relation fold offset"
        ),
        "fold_seed": fold_seed,
        "n_folds": folds,
        "n_unique_triples": len(assignment),
        "triple_groups_per_fold": {str(fold): group_sizes[fold] for fold in range(folds)},
        "observations_per_fold": {str(fold): observation_sizes[fold] for fold in range(folds)},
    }
    return assignment, audit


def select_fold_policy(
    train_rows: list[dict],
    alphas: tuple[float, ...],
    relation_min_support: int,
) -> dict:
    global_alpha, global_train_mrr = best_alpha(train_rows, alphas)
    by_relation: dict[int, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_relation[int(row["relation_id"])].append(row)

    relation_alpha = {}
    details = {}
    for relation_id, relation_rows in sorted(by_relation.items()):
        if len(relation_rows) >= relation_min_support:
            alpha, train_mrr = best_alpha(relation_rows, alphas)
            source = "relation_train_folds"
        else:
            alpha = global_alpha
            train_mrr = None
            source = "global_fallback"
        relation_alpha[relation_id] = alpha
        details[str(relation_id)] = {
            "support_observations": len(relation_rows),
            "support_seed_stripped_directional_queries": len(
                {str(row["query_key"]) for row in relation_rows}
            ),
            "support_original_triples": len({triple_key(row) for row in relation_rows}),
            "alpha": alpha,
            "source": source,
            "train_mrr": train_mrr,
        }
    return {
        "global_alpha": global_alpha,
        "global_train_mrr": global_train_mrr,
        "relation_alpha": relation_alpha,
        "relation_details": details,
    }


def metric(values: list[float]) -> dict:
    ranks = [int(round(1.0 / value)) for value in values]
    return {
        "count": len(values),
        "mrr": sum(values) / len(values),
        "hits@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hits@3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "hits@10": sum(rank <= 10 for rank in ranks) / len(ranks),
    }


def clustered_interval(rows: list[dict], column: str, reference: str = "rr_a") -> dict:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[triple_key(row)].append(float(row[column]) - float(row[reference]))
    values = [sum(cluster) / len(cluster) for cluster in clusters.values()]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    return {
        "n_triple_clusters": len(values),
        "mean_delta": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "note": "normal interval over original-triple cluster means",
    }


def summarize(rows: list[dict], expert_a: str, expert_b: str) -> list[dict]:
    methods = [
        (expert_a, "rr_a", "fixed expert"),
        (expert_b, "rr_b", "fixed expert"),
        ("Equal RRF", "rr_rrf", "fixed"),
        ("Query-zscore 0.5", "rr_equal", "fixed"),
        ("Global alpha (full-DEV fit)", "rr_global", "in-sample DEV selection"),
        ("Global alpha (5-fold cross-fit)", "rr_global_crossfit", "held-out triples"),
        ("Relation alpha (full-DEV fit)", "rr_relation", "in-sample DEV selection"),
        ("Relation alpha (5-fold cross-fit)", "rr_relation_crossfit", "held-out triples"),
        ("Oracle", "rr_oracle", "answer-aware upper bound"),
    ]
    anchor = metric([float(row["rr_a"]) for row in rows])["mrr"]
    oracle = metric([float(row["rr_oracle"]) for row in rows])["mrr"]
    oracle_gap = oracle - anchor
    output = []
    for method, column, notes in methods:
        result = metric([float(row[column]) for row in rows])
        result.update(
            {
                "method": method,
                "delta_vs_a": result["mrr"] - anchor,
                "oracle_gap_recovery": (
                    (result["mrr"] - anchor) / oracle_gap if oracle_gap > 0 else 0.0
                ),
                "notes": notes,
            }
        )
        output.append(result)
    return output


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Method | DEV MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs. A | Oracle gap recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mrr']:.6f} | {row['hits@1']:.6f} | "
            f"{row['hits@3']:.6f} | {row['hits@10']:.6f} | "
            f"{row['delta_vs_a']:+.6f} | {100.0 * row['oracle_gap_recovery']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    query_path = Path(args.query_rows)
    selection_path = Path(args.selection_json)
    out_dir = Path(args.output_dir)
    rows = read_csv(query_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not rows or {row["split"] for row in rows} != {"dev"}:
        raise RuntimeError("Cross-fitting requires non-empty DEV query rows")
    if {row["pair_name"] for row in rows} != {selection["pair_name"]}:
        raise RuntimeError("Query rows and selection pair_name differ")

    alphas = tuple(float(value) for value in selection["alpha_grid"])
    required = {
        "query_key",
        "head_id",
        "relation_id",
        "tail_id",
        "seed",
        "direction",
        "rr_a",
        "rr_b",
        "rr_rrf",
        "rr_equal",
        "rr_global",
        "rr_relation",
        "rr_oracle",
        *(alpha_column(alpha) for alpha in alphas),
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Query rows are missing required columns: {sorted(missing)}")

    seeds = sorted({int(row["seed"]) for row in rows})
    assignment, fold_audit = assign_grouped_folds(rows, args.folds, args.fold_seed)
    group_observation_counts = Counter(triple_key(row) for row in rows)
    expected_per_group = 2 * len(seeds)
    unexpected_groups = {
        key: count for key, count in group_observation_counts.items() if count != expected_per_group
    }
    if unexpected_groups:
        raise RuntimeError(
            f"Expected {expected_per_group} observations per original triple; "
            f"found {len(unexpected_groups)} malformed groups"
        )

    fold_policies = []
    fold_results = []
    for fold in range(args.folds):
        train_rows = [row for row in rows if assignment[triple_key(row)] != fold]
        heldout_rows = [row for row in rows if assignment[triple_key(row)] == fold]
        policy = select_fold_policy(
            train_rows,
            alphas,
            int(selection["relation_min_support"]),
        )
        for row in heldout_rows:
            relation_id = int(row["relation_id"])
            global_alpha = float(policy["global_alpha"])
            relation_alpha = float(policy["relation_alpha"].get(relation_id, global_alpha))
            row["crossfit_fold"] = fold + 1
            row["alpha_global_crossfit"] = global_alpha
            row["rr_global_crossfit"] = float(row[alpha_column(global_alpha)])
            row["alpha_relation_crossfit"] = relation_alpha
            row["rr_relation_crossfit"] = float(row[alpha_column(relation_alpha)])
        relation_specific = sum(
            detail["source"] == "relation_train_folds"
            for detail in policy["relation_details"].values()
        )
        fold_policies.append(
            {
                "fold": fold + 1,
                "train_triple_groups": len({triple_key(row) for row in train_rows}),
                "heldout_triple_groups": len({triple_key(row) for row in heldout_rows}),
                "train_observations": len(train_rows),
                "heldout_observations": len(heldout_rows),
                "global_alpha": policy["global_alpha"],
                "global_train_mrr": policy["global_train_mrr"],
                "relation_specific_count": relation_specific,
                "relation_fallback_count": len(policy["relation_details"])
                - relation_specific,
                "relation_details": policy["relation_details"],
            }
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "global_alpha": policy["global_alpha"],
                "global_mrr": metric(
                    [float(row["rr_global_crossfit"]) for row in heldout_rows]
                )["mrr"],
                "relation_mrr": metric(
                    [float(row["rr_relation_crossfit"]) for row in heldout_rows]
                )["mrr"],
                "mhyper_mrr": metric([float(row["rr_a"]) for row in heldout_rows])["mrr"],
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["crossfit_fold"]),
            int(row["seed"]),
            str(row["direction"]),
            int(row["relation_id"]),
            int(row["head_id"]),
            int(row["tail_id"]),
        )
    )
    results = summarize(
        rows,
        expert_a=str(selection["expert_a_name"]),
        expert_b=str(selection["expert_b_name"]),
    )
    by_seed = []
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for method, column in (
            ("M-Hyper", "rr_a"),
            ("Global alpha (5-fold cross-fit)", "rr_global_crossfit"),
            ("Relation alpha (5-fold cross-fit)", "rr_relation_crossfit"),
        ):
            by_seed.append({"seed": seed, "method": method, **metric([float(row[column]) for row in seed_rows])})

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_crossfit_query_rows.csv", rows)
    write_csv(out_dir / "dev_crossfit_results.csv", results)
    write_csv(out_dir / "dev_crossfit_results_by_seed.csv", by_seed)
    write_csv(out_dir / "dev_crossfit_results_by_fold.csv", fold_results)
    write_markdown(out_dir / "dev_crossfit_results.md", results)
    summary = {
        "schema_version": 1,
        "pair_name": selection["pair_name"],
        "dataset": selection["dataset"],
        "expert_a_name": selection["expert_a_name"],
        "expert_b_name": selection["expert_b_name"],
        "source_query_rows": str(query_path),
        "source_selection": str(selection_path),
        "seeds": seeds,
        "fold_audit": fold_audit,
        "leakage_guard": "all seeds and both directions of one original triple share one fold",
        "alpha_grid": list(alphas),
        "relation_min_support_observations": int(selection["relation_min_support"]),
        "results": results,
        "fold_results": fold_results,
        "fold_policies": fold_policies,
        "clustered_intervals": {
            "global_crossfit_vs_a": clustered_interval(rows, "rr_global_crossfit"),
            "relation_crossfit_vs_a": clustered_interval(rows, "rr_relation_crossfit"),
        },
        "interpretation_boundary": (
            "Cross-fit metrics are held out by original DEV triple. They remain DEV sanity checks "
            "and do not replace locked TEST evaluation."
        ),
    }
    (out_dir / "dev_crossfit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_dir / 'dev_crossfit_results.md'}")


if __name__ == "__main__":
    main()
