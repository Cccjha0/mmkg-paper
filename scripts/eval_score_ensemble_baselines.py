from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_candidate_soft_router_full import (
    score_full_matrix,
    target_ids_for_direction,
    target_ranks_and_rr,
)
from scripts.export_candidate_scores import build_filtered_indexes, load_run, load_split_triples, resolve_device
from scripts.build_candidate_router_paper_tables import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple Gate-only / Residual-only score-ensemble baselines."
    )
    parser.add_argument("--score-dir", default="outputs/candidate_router/scores")
    parser.add_argument("--output-dir", default="outputs/score_ensemble/eval")
    parser.add_argument("--paper-table-dir", default="docs/paper_tables")
    parser.add_argument(
        "--paper-figures-dir",
        default="docs/paper/figures",
        help="Optional mirror location for LaTeX table inputs used directly by docs/paper/manuscript_main.tex.",
    )
    parser.add_argument("--split", default="test", choices=["test"], help="Final reporting split.")
    parser.add_argument("--selection-split", default="dev", choices=["dev"], help="Split used for alpha selection.")
    parser.add_argument(
        "--alphas",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0",
    )
    parser.add_argument("--direction", default="both", choices=["both"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--relation-min-support",
        type=int,
        default=20,
        help="Minimum dev queries required before selecting a relation-specific alpha; lower-support relations fall back to global alpha.",
    )
    parser.add_argument("--baseline-summary", default="outputs/router/eval/clean/baseline_locked_summary.csv")
    parser.add_argument("--candidate-main-results", default="outputs/candidate_router/eval/tables/candidate_router_main_results.csv")
    return parser.parse_args()


def parse_alpha_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("alpha grid is empty")
    return values


def load_run_pairs(score_dir: Path, split: str) -> list[dict]:
    pairs = []
    for path in sorted(score_dir.glob(f"{split}_seed*_top100_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs.append(
            {
                "seed": int(payload["seed"]),
                "gate_run_dir": payload["gate_run_dir"],
                "residual_run_dir": payload["residual_run_dir"],
                "summary_path": str(path),
            }
        )
    if not pairs:
        raise FileNotFoundError(f"No {split} score summaries found in {score_dir}")
    return sorted(pairs, key=lambda row: row["seed"])


def safe_scores(scores: torch.Tensor) -> torch.Tensor:
    finite = scores[torch.isfinite(scores)]
    low = float(finite.min().item()) - 1.0 if finite.numel() else -100.0
    high = float(finite.max().item()) + 1.0 if finite.numel() else 100.0
    return torch.nan_to_num(scores, nan=low, posinf=high, neginf=low)


def query_features(gate_scores: torch.Tensor, residual_scores: torch.Tensor, direction: str, relations: torch.Tensor) -> np.ndarray:
    def stats(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe = safe_scores(scores)
        top = torch.topk(safe, k=min(5, safe.size(1)), dim=1).values
        top1 = top[:, 0]
        top2 = top[:, 1] if top.size(1) > 1 else top[:, 0]
        return top1, top[:, : min(5, top.size(1))].mean(dim=1), top1 - top2, safe.std(dim=1)

    g1, g5, gm, gs = stats(gate_scores)
    r1, r5, rm, rs = stats(residual_scores)
    direction_tail = torch.full_like(g1, 1.0 if direction == "tail" else 0.0)
    rel = relations.to(dtype=torch.float32)
    features = torch.stack(
        [
            direction_tail,
            rel,
            g1,
            g5,
            gm,
            gs,
            r1,
            r5,
            rm,
            rs,
            g1 - r1,
            g5 - r5,
            gm - rm,
            gs - rs,
        ],
        dim=1,
    )
    return features.detach().cpu().numpy().astype(np.float32)


def eval_mixed_rr(gate_scores: torch.Tensor, residual_scores: torch.Tensor, target_ids: torch.Tensor, alpha: float | np.ndarray) -> list[float]:
    gate_safe = safe_scores(gate_scores)
    residual_safe = safe_scores(residual_scores)
    if isinstance(alpha, np.ndarray):
        alpha_t = torch.tensor(alpha, dtype=gate_safe.dtype).view(-1, 1)
    else:
        alpha_t = torch.full((gate_safe.size(0), 1), float(alpha), dtype=gate_safe.dtype)
    mixed = alpha_t * gate_safe + (1.0 - alpha_t) * residual_safe
    both_filtered = (~torch.isfinite(gate_scores)) & (~torch.isfinite(residual_scores))
    mixed[both_filtered] = float("-inf")
    _, rr = target_ranks_and_rr(mixed, target_ids)
    return rr


def evaluate_split(
    *,
    run_pairs: list[dict],
    split: str,
    alphas: list[float],
    device_arg: str | None,
    chunk_size_arg: int | None,
    query_batch_size_arg: int | None,
    max_queries: int | None,
    query_model=None,
    selected_global_alpha: float | None = None,
    selected_direction_alpha: dict[str, float] | None = None,
    selected_relation_alpha: dict[int, float] | None = None,
    selected_relation_fallback_alpha: float | None = None,
) -> dict:
    alpha_rr: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    alpha_direction_rr: dict[str, dict[float, list[float]]] = {
        "head": {alpha: [] for alpha in alphas},
        "tail": {alpha: [] for alpha in alphas},
    }
    alpha_relation_rr: dict[int, dict[float, list[float]]] = defaultdict(lambda: {alpha: [] for alpha in alphas})
    global_rows: list[dict] = []
    direction_rows: list[dict] = []
    relation_rows: list[dict] = []
    query_rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for pair in run_pairs:
        gate_cfg_raw = json.loads((Path(pair["gate_run_dir"]) / "config_merged.json").read_text(encoding="utf-8"))
        device = resolve_device(device_arg, gate_cfg_raw.get("system", {}).get("device", "cuda"))
        gate_cfg, gate_model, gate_num_entities = load_run(pair["gate_run_dir"], device)
        residual_cfg, residual_model, residual_num_entities = load_run(pair["residual_run_dir"], device)
        if gate_num_entities != residual_num_entities:
            raise RuntimeError("Gate and Residual entity counts differ.")
        seed = int(gate_cfg.get("system", {}).get("seed", pair["seed"]))
        if seed != int(residual_cfg.get("system", {}).get("seed", seed)):
            raise RuntimeError(f"Seed mismatch for pair {pair}")
        triples = load_split_triples(gate_cfg, split)
        true_tails_idx, true_heads_idx = build_filtered_indexes(gate_cfg)
        ev_cfg = gate_cfg.get("evaluation", {})
        chunk_size = int(chunk_size_arg or ev_cfg.get("chunk_size", 4096))
        query_batch_size = int(query_batch_size_arg or ev_cfg.get("query_batch_size", 8))
        directions = ["head", "tail"]
        max_queries_per_direction = None if max_queries is None else max(1, max_queries // len(directions))

        for direction in directions:
            true_index = true_tails_idx if direction == "tail" else true_heads_idx
            triples_eval = triples[:max_queries_per_direction] if max_queries_per_direction else triples
            triples_t = torch.tensor(triples_eval, dtype=torch.long)
            for q_start in range(0, triples_t.size(0), query_batch_size):
                q_end = min(triples_t.size(0), q_start + query_batch_size)
                q_cpu = triples_t[q_start:q_end]
                gate_scores = score_full_matrix(gate_model, q_cpu, direction, true_index, gate_num_entities, chunk_size, device)
                residual_scores = score_full_matrix(
                    residual_model, q_cpu, direction, true_index, residual_num_entities, chunk_size, device
                )
                target_ids = target_ids_for_direction(q_cpu, direction)
                _, gate_rr = target_ranks_and_rr(gate_scores, target_ids)
                _, residual_rr = target_ranks_and_rr(residual_scores, target_ids)
                feats = query_features(gate_scores, residual_scores, direction, q_cpu[:, 1])
                feature_rows.append(feats)
                labels.extend([int(g > r) for g, r in zip(gate_rr, residual_rr)])

                for alpha in alphas:
                    rr = eval_mixed_rr(gate_scores, residual_scores, target_ids, alpha)
                    alpha_rr[alpha].extend(rr)
                    alpha_direction_rr[direction][alpha].extend(rr)
                    for relation_id, value in zip(q_cpu[:, 1].tolist(), rr):
                        alpha_relation_rr[int(relation_id)][alpha].append(value)

                if selected_global_alpha is not None:
                    rr = eval_mixed_rr(gate_scores, residual_scores, target_ids, selected_global_alpha)
                    global_rows.extend({"mixed_rr": value} for value in rr)
                if selected_direction_alpha is not None:
                    rr = eval_mixed_rr(gate_scores, residual_scores, target_ids, selected_direction_alpha[direction])
                    direction_rows.extend({"mixed_rr": value} for value in rr)
                if selected_relation_alpha is not None:
                    fallback_alpha = selected_relation_fallback_alpha
                    if fallback_alpha is None:
                        fallback_alpha = selected_global_alpha if selected_global_alpha is not None else 0.0
                    relation_alpha = np.array(
                        [
                            selected_relation_alpha.get(int(relation_id), float(fallback_alpha))
                            for relation_id in q_cpu[:, 1].tolist()
                        ],
                        dtype=np.float32,
                    )
                    rr = eval_mixed_rr(gate_scores, residual_scores, target_ids, relation_alpha)
                    relation_rows.extend({"mixed_rr": value} for value in rr)
                if query_model is not None:
                    pred_alpha = query_model.predict_proba(feats)[:, 1].astype(np.float32)
                    rr = eval_mixed_rr(gate_scores, residual_scores, target_ids, pred_alpha)
                    query_rows.extend({"mixed_rr": value, "alpha": float(a)} for value, a in zip(rr, pred_alpha))

    return {
        "alpha_rr": alpha_rr,
        "alpha_direction_rr": alpha_direction_rr,
        "alpha_relation_rr": {relation_id: dict(alpha_map) for relation_id, alpha_map in alpha_relation_rr.items()},
        "global_rows": global_rows,
        "direction_rows": direction_rows,
        "relation_rows": relation_rows,
        "query_rows": query_rows,
        "features": np.concatenate(feature_rows, axis=0) if feature_rows else np.empty((0, 0), dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def best_alpha(alpha_rr: dict[float, list[float]]) -> tuple[float, float]:
    scored = [(alpha, float(np.mean(rr)) if rr else 0.0) for alpha, rr in alpha_rr.items()]
    return max(scored, key=lambda item: (item[1], -item[0]))


def select_relation_alphas(
    alpha_relation_rr: dict[int, dict[float, list[float]]],
    *,
    fallback_alpha: float,
    min_support: int,
) -> tuple[dict[int, float], dict]:
    selected: dict[int, float] = {}
    summary_rows = []
    for relation_id, alpha_map in sorted(alpha_relation_rr.items()):
        support = max((len(values) for values in alpha_map.values()), default=0)
        if support >= min_support:
            alpha, dev_mrr = best_alpha(alpha_map)
            used_fallback = False
        else:
            alpha = fallback_alpha
            dev_mrr = float(np.mean(alpha_map.get(fallback_alpha, []))) if alpha_map.get(fallback_alpha) else 0.0
            used_fallback = True
        selected[int(relation_id)] = float(alpha)
        summary_rows.append(
            {
                "relation_id": int(relation_id),
                "support": int(support),
                "alpha": float(alpha),
                "dev_mrr": float(dev_mrr),
                "used_fallback": bool(used_fallback),
            }
        )
    summary = {
        "min_support": int(min_support),
        "fallback_alpha": float(fallback_alpha),
        "n_relations": len(summary_rows),
        "n_relation_specific": sum(1 for row in summary_rows if not row["used_fallback"]),
        "n_fallback": sum(1 for row in summary_rows if row["used_fallback"]),
        "relations": summary_rows,
    }
    return selected, summary


def rows_to_metrics(rows: list[dict]) -> dict:
    rr = np.array([float(row["mixed_rr"]) for row in rows], dtype=np.float64)
    ranks = np.rint(1.0 / np.maximum(rr, 1e-12)).astype(np.int64)
    return {
        "count": int(rr.size),
        "mrr": float(rr.mean()) if rr.size else 0.0,
        "hits1": float((ranks <= 1).mean()) if rr.size else 0.0,
        "hits3": float((ranks <= 3).mean()) if rr.size else 0.0,
        "hits10": float((ranks <= 10).mean()) if rr.size else 0.0,
    }


def load_reference_metrics(baseline_summary: Path, candidate_main: Path) -> dict:
    baseline = pd.read_csv(baseline_summary)
    by_method = {row["method"]: row for _, row in baseline.iterrows()}
    candidate = pd.read_csv(candidate_main)
    ca_s2 = candidate[candidate["Method"].eq("CA-S2 score-aware")]
    if ca_s2.empty:
        raise RuntimeError("Could not find CA-S2 row in candidate main results.")
    return {
        "residual": float(by_method["Residual-only"]["mrr"]),
        "e5": float(by_method["Regression-based clean router"]["mrr"]),
        "ca_s2": float(str(ca_s2.iloc[0]["MRR"]).split()[0]),
    }


def result_row(method: str, granularity: str, alpha_policy: str, metrics: dict, refs: dict, notes: str) -> dict:
    mrr = float(metrics["mrr"])
    return {
        "method": method,
        "level": "ensemble",
        "granularity": granularity,
        "selected_on": "dev",
        "alpha_policy": alpha_policy,
        "mrr": mrr,
        "hits1": float(metrics["hits1"]),
        "hits3": float(metrics["hits3"]),
        "hits10": float(metrics["hits10"]),
        "delta_vs_residual": mrr - refs["residual"],
        "delta_vs_e5": mrr - refs["e5"],
        "delta_vs_ca_s2": mrr - refs["ca_s2"],
        "notes": notes,
    }


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def write_latex_table(path: Path, rows: list[dict], refs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paper_rows = []
    for row in rows:
        paper_rows.append(
            {
                "Method": row["method"],
                "Level": row["level"],
                "Granularity": row["granularity"],
                "MRR": fmt(row["mrr"]),
                "Delta vs E5": fmt_delta(row["delta_vs_e5"]),
                "Delta vs CA-S2": "--" if row["method"].startswith("CA-S2") else fmt_delta(row["delta_vs_ca_s2"]),
            }
        )
    paper_rows.append(
        {
            "Method": "CA-S2 score-aware candidate router",
            "Level": "router",
            "Granularity": "candidate",
            "MRR": fmt(refs["ca_s2"]),
            "Delta vs E5": fmt_delta(refs["ca_s2"] - refs["e5"]),
            "Delta vs CA-S2": "--",
        }
    )
    frame = pd.DataFrame(paper_rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Simple score-ensemble baselines compared with CA-S2 under full filtered ranking. Ensemble baselines use fixed Gate-only and Residual-only scores, select their policies on the development split, and are evaluated on the test split.}",
        r"\label{tab:score_ensemble_baselines}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.30\textwidth}p{0.12\textwidth}p{0.13\textwidth}ccc}",
        r"\toprule",
        "Method & Level & Granularity & MRR & Delta vs E5 & Delta vs CA-S2" + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(str(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}\caption*{\footnotesize The baselines test whether CA-S2 can be explained by fixed, direction-specific, relation-specific, or query-level score averaging alone.}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    alphas = parse_alpha_grid(args.alphas)
    refs = load_reference_metrics(Path(args.baseline_summary), Path(args.candidate_main_results))

    dev_pairs = load_run_pairs(score_dir, args.selection_split)
    test_pairs = load_run_pairs(score_dir, args.split)

    print("[INFO] evaluating development split for policy selection")
    dev = evaluate_split(
        run_pairs=dev_pairs,
        split=args.selection_split,
        alphas=alphas,
        device_arg=args.device,
        chunk_size_arg=args.chunk_size,
        query_batch_size_arg=args.query_batch_size,
        max_queries=args.max_queries,
    )
    global_alpha, global_dev_mrr = best_alpha(dev["alpha_rr"])
    head_alpha, head_dev_mrr = best_alpha(dev["alpha_direction_rr"]["head"])
    tail_alpha, tail_dev_mrr = best_alpha(dev["alpha_direction_rr"]["tail"])
    relation_alpha, relation_summary = select_relation_alphas(
        dev["alpha_relation_rr"],
        fallback_alpha=global_alpha,
        min_support=args.relation_min_support,
    )

    query_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    query_model.fit(dev["features"], dev["labels"])

    print(
        "[INFO] selected policies: "
        f"global alpha={global_alpha:.2f} dev_mrr={global_dev_mrr:.4f}; "
        f"head alpha={head_alpha:.2f} dev_mrr={head_dev_mrr:.4f}; "
        f"tail alpha={tail_alpha:.2f} dev_mrr={tail_dev_mrr:.4f}; "
        f"relation-specific={relation_summary['n_relation_specific']} fallback={relation_summary['n_fallback']}"
    )
    print("[INFO] evaluating test split")
    test = evaluate_split(
        run_pairs=test_pairs,
        split=args.split,
        alphas=alphas,
        device_arg=args.device,
        chunk_size_arg=args.chunk_size,
        query_batch_size_arg=args.query_batch_size,
        max_queries=args.max_queries,
        query_model=query_model,
        selected_global_alpha=global_alpha,
        selected_direction_alpha={"head": head_alpha, "tail": tail_alpha},
        selected_relation_alpha=relation_alpha,
        selected_relation_fallback_alpha=global_alpha,
    )

    rows = [
        result_row(
            "Global score interpolation",
            "global",
            f"alpha={global_alpha:.2f}",
            rows_to_metrics(test["global_rows"]),
            refs,
            f"alpha selected by dev MRR ({global_dev_mrr:.4f})",
        ),
        result_row(
            "Direction-specific score interpolation",
            "direction",
            f"alpha_head={head_alpha:.2f}; alpha_tail={tail_alpha:.2f}",
            rows_to_metrics(test["direction_rows"]),
            refs,
            f"head/tail alphas selected independently on dev MRR ({head_dev_mrr:.4f}/{tail_dev_mrr:.4f})",
        ),
        result_row(
            "Relation-specific score interpolation",
            "relation",
            f"per-relation alpha; fallback alpha={global_alpha:.2f}; min_support={args.relation_min_support}",
            rows_to_metrics(test["relation_rows"]),
            refs,
            f"relation alphas selected on dev MRR; {relation_summary['n_relation_specific']} relations selected, {relation_summary['n_fallback']} used fallback",
        ),
        result_row(
            "Query-level soft score weighting",
            "query",
            "logistic p(Gate beats Residual) from score-distribution features",
            rows_to_metrics(test["query_rows"]),
            refs,
            "query-level soft alpha uses non-answer-aware score-distribution features; labels are dev-only expert wins",
        ),
    ]

    write_csv_rows(output_dir / "score_ensemble_baselines.csv", rows)
    (output_dir / "score_ensemble_baselines.json").write_text(
        json.dumps(
            {
                "selection": {
                    "global_alpha": global_alpha,
                    "global_dev_mrr": global_dev_mrr,
                    "head_alpha": head_alpha,
                    "head_dev_mrr": head_dev_mrr,
                    "tail_alpha": tail_alpha,
                    "tail_dev_mrr": tail_dev_mrr,
                    "relation": relation_summary,
                    "alpha_grid": alphas,
                },
                "reference_metrics": refs,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_frame = pd.DataFrame(rows)
    for col in ["mrr", "hits1", "hits3", "hits10", "delta_vs_residual", "delta_vs_e5", "delta_vs_ca_s2"]:
        md_frame[col] = md_frame[col].map(lambda value: f"{float(value):.4f}")
    (output_dir / "score_ensemble_baselines.md").write_text(markdown_table(md_frame) + "\n", encoding="utf-8")
    paper_table_path = Path(args.paper_table_dir) / "table_score_ensemble_baselines.tex"
    write_latex_table(paper_table_path, rows, refs)
    paper_figures_path = Path(args.paper_figures_dir) / "table_score_ensemble_baselines.tex"
    if paper_figures_path != paper_table_path:
        write_latex_table(paper_figures_path, rows, refs)
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.csv'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.json'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.md'}")
    print(f"[OK] wrote {paper_table_path}")
    if paper_figures_path != paper_table_path:
        print(f"[OK] wrote {paper_figures_path}")


if __name__ == "__main__":
    main()
