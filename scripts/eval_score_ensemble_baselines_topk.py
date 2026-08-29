from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_candidate_router_paper_tables import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple score-ensemble baselines on exported top-k candidate score sets."
    )
    parser.add_argument("--score-dir", default="outputs/candidate_router/scores")
    parser.add_argument("--output-dir", default="outputs/score_ensemble/eval")
    parser.add_argument("--paper-table-dir", default="docs/paper_tables")
    parser.add_argument(
        "--alphas",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0",
    )
    parser.add_argument("--baseline-summary", default="outputs/router/eval/clean/baseline_locked_summary.csv")
    parser.add_argument("--candidate-main-results", default="outputs/candidate_router/eval/tables/candidate_router_main_results.csv")
    parser.add_argument("--top-k", type=int, default=None, help="Select one exported top-k value; infer if unambiguous.")
    return parser.parse_args()


def parse_alpha_grid(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parquet_paths(score_dir: Path, split: str, requested_top_k: int | None = None) -> tuple[list[Path], int]:
    candidates = sorted(score_dir.glob(f"{split}_seed*_top*.parquet"))
    parsed: list[tuple[Path, int]] = []
    for path in candidates:
        match = re.search(r"_top(\d+)\.parquet$", path.name)
        if match:
            parsed.append((path, int(match.group(1))))
    if requested_top_k is not None:
        parsed = [item for item in parsed if item[1] == requested_top_k]
    top_ks = {top_k for _path, top_k in parsed}
    if len(top_ks) > 1:
        raise RuntimeError(f"Multiple top_k artifacts found for {split}: {sorted(top_ks)}; pass --top-k.")
    paths = [path for path, _top_k in parsed]
    if not paths:
        raise FileNotFoundError(f"No {split} parquet files found in {score_dir}")
    return paths, next(iter(top_ks))


def score_frame(df: pd.DataFrame, alpha: float | np.ndarray) -> pd.DataFrame:
    work = df[["query_id", "direction", "score_gate", "score_residual", "is_target"]].copy()
    if isinstance(alpha, np.ndarray):
        alpha_map = pd.Series(alpha, index=work["query_id"].drop_duplicates().to_numpy())
        work["_alpha"] = work["query_id"].map(alpha_map).astype(float)
    else:
        work["_alpha"] = float(alpha)
    work["_score"] = work["_alpha"] * work["score_gate"] + (1.0 - work["_alpha"]) * work["score_residual"]
    target_scores = work.loc[work["is_target"].eq(1), ["query_id", "_score"]].rename(columns={"_score": "_target_score"})
    work = work.merge(target_scores, on="query_id", how="left")
    outrank = work["_score"].gt(work["_target_score"]) & work["is_target"].ne(1)
    ranks = outrank.groupby(work["query_id"]).sum().astype(int) + 1
    meta = work.drop_duplicates("query_id")[["query_id", "direction"]].set_index("query_id")
    out = meta.join(ranks.rename("rank"))
    out["rr"] = 1.0 / out["rank"]
    return out.reset_index()


def metrics(rows: pd.DataFrame) -> dict:
    rank = rows["rank"].to_numpy()
    rr = rows["rr"].to_numpy()
    return {
        "count": int(len(rows)),
        "mrr": float(rr.mean()),
        "hits1": float((rank <= 1).mean()),
        "hits3": float((rank <= 3).mean()),
        "hits10": float((rank <= 10).mean()),
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("query_id", sort=False)
    feats = grouped.agg(
        direction=("direction", "first"),
        relation_id=("relation_id", "first"),
        gate_top1=("score_gate", "max"),
        gate_mean=("score_gate", "mean"),
        gate_std=("score_gate", "std"),
        residual_top1=("score_residual", "max"),
        residual_mean=("score_residual", "mean"),
        residual_std=("score_residual", "std"),
    )
    feats["direction_tail"] = feats["direction"].eq("tail").astype(float)
    feats["top1_diff"] = feats["gate_top1"] - feats["residual_top1"]
    feats["mean_diff"] = feats["gate_mean"] - feats["residual_mean"]
    feats["std_diff"] = feats["gate_std"].fillna(0.0) - feats["residual_std"].fillna(0.0)
    return feats.fillna(0.0)


def load_reference_metrics(baseline_summary: Path, candidate_main: Path) -> dict:
    baseline = pd.read_csv(baseline_summary)
    by_method = {row["method"]: row for _, row in baseline.iterrows()}
    candidate = pd.read_csv(candidate_main)
    ca_s2 = candidate[candidate["Method"].eq("CA-S2 score-aware")]
    return {
        "residual": float(by_method["Residual-only"]["mrr"]),
        "e5": float(by_method["Regression-based clean router"]["mrr"]),
        "ca_s2": float(str(ca_s2.iloc[0]["MRR"]).split()[0]),
    }


def select_policies(score_dir: Path, alphas: list[float], requested_top_k: int | None) -> tuple[dict, object, int]:
    alpha_rr = {alpha: [] for alpha in alphas}
    direction_rr = {"head": {alpha: [] for alpha in alphas}, "tail": {alpha: [] for alpha in alphas}}
    feature_blocks = []
    label_blocks = []
    dev_paths, top_k = parquet_paths(score_dir, "dev", requested_top_k)
    for path in dev_paths:
        print(f"[INFO] dev selection from {path}")
        df = pd.read_parquet(path)
        gate_rows = score_frame(df, 1.0).set_index("query_id")
        residual_rows = score_frame(df, 0.0).set_index("query_id")
        label = gate_rows["rr"].gt(residual_rows["rr"]).astype(int)
        feats = feature_frame(df)
        feature_blocks.append(feats[feature_columns()].to_numpy(dtype=np.float32))
        label_blocks.append(label.reindex(feats.index).fillna(0).to_numpy(dtype=np.int64))
        for alpha in alphas:
            rows = score_frame(df, alpha)
            alpha_rr[alpha].extend(rows["rr"].tolist())
            for direction in ["head", "tail"]:
                direction_rr[direction][alpha].extend(rows.loc[rows["direction"].eq(direction), "rr"].tolist())
    best_global = max(((alpha, float(np.mean(rr))) for alpha, rr in alpha_rr.items()), key=lambda item: (item[1], -item[0]))
    best_head = max(
        ((alpha, float(np.mean(rr))) for alpha, rr in direction_rr["head"].items()), key=lambda item: (item[1], -item[0])
    )
    best_tail = max(
        ((alpha, float(np.mean(rr))) for alpha, rr in direction_rr["tail"].items()), key=lambda item: (item[1], -item[0])
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0))
    model.fit(np.concatenate(feature_blocks, axis=0), np.concatenate(label_blocks, axis=0))
    return {
        "global_alpha": best_global[0],
        "global_dev_mrr": best_global[1],
        "head_alpha": best_head[0],
        "head_dev_mrr": best_head[1],
        "tail_alpha": best_tail[0],
        "tail_dev_mrr": best_tail[1],
    }, model, top_k


def feature_columns() -> list[str]:
    return [
        "direction_tail",
        "relation_id",
        "gate_top1",
        "gate_mean",
        "gate_std",
        "residual_top1",
        "residual_mean",
        "residual_std",
        "top1_diff",
        "mean_diff",
        "std_diff",
    ]


def evaluate_test(score_dir: Path, selection: dict, query_model, top_k: int) -> dict[str, pd.DataFrame]:
    outputs = {"global": [], "direction": [], "query": []}
    test_paths, test_top_k = parquet_paths(score_dir, "test", top_k)
    if test_top_k != top_k:
        raise RuntimeError(f"DEV top_k={top_k} does not match test top_k={test_top_k}")
    for path in test_paths:
        print(f"[INFO] test evaluation from {path}")
        df = pd.read_parquet(path)
        outputs["global"].append(score_frame(df, float(selection["global_alpha"])))
        head_alpha = float(selection["head_alpha"])
        tail_alpha = float(selection["tail_alpha"])
        q_alpha = df.drop_duplicates("query_id")["direction"].map({"head": head_alpha, "tail": tail_alpha}).to_numpy()
        outputs["direction"].append(score_frame(df, q_alpha))
        feats = feature_frame(df)
        pred_alpha = query_model.predict_proba(feats[feature_columns()].to_numpy(dtype=np.float32))[:, 1]
        outputs["query"].append(score_frame(df, pred_alpha.astype(np.float32)))
    return {key: pd.concat(parts, ignore_index=True) for key, parts in outputs.items()}


def result_row(method: str, granularity: str, alpha_policy: str, metric: dict, refs: dict, notes: str) -> dict:
    mrr = metric["mrr"]
    return {
        "method": method,
        "level": "ensemble",
        "granularity": granularity,
        "selected_on": "dev",
        "alpha_policy": alpha_policy,
        "mrr": mrr,
        "hits1": metric["hits1"],
        "hits3": metric["hits3"],
        "hits10": metric["hits10"],
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


def write_latex_table(path: Path, rows: list[dict], refs: dict, top_k: int) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Simple score-ensemble sanity baselines compared with CA-S2. Ensemble baselines use exported top-{top_k} candidate score sets, select policies on development data, and are reported as candidate-set sanity checks rather than official full-ranking replacements.}}",
        r"\label{tab:score_ensemble_baselines}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.30\textwidth}p{0.12\textwidth}p{0.13\textwidth}ccc}",
        r"\toprule",
        "Method & Level & Granularity & MRR & Delta vs E5 & Delta vs CA-S2" + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['level']} & {row['granularity']} & {fmt(row['mrr'])} & "
            f"{fmt_delta(row['delta_vs_e5'])} & {fmt_delta(row['delta_vs_ca_s2'])}" + r" \\"
        )
    lines.append(
        f"CA-S2 score-aware candidate router & router & candidate & {fmt(refs['ca_s2'])} & "
        f"{fmt_delta(refs['ca_s2'] - refs['e5'])} & --" + r" \\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\vspace{{0.4ex}}\caption*{{\footnotesize The ensemble rows are bounded sanity checks on the exported top-{top_k} candidate sets; CA-S2 is the official full-ranking result.}}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    refs = load_reference_metrics(Path(args.baseline_summary), Path(args.candidate_main_results))
    alphas = parse_alpha_grid(args.alphas)
    selection, query_model, top_k = select_policies(score_dir, alphas, args.top_k)
    test = evaluate_test(score_dir, selection, query_model, top_k)
    top_tag = f"top{top_k}"
    rows = [
        result_row(
            "Global score interpolation",
            f"{top_tag} global",
            f"alpha={selection['global_alpha']:.2f}",
            metrics(test["global"]),
            refs,
            f"{top_tag} candidate-set sanity; dev MRR={selection['global_dev_mrr']:.4f}",
        ),
        result_row(
            "Direction-specific score interpolation",
            f"{top_tag} direction",
            f"alpha_head={selection['head_alpha']:.2f}; alpha_tail={selection['tail_alpha']:.2f}",
            metrics(test["direction"]),
            refs,
            f"{top_tag} candidate-set sanity; dev head/tail MRR={selection['head_dev_mrr']:.4f}/{selection['tail_dev_mrr']:.4f}",
        ),
        result_row(
            "Query-level soft score weighting",
            f"{top_tag} query",
            "logistic p(Gate beats Residual) from score-distribution features",
            metrics(test["query"]),
            refs,
            f"{top_tag} candidate-set sanity; query-level soft alpha trained on dev-only expert wins",
        ),
    ]
    write_csv_rows(output_dir / "score_ensemble_baselines.csv", rows)
    (output_dir / "score_ensemble_baselines.json").write_text(
        json.dumps({"basis": f"{top_tag}_candidate_set", "top_k": top_k, "selection": selection, "reference_metrics": refs, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    md_frame = pd.DataFrame(rows)
    for col in ["mrr", "hits1", "hits3", "hits10", "delta_vs_residual", "delta_vs_e5", "delta_vs_ca_s2"]:
        md_frame[col] = md_frame[col].map(lambda value: f"{float(value):.4f}")
    (output_dir / "score_ensemble_baselines.md").write_text(markdown_table(md_frame) + "\n", encoding="utf-8")
    write_latex_table(Path(args.paper_table_dir) / "table_score_ensemble_baselines.tex", rows, refs, top_k)
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.csv'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.json'}")
    print(f"[OK] wrote {output_dir / 'score_ensemble_baselines.md'}")
    print(f"[OK] wrote {Path(args.paper_table_dir) / 'table_score_ensemble_baselines.tex'}")


if __name__ == "__main__":
    main()
