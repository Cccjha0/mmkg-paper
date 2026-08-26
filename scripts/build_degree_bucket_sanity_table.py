from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


MODEL_INPUTS = {
    "Residual-only": "residual_only_query_eval_seed*.csv",
    "Full Model": "full_model_query_eval_seed*.csv",
    "Gate-only": "gate_only_query_eval_seed*.csv",
}

MODEL_ORDER = ["Residual-only", "Full Model", "Gate-only"]
BUCKET_ORDER = ["0-1", "2-5", "6-20", ">20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a degree-bucket sanity table from locked OpenBG-IMG query-evaluation files."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/datasets/openbg_img/paper_split"),
        help="Directory containing OpenBG-IMG_paper_train.tsv.",
    )
    parser.add_argument(
        "--query-eval-dir",
        type=Path,
        default=Path("outputs/router/test"),
        help="Directory containing *_query_eval_seed*.csv files.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/paper_tables/table_degree_bucket_sanity_check.csv"),
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=Path("docs/paper_tables/table_degree_bucket_sanity_check.tex"),
    )
    return parser.parse_args()


def entity_id(text: str) -> int:
    if text.startswith("ent_"):
        return int(text.split("_", 1)[1])
    return int(text)


def load_train_degree(path: Path) -> Counter[int]:
    degree: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                head = entity_id(parts[0])
                tail = entity_id(parts[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} has an invalid entity id") from exc
            degree[head] += 1
            degree[tail] += 1
    return degree


def degree_bucket(value: int) -> str:
    if value <= 1:
        return "0-1"
    if value <= 5:
        return "2-5"
    if value <= 20:
        return "6-20"
    return ">20"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace(">", r"$>$")


def fmt_mrr(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def interpretation(row: pd.Series) -> str:
    bucket = row["degree_bucket"]
    best = row["best_model"]
    if bucket == ">20":
        return "high-degree tail-side queries drive the structural advantage"
    if bucket == "6-20":
        return "head-dominated bucket; all experts remain weak"
    if bucket == "2-5":
        return "low-degree mixed bucket favors the full model"
    return "small sparse bucket; non-structural experts are stronger"


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    train_path = args.split_dir / "OpenBG-IMG_paper_train.tsv"
    degree = load_train_degree(train_path)

    frames = []
    for model_name, pattern in MODEL_INPUTS.items():
        paths = sorted(args.query_eval_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No files matched {args.query_eval_dir / pattern}")
        for path in paths:
            frame = pd.read_csv(
                path,
                usecols=["query_id", "target_entity_id", "target_regime", "rr", "seed"],
            )
            frame["model"] = model_name
            frame["target_degree"] = frame["target_entity_id"].map(lambda item: degree[int(item)])
            frame["degree_bucket"] = frame["target_degree"].map(degree_bucket)
            frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["degree_bucket"] = pd.Categorical(all_rows["degree_bucket"], BUCKET_ORDER, ordered=True)

    metric_rows = []
    for bucket in BUCKET_ORDER:
        bucket_rows = all_rows[all_rows["degree_bucket"] == bucket]
        if bucket_rows.empty:
            continue

        first_model_rows = bucket_rows[bucket_rows["model"] == MODEL_ORDER[0]]
        unique_queries = first_model_rows.drop_duplicates(["seed", "query_id"])
        regime_counts = unique_queries["target_regime"].value_counts()
        dominant_regime = str(regime_counts.idxmax())
        dominant_share = float(regime_counts.max() / regime_counts.sum())

        by_model = bucket_rows.groupby("model", observed=True)["rr"].mean().to_dict()
        best_model = max(MODEL_ORDER, key=lambda name: by_model[name])
        metric_rows.append(
            {
                "degree_bucket": bucket,
                "n_seed_queries": int(len(first_model_rows)),
                "n_unique_test_queries": int(len(first_model_rows) / first_model_rows["seed"].nunique()),
                "mean_degree": float(first_model_rows["target_degree"].mean()),
                "dominant_regime": dominant_regime,
                "dominant_regime_share": dominant_share,
                "residual_only_mrr": float(by_model["Residual-only"]),
                "full_model_mrr": float(by_model["Full Model"]),
                "gate_only_mrr": float(by_model["Gate-only"]),
                "delta_residual_vs_full": float(by_model["Residual-only"] - by_model["Full Model"]),
                "best_model": best_model,
            }
        )

    result = pd.DataFrame(metric_rows)
    result["interpretation"] = result.apply(interpretation, axis=1)
    return result


def write_tex(path: Path, rows: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Degree-bucket sanity check for fixed experts.}",
        r"\label{tab:degree_bucket_sanity_check}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.10\textwidth}rp{0.19\textwidth}ccccp{0.25\textwidth}}",
        r"\toprule",
        (
            r"Degree bucket & \#Queries & Dominant regime & Residual-only & "
            r"Full Model & Gate-only & $\Delta$ Res.-Full & Interpretation \\"
        ),
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        dominant = f"{latex_escape(row['dominant_regime'])} ({fmt_percent(row['dominant_regime_share'])})"
        best_values = {
            "Residual-only": fmt_mrr(row["residual_only_mrr"]),
            "Full Model": fmt_mrr(row["full_model_mrr"]),
            "Gate-only": fmt_mrr(row["gate_only_mrr"]),
        }
        best_values[row["best_model"]] = r"\textbf{" + best_values[row["best_model"]] + "}"
        lines.append(
            f"{latex_escape(row['degree_bucket'])} & "
            f"{int(row['n_unique_test_queries']):,} & "
            f"{dominant} & "
            f"{best_values['Residual-only']} & "
            f"{best_values['Full Model']} & "
            f"{best_values['Gate-only']} & "
            f"{fmt_delta(row['delta_residual_vs_full'])} & "
            f"{row['interpretation']} "
            + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} Degree is the undirected count of training triples "
                r"in which the target entity appears. MRR is averaged over three seed-specific evaluations; "
                r"\#Queries reports the corresponding number of test queries per seed. The buckets expose the "
                r"strong coupling between target degree and prediction-side regime: low- and mid-degree buckets "
                r"are mostly head-side queries, whereas the highest-degree bucket is almost entirely tail-side.}"
            ),
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.csv_out, index=False)
    write_tex(args.tex_out, rows)
    print(f"[OK] wrote {args.csv_out}")
    print(f"[OK] wrote {args.tex_out}")


if __name__ == "__main__":
    main()
