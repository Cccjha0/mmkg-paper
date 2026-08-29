import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import write_csv, write_json
from router.prior_utils import compute_relation_gain_stats, summarize_relation_gain_stats


OUTPUT_HEADER = [
    "relation_id",
    "support",
    "fusion_mean_rr",
    "struct_mean_rr",
    "mean_rr_gain",
    "mean_rr_gain_raw",
    "mean_rr_gain_shrunk",
    "fusion_win_rate",
    "struct_win_rate",
    "tie_rate",
    "is_visual_prior",
]

GENERAL_OUTPUT_HEADER = [*OUTPUT_HEADER[:-1], "is_fusion_prior"]


LEGACY_OUTPUT_HEADER = [
    "relation_id",
    "relation_name",
    "n_queries",
    "mean_rr_gate",
    "mean_rr_residual",
    "mean_delta_rr",
    "mean_delta_rr_raw",
    "mean_delta_rr_shrunk",
    "fusion_win_rate",
    "struct_win_rate",
    "head_has_img_ratio",
    "tail_no_img_ratio",
    "is_visual_prior",
]

GENERAL_RELATION_OUTPUT_HEADER = [
    "relation_id",
    "relation_name",
    "n_queries",
    "mean_rr_gate",
    "mean_rr_residual",
    "mean_delta_rr",
    "mean_delta_rr_raw",
    "mean_delta_rr_shrunk",
    "fusion_win_rate",
    "struct_win_rate",
    *[
        f"{direction}_{tag}_ratio"
        for direction in ("head", "tail")
        for tag in ("T0V0", "T0V1", "T1V0", "T1V1")
    ],
    "is_fusion_prior",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dev-only relation priors for router features.")
    parser.add_argument("--gate-dev", default=None, help="Dev-side gate-only query outcomes parquet/csv")
    parser.add_argument("--residual-dev", default=None, help="Dev-side residual-only query outcomes parquet/csv")
    parser.add_argument("--out", default=None, help="Output CSV path for the checklist contract")

    parser.add_argument("--split", default="dev", choices=["dev"], help="Phase 2 currently uses dev only")
    parser.add_argument("--gamma", type=float, default=0.0, help="fusion-gain prior threshold on mean_rr_gain")
    parser.add_argument("--use-shrinkage", action="store_true", help="Apply shrinkage to mean delta RR before marking priors")
    parser.add_argument("--shrink-k", type=float, default=20.0, help="Shrinkage denominator constant")
    parser.add_argument("--gate-dir", default="outputs/router/dev")
    parser.add_argument("--residual-dir", default="outputs/router/dev")
    parser.add_argument("--out-dir", default="outputs/router/priors")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def gamma_tag(gamma: float) -> str:
    return f"{float(gamma):.3f}"


def infer_seed_from_path(path: Path) -> int:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        raise ValueError(f"Cannot infer seed from filename: {path}")
    return int(stem.split(marker)[-1])


def find_seed_files(base_dir: Path, expert_prefix: str) -> dict[int, Path]:
    files = sorted(base_dir.glob(f"{expert_prefix}_query_eval_seed*.csv"))
    return {infer_seed_from_path(path): path for path in files}


def normalize_query_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "query_id": str(row["query_id"]),
                "relation_id": int(row["relation_id"]),
                "relation_name": row.get("relation_name", f"rel_{int(row['relation_id']):04d}"),
                "rr": float(row["rr"]),
                "target_regime": str(row["target_regime"]),
            }
        )
    return out


def validate_dataset_local(gate_rows: list[dict], residual_rows: list[dict]) -> dict[str, str]:
    """Reject accidental cross-dataset priors when general exports carry metadata."""
    for expert, rows in (("gate", gate_rows), ("residual", residual_rows)):
        splits = {str(row.get("split", "")) for row in rows}
        if splits != {"dev"}:
            raise RuntimeError(f"Relation priors are validation-only; {expert} rows contain splits={sorted(splits)}")
    metadata: dict[str, str] = {}
    for field in ("dataset", "protocol_version"):
        gate_values = {str(row[field]) for row in gate_rows if row.get(field) not in (None, "")}
        residual_values = {str(row[field]) for row in residual_rows if row.get(field) not in (None, "")}
        if not gate_values and not residual_values:
            continue  # Frozen legacy CSVs predate these columns.
        if len(gate_values) != 1 or gate_values != residual_values:
            raise RuntimeError(
                f"Relation priors must be dataset-local: {field} gate={sorted(gate_values)} "
                f"residual={sorted(residual_values)}"
            )
        metadata[field] = next(iter(gate_values))
    return metadata


def convert_rows_for_contract(rows: list[dict], *, general_protocol: bool = False) -> list[dict]:
    out = []
    for row in rows:
        support = int(row["n_queries"])
        fusion_win_rate = float(row["fusion_win_rate"])
        struct_win_rate = float(row["struct_win_rate"])
        tie_rate = max(0.0, 1.0 - fusion_win_rate - struct_win_rate)
        record = {
                "relation_id": int(row["relation_id"]),
                "support": support,
                "fusion_mean_rr": float(row["mean_rr_gate"]),
                "struct_mean_rr": float(row["mean_rr_residual"]),
                "mean_rr_gain": float(row["mean_delta_rr"]),
                "mean_rr_gain_raw": float(row["mean_delta_rr_raw"]),
                "mean_rr_gain_shrunk": float(row["mean_delta_rr_shrunk"]),
                "fusion_win_rate": fusion_win_rate,
                "struct_win_rate": struct_win_rate,
                "tie_rate": tie_rate,
        }
        record["is_fusion_prior" if general_protocol else "is_visual_prior"] = int(row["is_visual_prior"])
        out.append(record)
    return out


def generalize_summary_names(summary: dict) -> dict:
    out = dict(summary)
    if "n_visual_prior" in out:
        out["n_fusion_prior"] = out.pop("n_visual_prior")
    if "visual_prior_rate" in out:
        out["fusion_prior_rate"] = out.pop("visual_prior_rate")
    return out


def build_from_contract_inputs(
    gate_dev: Path,
    residual_dev: Path,
    out_path: Path,
    gamma: float,
    summary_json: str | None,
    use_shrinkage: bool,
    shrink_k: float,
) -> None:
    raw_gate_rows = read_table(gate_dev)
    raw_residual_rows = read_table(residual_dev)
    dataset_metadata = validate_dataset_local(raw_gate_rows, raw_residual_rows)
    gate_rows = normalize_query_rows(raw_gate_rows)
    residual_rows = normalize_query_rows(raw_residual_rows)
    legacy_rows = compute_relation_gain_stats(
        gate_rows,
        residual_rows,
        gamma,
        use_shrinkage=use_shrinkage,
        shrink_k=shrink_k,
    )
    general_protocol = dataset_metadata.get("protocol_version") == "mmkg_general_v1"
    contract_rows = convert_rows_for_contract(legacy_rows, general_protocol=general_protocol)
    write_csv(out_path, contract_rows, GENERAL_OUTPUT_HEADER if general_protocol else OUTPUT_HEADER)
    print(f"[OK] wrote relation priors -> {out_path.as_posix()}")

    summary = summarize_relation_gain_stats(legacy_rows, gamma)
    if general_protocol:
        summary = generalize_summary_names(summary)
    summary["split"] = "dev"
    summary["use_shrinkage"] = bool(use_shrinkage)
    summary["shrink_k"] = float(shrink_k)
    summary["source_gate_dev"] = gate_dev.as_posix()
    summary["source_residual_dev"] = residual_dev.as_posix()
    summary.update(dataset_metadata)
    summary_path = Path(summary_json) if summary_json else out_path.with_name(out_path.stem + "_summary.json")
    write_json(summary_path, summary)
    print(f"[OK] wrote summary         -> {summary_path.as_posix()}")


def build_from_legacy_dirs(args: argparse.Namespace) -> None:
    gate_dir = Path(args.gate_dir)
    residual_dir = Path(args.residual_dir)
    out_dir = Path(args.out_dir)

    gate_files = find_seed_files(gate_dir, "gate_only")
    residual_files = find_seed_files(residual_dir, "residual_only")
    common_seeds = sorted(set(gate_files) & set(residual_files))
    if not common_seeds:
        raise RuntimeError("No overlapping gate/residual query_eval seed files found.")

    all_gate_rows: list[dict] = []
    all_residual_rows: list[dict] = []
    dataset_metadata: dict[str, str] = {}
    for seed in common_seeds:
        raw_gate = read_table(gate_files[seed])
        raw_residual = read_table(residual_files[seed])
        seed_metadata = validate_dataset_local(raw_gate, raw_residual)
        if dataset_metadata and seed_metadata != dataset_metadata:
            raise RuntimeError("Relation prior seeds contain different dataset/protocol metadata.")
        dataset_metadata = dataset_metadata or seed_metadata
        all_gate_rows.extend(normalize_query_rows(raw_gate))
        all_residual_rows.extend(normalize_query_rows(raw_residual))

    rows = compute_relation_gain_stats(
        all_gate_rows,
        all_residual_rows,
        args.gamma,
        use_shrinkage=args.use_shrinkage,
        shrink_k=args.shrink_k,
    )
    out_path = out_dir / f"relation_gain_stats_gamma_{gamma_tag(args.gamma)}.csv"
    general_protocol = dataset_metadata.get("protocol_version") == "mmkg_general_v1"
    header = GENERAL_RELATION_OUTPUT_HEADER if general_protocol else LEGACY_OUTPUT_HEADER
    write_csv(out_path, [{key: row[key] for key in header} for row in rows], header)
    print(f"[OK] wrote relation priors -> {out_path.as_posix()}")

    summary = summarize_relation_gain_stats(rows, args.gamma)
    if general_protocol:
        summary = generalize_summary_names(summary)
    summary["split"] = args.split
    summary["use_shrinkage"] = bool(args.use_shrinkage)
    summary["shrink_k"] = float(args.shrink_k)
    summary["seeds"] = common_seeds
    summary.update(dataset_metadata)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else out_dir / f"relation_gain_stats_gamma_{gamma_tag(args.gamma)}_summary.json"
    )
    write_json(summary_path, summary)
    print(f"[OK] wrote summary         -> {summary_path.as_posix()}")


def main() -> None:
    args = parse_args()
    if args.gate_dev and args.residual_dev:
        out_path = Path(args.out) if args.out else Path("outputs/router/raw/dev_relation_priors.csv")
        build_from_contract_inputs(
            Path(args.gate_dev),
            Path(args.residual_dev),
            out_path,
            args.gamma,
            args.summary_json,
            args.use_shrinkage,
            args.shrink_k,
        )
        return

    build_from_legacy_dirs(args)


if __name__ == "__main__":
    main()
