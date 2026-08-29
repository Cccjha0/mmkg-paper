import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import write_csv, write_json
from router.label_utils import build_binary_gain_label, compute_delta_rr, summarize_gain_distribution


OUTPUT_HEADER = [
    "query_id",
    "rr_fusion",
    "rr_struct",
    "delta_rr",
    "delta_threshold",
    "label_gain",
    "direction",
    "target_regime",
    "relation_id",
    "seed",
]
GENERAL_OUTPUT_HEADER = ["dataset", "protocol_version", *OUTPUT_HEADER]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, required=True, help="e.g. 0.01")
    ap.add_argument("--split", default="dev", choices=["dev"], help="Phase 2 currently uses dev only")
    ap.add_argument("--gate-dir", default="outputs/router/dev")
    ap.add_argument("--residual-dir", default="outputs/router/dev")
    ap.add_argument("--out-dir", default="outputs/router/dev")
    ap.add_argument("--summary-json", default=None)
    return ap.parse_args()


def delta_tag(delta: float) -> str:
    return f"{float(delta):.2f}"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def infer_seed_from_path(path: Path) -> int:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        raise ValueError(f"Cannot infer seed from filename: {path}")
    return int(stem.split(marker)[-1])


def find_seed_files(base_dir: Path, expert_prefix: str) -> dict[int, Path]:
    files = sorted(base_dir.glob(f"{expert_prefix}_query_eval_seed*.csv"))
    out: dict[int, Path] = {}
    for path in files:
        seed = infer_seed_from_path(path)
        out[seed] = path
    return out


def validate_sources(gate_rows: list[dict], residual_rows: list[dict]) -> dict[str, str]:
    for expert, rows in (("gate", gate_rows), ("residual", residual_rows)):
        splits = {str(row.get("split", "")) for row in rows}
        if splits != {"dev"}:
            raise RuntimeError(f"Gain labels are validation-only; {expert} rows contain splits={sorted(splits)}")
    metadata: dict[str, str] = {}
    for field in ("dataset", "protocol_version"):
        gate_values = {str(row[field]) for row in gate_rows if row.get(field) not in (None, "")}
        residual_values = {str(row[field]) for row in residual_rows if row.get(field) not in (None, "")}
        if not gate_values and not residual_values:
            continue  # Frozen OpenBG exports predate these columns.
        if len(gate_values) != 1 or gate_values != residual_values:
            raise RuntimeError(
                f"Gain labels must be dataset-local: {field} gate={sorted(gate_values)} "
                f"residual={sorted(residual_values)}"
            )
        metadata[field] = next(iter(gate_values))
    if bool(metadata) != (set(metadata) == {"dataset", "protocol_version"}):
        raise RuntimeError("General gain-label sources must provide both dataset and protocol_version.")
    return metadata


def build_seed_labels(gate_rows: list[dict], residual_rows: list[dict], delta: float) -> tuple[list[dict], dict[str, str]]:
    metadata = validate_sources(gate_rows, residual_rows)
    gate_by_id = {row["query_id"]: row for row in gate_rows}
    residual_by_id = {row["query_id"]: row for row in residual_rows}
    if len(gate_by_id) != len(gate_rows) or len(residual_by_id) != len(residual_rows):
        raise RuntimeError("Duplicate query_id detected in gate or residual query rows.")

    gate_ids = set(gate_by_id)
    residual_ids = set(residual_by_id)
    if gate_ids != residual_ids:
        missing_in_gate = len(residual_ids - gate_ids)
        missing_in_residual = len(gate_ids - residual_ids)
        raise RuntimeError(
            f"query_id mismatch between experts: missing_in_gate={missing_in_gate}, missing_in_residual={missing_in_residual}"
        )

    rows: list[dict] = []
    for query_id in sorted(gate_ids):
        gate = gate_by_id[query_id]
        residual = residual_by_id[query_id]
        for field in ("split", "seed", "direction", "target_regime", "relation_id"):
            if str(gate[field]) != str(residual[field]):
                raise RuntimeError(
                    f"Expert mismatch for query_id={query_id}: "
                    f"{field} gate={gate[field]!r}, residual={residual[field]!r}."
                )
        rr_fusion = float(gate["rr"])
        rr_struct = float(residual["rr"])
        delta_rr = compute_delta_rr(rr_fusion, rr_struct)
        row = {
            "query_id": query_id,
            "rr_fusion": rr_fusion,
            "rr_struct": rr_struct,
            "delta_rr": delta_rr,
            "delta_threshold": float(delta),
            "label_gain": build_binary_gain_label(delta_rr, delta),
            "direction": gate["direction"],
            "target_regime": gate["target_regime"],
            "relation_id": int(gate["relation_id"]),
            "seed": int(gate["seed"]),
        }
        row.update(metadata)
        rows.append(row)
    return rows, metadata


def main() -> None:
    args = parse_args()
    gate_dir = Path(args.gate_dir)
    residual_dir = Path(args.residual_dir)
    out_dir = Path(args.out_dir)
    delta_str = delta_tag(args.delta)

    gate_files = find_seed_files(gate_dir, "gate_only")
    residual_files = find_seed_files(residual_dir, "residual_only")
    common_seeds = sorted(set(gate_files) & set(residual_files))
    if not common_seeds:
        raise RuntimeError("No overlapping gate/residual query_eval seed files found.")

    all_rows: list[dict] = []
    dataset_metadata: dict[str, str] = {}
    for seed in common_seeds:
        gate_rows = load_csv(gate_files[seed])
        residual_rows = load_csv(residual_files[seed])
        label_rows, seed_metadata = build_seed_labels(gate_rows, residual_rows, args.delta)
        if dataset_metadata and seed_metadata != dataset_metadata:
            raise RuntimeError("Gain-label seeds contain different dataset/protocol metadata.")
        dataset_metadata = dataset_metadata or seed_metadata
        out_path = out_dir / f"gain_labels_delta_{delta_str}_seed{seed}.csv"
        header = GENERAL_OUTPUT_HEADER if seed_metadata else OUTPUT_HEADER
        write_csv(out_path, label_rows, header)
        print(f"[OK] wrote gain labels -> {out_path.as_posix()}")
        all_rows.extend(label_rows)

    summary = summarize_gain_distribution(all_rows)
    summary["delta"] = float(args.delta)
    summary["split"] = args.split
    summary["seeds"] = common_seeds
    summary.update(dataset_metadata)
    summary_path = Path(args.summary_json) if args.summary_json else out_dir / f"gain_label_summary_delta_{delta_str}.json"
    write_json(summary_path, summary)
    print(f"[OK] wrote summary     -> {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
