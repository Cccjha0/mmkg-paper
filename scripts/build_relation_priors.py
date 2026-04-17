import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import write_csv, write_json
from router.prior_utils import compute_relation_gain_stats, summarize_relation_gain_stats


OUTPUT_HEADER = [
    "relation_id",
    "relation_name",
    "n_queries",
    "mean_rr_gate",
    "mean_rr_residual",
    "mean_delta_rr",
    "fusion_win_rate",
    "struct_win_rate",
    "head_has_img_ratio",
    "tail_no_img_ratio",
    "is_visual_prior",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"], help="Phase 2 currently uses dev only")
    ap.add_argument("--gamma", type=float, default=0.0, help="visual prior threshold on mean_delta_rr")
    ap.add_argument("--gate-dir", default="outputs/router/dev")
    ap.add_argument("--residual-dir", default="outputs/router/dev")
    ap.add_argument("--out-dir", default="outputs/router/priors")
    ap.add_argument("--summary-json", default=None)
    return ap.parse_args()


def gamma_tag(gamma: float) -> str:
    return f"{float(gamma):.3f}"


def infer_seed_from_path(path: Path) -> int:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        raise ValueError(f"Cannot infer seed from filename: {path}")
    return int(stem.split(marker)[-1])


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_seed_files(base_dir: Path, expert_prefix: str) -> dict[int, Path]:
    files = sorted(base_dir.glob(f"{expert_prefix}_query_eval_seed*.csv"))
    out: dict[int, Path] = {}
    for path in files:
        out[infer_seed_from_path(path)] = path
    return out


def main() -> None:
    args = parse_args()
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
    for seed in common_seeds:
        gate_rows = load_csv(gate_files[seed])
        residual_rows = load_csv(residual_files[seed])
        all_gate_rows.extend(gate_rows)
        all_residual_rows.extend(residual_rows)

    rows = compute_relation_gain_stats(all_gate_rows, all_residual_rows, args.gamma)
    out_path = out_dir / f"relation_gain_stats_gamma_{gamma_tag(args.gamma)}.csv"
    write_csv(out_path, rows, OUTPUT_HEADER)
    print(f"[OK] wrote relation priors -> {out_path.as_posix()}")

    summary = summarize_relation_gain_stats(rows, args.gamma)
    summary["split"] = args.split
    summary["seeds"] = common_seeds
    summary_path = Path(args.summary_json) if args.summary_json else out_dir / f"relation_gain_stats_gamma_{gamma_tag(args.gamma)}_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] wrote summary         -> {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
