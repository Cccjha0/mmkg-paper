import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.feature_utils import (
    build_feature_rows,
    infer_cache_dir,
    load_cache_bundle,
    load_relation_prior_map,
    summarize_feature_rows,
)
from router.io_utils import read_csv, write_csv, write_json
from router.schemas import ROUTER_FEATURE_HEADER


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", default=["0.00", "0.01", "0.02"], help="delta tags like 0.00 0.01 0.02")
    ap.add_argument("--gate-dev-dir", default="outputs/router/dev")
    ap.add_argument("--residual-dev-dir", default="outputs/router/dev")
    ap.add_argument("--label-dir", default="outputs/router/dev")
    ap.add_argument("--gate-test-dir", default="outputs/router/test")
    ap.add_argument("--residual-test-dir", default="outputs/router/test")
    ap.add_argument("--prior-csv", default="outputs/router/priors/relation_gain_stats_gamma_0.000.csv")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--run-dir", default="ml/artifacts/outputs/openbg_img_gate_only/20260327_173820_seed1")
    ap.add_argument("--out-dir", default="outputs/router/features")
    ap.add_argument("--summary-json", default=None)
    return ap.parse_args()


def infer_seed_from_path(path: Path) -> int:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        raise ValueError(f"Cannot infer seed from filename: {path}")
    return int(stem.split(marker)[-1])


def find_seed_files(base_dir: Path, pattern: str) -> dict[int, Path]:
    files = sorted(base_dir.glob(pattern))
    return {infer_seed_from_path(path): path for path in files}


def load_label_map(path: Path) -> dict[str, dict]:
    return {row["query_id"]: row for row in read_csv(path)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    cache_dir = infer_cache_dir(args.cache_dir, args.run_dir)
    cache_bundle = load_cache_bundle(cache_dir)
    prior_map = load_relation_prior_map(read_csv(args.prior_csv))

    gate_dev_files = find_seed_files(Path(args.gate_dev_dir), "gate_only_query_eval_seed*.csv")
    residual_dev_files = find_seed_files(Path(args.residual_dev_dir), "residual_only_query_eval_seed*.csv")
    gate_test_files = find_seed_files(Path(args.gate_test_dir), "gate_only_query_eval_seed*.csv")
    residual_test_files = find_seed_files(Path(args.residual_test_dir), "residual_only_query_eval_seed*.csv")

    common_dev_seeds = sorted(set(gate_dev_files) & set(residual_dev_files))
    common_test_seeds = sorted(set(gate_test_files) & set(residual_test_files))
    if not common_dev_seeds:
        raise RuntimeError("No overlapping dev query_eval seed files found.")
    if not common_test_seeds:
        raise RuntimeError("No overlapping test query_eval seed files found.")

    train_rows_by_delta: dict[str, list[dict]] = {}
    for delta_tag in args.deltas:
        all_rows: list[dict] = []
        label_files = find_seed_files(Path(args.label_dir), f"gain_labels_delta_{delta_tag}_seed*.csv")
        common_label_seeds = sorted(set(common_dev_seeds) & set(label_files))
        if not common_label_seeds:
            raise RuntimeError(f"No label files found for delta={delta_tag}")

        for seed in common_label_seeds:
            gate_rows = read_csv(gate_dev_files[seed])
            residual_rows = read_csv(residual_dev_files[seed])
            label_map = load_label_map(label_files[seed])
            rows = build_feature_rows(gate_rows, residual_rows, prior_map, cache_bundle, label_by_query_id=label_map)
            all_rows.extend(rows)

        out_path = out_dir / f"router_train_dev_delta_{delta_tag}.csv"
        write_csv(out_path, all_rows, ROUTER_FEATURE_HEADER)
        print(f"[OK] wrote train features -> {out_path.as_posix()}")
        train_rows_by_delta[delta_tag] = all_rows

    test_rows: list[dict] = []
    for seed in common_test_seeds:
        gate_rows = read_csv(gate_test_files[seed])
        residual_rows = read_csv(residual_test_files[seed])
        rows = build_feature_rows(gate_rows, residual_rows, prior_map, cache_bundle, label_by_query_id=None)
        test_rows.extend(rows)

    test_out = out_dir / "router_test_features.csv"
    write_csv(test_out, test_rows, ROUTER_FEATURE_HEADER)
    print(f"[OK] wrote test features  -> {test_out.as_posix()}")

    summary = summarize_feature_rows(train_rows_by_delta, test_rows)
    summary["cache_dir"] = cache_bundle["cache_dir"]
    summary["prior_csv"] = str(Path(args.prior_csv).as_posix())
    summary["train_seeds"] = common_dev_seeds
    summary["test_seeds"] = common_test_seeds
    summary_path = Path(args.summary_json) if args.summary_json else out_dir / "router_feature_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] wrote summary        -> {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
