import argparse
import tempfile
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv
from scripts.export_query_eval import (
    export_query_eval,
    infer_expert_name,
    load_cfg_and_ckpt,
    resolve_device,
)


EXPECTED_EXPERTS = {"gate_only", "residual_only", "full_model"}
EXPECTED_LINES = {"routing_compatible", "official"}


OUTPUT_COLUMNS = [
    "query_id",
    "split",
    "model_name",
    "evaluation_line",
    "source_run_dir",
    "source_seed",
    "direction",
    "head_id",
    "relation_id",
    "tail_id",
    "target_entity_id",
    "target_has_img",
    "target_regime",
    "rank",
    "rr",
    "correct_score",
    "top1_score",
    "top2_score",
    "margin",
    "is_correct_at_1",
    "is_correct_at_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a unified query-level outcome table for one expert/split into parquet."
    )
    parser.add_argument("--run-dir", default=None, help="Run directory containing config_merged.json and best.ckpt")
    parser.add_argument("--input-csvs", nargs="+", default=None, help="Optional existing query_eval csv files to convert")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--model-name", required=True, choices=sorted(EXPECTED_EXPERTS))
    parser.add_argument("--evaluation-line", required=True, choices=sorted(EXPECTED_LINES))
    parser.add_argument("--out", required=True, help="Parquet output path")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional explicit source seed override")
    return parser.parse_args()


def transform_rows(rows: list[dict], model_name: str, evaluation_line: str, run_dir: Path) -> list[dict]:
    output_rows = []
    for row in rows:
        rank = int(row["rank"])
        output_rows.append(
            {
                "query_id": row["query_id"],
                "split": row["split"],
                "model_name": model_name,
                "evaluation_line": evaluation_line,
                "source_run_dir": run_dir.as_posix(),
                "source_seed": int(row["seed"]),
                "direction": row["direction"],
                "head_id": int(row["head_id"]),
                "relation_id": int(row["relation_id"]),
                "tail_id": int(row["tail_id"]),
                "target_entity_id": int(row["target_entity_id"]),
                "target_has_img": int(row["target_has_img"]),
                "target_regime": row["target_regime"],
                "rank": rank,
                "rr": float(row["rr"]),
                "correct_score": float(row["correct_score"]),
                "top1_score": float(row["top1_score"]),
                "top2_score": float(row["top2_score"]),
                "margin": float(row["score_margin"]),
                "is_correct_at_1": int(rank == 1),
                "is_correct_at_10": int(rank <= 10),
            }
        )
    return output_rows


def validate_rows(rows: list[dict], split: str) -> None:
    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError(f"query_id is not unique within split={split}.")

    allowed_regimes = {"head_has_img", "head_no_img", "tail_no_img"}
    found_regimes = {row["target_regime"] for row in rows}
    if not found_regimes.issubset(allowed_regimes):
        raise RuntimeError(f"Unexpected target_regime values: {sorted(found_regimes - allowed_regimes)}")

    bad_rr = [row for row in rows if abs(float(row["rr"]) - (1.0 / int(row["rank"]))) > 1e-12]
    if bad_rr:
        raise RuntimeError("Found rows where rr != 1 / rank.")

    bad_margin = [row for row in rows if float(row["margin"]) < 0.0]
    if bad_margin:
        raise RuntimeError("Found rows where margin < 0.")


def main() -> None:
    args = parse_args()
    raw_rows: list[dict]
    run_dir = Path(args.run_dir) if args.run_dir else Path(".")

    if args.input_csvs:
        raw_rows = []
        for csv_path in args.input_csvs:
            raw_rows.extend(read_csv(csv_path))
    else:
        if not args.run_dir:
            raise ValueError("Either --run-dir or --input-csvs is required.")
        cfg, ckpt_path = load_cfg_and_ckpt(args)
        seed = int(args.seed if args.seed is not None else cfg.get("system", {}).get("seed", 1))
        device = resolve_device(args.device or cfg.get("system", {}).get("device", "cuda"))
        cfg.setdefault("system", {})["device"] = device
        ev_cfg = cfg.get("evaluation", {})
        chunk_size = int(args.chunk_size or ev_cfg.get("chunk_size", 4096))
        query_batch_size = int(args.query_batch_size or ev_cfg.get("query_batch_size", 8))
        expert_name = infer_expert_name(cfg, args.model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_csv = Path(tmp_dir) / "query_eval.csv"
            export_query_eval(
                cfg=cfg,
                ckpt_path=ckpt_path,
                expert_name=expert_name,
                split=args.split,
                out_path=str(tmp_csv),
                seed=seed,
                device=device,
                chunk_size=chunk_size,
                query_batch_size=query_batch_size,
                summary_json=None,
            )
            raw_rows = read_csv(tmp_csv)

    rows = transform_rows(raw_rows, args.model_name, args.evaluation_line, run_dir)
    validate_rows(rows, args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_parquet(out_path, index=False)
    print(f"[OK] wrote parquet -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
