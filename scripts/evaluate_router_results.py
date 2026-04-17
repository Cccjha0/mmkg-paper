import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import write_csv


MAIN_HEADER = [
    "model",
    "delta",
    "tau",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
]

SUBGROUP_HEADER = [
    "model",
    "delta",
    "tau",
    "target_regime",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/router/eval")
    ap.add_argument("--main-out", default="outputs/router/eval/main_results_table.csv")
    ap.add_argument("--subgroup-out", default="outputs/router/eval/subgroup_results_table.csv")
    return ap.parse_args()


def load_eval_jsons(eval_dir: Path) -> list[dict]:
    payload = []
    for path in sorted(eval_dir.glob("router_eval_*_delta_*_tau_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path.as_posix()
        payload.append(data)
    return payload


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    payloads = load_eval_jsons(eval_dir)
    if not payloads:
        raise RuntimeError(f"No router eval json files found under {eval_dir}")

    main_rows = []
    subgroup_rows = []
    for data in payloads:
        overall = data["overall"]
        main_rows.append(
            {
                "model": data["model"],
                "delta": data["delta"],
                "tau": data["tau"],
                "n_queries": overall["n_queries"],
                "mrr": overall["mrr"],
                "hits1": overall["hits1"],
                "hits3": overall["hits3"],
                "hits10": overall["hits10"],
                "fusion_coverage": overall["fusion_coverage"],
            }
        )
        for regime, stats in sorted(data.get("by_regime", {}).items()):
            subgroup_rows.append(
                {
                    "model": data["model"],
                    "delta": data["delta"],
                    "tau": data["tau"],
                    "target_regime": regime,
                    "n_queries": stats["n_queries"],
                    "mrr": stats["mrr"],
                    "hits1": stats["hits1"],
                    "hits3": stats["hits3"],
                    "hits10": stats["hits10"],
                    "fusion_coverage": stats["fusion_coverage"],
                }
            )

    main_rows.sort(key=lambda row: (row["delta"], row["model"], float(row["tau"])))
    subgroup_rows.sort(key=lambda row: (row["delta"], row["model"], float(row["tau"]), row["target_regime"]))

    write_csv(args.main_out, main_rows, MAIN_HEADER)
    print(f"[OK] wrote main results  -> {Path(args.main_out).as_posix()}")
    write_csv(args.subgroup_out, subgroup_rows, SUBGROUP_HEADER)
    print(f"[OK] wrote subgroup rows -> {Path(args.subgroup_out).as_posix()}")


if __name__ == "__main__":
    main()
