from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ORDER = [
    "M-Hyper",
    "NativE",
    "AdaMF-MAT",
    "APKGC",
    "Gate-only",
    "Residual-only",
]
DEFAULT_BASELINE_CANDIDATES = DEFAULT_MODEL_ORDER[:4]
PAIR_FIELDS = (
    "split",
    "seed",
    "direction",
    "relation_id",
    "head_id",
    "tail_id",
    "target_entity_id",
)


@dataclass(frozen=True)
class ModelRows:
    name: str
    sources: tuple[str, ...]
    rows: dict[str, dict[str, str]]
    seeds: tuple[int, ...]
    seed_mrr: dict[int, float]
    seed_hits10: dict[int, float]

    @property
    def dev_mrr(self) -> float:
        return sum(self.seed_mrr.values()) / len(self.seed_mrr)

    @property
    def dev_hits10(self) -> float:
        return sum(self.seed_hits10.values()) / len(self.seed_hits10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the OpenBG-IMG DEV complementarity table against the strongest "
            "recent baseline. This script only reads query-eval CSV files."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help=(
            "Model name and query-eval CSV glob; repeat once per model. "
            "Quote patterns containing * in PowerShell."
        ),
    )
    parser.add_argument(
        "--baseline-candidate",
        action="append",
        default=None,
        help=(
            "Model eligible to become B; repeat as needed. Defaults to M-Hyper, "
            "NativE, AdaMF-MAT, and APKGC."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Explicit B override. Otherwise B is selected by three-seed mean DEV MRR.",
    )
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--output-dir",
        default="outputs/openbg_img/recent_baseline_complementarity",
    )
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model value {spec!r}; expected NAME=GLOB")
    name, pattern = spec.split("=", 1)
    name = name.strip()
    pattern = pattern.strip()
    if not name or not pattern:
        raise ValueError(f"Invalid --model value {spec!r}; name and glob are required")
    return name, pattern


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def query_key(row: dict[str, str]) -> str:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id:
        raise RuntimeError("Every query-eval row must contain query_id")
    return query_id


def load_model(name: str, pattern: str, split: str, expected_seeds: tuple[int, ...]) -> ModelRows:
    paths = tuple(Path(value) for value in sorted(glob.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No query-eval CSV matched {pattern!r} for {name}")

    rows_by_key: dict[str, dict[str, str]] = {}
    sources_by_seed: dict[int, Path] = {}
    rr_by_seed: dict[int, list[float]] = {seed: [] for seed in expected_seeds}
    hit10_by_seed: dict[int, list[float]] = {seed: [] for seed in expected_seeds}
    required = {"query_id", "split", "seed", "direction", "relation_id", "rr"}

    for path in paths:
        rows = read_csv(path)
        if not rows:
            raise RuntimeError(f"Empty query-eval CSV: {path}")
        missing = required - set(rows[0])
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
        file_seeds = {int(row["seed"]) for row in rows}
        if len(file_seeds) != 1:
            raise RuntimeError(f"{path} must contain exactly one seed; found {sorted(file_seeds)}")
        seed = next(iter(file_seeds))
        if seed not in expected_seeds:
            raise RuntimeError(f"{path} contains unexpected seed {seed}; expected {expected_seeds}")
        if seed in sources_by_seed:
            raise RuntimeError(
                f"Multiple CSV files supplied for {name} seed {seed}: "
                f"{sources_by_seed[seed]} and {path}"
            )
        sources_by_seed[seed] = path

        for row in rows:
            if row["split"] != split:
                raise RuntimeError(f"{path} contains split={row['split']!r}; expected {split!r}")
            key = query_key(row)
            if key in rows_by_key:
                raise RuntimeError(f"Duplicate query_id for {name}: {key}")
            rr = float(row["rr"])
            if not math.isfinite(rr) or not 0.0 < rr <= 1.0:
                raise RuntimeError(f"Invalid reciprocal rank {row['rr']!r} in {path}")
            hit10 = float(row.get("hit10", int(rr >= 0.1)))
            rows_by_key[key] = row
            rr_by_seed[seed].append(rr)
            hit10_by_seed[seed].append(hit10)

    missing_seeds = set(expected_seeds) - set(sources_by_seed)
    if missing_seeds:
        raise RuntimeError(f"Missing {name} query exports for seeds {sorted(missing_seeds)}")

    return ModelRows(
        name=name,
        sources=tuple(str(sources_by_seed[seed]) for seed in expected_seeds),
        rows=rows_by_key,
        seeds=expected_seeds,
        seed_mrr={seed: sum(values) / len(values) for seed, values in rr_by_seed.items()},
        seed_hits10={
            seed: sum(values) / len(values) for seed, values in hit10_by_seed.items()
        },
    )


def resolve_name(requested: str, models: dict[str, ModelRows]) -> str:
    matches = [name for name in models if name.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Model {requested!r} is not uniquely available; supplied models are {sorted(models)}"
        )
    return matches[0]


def choose_baseline(
    models: dict[str, ModelRows],
    requested: str | None,
    candidates: list[str],
) -> str:
    if requested:
        return resolve_name(requested, models)

    resolved = [resolve_name(candidate, models) for candidate in candidates]
    scores = {
        name: (models[name].dev_mrr, models[name].dev_hits10)
        for name in resolved
    }
    best_score = max(scores.values())
    winners = [name for name, score in scores.items() if score == best_score]
    if len(winners) != 1:
        raise RuntimeError(
            "The baseline candidates tie on mean DEV MRR and Hits@10. "
            "Apply the declared training-budget tie-break and pass --baseline-model. "
            f"Tied models: {winners}"
        )
    return winners[0]


def validate_pairing(baseline: ModelRows, candidate: ModelRows) -> None:
    baseline_keys = set(baseline.rows)
    candidate_keys = set(candidate.rows)
    if baseline_keys != candidate_keys:
        missing = len(baseline_keys - candidate_keys)
        extra = len(candidate_keys - baseline_keys)
        raise RuntimeError(
            f"{candidate.name} does not match {baseline.name}'s query set: "
            f"missing={missing}, extra={extra}"
        )
    for key in baseline_keys:
        left = baseline.rows[key]
        right = candidate.rows[key]
        for field in PAIR_FIELDS:
            if str(left.get(field, "")) != str(right.get(field, "")):
                raise RuntimeError(
                    f"Query metadata mismatch for {key}: {field} "
                    f"({baseline.name}={left.get(field)!r}, "
                    f"{candidate.name}={right.get(field)!r})"
                )


def summarize_pair(baseline: ModelRows, candidate: ModelRows) -> dict[str, float | int]:
    validate_pairing(baseline, candidate)
    wins = 0
    losses = 0
    oracle_total = 0.0
    for key, baseline_row in baseline.rows.items():
        baseline_rr = float(baseline_row["rr"])
        candidate_rr = float(candidate.rows[key]["rr"])
        wins += candidate_rr > baseline_rr
        losses += candidate_rr < baseline_rr
        oracle_total += max(baseline_rr, candidate_rr)
    count = len(baseline.rows)
    oracle_mrr = oracle_total / count
    return {
        "count": count,
        "wins": wins,
        "win_pct": wins / count,
        "losses": losses,
        "loss_pct": losses / count,
        "ties": count - wins - losses,
        "tie_pct": (count - wins - losses) / count,
        "oracle_mrr": oracle_mrr,
        "oracle_headroom": max(
            0.0, oracle_mrr - max(baseline.dev_mrr, candidate.dev_mrr)
        ),
    }


def ordered_names(models: dict[str, ModelRows]) -> list[str]:
    resolved = []
    for preferred in DEFAULT_MODEL_ORDER:
        matches = [name for name in models if name.casefold() == preferred.casefold()]
        if matches:
            resolved.append(matches[0])
    resolved.extend(name for name in models if name not in resolved)
    return resolved


def write_outputs(
    out_dir: Path,
    models: dict[str, ModelRows],
    baseline_name: str,
    rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "openbg_baseline_complementarity_table.csv"
    fieldnames = [
        "Model",
        "Dev MRR",
        "Win vs. B",
        "Lose vs. B",
        "Oracle(B,M)",
        "Oracle headroom",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "Model": row["model"],
                "Dev MRR": f"{row['dev_mrr']:.6f}",
                "Win vs. B": f"{row['wins']} ({100.0 * row['win_pct']:.2f}%)",
                "Lose vs. B": f"{row['losses']} ({100.0 * row['loss_pct']:.2f}%)",
                "Oracle(B,M)": f"{row['oracle_mrr']:.6f}",
                "Oracle headroom": f"{row['oracle_headroom']:.6f}",
            }
            for row in rows
        )

    markdown_lines = [
        "| Model | Dev MRR | Win vs. B | Lose vs. B | Oracle(B,M) | Oracle headroom |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        marker = " **(B)**" if row["model"] == baseline_name else ""
        markdown_lines.append(
            f"| {row['model']}{marker} | {row['dev_mrr']:.6f} | "
            f"{row['wins']:,} ({100.0 * row['win_pct']:.2f}%) | "
            f"{row['losses']:,} ({100.0 * row['loss_pct']:.2f}%) | "
            f"{row['oracle_mrr']:.6f} | {row['oracle_headroom']:.6f} |"
        )
    markdown_lines.extend(
        [
            "",
            f"B = {baseline_name}, selected by mean DEV MRR across seeds "
            f"{', '.join(map(str, models[baseline_name].seeds))} among the declared recent-baseline candidates.",
            "Win/Lose percentages pool the same aligned seed-query observations. "
            "Oracle headroom is Oracle(B,M) minus the better fixed DEV MRR of B and M.",
        ]
    )
    (out_dir / "openbg_baseline_complementarity_table.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "dataset": "OpenBG-IMG paper_split",
        "split": "dev",
        "baseline_model": baseline_name,
        "baseline_selection": "mean_dev_mrr_then_mean_dev_hits10",
        "models": {
            name: {
                "sources": list(model.sources),
                "seed_mrr": model.seed_mrr,
                "mean_dev_mrr": model.dev_mrr,
                "seed_hits10": model.seed_hits10,
                "mean_dev_hits10": model.dev_hits10,
            }
            for name, model in models.items()
        },
        "rows": rows,
        "definitions": {
            "win": "RR_M > RR_B on the same seed-query observation",
            "lose": "RR_M < RR_B on the same seed-query observation",
            "oracle": "mean(max(RR_B, RR_M))",
            "oracle_headroom": "Oracle(B,M) - max(MRR_B, MRR_M)",
        },
    }
    (out_dir / "openbg_baseline_complementarity_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    expected_seeds = tuple(args.seeds)
    models: dict[str, ModelRows] = {}
    for spec in args.model:
        name, pattern = parse_model_spec(spec)
        if any(existing.casefold() == name.casefold() for existing in models):
            raise RuntimeError(f"Duplicate model name: {name}")
        models[name] = load_model(name, pattern, args.split, expected_seeds)

    baseline_candidates = args.baseline_candidate or DEFAULT_BASELINE_CANDIDATES
    baseline_name = choose_baseline(models, args.baseline_model, baseline_candidates)
    baseline = models[baseline_name]
    rows = []
    for name in ordered_names(models):
        candidate = models[name]
        summary = summarize_pair(baseline, candidate)
        rows.append(
            {
                "model": name,
                "dev_mrr": candidate.dev_mrr,
                **summary,
            }
        )

    out_dir = Path(args.output_dir)
    write_outputs(out_dir, models, baseline_name, rows)
    print(f"[OK] B = {baseline_name}")
    print(f"[OK] wrote OpenBG-IMG complementarity table -> {out_dir}")


if __name__ == "__main__":
    main()
