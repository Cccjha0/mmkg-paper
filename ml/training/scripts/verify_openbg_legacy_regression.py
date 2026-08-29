"""Read-only verifier for the frozen OpenBG legacy artifacts.

This script never loads a model or runs training/evaluation.  It verifies the
existing metrics files and byte-exact query exports against the checked-in
pre-generalization fixture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = ROOT / "ml/training/tests/fixtures/openbg_legacy_regression_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_vector_sha256(path: Path) -> tuple[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            (row["query_id"], row["direction"], int(row["rank"]))
            for row in csv.DictReader(handle)
        ]
    rows.sort()
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest(), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=None,
        help="Optional directory containing newly re-exported query CSVs with the fixture basenames.",
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    failures: list[str] = []
    for model_name, expected in fixture["models"].items():
        metrics_path = ROOT / expected["run_dir"] / "test_metrics.json"
        actual_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key, value in expected["metrics"].items():
            actual = float(actual_metrics[key])
            tolerance = args.atol + args.rtol * abs(float(value))
            if abs(actual - float(value)) > tolerance:
                failures.append(f"{model_name}.{key}: expected {value}, got {actual}")
        fixture_query_path = ROOT / expected["query_eval"]
        query_path = args.query_dir / fixture_query_path.name if args.query_dir else fixture_query_path
        if args.query_dir is None and sha256_file(query_path) != expected["query_sha256"]:
            failures.append(f"{model_name}: query export SHA-256 changed")
        rank_hash, rows = rank_vector_sha256(query_path)
        if rank_hash != expected["rank_vector_sha256"]:
            failures.append(f"{model_name}: semantic query_id/direction/rank vector changed")
        if rows != int(expected["query_rows"]):
            failures.append(f"{model_name}: expected {expected['query_rows']} query rows, got {rows}")
    if failures:
        raise SystemExit("OpenBG legacy regression failed:\n- " + "\n- ".join(failures))
    print("[OK] OpenBG legacy metrics and query exports match openbg_legacy_v1 fixture.")


if __name__ == "__main__":
    main()
