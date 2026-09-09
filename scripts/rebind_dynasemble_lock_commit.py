from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind only DynaSemble repository-commit provenance after verifying "
            "that every execution-critical locked asset is byte-identical."
        )
    )
    parser.add_argument("--lock-json", required=True)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--baseline-selection-json", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Refusing provenance rebind: {role} hash differs; "
            f"expected={expected}, actual={actual}, path={path}"
        )
    return actual


def current_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def atomic_write_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = Path(args.lock_json)
    protocol_path = Path(args.protocol_path)
    baseline_path = Path(args.baseline_selection_json)
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = lock.get("source_provenance") or {}
    required_source_fields = {
        "repository_commit",
        "evaluator_sha256",
        "shared_full_ranking_evaluator_sha256",
    }
    missing = required_source_fields - set(source)
    if missing:
        raise RuntimeError(f"DEV lock lacks source provenance fields: {sorted(missing)}")

    verified = {
        "dynasemble_evaluator_sha256": require_hash(
            repo_root / "scripts" / "eval_openbg_dynasemble.py",
            source["evaluator_sha256"],
            "DynaSemble evaluator",
        ),
        "shared_full_ranking_evaluator_sha256": require_hash(
            repo_root / "scripts" / "eval_heterogeneous_complementarity.py",
            source["shared_full_ranking_evaluator_sha256"],
            "shared exact full-ranking evaluator",
        ),
        "paper_a_protocol_sha256": require_hash(
            protocol_path,
            (lock.get("paper_a_protocol") or {}).get("sha256", ""),
            "Paper A protocol",
        ),
        "baseline_selection_sha256": require_hash(
            baseline_path,
            lock.get("baseline_selection_sha256", ""),
            "DEV baseline selection",
        ),
    }
    for seed, record in sorted((lock.get("selectors") or {}).items()):
        selector_path = Path(record["path"])
        verified[f"selector_seed_{seed}_sha256"] = require_hash(
            selector_path, record["sha256"], f"DynaSemble selector seed {seed}"
        )

    old_commit = str(source["repository_commit"])
    new_commit = current_commit(repo_root)
    if old_commit == new_commit:
        print(f"[LOCK OK] repository commit already matches {new_commit}")
        return

    before_hash = sha256_file(lock_path)
    history = list(lock.get("repository_commit_rebind_history") or [])
    history.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "from_repository_commit": old_commit,
            "to_repository_commit": new_commit,
            "reason": (
                "Operational wrapper/documentation commits after DEV; all execution-critical "
                "assets verified byte-identical before provenance-only rebind"
            ),
            "test_outcomes_read": False,
            "method_or_hyperparameter_changed": False,
            "verified_locked_hashes": verified,
            "lock_sha256_before_rebind": before_hash,
        }
    )
    source["repository_commit"] = new_commit
    lock["source_provenance"] = source
    lock["repository_commit_rebind_history"] = history
    atomic_write_json(lock_path, lock)
    print(
        f"[PROVENANCE REBOUND] {lock_path}: {old_commit} -> {new_commit}; "
        "core hashes unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
