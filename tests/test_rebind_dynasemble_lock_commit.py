from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebind_dynasemble_lock_commit.py"
SPEC = importlib.util.spec_from_file_location("rebind_lock", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_require_hash_accepts_identical_file(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"locked")
    expected = hashlib.sha256(b"locked").hexdigest()
    assert MODULE.require_hash(path, expected, "asset") == expected


def test_require_hash_rejects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"changed")
    try:
        MODULE.require_hash(path, hashlib.sha256(b"locked").hexdigest(), "asset")
    except RuntimeError as error:
        assert "Refusing provenance rebind" in str(error)
    else:
        raise AssertionError("Changed locked asset was accepted")
