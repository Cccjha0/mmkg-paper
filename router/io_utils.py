import csv
import json
from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> Path:
    out_path = ensure_parent_dir(path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def read_csv(path: str | Path) -> list[dict]:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, payload: dict) -> Path:
    out_path = ensure_parent_dir(path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
