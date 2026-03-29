from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_entity_id(token: str) -> int:
    token = token.strip()
    if token.startswith("ent_"):
        return int(token[4:])
    return int(token)


def load_triples(path: Path) -> torch.LongTensor:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path} contains a non-3-column line: {line}")
            rows.append((parse_entity_id(parts[0]), 0, parse_entity_id(parts[2])))
    return torch.tensor(rows, dtype=torch.long) if rows else torch.empty((0, 3), dtype=torch.long)


def count_side(has_img: torch.Tensor, entity_ids: torch.Tensor) -> dict:
    entity_ids = entity_ids.to(dtype=torch.long)
    total = int(entity_ids.numel())
    if total == 0:
        return {
            "count": 0,
            "has_img_count": 0,
            "no_img_count": 0,
            "has_img_ratio": 0.0,
            "no_img_ratio": 0.0,
        }

    flags = has_img[entity_ids].to(dtype=torch.bool)
    has_count = int(flags.sum().item())
    no_count = total - has_count
    return {
        "count": total,
        "has_img_count": has_count,
        "no_img_count": no_count,
        "has_img_ratio": has_count / total,
        "no_img_ratio": no_count / total,
    }


def summarize_split(has_img: torch.Tensor, triples: torch.LongTensor) -> dict:
    heads = triples[:, 0] if triples.numel() > 0 else torch.empty(0, dtype=torch.long)
    tails = triples[:, 2] if triples.numel() > 0 else torch.empty(0, dtype=torch.long)
    return {
        "num_triples": int(triples.size(0)),
        "head": count_side(has_img, heads),
        "tail": count_side(has_img, tails),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check has_img distribution on OpenBG paper_split.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/datasets/openbg_img/paper_split"),
        help="Directory containing OpenBG-IMG_paper_train/dev/test.tsv",
    )
    parser.add_argument(
        "--has-img",
        type=Path,
        default=Path("data/cache/openbg_img/has_img.pt"),
        help="Path to has_img.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON",
    )
    args = parser.parse_args()

    has_img = torch.load(args.has_img, map_location="cpu").to(dtype=torch.bool)

    split_files = {
        "train": args.split_dir / "OpenBG-IMG_paper_train.tsv",
        "dev": args.split_dir / "OpenBG-IMG_paper_dev.tsv",
        "test": args.split_dir / "OpenBG-IMG_paper_test.tsv",
    }

    summary = {}
    for split_name, path in split_files.items():
        triples = load_triples(path)
        summary[split_name] = summarize_split(has_img, triples)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    main()
