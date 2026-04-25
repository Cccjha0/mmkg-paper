import argparse
import json
from pathlib import Path


def count_mapping_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} expected 3 tab-separated columns, got {len(parts)}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def count_image_entities(has_img_path: Path) -> tuple[int, int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Loading has_img.pt requires torch. Install project dependencies or provide a different image metadata source."
        ) from exc

    has_img = torch.load(has_img_path, map_location="cpu")
    if hasattr(has_img, "detach"):
        total = int(has_img.numel())
        image_entities = int(has_img.to(dtype=torch.bool).sum().item())
        return total, image_entities
    if isinstance(has_img, (list, tuple)):
        total = len(has_img)
        image_entities = sum(bool(x) for x in has_img)
        return total, image_entities
    raise TypeError(f"Unsupported has_img.pt payload type: {type(has_img)!r}")


def format_int(value: int) -> str:
    return f"{value:,}"


def format_latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenBG-IMG paper split dataset statistics.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/openbg_img"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/openbg_img"))
    parser.add_argument("--json-out", type=Path, default=Path("docs/dataset_statistics.json"))
    parser.add_argument("--tex-out", type=Path, default=Path("docs/paper_tables/table_dataset_statistics.tex"))
    args = parser.parse_args()

    raw_root = args.dataset_root / "raw"
    split_root = args.dataset_root / "paper_split"

    entity_path = raw_root / "OpenBG-IMG_entity2text.tsv"
    relation_path = raw_root / "OpenBG-IMG_relation2text.tsv"
    train_path = split_root / "OpenBG-IMG_paper_train.tsv"
    valid_path = split_root / "OpenBG-IMG_paper_dev.tsv"
    test_path = split_root / "OpenBG-IMG_paper_test.tsv"
    has_img_path = args.cache_root / "has_img.pt"

    num_entities = count_mapping_rows(entity_path)
    num_relations = count_mapping_rows(relation_path)
    train_triples = read_triples(train_path)
    valid_triples = read_triples(valid_path)
    test_triples = read_triples(test_path)
    has_img_total, image_entities = count_image_entities(has_img_path)
    image_coverage = image_entities / num_entities * 100.0

    if has_img_total != num_entities:
        raise ValueError(
            f"has_img length ({has_img_total}) does not match entity mapping count ({num_entities})"
        )

    split_entities = {h for h, _, _ in train_triples + valid_triples + test_triples}
    split_entities.update(t for _, _, t in train_triples + valid_triples + test_triples)
    split_relations = {r for _, r, _ in train_triples + valid_triples + test_triples}

    stats = {
        "dataset": "OpenBG-IMG",
        "split": "paper_split",
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_triples": len(train_triples),
        "num_valid_triples": len(valid_triples),
        "num_test_triples": len(test_triples),
        "num_image_entities": image_entities,
        "image_coverage_percent": image_coverage,
        "split_unique_entities": len(split_entities),
        "split_unique_relations": len(split_relations),
        "sources": {
            "entities": str(entity_path),
            "relations": str(relation_path),
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
            "has_img": str(has_img_path),
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Dataset statistics of the OpenBG-IMG \\texttt{{paper\\_split}}.}}
\\label{{tab:dataset_statistics}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
Dataset & \\#Entities & \\#Relations & \\#Train & \\#Valid & \\#Test & \\#Image entities & Image coverage \\\\
\\midrule
OpenBG-IMG & {format_latex_int(num_entities)} & {format_latex_int(num_relations)} & {format_latex_int(len(train_triples))} & {format_latex_int(len(valid_triples))} & {format_latex_int(len(test_triples))} & {format_latex_int(image_entities)} & {image_coverage:.2f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    args.tex_out.write_text(tex, encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
