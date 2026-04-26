from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch


Triple = tuple[str, str, str]


def read_triples(path: Path) -> list[Triple]:
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} is not a 3-column triple: {line}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def parse_entity_id(token: str) -> int:
    token = token.strip()
    if token.startswith("ent_"):
        return int(token[4:])
    return int(token)


def entity_set(triples: Iterable[Triple]) -> set[str]:
    entities: set[str] = set()
    for h, _, t in triples:
        entities.add(h)
        entities.add(t)
    return entities


def relation_counter(triples: Iterable[Triple]) -> Counter[str]:
    return Counter(r for _, r, _ in triples)


def count_missing_against_train(train: list[Triple], split: list[Triple]) -> dict[str, int]:
    train_entities = entity_set(train)
    train_relations = set(relation_counter(train))
    missing_entities: set[str] = set()
    missing_relations: set[str] = set()
    for h, r, t in split:
        if h not in train_entities:
            missing_entities.add(h)
        if t not in train_entities:
            missing_entities.add(t)
        if r not in train_relations:
            missing_relations.add(r)
    return {
        "missing_entities_vs_train": len(missing_entities),
        "missing_relations_vs_train": len(missing_relations),
    }


def summarize_target_regimes(test_triples: list[Triple], has_img: torch.Tensor) -> dict[str, object]:
    counts = Counter(
        {
            "head_has_img": 0,
            "head_no_img": 0,
            "tail_has_img": 0,
            "tail_no_img": 0,
        }
    )

    for h, _, t in test_triples:
        head_has_img = bool(has_img[parse_entity_id(h)].item())
        tail_has_img = bool(has_img[parse_entity_id(t)].item())
        counts["head_has_img" if head_has_img else "head_no_img"] += 1
        counts["tail_has_img" if tail_has_img else "tail_no_img"] += 1

    total_bidirectional_queries = 2 * len(test_triples)
    ratios = {
        key: (counts[key] / total_bidirectional_queries if total_bidirectional_queries else 0.0)
        for key in ("head_has_img", "head_no_img", "tail_has_img", "tail_no_img")
    }
    side_ratios = {
        "head_has_img_within_head_queries": counts["head_has_img"] / len(test_triples) if test_triples else 0.0,
        "tail_has_img_within_tail_queries": counts["tail_has_img"] / len(test_triples) if test_triples else 0.0,
    }
    return {
        "total_bidirectional_queries": total_bidirectional_queries,
        "counts": dict(counts),
        "bidirectional_ratios": ratios,
        "side_ratios": side_ratios,
    }


def relation_validation(train: list[Triple], test: list[Triple], top_k: int) -> dict[str, object]:
    train_counts = relation_counter(train)
    test_counts = relation_counter(test)
    top_train_relations = train_counts.most_common(top_k)
    missing_top_train_relations = [
        {"relation": rel, "train_count": count}
        for rel, count in top_train_relations
        if test_counts.get(rel, 0) == 0
    ]
    top_rows = [
        {
            "relation": rel,
            "train_count": count,
            "test_count": test_counts.get(rel, 0),
            "test_to_train_ratio": test_counts.get(rel, 0) / count if count else 0.0,
        }
        for rel, count in top_train_relations
    ]
    return {
        "top_k": top_k,
        "num_train_relations": len(train_counts),
        "num_test_relations": len(test_counts),
        "missing_top_train_relations_in_test": missing_top_train_relations,
        "top_train_relation_test_support": top_rows,
    }


def build_acceptance(summary: dict[str, object]) -> dict[str, object]:
    counts = summary["final_counts"]
    coverage = summary["coverage_check"]
    disjoint = summary["disjoint_check"]
    regime = summary["test_target_regimes"]
    relation = summary["relation_coverage_check"]

    regime_counts = regime["counts"]
    regime_ratios = regime["bidirectional_ratios"]

    checks = {
        "train_size_is_220087": counts["train"] == 220087,
        "dev_size_is_5000": counts["dev"] == 5000,
        "test_size_is_10000": counts["test"] == 10000,
        "dev_missing_entities_vs_train_is_0": coverage["dev"]["missing_entities_vs_train"] == 0,
        "dev_missing_relations_vs_train_is_0": coverage["dev"]["missing_relations_vs_train"] == 0,
        "test_missing_entities_vs_train_is_0": coverage["test"]["missing_entities_vs_train"] == 0,
        "test_missing_relations_vs_train_is_0": coverage["test"]["missing_relations_vs_train"] == 0,
        "train_dev_test_are_disjoint": all(disjoint.values()),
        "test_head_has_img_positive": regime_counts["head_has_img"] > 0,
        "test_head_no_img_positive": regime_counts["head_no_img"] > 0,
        "test_tail_has_img_below_1pct_bidirectional": regime_ratios["tail_has_img"] < 0.01,
        "test_tail_no_img_near_50pct_bidirectional": 0.45 <= regime_ratios["tail_no_img"] <= 0.55,
        "top_train_relations_have_test_support": len(relation["missing_top_train_relations_in_test"]) == 0,
    }
    return {
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    acceptance = summary["acceptance"]
    regimes = summary["test_target_regimes"]
    coverage = summary["coverage_check"]
    relation = summary["relation_coverage_check"]

    lines = [
        "# OpenBG-IMG Appendix Split Validation",
        "",
        f"- Split directory: `{summary['split_dir']}`",
        f"- Seed: `{summary.get('seed', 'unknown')}`",
        f"- Overall pass: `{acceptance['overall_pass']}`",
        "",
        "## Size And Coverage",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| train triples | {summary['final_counts']['train']} |",
        f"| dev triples | {summary['final_counts']['dev']} |",
        f"| test triples | {summary['final_counts']['test']} |",
        f"| dev missing entities vs train | {coverage['dev']['missing_entities_vs_train']} |",
        f"| dev missing relations vs train | {coverage['dev']['missing_relations_vs_train']} |",
        f"| test missing entities vs train | {coverage['test']['missing_entities_vs_train']} |",
        f"| test missing relations vs train | {coverage['test']['missing_relations_vs_train']} |",
        "",
        "## Test Target-side Regimes",
        "",
        "| Regime | Count | Bidirectional ratio |",
        "|---|---:|---:|",
    ]

    for key in ("head_has_img", "head_no_img", "tail_has_img", "tail_no_img"):
        lines.append(
            f"| `{key}` | {regimes['counts'][key]} | {regimes['bidirectional_ratios'][key]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance Checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for key, value in acceptance["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Top Relation Test Support",
            "",
            f"- Top-k train relations checked: `{relation['top_k']}`",
            f"- Missing top train relations in test: `{len(relation['missing_top_train_relations_in_test'])}`",
            "",
            "| Relation | Train count | Test count | Test/train ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in relation["top_train_relation_test_support"]:
        lines.append(
            f"| `{row['relation']}` | {row['train_count']} | {row['test_count']} | {row['test_to_train_ratio']:.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an OpenBG-IMG appendix robustness split with protocol-regime statistics."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--has-img", type=Path, default=Path("data/cache/openbg_img/has_img.pt"))
    parser.add_argument("--train-file", default="OpenBG-IMG_paper_train.tsv")
    parser.add_argument("--dev-file", default="OpenBG-IMG_paper_dev.tsv")
    parser.add_argument("--test-file", default="OpenBG-IMG_paper_test.tsv")
    parser.add_argument("--top-k-relations", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    split_dir = args.split_dir
    train = read_triples(split_dir / args.train_file)
    dev = read_triples(split_dir / args.dev_file)
    test = read_triples(split_dir / args.test_file)
    has_img = torch.load(args.has_img, map_location="cpu").to(dtype=torch.bool)

    train_set = set(train)
    dev_set = set(dev)
    test_set = set(test)

    meta_path = split_dir / "split_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    summary: dict[str, object] = {
        "split_dir": split_dir.as_posix(),
        "seed": meta.get("seed"),
        "source_files": meta.get("source_files"),
        "final_counts": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "coverage_check": {
            "dev": count_missing_against_train(train, dev),
            "test": count_missing_against_train(train, test),
        },
        "disjoint_check": {
            "train_dev_disjoint": train_set.isdisjoint(dev_set),
            "train_test_disjoint": train_set.isdisjoint(test_set),
            "dev_test_disjoint": dev_set.isdisjoint(test_set),
        },
        "test_target_regimes": summarize_target_regimes(test, has_img),
        "relation_coverage_check": relation_validation(train, test, args.top_k_relations),
    }
    summary["acceptance"] = build_acceptance(summary)

    output_json = args.output_json or (split_dir / "appendix_split_validation.json")
    output_md = args.output_md or (split_dir / "appendix_split_validation.md")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_md)

    print(json.dumps(summary["acceptance"], indent=2, ensure_ascii=False))
    print(f"[OK] wrote json -> {output_json}")
    print(f"[OK] wrote markdown -> {output_md}")


if __name__ == "__main__":
    main()
