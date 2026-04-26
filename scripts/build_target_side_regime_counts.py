from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def parse_entity_id(token: str) -> int:
    token = token.strip()
    if token.startswith("ent_"):
        return int(token[4:])
    return int(token)


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


def count_target_side_regimes(triples: list[tuple[str, str, str]], has_img: torch.Tensor) -> dict[str, object]:
    counts = Counter(
        {
            "head_has_img": 0,
            "head_no_img": 0,
            "tail_has_img": 0,
            "tail_no_img": 0,
        }
    )
    for h, _, t in triples:
        h_has_img = bool(has_img[parse_entity_id(h)].item())
        t_has_img = bool(has_img[parse_entity_id(t)].item())
        counts["head_has_img" if h_has_img else "head_no_img"] += 1
        counts["tail_has_img" if t_has_img else "tail_no_img"] += 1

    total = 2 * len(triples)
    return {
        "num_test_triples": len(triples),
        "num_bidirectional_queries": total,
        "counts": dict(counts),
        "bidirectional_ratios": {
            key: (counts[key] / total if total else 0.0)
            for key in ("head_has_img", "head_no_img", "tail_has_img", "tail_no_img")
        },
        "side_ratios": {
            "head_has_img_within_head_queries": counts["head_has_img"] / len(triples) if triples else 0.0,
            "tail_has_img_within_tail_queries": counts["tail_has_img"] / len(triples) if triples else 0.0,
        },
    }


def format_latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def write_tex(summary: dict[str, object], path: Path, split_name: str) -> None:
    counts = summary["counts"]
    ratios = summary["bidirectional_ratios"]
    split_caption = split_name.replace("_", "\\_")
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Target-side image-availability regimes for the OpenBG-IMG \\texttt{{{split_caption}}} test set.}}
\\label{{tab:target_side_regime_counts_{split_name}}}
\\small
\\begin{{tabular}}{{lrr}}
\\toprule
Regime & \\#Queries & Bidirectional share \\\\
\\midrule
\\texttt{{head\\_has\\_img}} & {format_latex_int(counts["head_has_img"])} & {ratios["head_has_img"]:.4f} \\\\
\\texttt{{head\\_no\\_img}} & {format_latex_int(counts["head_no_img"])} & {ratios["head_no_img"]:.4f} \\\\
\\texttt{{tail\\_has\\_img}} & {format_latex_int(counts["tail_has_img"])} & {ratios["tail_has_img"]:.4f} \\\\
\\texttt{{tail\\_no\\_img}} & {format_latex_int(counts["tail_no_img"])} & {ratios["tail_no_img"]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-side regime counts for an OpenBG-IMG split.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="appendix_split_seed20260427")
    parser.add_argument("--test-file", default="OpenBG-IMG_paper_test.tsv")
    parser.add_argument("--has-img", type=Path, default=Path("data/cache/openbg_img/has_img.pt"))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--tex-out", type=Path, required=True)
    args = parser.parse_args()

    triples = read_triples(args.split_dir / args.test_file)
    has_img = torch.load(args.has_img, map_location="cpu").to(dtype=torch.bool)
    summary = {
        "split": args.split_name,
        "split_dir": args.split_dir.as_posix(),
        "test_file": (args.split_dir / args.test_file).as_posix(),
        "has_img": args.has_img.as_posix(),
        **count_target_side_regimes(triples, has_img),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tex(summary, args.tex_out, args.split_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] wrote json -> {args.json_out}")
    print(f"[OK] wrote tex -> {args.tex_out}")


if __name__ == "__main__":
    main()
