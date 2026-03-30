from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.tsv_reader import read_allow_2or3
from ml.training.src.eval.filtered_ranking import filtered_ranking_eval
from ml.training.src.models.build_model import build_model


MODEL_LABEL_OVERRIDES = {
    "openbg_img_complex": "ComplEx",
    "openbg_img_tucker": "TuckER",
    "openbg_img_text_only": "Text-only",
    "openbg_img_early": "Early Fusion",
    "openbg_img_gate_only": "Gate-only",
    "openbg_img_residual_only": "Residual-only",
    "openbg_img_gated_vec_res_rel": "Full Model",
}

PRIMARY_ORDER = [
    "ComplEx",
    "TuckER",
    "Text-only",
    "Early Fusion",
    "Gate-only",
    "Full Model",
    "Residual-only",
]

DEFAULT_MODEL_SET = ["Gate-only", "Full Model", "Residual-only"]
METRIC_KEYS = ["mrr", "hits@1", "hits@3", "hits@10", "tail_mrr", "head_mrr"]
RESERVED_GROUP_KEYS = {"source_files", "grouping_version", "grouping_principle"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out[parts[0]] = parts[1]
    return out


def safe_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def resolve_device(requested: str) -> str:
    requested = (requested or "cuda").lower()

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            print("[WARN] cuda not available, switching to mps")
            return "mps"
        print("[WARN] cuda not available, switching to cpu")
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            print("[WARN] mps not available, switching to cuda")
            return "cuda"
        print("[WARN] mps not available, switching to cpu")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def relation_token_to_id(token: str) -> int:
    if not token.startswith("rel_"):
        raise ValueError(f"Bad relation token: {token}")
    return int(token.replace("rel_", ""))


def relation_id_to_token(rel_id: int) -> str:
    return f"rel_{rel_id:04d}"


def ordered_labels(labels: list[str]) -> list[str]:
    out = [label for label in PRIMARY_ORDER if label in labels]
    out.extend(sorted(label for label in labels if label not in out))
    return out


def normalize_requested_models(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_MODEL_SET)

    out = []
    for value in values:
        if value in MODEL_LABEL_OVERRIDES:
            out.append(MODEL_LABEL_OVERRIDES[value])
        else:
            out.append(value)
    return out


def build_group_definitions(groups_json: dict, zh_map: dict[str, str], en_map: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group_name, rel_tokens in groups_json.items():
        if group_name in RESERVED_GROUP_KEYS:
            continue
        if not isinstance(rel_tokens, list):
            continue

        relation_ids = sorted(relation_token_to_id(token) for token in rel_tokens)
        relations = []
        for rel_id in relation_ids:
            token = relation_id_to_token(rel_id)
            relations.append(
                {
                    "relation_id": rel_id,
                    "relation_token": token,
                    "text_zh": zh_map.get(token, ""),
                    "text_en": en_map.get(token, ""),
                }
            )

        out[group_name] = {
            "relation_ids": relation_ids,
            "relations": relations,
        }
    return out


def select_latest_runs(
    outputs_root: Path,
    requested_labels: set[str],
) -> tuple[dict[str, list[Path]], dict[str, dict[str, list[str]]]]:
    by_label_seed: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    duplicates: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for cfg_path in sorted(outputs_root.rglob("config_merged.json")):
        run_dir = cfg_path.parent
        exp_name = run_dir.parent.name
        label = MODEL_LABEL_OVERRIDES.get(exp_name, exp_name)
        if label not in requested_labels:
            continue
        if not (run_dir / "best.ckpt").exists():
            continue

        cfg = load_json(cfg_path)
        seed = int(cfg.get("system", {}).get("seed", -1))
        if seed < 0:
            continue
        by_label_seed[label][seed].append(run_dir)

    selected: dict[str, list[Path]] = {}
    for label, seed_map in by_label_seed.items():
        selected[label] = []
        for seed, candidates in sorted(seed_map.items()):
            candidates = sorted(candidates, key=lambda p: p.name)
            chosen = candidates[-1]
            selected[label].append(chosen)
            if len(candidates) > 1:
                duplicates[label][str(seed)] = [p.relative_to(outputs_root).as_posix() for p in candidates]

    return selected, duplicates


def summarize_relation_counts(triples: list[tuple[int, int, int]]) -> Counter:
    counter: Counter = Counter()
    for _, rel_id, _ in triples:
        counter[rel_id] += 1
    return counter


def subset_triples_by_relations(
    triples: list[tuple[int, int, int]],
    relation_ids: set[int],
) -> list[tuple[int, int, int]]:
    return [triple for triple in triples if triple[1] in relation_ids]


def aggregate_metric_rows(rows: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        stats[key] = {
            "mean": statistics.mean(values),
            "std": safe_stdev(values),
            "values": values,
        }
    return stats


def evaluate_run_on_groups(
    run_dir: Path,
    group_defs: dict[str, dict],
    device_override: str | None = None,
    per_relation: bool = True,
) -> dict:
    cfg = load_json(run_dir / "config_merged.json")
    seed = int(cfg.get("system", {}).get("seed", -1))
    exp_name = run_dir.parent.name
    label = MODEL_LABEL_OVERRIDES.get(exp_name, exp_name)

    device = resolve_device(device_override or cfg.get("system", {}).get("device", "cuda"))
    cfg["system"]["device"] = device

    train3, _, bad_train = read_allow_2or3(cfg["dataset"]["train"])
    dev3, _, bad_dev = read_allow_2or3(cfg["dataset"]["dev"])
    test3, _, bad_test = read_allow_2or3(cfg["dataset"]["test"])
    if bad_train or bad_dev or bad_test:
        print(f"[WARN] malformed lines skipped for {run_dir.name}: train={bad_train}, dev={bad_dev}, test={bad_test}")
    if not test3:
        raise RuntimeError(f"No labeled 3-column test triples found for run: {run_dir}")

    true_tails, true_heads = build_true_facts(train3 + dev3 + test3)
    model, num_entities = build_model(cfg)
    model = model.to(device)

    state = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    base_eval_kwargs = {
        "model": model,
        "true_tails": true_tails,
        "true_heads": true_heads,
        "num_entities": num_entities,
        "chunk_size": int(cfg.get("evaluation", {}).get("chunk_size", 10000)),
        "query_batch_size": int(cfg.get("evaluation", {}).get("query_batch_size", 1)),
        "device": device,
        "ks": (1, 3, 10),
        "direction": cfg.get("evaluation", {}).get("direction", "both"),
        "entity_has_img": None,
    }

    relation_counter = summarize_relation_counts(test3)
    per_group: dict[str, dict] = {}

    for group_name, group_info in group_defs.items():
        relation_ids = set(group_info["relation_ids"])
        group_triples = subset_triples_by_relations(test3, relation_ids)

        if not group_triples:
            per_group[group_name] = {
                "triple_count": 0,
                "relation_count": 0,
                "metrics": {},
                "per_relation": {},
            }
            continue

        group_metrics = filtered_ranking_eval(
            triples=torch.tensor(group_triples, dtype=torch.long),
            **base_eval_kwargs,
        )

        relation_rows = {}
        if per_relation:
            for rel_id in group_info["relation_ids"]:
                rel_triples = [triple for triple in group_triples if triple[1] == rel_id]
                if not rel_triples:
                    continue
                relation_rows[relation_id_to_token(rel_id)] = {
                    "relation_id": rel_id,
                    "relation_token": relation_id_to_token(rel_id),
                    "triple_count": len(rel_triples),
                    "metrics": filtered_ranking_eval(
                        triples=torch.tensor(rel_triples, dtype=torch.long),
                        **base_eval_kwargs,
                    ),
                }

        per_group[group_name] = {
            "triple_count": len(group_triples),
            "relation_count": sum(1 for rel_id in group_info["relation_ids"] if relation_counter[rel_id] > 0),
            "metrics": group_metrics,
            "per_relation": relation_rows,
        }

    return {
        "label": label,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "relative_run_dir": run_dir.relative_to(run_dir.parents[2]).as_posix(),
        "device": device,
        "test_triple_count": len(test3),
        "relation_counter": {relation_id_to_token(rel_id): int(count) for rel_id, count in sorted(relation_counter.items())},
        "groups": per_group,
    }


def build_summary(
    run_results: dict[str, list[dict]],
    group_defs: dict[str, dict],
    test_relation_counter: Counter,
) -> dict:
    summary = {
        "groups": {},
        "models": {},
        "per_relation": {},
    }

    for group_name, group_info in group_defs.items():
        present_relations = [rel for rel in group_info["relations"] if test_relation_counter[rel["relation_id"]] > 0]
        summary["groups"][group_name] = {
            "relation_ids": group_info["relation_ids"],
            "relation_count_defined": len(group_info["relation_ids"]),
            "relation_count_in_test": len(present_relations),
            "triple_count_in_test": int(sum(test_relation_counter[rel["relation_id"]] for rel in group_info["relations"])),
            "relations_in_test": present_relations,
            "relations_missing_in_test": [
                rel for rel in group_info["relations"] if test_relation_counter[rel["relation_id"]] == 0
            ],
        }

    for label, rows in run_results.items():
        model_summary = {
            "num_seeds": len(rows),
            "runs": rows,
            "group_stats": {},
        }
        for group_name in group_defs:
            group_rows = []
            for row in rows:
                group_payload = row["groups"].get(group_name, {})
                metrics = group_payload.get("metrics", {})
                if not metrics:
                    continue
                group_rows.append(
                    {
                        "seed": row["seed"],
                        "run_dir": row["relative_run_dir"],
                        "triple_count": group_payload.get("triple_count", 0),
                        **{key: metrics[key] for key in METRIC_KEYS if key in metrics},
                    }
                )
            model_summary["group_stats"][group_name] = {
                "num_seeds": len(group_rows),
                "rows": group_rows,
                "stats": aggregate_metric_rows(group_rows),
            }
        summary["models"][label] = model_summary

    for group_name, group_info in group_defs.items():
        relation_bucket: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

        for label, rows in run_results.items():
            for row in rows:
                per_rel = row["groups"].get(group_name, {}).get("per_relation", {})
                for rel_token, rel_payload in per_rel.items():
                    relation_bucket[rel_token][label].append(
                        {
                            "seed": row["seed"],
                            "run_dir": row["relative_run_dir"],
                            "triple_count": rel_payload.get("triple_count", 0),
                            **{
                                key: rel_payload["metrics"][key]
                                for key in METRIC_KEYS
                                if key in rel_payload.get("metrics", {})
                            },
                        }
                    )

        summary["per_relation"][group_name] = {}
        for rel in group_info["relations"]:
            rel_token = rel["relation_token"]
            model_rows = relation_bucket.get(rel_token, {})
            if not model_rows:
                continue
            summary["per_relation"][group_name][rel_token] = {
                "relation_id": rel["relation_id"],
                "relation_token": rel_token,
                "text_zh": rel["text_zh"],
                "text_en": rel["text_en"],
                "test_triple_count": int(test_relation_counter[rel["relation_id"]]),
                "models": {
                    label: {
                        "num_seeds": len(rows),
                        "rows": rows,
                        "stats": aggregate_metric_rows(rows),
                    }
                    for label, rows in model_rows.items()
                },
            }

    return summary


def top_relations_preview(group_summary: dict, limit: int = 8) -> list[str]:
    relations = sorted(group_summary.get("relations_in_test", []), key=lambda item: item.get("relation_id", 0))
    preview = []
    for rel in relations[:limit]:
        zh = rel.get("text_zh") or rel["relation_token"]
        preview.append(f"`{rel['relation_token']}` ({zh})")
    return preview


def render_markdown(
    summary: dict,
    selected_runs: dict[str, list[Path]],
    duplicates: dict[str, dict[str, list[str]]],
    outputs_root: Path,
    groups_json_path: Path,
    min_relation_test_triples: int = 0,
) -> str:
    labels = ordered_labels(list(summary["models"].keys()))
    group_names = list(summary["groups"].keys())

    lines = [
        "# Relation Type Summary",
        "",
        "## 1. Purpose",
        "",
        "This document summarizes relation-type grouped evaluation under the current unified paper protocol.",
        "",
        "Current protocol:",
        "",
        "- dataset split: `paper_split`",
        "- evaluation: filtered ranking",
        "- direction: `both`",
        "- selection/reporting: same `best.ckpt` and `test` split used in main experiments",
        f"- grouping source: `{groups_json_path.as_posix()}`",
        "",
        "This analysis is used to answer:",
        "",
        "- whether multimodal gain is relation-dependent",
        "- whether multimodal gain is only locally effective",
        "- whether the paper can support a gain-boundary conclusion",
        "",
        "## 2. Selected Runs",
        "",
        f"Outputs root: `{outputs_root.as_posix()}`",
        "",
    ]

    for label in labels:
        rel_paths = [run.relative_to(outputs_root).as_posix() for run in selected_runs.get(label, [])]
        lines.append(f"- `{label}`: {', '.join(f'`{path}`' for path in rel_paths)}")

    duplicate_lines = []
    for label in labels:
        for seed, candidates in sorted(duplicates.get(label, {}).items()):
            duplicate_lines.append(
                f"- `{label}` seed `{seed}` had multiple runs; selected latest: `{candidates[-1]}`"
            )
    if duplicate_lines:
        lines.extend(["", "Duplicate handling:"] + duplicate_lines)

    lines.extend(["", "## 3. Group Definition and Test Coverage", ""])
    lines.append("| Group | Defined Relations | Relations in Test | Test Triples | Preview |")
    lines.append("|---|---:|---:|---:|---|")
    for group_name in group_names:
        group_summary = summary["groups"][group_name]
        preview = ", ".join(top_relations_preview(group_summary))
        lines.append(
            "| "
            f"`{group_name}` | "
            f"{group_summary['relation_count_defined']} | "
            f"{group_summary['relation_count_in_test']} | "
            f"{group_summary['triple_count_in_test']} | "
            f"{preview} |"
        )

    for idx, group_name in enumerate(group_names, start=1):
        lines.extend(["", f"## 4.{idx} `{group_name}`", ""])
        lines.append("| Model | Seeds | Group MRR | Hits@1 | Hits@3 | Hits@10 | Tail MRR | Head MRR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        ranking = []
        for label in labels:
            payload = summary["models"][label]["group_stats"].get(group_name, {})
            stats = payload.get("stats", {})
            if not stats:
                continue
            ranking.append((label, stats["mrr"]["mean"]))
            lines.append(
                "| "
                f"{label} | {payload['num_seeds']} | "
                f"{fmt(stats['mrr']['mean'], stats['mrr']['std'])} | "
                f"{fmt(stats['hits@1']['mean'], stats['hits@1']['std'])} | "
                f"{fmt(stats['hits@3']['mean'], stats['hits@3']['std'])} | "
                f"{fmt(stats['hits@10']['mean'], stats['hits@10']['std'])} | "
                f"{fmt(stats['tail_mrr']['mean'], stats['tail_mrr']['std'])} | "
                f"{fmt(stats['head_mrr']['mean'], stats['head_mrr']['std'])} |"
            )

        if ranking:
            lines.extend(["", "Ranking:"])
            for rank_idx, (label, mean_mrr) in enumerate(sorted(ranking, key=lambda item: item[1], reverse=True), start=1):
                lines.append(f"{rank_idx}. `{label}`: {mean_mrr:.4f}")

        group_relations = summary["per_relation"].get(group_name, {})
        kept_relation_count = sum(
            1
            for rel_payload in group_relations.values()
            if rel_payload.get("test_triple_count", 0) >= min_relation_test_triples
        )

        if min_relation_test_triples > 0:
            lines.extend(
                [
                    "",
                    f"Per-relation details (`test triples >= {min_relation_test_triples}`; kept {kept_relation_count} / {len(group_relations)} relations):",
                    "",
                ]
            )
        else:
            lines.extend(["", "Per-relation details:", ""])
        lines.append("| Relation | Chinese | English | Model | Seeds | MRR | Tail MRR | Head MRR | Test Triples |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")

        for rel_token in sorted(group_relations.keys(), key=lambda token: group_relations[token]["relation_id"]):
            rel_payload = group_relations[rel_token]
            if rel_payload.get("test_triple_count", 0) < min_relation_test_triples:
                continue
            for label in labels:
                model_payload = rel_payload["models"].get(label)
                if not model_payload:
                    continue
                stats = model_payload["stats"]
                triple_count = model_payload["rows"][0]["triple_count"] if model_payload["rows"] else 0
                lines.append(
                    "| "
                    f"`{rel_token}` | "
                    f"{rel_payload['text_zh']} | "
                    f"{rel_payload['text_en']} | "
                    f"{label} | {model_payload['num_seeds']} | "
                    f"{fmt(stats['mrr']['mean'], stats['mrr']['std'])} | "
                    f"{fmt(stats['tail_mrr']['mean'], stats['tail_mrr']['std'])} | "
                    f"{fmt(stats['head_mrr']['mean'], stats['head_mrr']['std'])} | "
                    f"{triple_count} |"
                )

    lines.extend(
        [
            "",
            "## 5. Next Step",
            "",
            "- Compare `visual_relations` and `weak_visual_relations` first to judge whether multimodal gain is relation-dependent.",
            "- Use this grouped result together with `has_img / no_img` analysis to decide whether the paper should emphasize a gain-boundary narrative.",
            "- If relation dependence is clear, continue to behavior analysis on gate and residual by relation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--groups-json", default="docs/relation_type_groups_draft.json")
    ap.add_argument("--output-md", default="docs/RELATION_TYPE_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/relation_type_summary.json")
    ap.add_argument("--device", default=None, help="override device, e.g. cpu/cuda/mps")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="model labels or experiment names; default: Gate-only Full Model Residual-only",
    )
    ap.add_argument(
        "--skip-per-relation",
        action="store_true",
        help="skip per-relation evaluation and only compute group-level metrics",
    )
    ap.add_argument(
        "--min-relation-test-triples",
        type=int,
        default=0,
        help="only render per-relation rows whose test triple count is at least this value",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    groups_json_path = Path(args.groups_json)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)

    groups_json = load_json(groups_json_path)
    source_files = groups_json.get("source_files", {})
    if "relation2text_zh" not in source_files:
        raise RuntimeError("groups json must define source_files.relation2text_zh")

    zh_map = load_tsv_map(Path(source_files["relation2text_zh"]))
    en_path = source_files.get("relation2text_en")
    en_map = load_tsv_map(Path(en_path)) if en_path else {}
    group_defs = build_group_definitions(groups_json, zh_map=zh_map, en_map=en_map)
    if not group_defs:
        raise RuntimeError("No valid relation groups found in groups json.")

    requested_labels = set(normalize_requested_models(args.models))
    selected_runs, duplicates = select_latest_runs(outputs_root, requested_labels)
    if not selected_runs:
        raise RuntimeError("No matching runs found for requested models.")

    run_results: dict[str, list[dict]] = defaultdict(list)
    test_relation_counter: Counter = Counter()
    for label in ordered_labels(list(selected_runs.keys())):
        for run_dir in selected_runs[label]:
            result = evaluate_run_on_groups(
                run_dir=run_dir,
                group_defs=group_defs,
                device_override=args.device,
                per_relation=not args.skip_per_relation,
            )
            run_results[label].append(result)
            if not test_relation_counter:
                test_relation_counter.update(
                    {
                        relation_token_to_id(token): count
                        for token, count in result["relation_counter"].items()
                    }
                )

    summary = build_summary(run_results, group_defs, test_relation_counter)
    summary["meta"] = {
        "outputs_root": outputs_root.as_posix(),
        "groups_json": groups_json_path.as_posix(),
        "selected_models": ordered_labels(list(run_results.keys())),
        "selected_runs": {
            label: [run.relative_to(outputs_root).as_posix() for run in runs]
            for label, runs in selected_runs.items()
        },
        "duplicate_candidates": duplicates,
        "skip_per_relation": bool(args.skip_per_relation),
        "min_relation_test_triples": int(args.min_relation_test_triples),
    }

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        render_markdown(
            summary=summary,
            selected_runs=selected_runs,
            duplicates=duplicates,
            outputs_root=outputs_root,
            groups_json_path=groups_json_path,
            min_relation_test_triples=max(0, int(args.min_relation_test_triples)),
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {output_md.as_posix()}")
    print(f"[OK] wrote {output_json.as_posix()}")


if __name__ == "__main__":
    main()
