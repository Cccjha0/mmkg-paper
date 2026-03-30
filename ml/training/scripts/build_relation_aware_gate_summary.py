from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.training.src.data.tsv_reader import read_allow_2or3
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

PRIMARY_ORDER = ["Gate-only", "Full Model"]
DEFAULT_MODEL_SET = ["Gate-only", "Full Model"]
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
    return [MODEL_LABEL_OVERRIDES.get(value, value) for value in values]


def resolve_device(requested: str) -> str:
    requested = (requested or "cuda").lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_group_definitions(groups_json: dict, zh_map: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group_name, rel_tokens in groups_json.items():
        if group_name in RESERVED_GROUP_KEYS or not isinstance(rel_tokens, list):
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
                }
            )
        out[group_name] = {"relation_ids": relation_ids, "relations": relations}
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
        if cfg.get("model", {}).get("name") not in {"openbg_img_gate_only", "openbg_img_gate_residual"}:
            continue

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


def summarize_relation_counts(triples: list[tuple[int, int, int]]) -> dict[int, int]:
    counter: dict[int, int] = defaultdict(int)
    for _, rel_id, _ in triples:
        counter[rel_id] += 1
    return dict(counter)


def summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": safe_stdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_gate_payload(head_values: list[float], tail_values: list[float], head_has_img: list[float], head_noimg: list[float], tail_has_img: list[float], tail_noimg: list[float]) -> dict:
    all_values = head_values + tail_values
    img_values = head_has_img + tail_has_img
    noimg_values = head_noimg + tail_noimg
    return {
        "all": summarize_values(all_values),
        "head": summarize_values(head_values),
        "tail": summarize_values(tail_values),
        "target_has_img": summarize_values(img_values),
        "target_noimg": summarize_values(noimg_values),
        "head_has_img": summarize_values(head_has_img),
        "head_noimg": summarize_values(head_noimg),
        "tail_has_img": summarize_values(tail_has_img),
        "tail_noimg": summarize_values(tail_noimg),
    }


@torch.no_grad()
def compute_gate_scalars(model, eids: torch.Tensor, rids: torch.Tensor, chunk_size: int) -> list[float]:
    out: list[float] = []
    for start in range(0, eids.size(0), chunk_size):
        end = min(start + chunk_size, eids.size(0))
        _, g = model._entity_with_relation(eids[start:end], rids[start:end])
        if g is None:
            raise RuntimeError("Model did not return gate tensor; relation-aware gate analysis requires fusion-enabled models.")
        out.extend(g.mean(dim=-1).detach().cpu().tolist())
    return out


def collect_side_values(model, triples: list[tuple[int, int, int]], side: str, device: str, chunk_size: int) -> tuple[list[float], list[bool]]:
    if not triples:
        return [], []
    entity_index = 0 if side == "head" else 2
    eids = torch.tensor([triple[entity_index] for triple in triples], dtype=torch.long, device=device)
    rids = torch.tensor([triple[1] for triple in triples], dtype=torch.long, device=device)
    values = compute_gate_scalars(model, eids, rids, chunk_size)
    has_img = model.has_img[eids].detach().cpu().tolist()
    return values, [bool(v) for v in has_img]


def split_by_bool(values: list[float], flags: list[bool]) -> tuple[list[float], list[float]]:
    positives = [value for value, flag in zip(values, flags) if flag]
    negatives = [value for value, flag in zip(values, flags) if not flag]
    return positives, negatives


def evaluate_run_on_groups(
    run_dir: Path,
    group_defs: dict[str, dict],
    device_override: str | None = None,
    min_relation_test_triples: int = 20,
    chunk_size: int = 4096,
) -> dict:
    cfg = load_json(run_dir / "config_merged.json")
    label = MODEL_LABEL_OVERRIDES.get(run_dir.parent.name, run_dir.parent.name)
    seed = int(cfg.get("system", {}).get("seed", -1))
    device = resolve_device(device_override or cfg.get("system", {}).get("device", "cuda"))
    cfg["system"]["device"] = device

    test3, _, bad_test = read_allow_2or3(cfg["dataset"]["test"])
    if bad_test:
        print(f"[WARN] malformed test lines skipped for {run_dir.name}: test={bad_test}")
    if not test3:
        raise RuntimeError(f"No labeled 3-column test triples found for run: {run_dir}")

    model, _ = build_model(cfg)
    model = model.to(device)
    state = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    relation_counter = summarize_relation_counts(test3)
    per_group = {}
    per_relation = {}

    for group_name, group_info in group_defs.items():
        relation_ids = set(group_info["relation_ids"])
        group_triples = [triple for triple in test3 if triple[1] in relation_ids]
        if not group_triples:
            per_group[group_name] = {"triple_count": 0, "gate_stats": {}}
            per_relation[group_name] = {}
            continue

        head_values, head_has_flags = collect_side_values(model, group_triples, "head", device, chunk_size)
        tail_values, tail_has_flags = collect_side_values(model, group_triples, "tail", device, chunk_size)
        head_has_img, head_noimg = split_by_bool(head_values, head_has_flags)
        tail_has_img, tail_noimg = split_by_bool(tail_values, tail_has_flags)

        per_group[group_name] = {
            "triple_count": len(group_triples),
            "gate_stats": summarize_gate_payload(
                head_values=head_values,
                tail_values=tail_values,
                head_has_img=head_has_img,
                head_noimg=head_noimg,
                tail_has_img=tail_has_img,
                tail_noimg=tail_noimg,
            ),
        }

        group_relations = {}
        for rel in group_info["relations"]:
            rel_id = rel["relation_id"]
            rel_triples = [triple for triple in group_triples if triple[1] == rel_id]
            if len(rel_triples) < min_relation_test_triples:
                continue
            rel_head_values, rel_head_flags = collect_side_values(model, rel_triples, "head", device, chunk_size)
            rel_tail_values, rel_tail_flags = collect_side_values(model, rel_triples, "tail", device, chunk_size)
            rel_head_has_img, rel_head_noimg = split_by_bool(rel_head_values, rel_head_flags)
            rel_tail_has_img, rel_tail_noimg = split_by_bool(rel_tail_values, rel_tail_flags)
            group_relations[rel["relation_token"]] = {
                "relation_id": rel_id,
                "relation_token": rel["relation_token"],
                "text_zh": rel["text_zh"],
                "test_triple_count": len(rel_triples),
                "gate_stats": summarize_gate_payload(
                    head_values=rel_head_values,
                    tail_values=rel_tail_values,
                    head_has_img=rel_head_has_img,
                    head_noimg=rel_head_noimg,
                    tail_has_img=rel_tail_has_img,
                    tail_noimg=rel_tail_noimg,
                ),
            }
        per_relation[group_name] = group_relations

    return {
        "label": label,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "relative_run_dir": run_dir.relative_to(run_dir.parents[2]).as_posix(),
        "device": device,
        "relation_counter": {relation_id_to_token(rel_id): count for rel_id, count in sorted(relation_counter.items())},
        "groups": per_group,
        "per_relation": per_relation,
    }


def aggregate_gate_rows(rows: list[dict]) -> dict[str, dict]:
    views = ["all", "head", "tail", "target_has_img", "target_noimg", "head_has_img", "head_noimg", "tail_has_img", "tail_noimg"]
    out = {}
    for view in views:
        mean_values = [float(row[view]["mean"]) for row in rows if row.get(view)]
        within_std_values = [float(row[view]["std"]) for row in rows if row.get(view)]
        if not mean_values:
            continue
        out[view] = {
            "mean_of_means": statistics.mean(mean_values),
            "std_of_means": safe_stdev(mean_values),
            "mean_values": mean_values,
            "mean_of_within_std": statistics.mean(within_std_values) if within_std_values else 0.0,
            "std_of_within_std": safe_stdev(within_std_values) if within_std_values else 0.0,
            "within_std_values": within_std_values,
        }
    return out


def build_summary(run_results: dict[str, list[dict]], group_defs: dict[str, dict], test_relation_counter: dict[int, int]) -> dict:
    summary = {"groups": {}, "models": {}, "per_relation": {}}

    for group_name, group_info in group_defs.items():
        present_relations = [rel for rel in group_info["relations"] if test_relation_counter.get(rel["relation_id"], 0) > 0]
        summary["groups"][group_name] = {
            "relation_ids": group_info["relation_ids"],
            "relation_count_defined": len(group_info["relation_ids"]),
            "relation_count_in_test": len(present_relations),
            "triple_count_in_test": int(sum(test_relation_counter.get(rel["relation_id"], 0) for rel in group_info["relations"])),
            "relations_in_test": present_relations,
        }

    for label, rows in run_results.items():
        model_summary = {"num_seeds": len(rows), "runs": rows, "group_stats": {}}
        for group_name in group_defs:
            group_rows = []
            for row in rows:
                payload = row["groups"].get(group_name, {})
                gate_stats = payload.get("gate_stats", {})
                if not gate_stats:
                    continue
                group_rows.append({"seed": row["seed"], "run_dir": row["relative_run_dir"], **gate_stats})
            model_summary["group_stats"][group_name] = {
                "num_seeds": len(group_rows),
                "rows": group_rows,
                "stats": aggregate_gate_rows(group_rows),
            }
        summary["models"][label] = model_summary

    for group_name, group_info in group_defs.items():
        relation_bucket: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for label, rows in run_results.items():
            for row in rows:
                for rel_token, rel_payload in row["per_relation"].get(group_name, {}).items():
                    relation_bucket[rel_token][label].append(
                        {
                            "seed": row["seed"],
                            "run_dir": row["relative_run_dir"],
                            "test_triple_count": rel_payload["test_triple_count"],
                            **rel_payload["gate_stats"],
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
                "test_triple_count": int(test_relation_counter.get(rel["relation_id"], 0)),
                "models": {
                    label: {
                        "num_seeds": len(rows),
                        "rows": rows,
                        "stats": aggregate_gate_rows(rows),
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


def render_markdown(summary: dict, selected_runs: dict[str, list[Path]], duplicates: dict[str, dict[str, list[str]]], outputs_root: Path, groups_json_path: Path, min_relation_test_triples: int) -> str:
    labels = ordered_labels(list(summary["models"].keys()))
    group_names = list(summary["groups"].keys())
    lines = [
        "# Relation-Aware Gate Summary",
        "",
        "## 1. Purpose",
        "",
        "This document summarizes gate behavior on real `(entity, relation)` pairs from the test split.",
        "",
        "Unlike the training-time gate summary, this script does not sample random relation ids. It measures gate values under the actual relations that appear in test triples.",
        "",
        "Current focus:",
        "",
        "- relation-group-aware gate mean and std",
        "- head vs tail target gate behavior",
        "- image-available vs no-image target gate behavior",
        f"- per-relation gate details with `test triples >= {min_relation_test_triples}`",
        "",
        "## 2. Selected Runs",
        "",
        f"Outputs root: `{outputs_root.as_posix()}`",
        "",
        f"Grouping source: `{groups_json_path.as_posix()}`",
        "",
    ]
    for label in labels:
        rel_paths = [run.relative_to(outputs_root).as_posix() for run in selected_runs.get(label, [])]
        lines.append(f"- `{label}`: {', '.join(f'`{path}`' for path in rel_paths)}")
    duplicate_lines = []
    for label in labels:
        for seed, candidates in sorted(duplicates.get(label, {}).items()):
            duplicate_lines.append(f"- `{label}` seed `{seed}` had multiple runs; selected latest: `{candidates[-1]}`")
    if duplicate_lines:
        lines.extend(["", "Duplicate handling:"] + duplicate_lines)

    lines.extend(["", "## 3. Group Definition and Test Coverage", ""])
    lines.append("| Group | Defined Relations | Relations in Test | Test Triples | Preview |")
    lines.append("|---|---:|---:|---:|---|")
    for group_name in group_names:
        group_summary = summary["groups"][group_name]
        lines.append(
            "| "
            f"`{group_name}` | {group_summary['relation_count_defined']} | {group_summary['relation_count_in_test']} | "
            f"{group_summary['triple_count_in_test']} | {', '.join(top_relations_preview(group_summary))} |"
        )

    for idx, group_name in enumerate(group_names, start=1):
        lines.extend(["", f"## 4.{idx} `{group_name}`", ""])
        lines.append("| Model | Seeds | Gate Mean (All Targets) | Gate Std (All Targets) | Head Mean | Tail Mean | Target Img Mean | Target NoImg Mean | Img-NoImg Gap |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label in labels:
            payload = summary["models"][label]["group_stats"].get(group_name, {})
            stats = payload.get("stats", {})
            if "all" not in stats:
                continue
            img_mean = stats.get("target_has_img", {}).get("mean_of_means", 0.0)
            img_std = stats.get("target_has_img", {}).get("std_of_means", 0.0)
            noimg_mean = stats.get("target_noimg", {}).get("mean_of_means", 0.0)
            noimg_std = stats.get("target_noimg", {}).get("std_of_means", 0.0)
            lines.append(
                "| "
                f"{label} | {payload['num_seeds']} | "
                f"{fmt(stats['all']['mean_of_means'], stats['all']['std_of_means'])} | "
                f"{fmt(stats['all']['mean_of_within_std'], stats['all']['std_of_within_std'])} | "
                f"{fmt(stats.get('head', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('head', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                f"{fmt(stats.get('tail', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('tail', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                f"{fmt(img_mean, img_std)} | "
                f"{fmt(noimg_mean, noimg_std)} | "
                f"{fmt(img_mean - noimg_mean, 0.0)} |"
            )

        group_relations = summary["per_relation"].get(group_name, {})
        kept_relation_count = len(group_relations)
        lines.extend(
            [
                "",
                f"Per-relation details (`test triples >= {min_relation_test_triples}`; kept {kept_relation_count} relations):",
                "",
            ]
        )
        lines.append("| Relation | Chinese | Model | Seeds | Gate Mean (All) | Head Mean | Tail Mean | Target Img Mean | Target NoImg Mean | Test Triples |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for rel_token in sorted(group_relations.keys(), key=lambda token: group_relations[token]["relation_id"]):
            rel_payload = group_relations[rel_token]
            for label in labels:
                model_payload = rel_payload["models"].get(label)
                if not model_payload:
                    continue
                stats = model_payload["stats"]
                lines.append(
                    "| "
                    f"`{rel_token}` | {rel_payload['text_zh']} | {label} | {model_payload['num_seeds']} | "
                    f"{fmt(stats['all']['mean_of_means'], stats['all']['std_of_means'])} | "
                    f"{fmt(stats.get('head', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('head', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                    f"{fmt(stats.get('tail', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('tail', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                    f"{fmt(stats.get('target_has_img', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('target_has_img', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                    f"{fmt(stats.get('target_noimg', {'mean_of_means': 0.0, 'std_of_means': 0.0})['mean_of_means'], stats.get('target_noimg', {'mean_of_means': 0.0, 'std_of_means': 0.0})['std_of_means'])} | "
                    f"{rel_payload['test_triple_count']} |"
                )

    lines.extend(
        [
            "",
            "## 5. Next Step",
            "",
            "- Compare `Gate-only` and `Full Model` group-by-group to judge whether gate behavior really changes with relation characteristics.",
            "- Connect relation-aware gate patterns back to the completed `6.2 relation type` results.",
            "- Extend this with residual subgroup analysis to finish the remaining parts of `7.2` and `7.3`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--groups-json", default="docs/relation_type_groups_draft.json")
    ap.add_argument("--output-md", default="docs/RELATION_AWARE_GATE_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/relation_aware_gate_summary.json")
    ap.add_argument("--device", default=None, help="override device, e.g. cpu/cuda/mps")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--min-relation-test-triples", type=int, default=20)
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="model labels or experiment names; default: Gate-only Full Model",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    groups_json_path = Path(args.groups_json)
    groups_json = load_json(groups_json_path)
    source_files = groups_json.get("source_files", {})
    if "relation2text_zh" not in source_files:
        raise RuntimeError("groups json must define source_files.relation2text_zh")
    zh_map = load_tsv_map(Path(source_files["relation2text_zh"]))
    group_defs = build_group_definitions(groups_json, zh_map)
    if not group_defs:
        raise RuntimeError("No valid relation groups found in groups json.")

    requested_labels = set(normalize_requested_models(args.models))
    selected_runs, duplicates = select_latest_runs(outputs_root, requested_labels)
    if not selected_runs:
        raise RuntimeError("No matching fusion-enabled runs found for requested models.")

    run_results: dict[str, list[dict]] = defaultdict(list)
    test_relation_counter: dict[int, int] = {}
    for label in ordered_labels(list(selected_runs.keys())):
        for run_dir in selected_runs[label]:
            result = evaluate_run_on_groups(
                run_dir=run_dir,
                group_defs=group_defs,
                device_override=args.device,
                min_relation_test_triples=max(0, int(args.min_relation_test_triples)),
                chunk_size=max(1, int(args.chunk_size)),
            )
            run_results[label].append(result)
            if not test_relation_counter:
                test_relation_counter = {
                    relation_token_to_id(token): count
                    for token, count in result["relation_counter"].items()
                }

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
        "min_relation_test_triples": int(args.min_relation_test_triples),
        "chunk_size": int(args.chunk_size),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        render_markdown(
            summary=summary,
            selected_runs=selected_runs,
            duplicates=duplicates,
            outputs_root=outputs_root,
            groups_json_path=groups_json_path,
            min_relation_test_triples=int(args.min_relation_test_triples),
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {output_md.as_posix()}")
    print(f"[OK] wrote {output_json.as_posix()}")


if __name__ == "__main__":
    main()
