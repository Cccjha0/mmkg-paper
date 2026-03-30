from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

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

PRIMARY_ORDER = ["Full Model", "Residual-only"]
DEFAULT_MODEL_SET = ["Full Model", "Residual-only"]
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
        if cfg.get("model", {}).get("name") not in {"openbg_img_residual_only", "openbg_img_gate_residual"}:
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


def aggregate_stats(rows: list[dict]) -> dict:
    keys = [
        "residual_norm",
        "effective_residual_norm",
        "fused_norm",
        "effective_fused_norm",
        "residual_to_fused_ratio",
        "residual_minus_fused",
        "residual_scale_value",
        "mix_w_fusion",
        "mix_w_residual",
    ]
    out = {}
    for key in keys:
        values = [row[key]["mean"] for row in rows if row.get(key)]
        within = [row[key]["std"] for row in rows if row.get(key)]
        if not values:
            continue
        out[key] = {
            "mean_of_means": statistics.mean(values),
            "std_of_means": safe_stdev(values),
            "mean_of_within_std": statistics.mean(within) if within else 0.0,
            "std_of_within_std": safe_stdev(within) if within else 0.0,
        }
    return out


def build_side_subsets(triples: list[tuple[int, int, int]], has_img: torch.Tensor) -> dict[str, list[tuple[int, int, int]]]:
    head_has_img = [triple for triple in triples if bool(has_img[triple[0]].item())]
    head_noimg = [triple for triple in triples if not bool(has_img[triple[0]].item())]
    tail_has_img = [triple for triple in triples if bool(has_img[triple[2]].item())]
    tail_noimg = [triple for triple in triples if not bool(has_img[triple[2]].item())]
    return {
        "all_targets": triples,
        "head_all": triples,
        "tail_all": triples,
        "head_has_img": head_has_img,
        "head_noimg": head_noimg,
        "tail_has_img": tail_has_img,
        "tail_noimg": tail_noimg,
        "target_has_img": head_has_img + tail_has_img,
        "target_noimg": head_noimg + tail_noimg,
    }


@torch.no_grad()
def compute_side_payload(model, eids: torch.Tensor, rids: torch.Tensor) -> dict[str, list[float]]:
    if hasattr(model, "entity_residual"):
        scale = F.softplus(model.residual_scale).detach()
        residual = scale * model.entity_residual(eids)
    else:
        residual = torch.zeros((eids.size(0), model.d), device=eids.device)
        scale = torch.zeros((), device=eids.device)

    residual_norm = residual.norm(dim=-1)

    has_fusion_branch = hasattr(model, "fusion")
    if has_fusion_branch:
        fused, _ = model._entity_with_relation(eids, rids)
        if getattr(model, "use_residual", False):
            if getattr(model, "use_normalized_mix", False):
                a = F.softplus(model.mix_fusion_raw).detach()
                b = F.softplus(model.mix_residual_raw).detach()
                denom = a + b + 1e-12
                mix_fusion = (a / denom).item()
                mix_residual = (b / denom).item()
            else:
                mix_fusion = 1.0
                mix_residual = 1.0
            fused_branch = fused - residual * mix_residual if getattr(model, "use_normalized_mix", False) else fused - residual
        else:
            fused_branch = fused
            mix_fusion = 1.0
            mix_residual = 0.0
    else:
        fused_branch = torch.zeros_like(residual)
        mix_fusion = 0.0
        mix_residual = 1.0 if getattr(model, "use_residual", False) else 0.0

    fused_norm = fused_branch.norm(dim=-1)
    effective_residual_norm = residual_norm * float(mix_residual)
    effective_fused_norm = fused_norm * float(mix_fusion)
    minus = effective_residual_norm - effective_fused_norm

    payload = {
        "residual_norm": residual_norm.detach().cpu().tolist(),
        "effective_residual_norm": effective_residual_norm.detach().cpu().tolist(),
        "fused_norm": fused_norm.detach().cpu().tolist(),
        "effective_fused_norm": effective_fused_norm.detach().cpu().tolist(),
        "residual_minus_fused": minus.detach().cpu().tolist(),
        "residual_scale_value": [float(scale.item())] * eids.size(0),
        "mix_w_fusion": [float(mix_fusion)] * eids.size(0),
        "mix_w_residual": [float(mix_residual)] * eids.size(0),
    }
    if has_fusion_branch:
        ratio = effective_residual_norm / (effective_fused_norm + 1e-12)
        payload["residual_to_fused_ratio"] = ratio.detach().cpu().tolist()
    return payload


def subset_stats(
    model,
    triples: list[tuple[int, int, int]],
    side: str,
    device: str,
    chunk_size: int,
) -> dict[str, dict[str, float]] | None:
    if not triples:
        return None
    entity_index = 0 if side == "head" else 2
    eids = torch.tensor([triple[entity_index] for triple in triples], dtype=torch.long, device=device)
    rids = torch.tensor([triple[1] for triple in triples], dtype=torch.long, device=device)
    payload_acc: dict[str, list[float]] = defaultdict(list)
    for start in range(0, eids.size(0), chunk_size):
        end = min(start + chunk_size, eids.size(0))
        chunk = compute_side_payload(model, eids[start:end], rids[start:end])
        for key, values in chunk.items():
            payload_acc[key].extend(values)
    return {key: summarize_values(values) for key, values in payload_acc.items()}


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
        subsets = build_side_subsets(group_triples, model.has_img)
        group_payload = {
            "test_triple_count": len(group_triples),
            "subgroups": {
                subset_name: {
                    "count": len(subset_triples),
                    "head": subset_stats(model, subset_triples, "head", device, chunk_size),
                    "tail": subset_stats(model, subset_triples, "tail", device, chunk_size),
                }
                for subset_name, subset_triples in subsets.items()
            },
        }
        per_group[group_name] = group_payload

        per_relation[group_name] = {}
        for rel in group_info["relations"]:
            rel_id = rel["relation_id"]
            rel_count = relation_counter.get(rel_id, 0)
            if rel_count < min_relation_test_triples:
                continue
            rel_triples = [triple for triple in group_triples if triple[1] == rel_id]
            rel_subsets = build_side_subsets(rel_triples, model.has_img)
            per_relation[group_name][rel["relation_token"]] = {
                "relation_id": rel_id,
                "text_zh": rel["text_zh"],
                "test_triple_count": rel_count,
                "subgroups": {
                    subset_name: {
                        "count": len(subset_triples),
                        "head": subset_stats(model, subset_triples, "head", device, chunk_size),
                        "tail": subset_stats(model, subset_triples, "tail", device, chunk_size),
                    }
                    for subset_name, subset_triples in rel_subsets.items()
                },
            }

    return {
        "label": label,
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "relative_run_dir": run_dir.as_posix(),
        "groups": per_group,
        "per_relation": per_relation,
        "relation_counter": {relation_id_to_token(k): v for k, v in relation_counter.items()},
    }


def aggregate_subgroup_rows(rows: list[dict], side_key: str) -> dict:
    payloads = [row[side_key] for row in rows if row.get(side_key)]
    if not payloads:
        return {}
    return aggregate_stats(payloads)


def build_summary(run_results: dict[str, list[dict]], group_defs: dict[str, dict], test_relation_counter: dict[int, int]) -> dict:
    summary = {"groups": {}, "models": {}, "per_relation": {}}

    for group_name, group_info in group_defs.items():
        relation_ids = group_info["relation_ids"]
        relations_in_test = [rel for rel in group_info["relations"] if test_relation_counter.get(rel["relation_id"], 0) > 0]
        triple_count_in_test = sum(test_relation_counter.get(rel_id, 0) for rel_id in relation_ids)
        summary["groups"][group_name] = {
            "relation_count_defined": len(relation_ids),
            "relation_count_in_test": len(relations_in_test),
            "triple_count_in_test": triple_count_in_test,
            "relations_in_test": relations_in_test,
        }

    for label, rows in run_results.items():
        model_summary = {"num_seeds": len(rows), "runs": rows, "group_stats": {}}
        for group_name in group_defs.keys():
            subgroup_bucket: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                payload = row["groups"].get(group_name, {})
                for subgroup_name, subgroup_payload in payload.get("subgroups", {}).items():
                    subgroup_bucket[subgroup_name].append(
                        {
                            "seed": row["seed"],
                            "run_dir": row["relative_run_dir"],
                            "count": subgroup_payload.get("count", 0),
                            "head": subgroup_payload.get("head"),
                            "tail": subgroup_payload.get("tail"),
                        }
                    )
            model_summary["group_stats"][group_name] = {
                subgroup_name: {
                    "num_seeds": len(subgroup_rows),
                    "rows": subgroup_rows,
                    "head_stats": aggregate_subgroup_rows(subgroup_rows, "head"),
                    "tail_stats": aggregate_subgroup_rows(subgroup_rows, "tail"),
                }
                for subgroup_name, subgroup_rows in subgroup_bucket.items()
            }
        summary["models"][label] = model_summary

    for group_name, group_info in group_defs.items():
        relation_bucket: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for label, rows in run_results.items():
            for row in rows:
                for rel_token, rel_payload in row["per_relation"].get(group_name, {}).items():
                    for subgroup_name, subgroup_payload in rel_payload.get("subgroups", {}).items():
                        relation_bucket[rel_token][label][subgroup_name].append(
                            {
                                "seed": row["seed"],
                                "run_dir": row["relative_run_dir"],
                                "count": subgroup_payload.get("count", 0),
                                "head": subgroup_payload.get("head"),
                                "tail": subgroup_payload.get("tail"),
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
                        subgroup_name: {
                            "num_seeds": len(subgroup_rows),
                            "rows": subgroup_rows,
                            "head_stats": aggregate_subgroup_rows(subgroup_rows, "head"),
                            "tail_stats": aggregate_subgroup_rows(subgroup_rows, "tail"),
                        }
                        for subgroup_name, subgroup_rows in subgroup_rows_by_name.items()
                    }
                    for label, subgroup_rows_by_name in model_rows.items()
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


def stats_cell(stats: dict, key: str) -> str:
    payload = stats.get(key)
    if not payload:
        return "n/a"
    return fmt(payload["mean_of_means"], payload["std_of_means"])


def render_markdown(summary: dict, selected_runs: dict[str, list[Path]], duplicates: dict[str, dict[str, list[str]]], outputs_root: Path, groups_json_path: Path, min_relation_test_triples: int) -> str:
    labels = ordered_labels(list(summary["models"].keys()))
    group_names = list(summary["groups"].keys())
    lines = [
        "# Relation-Aware Residual Summary",
        "",
        "## 1. Purpose",
        "",
        "This document summarizes residual-branch behavior on real `(entity, relation)` pairs from the test split.",
        "",
        "Current focus:",
        "",
        "- residual norm and effective residual contribution",
        "- fused-branch norm and effective fused contribution",
        "- residual-to-fused ratio on real subgroup and relation-group slices",
        f"- per-relation residual details with `test triples >= {min_relation_test_triples}`",
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

    subgroup_order = ["head_has_img", "head_noimg", "tail_noimg", "target_has_img", "target_noimg", "all_targets"]
    for idx, group_name in enumerate(group_names, start=1):
        lines.extend(["", f"## 4.{idx} `{group_name}`", ""])
        for subgroup_name in subgroup_order:
            if not any(subgroup_name in summary["models"][label]["group_stats"].get(group_name, {}) for label in labels):
                continue
            lines.extend(["", f"### {subgroup_name}", ""])
            lines.append("| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for label in labels:
                subgroup_payload = summary["models"][label]["group_stats"].get(group_name, {}).get(subgroup_name)
                if not subgroup_payload:
                    continue
                for side_key in ["head_stats", "tail_stats"]:
                    side_label = "head" if side_key == "head_stats" else "tail"
                    stats = subgroup_payload.get(side_key, {})
                    if not stats:
                        continue
                    lines.append(
                        "| "
                        f"{label} | {side_label} | {subgroup_payload['num_seeds']} | "
                        f"{stats_cell(stats, 'residual_norm')} | "
                        f"{stats_cell(stats, 'effective_residual_norm')} | "
                        f"{stats_cell(stats, 'fused_norm')} | "
                        f"{stats_cell(stats, 'effective_fused_norm')} | "
                        f"{stats_cell(stats, 'residual_to_fused_ratio')} | "
                        f"{stats_cell(stats, 'residual_minus_fused')} | "
                        f"{stats_cell(stats, 'residual_scale_value')} | "
                        f"{stats_cell(stats, 'mix_w_fusion')} | "
                        f"{stats_cell(stats, 'mix_w_residual')} |"
                    )

        group_relations = summary["per_relation"].get(group_name, {})
        kept_relation_count = len(group_relations)
        lines.extend(["", f"Per-relation residual preview (`test triples >= {min_relation_test_triples}`; kept {kept_relation_count} relations):", ""])
        lines.append("| Relation | Chinese | Model | Subgroup | Side | Seeds | Effective Residual | Effective Fused | Residual/Fused Ratio | Test Triples |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|")
        for rel_token in sorted(group_relations.keys(), key=lambda token: group_relations[token]["relation_id"]):
            rel_payload = group_relations[rel_token]
            for label in labels:
                model_payload = rel_payload["models"].get(label, {})
                for subgroup_name in ["head_has_img", "head_noimg", "tail_noimg"]:
                    subgroup_payload = model_payload.get(subgroup_name)
                    if not subgroup_payload:
                        continue
                    for side_key in ["head_stats", "tail_stats"]:
                        side_label = "head" if side_key == "head_stats" else "tail"
                        stats = subgroup_payload.get(side_key, {})
                        if not stats:
                            continue
                        lines.append(
                            "| "
                            f"`{rel_token}` | {rel_payload['text_zh']} | {label} | {subgroup_name} | {side_label} | {subgroup_payload['num_seeds']} | "
                            f"{stats_cell(stats, 'effective_residual_norm')} | "
                            f"{stats_cell(stats, 'effective_fused_norm')} | "
                            f"{stats_cell(stats, 'residual_to_fused_ratio')} | "
                            f"{rel_payload['test_triple_count']} |"
                        )

    lines.extend(
        [
            "",
            "## 5. Next Step",
            "",
            "- Compare `head_has_img / head_noimg / tail_noimg` to judge whether residual dominance is stronger on missing-image targets.",
            "- Connect residual subgroup patterns back to the completed `6.1` and `6.2` analyses.",
            "- Combine this with relation-aware gate results to finish `7.2` and prepare `7.3 fusion vs residual`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--groups-json", default="docs/relation_type_groups_draft.json")
    ap.add_argument("--output-md", default="docs/RELATION_AWARE_RESIDUAL_SUMMARY.md")
    ap.add_argument("--output-json", default="docs/relation_aware_residual_summary.json")
    ap.add_argument("--device", default=None, help="override device, e.g. cpu/cuda/mps")
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--min-relation-test-triples", type=int, default=20)
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="model labels or experiment names; default: Full Model Residual-only",
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
        raise RuntimeError("No matching residual-enabled runs found for requested models.")

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
            result["relative_run_dir"] = run_dir.relative_to(outputs_root).as_posix()
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
