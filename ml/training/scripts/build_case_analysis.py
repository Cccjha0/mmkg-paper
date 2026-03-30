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

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.tsv_reader import read_allow_2or3
from ml.training.src.eval.filtered_ranking import prepare_true_heads_index, prepare_true_tails_index
from ml.training.src.models.build_model import build_model


MODEL_LABEL_OVERRIDES = {
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


def relation_token_to_id(token: str) -> int:
    return int(token.replace("rel_", ""))


def relation_id_to_token(rel_id: int) -> str:
    return f"rel_{rel_id:04d}"


def entity_id_to_token(entity_id: int) -> str:
    return f"ent_{entity_id:06d}"


def normalize_requested_models(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_MODEL_SET)
    return [MODEL_LABEL_OVERRIDES.get(value, value) for value in values]


def resolve_device(requested: str | None) -> str:
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


def build_group_definitions(groups_json: dict, rel_map: dict[str, str]) -> tuple[dict[str, dict], dict[int, str]]:
    out: dict[str, dict] = {}
    rel_to_group: dict[int, str] = {}
    for group_name, rel_tokens in groups_json.items():
        if group_name in RESERVED_GROUP_KEYS or not isinstance(rel_tokens, list):
            continue
        relation_ids = sorted(relation_token_to_id(token) for token in rel_tokens)
        relations = []
        for rel_id in relation_ids:
            rel_to_group[rel_id] = group_name
            token = relation_id_to_token(rel_id)
            relations.append({"relation_id": rel_id, "relation_token": token, "text_zh": rel_map.get(token, "")})
        out[group_name] = {"relation_ids": relation_ids, "relations": relations}
    return out, rel_to_group


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def select_latest_runs(outputs_root: Path, requested_labels: set[str]) -> tuple[dict[str, dict[int, Path]], dict[str, dict[str, list[str]]]]:
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

    selected: dict[str, dict[int, Path]] = {}
    for label, seed_map in by_label_seed.items():
        selected[label] = {}
        for seed, candidates in sorted(seed_map.items()):
            candidates = sorted(candidates, key=lambda p: p.name)
            chosen = candidates[-1]
            selected[label][seed] = chosen
            if len(candidates) > 1:
                duplicates[label][str(seed)] = [p.relative_to(outputs_root).as_posix() for p in candidates]
    return selected, duplicates


@torch.inference_mode()
def compute_tail_ranks(model, triples: torch.LongTensor, true_tails: dict, num_entities: int, chunk_size: int, query_batch_size: int, device: str) -> torch.Tensor:
    device_t = torch.device(device)
    all_entities = torch.arange(num_entities, dtype=torch.long)
    ranks = []
    neg_inf = float("-inf")
    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]
        bq = q.size(0)
        h = q[:, 0].to(device_t)
        r = q[:, 1].to(device_t)
        t = q[:, 2].to(device_t)
        t_cpu = q[:, 2]
        target = model.score(torch.stack([h, r, t], dim=1)).unsqueeze(1)
        filt_excl = []
        for j in range(bq):
            key = (int(q[j, 0].item()), int(q[j, 1].item()))
            idx = true_tails.get(key, torch.empty(0, dtype=torch.long))
            if idx.numel() > 0:
                idx = idx[idx != int(t_cpu[j].item())]
            filt_excl.append(idx)
        greater = torch.zeros(bq, dtype=torch.long, device=device_t)
        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device_t)
            h_g = h.unsqueeze(1).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = cand.unsqueeze(0).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)
            scores = model.score(batch).view(bq, c)
            row_chunks = []
            col_chunks = []
            for j in range(bq):
                idx = filt_excl[j]
                if idx.numel() == 0:
                    continue
                l = int(torch.searchsorted(idx, start, right=False).item())
                rr = int(torch.searchsorted(idx, end, right=False).item())
                local = idx[l:rr]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)
            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device_t)
                cols = torch.cat(col_chunks, dim=0).to(device_t)
                scores[rows, cols] = neg_inf
            greater += (scores > target).sum(dim=1)
        ranks.append((greater + 1).detach().cpu())
    return torch.cat(ranks, dim=0) if ranks else torch.empty(0, dtype=torch.long)


@torch.inference_mode()
def compute_head_ranks(model, triples: torch.LongTensor, true_heads: dict, num_entities: int, chunk_size: int, query_batch_size: int, device: str) -> torch.Tensor:
    device_t = torch.device(device)
    all_entities = torch.arange(num_entities, dtype=torch.long)
    ranks = []
    neg_inf = float("-inf")
    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]
        bq = q.size(0)
        h = q[:, 0].to(device_t)
        r = q[:, 1].to(device_t)
        t = q[:, 2].to(device_t)
        h_cpu = q[:, 0]
        target = model.score(torch.stack([h, r, t], dim=1)).unsqueeze(1)
        filt_excl = []
        for j in range(bq):
            key = (int(q[j, 1].item()), int(q[j, 2].item()))
            idx = true_heads.get(key, torch.empty(0, dtype=torch.long))
            if idx.numel() > 0:
                idx = idx[idx != int(h_cpu[j].item())]
            filt_excl.append(idx)
        greater = torch.zeros(bq, dtype=torch.long, device=device_t)
        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device_t)
            h_g = cand.unsqueeze(0).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = t.unsqueeze(1).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)
            scores = model.score(batch).view(bq, c)
            row_chunks = []
            col_chunks = []
            for j in range(bq):
                idx = filt_excl[j]
                if idx.numel() == 0:
                    continue
                l = int(torch.searchsorted(idx, start, right=False).item())
                rr = int(torch.searchsorted(idx, end, right=False).item())
                local = idx[l:rr]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)
            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device_t)
                cols = torch.cat(col_chunks, dim=0).to(device_t)
                scores[rows, cols] = neg_inf
            greater += (scores > target).sum(dim=1)
        ranks.append((greater + 1).detach().cpu())
    return torch.cat(ranks, dim=0) if ranks else torch.empty(0, dtype=torch.long)


@torch.inference_mode()
def compute_full_model_behavior(model, entity_id: int, relation_id: int, device: str) -> dict:
    device_t = torch.device(device)
    eids = torch.tensor([entity_id], dtype=torch.long, device=device_t)
    rids = torch.tensor([relation_id], dtype=torch.long, device=device_t)
    t = model.t_adapter(model._entity_text(eids))
    v = model.v_adapter(model._entity_image(eids))
    z_fused, g = model.fusion(t, v, rids)
    res = F.softplus(model.residual_scale) * model.entity_residual(eids)
    if getattr(model, "use_normalized_mix", False):
        a = F.softplus(model.mix_fusion_raw)
        b = F.softplus(model.mix_residual_raw)
        denom = a + b + 1e-12
        mix_f = float((a / denom).detach().cpu().item())
        mix_r = float((b / denom).detach().cpu().item())
    else:
        mix_f = 1.0
        mix_r = 1.0
    eff_res = float((res.norm(dim=-1) * mix_r).detach().cpu().item())
    eff_fused = float((z_fused.norm(dim=-1) * mix_f).detach().cpu().item())
    ratio = eff_res / (eff_fused + 1e-12)
    gate_mean = float(g.mean(dim=-1).detach().cpu().item()) if g is not None else 0.0
    return {
        "gate_mean": gate_mean,
        "effective_residual_norm": eff_res,
        "effective_fused_norm": eff_fused,
        "residual_to_fused_ratio": ratio,
        "mix_w_fusion": mix_f,
        "mix_w_residual": mix_r,
    }


@torch.inference_mode()
def compute_filtered_topk(model, triple: tuple[int, int, int], direction: str, true_tails: dict, true_heads: dict, num_entities: int, device: str, topk: int = 5) -> list[dict]:
    device_t = torch.device(device)
    h, r, t = triple
    cand = torch.arange(num_entities, dtype=torch.long, device=device_t)
    if direction == "tail":
        h_g = torch.full((num_entities,), h, dtype=torch.long, device=device_t)
        r_g = torch.full((num_entities,), r, dtype=torch.long, device=device_t)
        batch = torch.stack([h_g, r_g, cand], dim=1)
        scores = model.score(batch)
        filt_idx = true_tails.get((h, r), torch.empty(0, dtype=torch.long))
        if filt_idx.numel() > 0:
            filt_idx = filt_idx[filt_idx != t].to(device_t)
            scores[filt_idx] = float("-inf")
        target_id = t
    else:
        r_g = torch.full((num_entities,), r, dtype=torch.long, device=device_t)
        t_g = torch.full((num_entities,), t, dtype=torch.long, device=device_t)
        batch = torch.stack([cand, r_g, t_g], dim=1)
        scores = model.score(batch)
        filt_idx = true_heads.get((r, t), torch.empty(0, dtype=torch.long))
        if filt_idx.numel() > 0:
            filt_idx = filt_idx[filt_idx != h].to(device_t)
            scores[filt_idx] = float("-inf")
        target_id = h
    k = min(topk, num_entities)
    top_scores, top_idx = torch.topk(scores, k=k, largest=True)
    out = []
    for idx, score in zip(top_idx.detach().cpu().tolist(), top_scores.detach().cpu().tolist()):
        out.append({"entity_id": idx, "entity_token": entity_id_to_token(idx), "score": float(score), "is_gold": idx == target_id})
    return out


def cue_hint(direction: str, target_has_img: bool, relation_group: str) -> str:
    if direction == "head" and target_has_img and relation_group == "visual_relations":
        return "likely multimodal-favorable: head target has image and relation is visually grounded"
    if direction == "head" and not target_has_img:
        return "likely structure-favorable: head target has no image, residual compensation should matter more"
    if direction == "tail":
        return "likely structure-favorable under current split: tail target is typically no-image"
    return "mixed cue regime"


def build_case_records(
    triples: list[tuple[int, int, int]],
    per_seed_ranks: dict[str, dict[int, dict[str, torch.Tensor]]],
    relation_group_map: dict[int, str],
    has_img: torch.Tensor,
) -> list[dict]:
    common_seeds = sorted(set(per_seed_ranks["Full Model"].keys()) & set(per_seed_ranks["Residual-only"].keys()))
    records = []
    for idx, triple in enumerate(triples):
        h, r, t = triple
        for direction in ["head", "tail"]:
            target = h if direction == "head" else t
            target_has_img = bool(has_img[target].item())
            full_ranks = [int(per_seed_ranks["Full Model"][seed][direction][idx].item()) for seed in common_seeds]
            res_ranks = [int(per_seed_ranks["Residual-only"][seed][direction][idx].item()) for seed in common_seeds]
            full_mrrs = [1.0 / rank for rank in full_ranks]
            res_mrrs = [1.0 / rank for rank in res_ranks]
            records.append(
                {
                    "case_id": f"{direction}_{idx}",
                    "triple_index": idx,
                    "direction": direction,
                    "triple": [h, r, t],
                    "target_entity_id": target,
                    "target_has_img": target_has_img,
                    "head_has_img": bool(has_img[h].item()),
                    "tail_has_img": bool(has_img[t].item()),
                    "relation_group": relation_group_map.get(r, "ungrouped"),
                    "full_ranks": full_ranks,
                    "residual_ranks": res_ranks,
                    "full_mean_rank": safe_mean(full_ranks),
                    "residual_mean_rank": safe_mean(res_ranks),
                    "full_mean_mrr": safe_mean(full_mrrs),
                    "residual_mean_mrr": safe_mean(res_mrrs),
                    "delta_mean_rank": safe_mean(res_ranks) - safe_mean(full_ranks),
                    "delta_mean_mrr": safe_mean(full_mrrs) - safe_mean(res_mrrs),
                    "full_better_all_seeds": all(fr < rr for fr, rr in zip(full_ranks, res_ranks)),
                    "residual_better_all_seeds": all(rr < fr for fr, rr in zip(full_ranks, res_ranks)),
                }
            )
    return records


def select_cases(records: list[dict], topn: int) -> tuple[list[dict], list[dict]]:
    success_pool = [
        row for row in records
        if row["full_better_all_seeds"] and row["full_mean_rank"] <= 10 and row["residual_mean_rank"] >= 20
    ]
    failure_pool = [
        row for row in records
        if row["residual_better_all_seeds"] and row["residual_mean_rank"] <= 10 and row["full_mean_rank"] >= 20
    ]
    if len(success_pool) < topn:
        success_pool = [row for row in records if row["full_better_all_seeds"]]
    if len(failure_pool) < topn:
        failure_pool = [row for row in records if row["residual_better_all_seeds"]]
    success = sorted(success_pool, key=lambda row: (row["delta_mean_mrr"], row["delta_mean_rank"]), reverse=True)[:topn]
    failure = sorted(failure_pool, key=lambda row: (row["delta_mean_mrr"], row["delta_mean_rank"]))[:topn]
    return success, failure


def enrich_cases(
    cases: list[dict],
    representative_seed: int,
    full_model,
    residual_model,
    true_tails: dict,
    true_heads: dict,
    num_entities: int,
    device: str,
    entity_map: dict[str, str],
    relation_map: dict[str, str],
) -> list[dict]:
    del representative_seed
    out = []
    for row in cases:
        h, r, t = row["triple"]
        direction = row["direction"]
        target = row["target_entity_id"]
        behavior = compute_full_model_behavior(full_model, target, r, device)
        full_top = compute_filtered_topk(full_model, (h, r, t), direction, true_tails, true_heads, num_entities, device, topk=5)
        res_top = compute_filtered_topk(residual_model, (h, r, t), direction, true_tails, true_heads, num_entities, device, topk=5)
        for item in full_top:
            item["entity_text_zh"] = entity_map.get(item["entity_token"], "")
        for item in res_top:
            item["entity_text_zh"] = entity_map.get(item["entity_token"], "")
        enriched = dict(row)
        enriched.update(
            {
                "head_token": entity_id_to_token(h),
                "head_text_zh": entity_map.get(entity_id_to_token(h), ""),
                "relation_token": relation_id_to_token(r),
                "relation_text_zh": relation_map.get(relation_id_to_token(r), ""),
                "tail_token": entity_id_to_token(t),
                "tail_text_zh": entity_map.get(entity_id_to_token(t), ""),
                "target_entity_token": entity_id_to_token(target),
                "target_entity_text_zh": entity_map.get(entity_id_to_token(target), ""),
                "full_behavior": behavior,
                "cue_hint": cue_hint(direction, row["target_has_img"], row["relation_group"]),
                "full_top5_filtered": full_top,
                "residual_top5_filtered": res_top,
            }
        )
        out.append(enriched)
    return out


def render_case(case: dict, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"- Query: `({case['head_token']}, {case['relation_token']}, {case['tail_token']})`",
        f"- Text: `{case['head_text_zh']}` --`{case['relation_text_zh']}`--> `{case['tail_text_zh']}`",
        f"- Direction: `{case['direction']}` | Relation group: `{case['relation_group']}`",
        f"- Target: `{case['target_entity_token']}` / `{case['target_entity_text_zh']}` | `target_has_img={case['target_has_img']}`",
        f"- Cue hint: {case['cue_hint']}",
        f"- Full Model mean rank / MRR: `{case['full_mean_rank']:.2f}` / `{case['full_mean_mrr']:.4f}`",
        f"- Residual-only mean rank / MRR: `{case['residual_mean_rank']:.2f}` / `{case['residual_mean_mrr']:.4f}`",
        f"- Delta MRR (Full - Residual): `{case['delta_mean_mrr']:.4f}`",
        f"- Full gate mean: `{case['full_behavior']['gate_mean']:.4f}`",
        f"- Full effective residual / fused: `{case['full_behavior']['effective_residual_norm']:.4f}` / `{case['full_behavior']['effective_fused_norm']:.4f}`",
        f"- Full residual-to-fused ratio: `{case['full_behavior']['residual_to_fused_ratio']:.4f}`",
        "",
        "Full Model filtered top-5:",
    ]
    for item in case["full_top5_filtered"]:
        lines.append(f"- `{item['entity_token']}` / `{item['entity_text_zh']}` | score `{item['score']:.4f}`" + (" | GOLD" if item["is_gold"] else ""))
    lines.extend(["", "Residual-only filtered top-5:"])
    for item in case["residual_top5_filtered"]:
        lines.append(f"- `{item['entity_token']}` / `{item['entity_text_zh']}` | score `{item['score']:.4f}`" + (" | GOLD" if item["is_gold"] else ""))
    lines.append("")
    return lines


def render_markdown(summary: dict, outputs_root: Path, groups_json_path: Path) -> str:
    lines = [
        "# Case Analysis",
        "",
        "## 1. Purpose",
        "",
        "This document extracts concrete success and failure cases from the completed paper-stage models.",
        "",
        "Current focus:",
        "",
        "- success cases where `Full Model` consistently beats `Residual-only`",
        "- failure cases where `Residual-only` consistently beats `Full Model`",
        "- case-level links to image availability, relation group, gate behavior, and residual/fused balance",
        "",
        "## 2. Setup",
        "",
        f"- Outputs root: `{outputs_root.as_posix()}`",
        f"- Grouping source: `{groups_json_path.as_posix()}`",
        f"- Compared models: `{', '.join(summary['meta']['selected_models'])}`",
        f"- Common seeds: `{summary['meta']['common_seeds']}`",
        f"- Candidate query directions: `both`",
        f"- Selected success cases: `{len(summary['success_cases'])}`",
        f"- Selected failure cases: `{len(summary['failure_cases'])}`",
        "",
        "## 3. Success Cases",
        "",
    ]
    for idx, case in enumerate(summary["success_cases"], start=1):
        lines.extend(render_case(case, f"Success {idx}"))
    lines.extend(["## 4. Failure Cases", ""])
    for idx, case in enumerate(summary["failure_cases"], start=1):
        lines.extend(render_case(case, f"Failure {idx}"))
    lines.extend(
        [
            "## 5. Takeaways",
            "",
            "- Success cases should be read as sample-level evidence for when multimodal fusion adds useful signal beyond pure residual compensation.",
            "- Failure cases should be read as sample-level evidence for when structure compensation remains the safer and stronger route.",
            "- These cases complement the completed `6` and `7` stages by grounding the gain-boundary narrative in concrete triples.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="ml/artifacts/outputs")
    ap.add_argument("--groups-json", default="docs/relation_type_groups_draft.json")
    ap.add_argument("--output-md", default="docs/CASE_ANALYSIS.md")
    ap.add_argument("--output-json", default="docs/case_analysis.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--query-batch-size", type=int, default=8)
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    groups_json_path = Path(args.groups_json)
    groups_json = load_json(groups_json_path)
    source_files = groups_json.get("source_files", {})
    relation_map = load_tsv_map(Path(source_files["relation2text_zh"]))
    entity_map = load_tsv_map(Path("data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv"))
    _, relation_group_map = build_group_definitions(groups_json, relation_map)

    selected_runs, duplicates = select_latest_runs(outputs_root, set(DEFAULT_MODEL_SET))
    common_seeds = sorted(set(selected_runs["Full Model"].keys()) & set(selected_runs["Residual-only"].keys()))
    if not common_seeds:
        raise RuntimeError("No common seeds found between Full Model and Residual-only.")

    ref_cfg = load_json(selected_runs["Full Model"][common_seeds[0]] / "config_merged.json")
    train3, _, _ = read_allow_2or3(ref_cfg["dataset"]["train"])
    dev3, _, _ = read_allow_2or3(ref_cfg["dataset"]["dev"])
    test3, _, _ = read_allow_2or3(ref_cfg["dataset"]["test"])
    true_tails, true_heads = build_true_facts(train3 + dev3 + test3)
    true_tails_index = prepare_true_tails_index(true_tails)
    true_heads_index = prepare_true_heads_index(true_heads)

    per_seed_ranks: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(dict)
    representative_models = {}
    num_entities = None
    device_used = None
    for label in PRIMARY_ORDER:
        for seed in common_seeds:
            run_dir = selected_runs[label][seed]
            cfg = load_json(run_dir / "config_merged.json")
            device = resolve_device(args.device or cfg.get("system", {}).get("device", "cuda"))
            cfg["system"]["device"] = device
            model, num_entities_local = build_model(cfg)
            state = torch.load(run_dir / "best.ckpt", map_location=device)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            num_entities = num_entities_local
            device_used = device
            triples_t = torch.tensor(test3, dtype=torch.long)
            per_seed_ranks[label][seed] = {
                "head": compute_head_ranks(model, triples_t, true_heads_index, num_entities_local, args.chunk_size, args.query_batch_size, device),
                "tail": compute_tail_ranks(model, triples_t, true_tails_index, num_entities_local, args.chunk_size, args.query_batch_size, device),
            }
            if seed == common_seeds[0]:
                representative_models[label] = model

    has_img = representative_models["Full Model"].has_img.detach().cpu()
    records = build_case_records(test3, per_seed_ranks, relation_group_map, has_img)
    success_cases, failure_cases = select_cases(records, topn=max(1, int(args.topn)))
    success_cases = enrich_cases(success_cases, common_seeds[0], representative_models["Full Model"], representative_models["Residual-only"], true_tails_index, true_heads_index, num_entities, device_used, entity_map, relation_map)
    failure_cases = enrich_cases(failure_cases, common_seeds[0], representative_models["Full Model"], representative_models["Residual-only"], true_tails_index, true_heads_index, num_entities, device_used, entity_map, relation_map)

    summary = {
        "meta": {
            "outputs_root": outputs_root.as_posix(),
            "groups_json": groups_json_path.as_posix(),
            "selected_models": PRIMARY_ORDER,
            "common_seeds": common_seeds,
            "duplicates": duplicates,
            "chunk_size": int(args.chunk_size),
            "query_batch_size": int(args.query_batch_size),
            "topn": int(args.topn),
        },
        "success_cases": success_cases,
        "failure_cases": failure_cases,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary, outputs_root, groups_json_path), encoding="utf-8")
    print(f"[OK] wrote {output_md.as_posix()}")
    print(f"[OK] wrote {output_json.as_posix()}")


if __name__ == "__main__":
    main()
