import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.dataset_loader import load_dataset_bundle
from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from ml.training.src.eval.filtered_ranking import prepare_true_heads_index, prepare_true_tails_index
from ml.training.src.models.build_model import build_model
from ml.training.src.utils.config import load_config
from ml.training.src.utils.seed import set_seed
from router.io_utils import write_csv, write_json
from router.schemas import QUERY_EVAL_HEADER, QueryEvalRecord


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, help="gate_only | residual_only | full_model")
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="experiment yaml path")
    ap.add_argument("--common", default="ml/configs/common.yaml", help="common yaml path")
    ap.add_argument("--run-dir", help="run directory containing config_merged.json and best.ckpt")
    ap.add_argument("--ckpt", help="checkpoint path; defaults to best.ckpt under --run-dir")
    ap.add_argument("--seed", type=int, help="override/export seed")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--query-batch-size", type=int, default=None)
    ap.add_argument("--summary-json", default=None, help="optional sidecar summary json path")
    return ap.parse_args()


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


def infer_expert_name(cfg: dict, requested: str) -> str:
    model_name = cfg.get("model", {}).get("name", "")
    mapping = {
        "openbg_img_gate_only": "gate_only",
        "openbg_img_residual_only": "residual_only",
        "openbg_img_gated_vec_res_rel": "full_model",
        "mmkg_gate_only": "gate_only",
        "mmkg_residual_only": "residual_only",
        "mmkg_gate_residual": "full_model",
        "mmkg_gate_only_v2": "fusion_v2",
        "mmkg_structural_v2": "structural_v2",
    }
    return mapping.get(model_name, requested.lower())


def load_cfg_and_ckpt(args: argparse.Namespace) -> tuple[dict, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        cfg_path = run_dir / "config_merged.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config_merged.json under run dir: {run_dir}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "best.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        return cfg, ckpt_path

    if not args.config or not args.ckpt:
        raise ValueError("Either --run-dir, or both --config and --ckpt, are required.")
    cfg = load_config(args.config, args.common)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return cfg, ckpt_path


def relation_name(rel_id: int, id2relation: dict[int, str] | None = None) -> str:
    if id2relation is not None:
        return id2relation[rel_id]
    return f"rel_{rel_id:04d}"


def make_query_id(split: str, seed: int, direction: str, relation_id: int, head_id: int, tail_id: int, target_id: int) -> str:
    return f"{split}|{seed}|{direction}|r={relation_id}|h={head_id}|t={tail_id}|target={target_id}"


def target_regime(direction: str, target_has_img: bool) -> str:
    """Frozen OpenBG legacy target-regime definition."""
    if direction == "head":
        return "head_has_img" if target_has_img else "head_no_img"
    return "tail_no_img"


def general_target_regime(direction: str, target_has_text: bool, target_has_img: bool) -> str:
    return f"{direction}_T{int(target_has_text)}V{int(target_has_img)}"


def filtered_direction_details(
    model,
    triples: torch.LongTensor,
    direction: str,
    true_index: dict,
    num_entities: int,
    device: str,
    chunk_size: int,
    query_batch_size: int,
) -> list[dict]:
    device_t = torch.device(device)
    scorer_name = "score_head" if direction == "head" else "score_tail"
    scorer = getattr(model, scorer_name, None)
    if scorer is None:
        scorer = model.score
    all_entities = torch.arange(num_entities, dtype=torch.long)
    neg_inf = float("-inf")
    records: list[dict] = []

    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]
        bq = q.size(0)

        h = q[:, 0].to(device_t)
        r = q[:, 1].to(device_t)
        t = q[:, 2].to(device_t)
        h_cpu = q[:, 0]
        t_cpu = q[:, 2]
        target_e = h if direction == "head" else t

        target_scores = scorer(torch.stack([h, r, t], dim=1))
        target_scores_cpu = target_scores.detach().cpu()
        greater = torch.zeros(bq, dtype=torch.long, device=device_t)
        top_scores = torch.full((bq, 2), neg_inf, dtype=torch.float32, device=device_t)
        top_entities = torch.full((bq, 2), -1, dtype=torch.long, device=device_t)

        filt_excl = []
        for j in range(bq):
            if direction == "tail":
                key = (int(q[j, 0].item()), int(q[j, 1].item()))
                idx = true_index.get(key, torch.empty(0, dtype=torch.long))
                gold = int(t_cpu[j].item())
            else:
                key = (int(q[j, 1].item()), int(q[j, 2].item()))
                idx = true_index.get(key, torch.empty(0, dtype=torch.long))
                gold = int(h_cpu[j].item())
            if idx.numel() > 0:
                idx = idx[idx != gold]
            filt_excl.append(idx)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device_t)
            if direction == "tail":
                h_g = h.unsqueeze(1).expand(bq, c)
                r_g = r.unsqueeze(1).expand(bq, c)
                e_g = cand.unsqueeze(0).expand(bq, c)
                batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), e_g.reshape(-1)], dim=1)
            else:
                e_g = cand.unsqueeze(0).expand(bq, c)
                r_g = r.unsqueeze(1).expand(bq, c)
                t_g = t.unsqueeze(1).expand(bq, c)
                batch = torch.stack([e_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = scorer(batch).view(bq, c)
            cand_matrix = cand.unsqueeze(0).expand(bq, c)

            row_chunks = []
            col_chunks = []
            for j in range(bq):
                idx = filt_excl[j]
                if idx.numel() == 0:
                    continue
                left = int(torch.searchsorted(idx, start, right=False).item())
                right = int(torch.searchsorted(idx, end, right=False).item())
                local = idx[left:right]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)

            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device_t)
                cols = torch.cat(col_chunks, dim=0).to(device_t)
                scores[rows, cols] = neg_inf

            greater += (scores > target_scores.unsqueeze(1)).sum(dim=1)

            combined_scores = torch.cat([top_scores, scores], dim=1)
            combined_entities = torch.cat([top_entities, cand_matrix], dim=1)
            new_top_scores, new_top_idx = torch.topk(combined_scores, k=2, dim=1)
            new_top_entities = torch.gather(combined_entities, 1, new_top_idx)
            top_scores = new_top_scores
            top_entities = new_top_entities

        ranks = (greater + 1).detach().cpu()
        top_scores_cpu = top_scores.detach().cpu()
        top_entities_cpu = top_entities.detach().cpu()
        target_cpu = target_e.detach().cpu()
        r_cpu = q[:, 1].detach().cpu()

        for j in range(bq):
            records.append(
                {
                    "direction": direction,
                    "relation_id": int(r_cpu[j].item()),
                    "head_id": int(h_cpu[j].item()),
                    "tail_id": int(t_cpu[j].item()),
                    "target_entity_id": int(target_cpu[j].item()),
                    "rank": int(ranks[j].item()),
                    "correct_score": float(target_scores_cpu[j].item()),
                    "top1_score": float(top_scores_cpu[j, 0].item()),
                    "top2_score": float(top_scores_cpu[j, 1].item()),
                    "top1_entity_id": int(top_entities_cpu[j, 0].item()),
                    "top2_entity_id": int(top_entities_cpu[j, 1].item()),
                }
            )
    return records


@torch.inference_mode()
def export_query_eval(cfg: dict, ckpt_path: Path, expert_name: str, split: str, out_path: str, seed: int, device: str, chunk_size: int, query_batch_size: int, summary_json: str | None) -> None:
    dataset_bundle = load_dataset_bundle(cfg)
    triples3 = dataset_bundle.valid_triples if split == "dev" else dataset_bundle.test_triples
    if not triples3:
        raise RuntimeError(f"No labeled 3-column triples found for split={split}")

    train3 = dataset_bundle.train_triples
    dev3 = dataset_bundle.valid_triples
    test3 = dataset_bundle.test_triples
    true_tails, true_heads = build_true_facts(train3 + dev3 + test3)
    true_tails_idx = prepare_true_tails_index(true_tails)
    true_heads_idx = prepare_true_heads_index(true_heads)

    model, num_entities = build_model(cfg, dataset_bundle=dataset_bundle)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    triples_t = torch.tensor(triples3, dtype=torch.long)
    tail_rows = filtered_direction_details(
        model=model,
        triples=triples_t,
        direction="tail",
        true_index=true_tails_idx,
        num_entities=num_entities,
        device=device,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
    )
    head_rows = filtered_direction_details(
        model=model,
        triples=triples_t,
        direction="head",
        true_index=true_heads_idx,
        num_entities=num_entities,
        device=device,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
    )

    has_img = getattr(model, "has_img", None)
    if has_img is None:
        if dataset_bundle.protocol_version != MMKG_GENERAL_V1:
            raise RuntimeError("Legacy model does not expose the canonical has_img mask.")
        # A pure general-v2 structural expert must not register modality masks;
        # target masks are dataset metadata used only for post-hoc reporting.
        has_img = dataset_bundle.features.has_img
    has_img_cpu = has_img.detach().cpu()
    protocol_version = dataset_bundle.protocol_version
    has_text_cpu = dataset_bundle.features.has_text.detach().cpu()
    id2relation = {relation_id: token for token, relation_id in dataset_bundle.relation2id.items()}

    rows: list[dict] = []
    for raw in tail_rows + head_rows:
        target_has_img = bool(has_img_cpu[raw["target_entity_id"]].item())
        target_has_text = bool(has_text_cpu[raw["target_entity_id"]].item())
        rank = raw["rank"]
        top1 = raw["top1_score"]
        top2 = raw["top2_score"]
        regime = (
            target_regime(raw["direction"], target_has_img)
            if protocol_version == OPENBG_LEGACY_V1
            else general_target_regime(raw["direction"], target_has_text, target_has_img)
        )
        row = QueryEvalRecord(
                query_id=make_query_id(
                    split=split,
                    seed=seed,
                    direction=raw["direction"],
                    relation_id=raw["relation_id"],
                    head_id=raw["head_id"],
                    tail_id=raw["tail_id"],
                    target_id=raw["target_entity_id"],
                ),
                split=split,
                direction=raw["direction"],
                relation_id=raw["relation_id"],
                relation_name=relation_name(raw["relation_id"], id2relation),
                head_id=raw["head_id"],
                tail_id=raw["tail_id"],
                target_entity_id=raw["target_entity_id"],
                target_position=raw["direction"],
                target_has_img=int(target_has_img),
                target_regime=regime,
                expert_name=expert_name,
                rank=rank,
                rr=float(1.0 / rank),
                hit1=int(rank <= 1),
                hit3=int(rank <= 3),
                hit10=int(rank <= 10),
                top1_score=top1,
                top2_score=top2,
                score_margin=float(top1 - top2),
                correct_score=raw["correct_score"],
                seed=seed,
            ).to_dict()
        if protocol_version == MMKG_GENERAL_V1:
            row["dataset"] = dataset_bundle.name
            row["protocol_version"] = protocol_version
            row["target_has_text"] = int(target_has_text)
            row["target_modality_count"] = int(target_has_text) + int(target_has_img)
        rows.append(row)

    rows.sort(key=lambda item: (item["direction"], item["relation_id"], item["head_id"], item["tail_id"], item["target_entity_id"]))
    header = list(QUERY_EVAL_HEADER)
    if protocol_version == MMKG_GENERAL_V1:
        header = ["dataset", "protocol_version", *header]
        target_img_index = header.index("target_has_img")
        header.insert(target_img_index, "target_has_text")
        header.insert(target_img_index + 2, "target_modality_count")
    write_csv(out_path, rows, header)

    summary = {
        "expert_name": expert_name,
        "split": split,
        "seed": seed,
        "dataset": dataset_bundle.name,
        "protocol_version": protocol_version,
        "n_rows": len(rows),
        "chunk_size": chunk_size,
        "query_batch_size": query_batch_size,
        "direction_counts": {
            "head": sum(1 for row in rows if row["direction"] == "head"),
            "tail": sum(1 for row in rows if row["direction"] == "tail"),
        },
        "target_regime_counts": {
            regime: sum(1 for row in rows if row["target_regime"] == regime)
            for regime in sorted({row["target_regime"] for row in rows})
        },
        "ckpt_path": str(ckpt_path),
        "out_path": str(out_path),
    }
    if summary_json:
        write_json(summary_json, summary)

    print(f"[OK] wrote query eval -> {out_path}")
    if summary_json:
        print(f"[OK] wrote summary    -> {summary_json}")


def main() -> None:
    args = parse_args()
    cfg, ckpt_path = load_cfg_and_ckpt(args)
    seed = int(args.seed if args.seed is not None else cfg.get("system", {}).get("seed", 1))
    set_seed(seed, deterministic=bool(cfg.get("system", {}).get("deterministic", False)))
    device = resolve_device(args.device or cfg.get("system", {}).get("device", "cuda"))
    cfg.setdefault("system", {})["device"] = device
    ev_cfg = cfg.get("evaluation", {})
    chunk_size = int(args.chunk_size or ev_cfg.get("chunk_size", 4096))
    query_batch_size = int(args.query_batch_size or ev_cfg.get("query_batch_size", 8))
    expert_name = infer_expert_name(cfg, args.expert)

    export_query_eval(
        cfg=cfg,
        ckpt_path=ckpt_path,
        expert_name=expert_name,
        split=args.split,
        out_path=args.out,
        seed=seed,
        device=device,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    main()
