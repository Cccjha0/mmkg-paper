from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.tsv_reader import read_allow_2or3
from ml.training.src.eval.filtered_ranking import prepare_true_heads_index, prepare_true_tails_index
from ml.training.src.models.build_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export candidate-level Gate-only and Residual-only scores for candidate-aware router experiments."
    )
    parser.add_argument("--gate-run-dir", required=True)
    parser.add_argument("--residual-run-dir", required=True)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--direction", default="both", choices=["head", "tail", "both"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--include-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None, help="cuda | cpu | mps | auto; defaults to run config")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--out-parquet", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def resolve_device(requested: str | None, fallback: str | None = None) -> str:
    requested = (requested or fallback or "cuda").lower()
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


def load_run(run_dir: str | Path, device: str):
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_merged.json"
    ckpt_path = run_dir / "best.ckpt"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("system", {})["device"] = device
    model, num_entities = build_model(cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return cfg, model, num_entities


def make_query_id(split: str, seed: int, direction: str, relation_id: int, head_id: int, tail_id: int, target_id: int) -> str:
    return f"{split}|{seed}|{direction}|r={relation_id}|h={head_id}|t={tail_id}|target={target_id}"


def target_regime(direction: str, target_has_img: bool) -> str:
    if direction == "head":
        return "head_has_img" if target_has_img else "head_no_img"
    return "tail_no_img"


def build_filtered_indexes(cfg: dict):
    train3, _, bad_train = read_allow_2or3(cfg["dataset"]["train"])
    dev3, _, bad_dev = read_allow_2or3(cfg["dataset"]["dev"])
    test3, _, bad_test = read_allow_2or3(cfg["dataset"]["test"])
    if bad_train or bad_dev or bad_test:
        print(f"[WARN] malformed lines skipped: train={bad_train}, dev={bad_dev}, test={bad_test}")
    true_tails, true_heads = build_true_facts(train3 + dev3 + test3)
    return prepare_true_tails_index(true_tails), prepare_true_heads_index(true_heads)


def load_split_triples(cfg: dict, split: str) -> list[tuple[int, int, int]]:
    triples3, _, bad = read_allow_2or3(cfg["dataset"][split])
    if bad:
        print(f"[WARN] malformed {split} lines skipped: {bad}")
    if not triples3:
        raise RuntimeError(f"No labeled triples found for split={split}")
    return triples3


def filter_scores_(
    scores: torch.Tensor,
    q_cpu: torch.Tensor,
    start: int,
    direction: str,
    true_index: dict,
) -> None:
    row_chunks = []
    col_chunks = []
    end = start + scores.size(1)
    for j in range(q_cpu.size(0)):
        if direction == "tail":
            key = (int(q_cpu[j, 0].item()), int(q_cpu[j, 1].item()))
            gold = int(q_cpu[j, 2].item())
        else:
            key = (int(q_cpu[j, 1].item()), int(q_cpu[j, 2].item()))
            gold = int(q_cpu[j, 0].item())
        idx = true_index.get(key, torch.empty(0, dtype=torch.long))
        if idx.numel() > 0:
            idx = idx[idx != gold]
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
        rows = torch.cat(row_chunks, dim=0).to(scores.device)
        cols = torch.cat(col_chunks, dim=0).to(scores.device)
        scores[rows, cols] = float("-inf")


@torch.inference_mode()
def score_candidates(model, q_cpu: torch.Tensor, candidate_ids_cpu: torch.Tensor, direction: str, device: str) -> torch.Tensor:
    q = q_cpu.to(device)
    cand = candidate_ids_cpu.to(device)
    bq, c = cand.shape
    h = q[:, 0]
    r = q[:, 1]
    t = q[:, 2]
    if direction == "tail":
        h_g = h.unsqueeze(1).expand(bq, c)
        r_g = r.unsqueeze(1).expand(bq, c)
        batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), cand.reshape(-1)], dim=1)
    else:
        r_g = r.unsqueeze(1).expand(bq, c)
        t_g = t.unsqueeze(1).expand(bq, c)
        batch = torch.stack([cand.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)
    return model.score(batch).view(bq, c).detach().cpu()


@torch.inference_mode()
def export_direction(
    *,
    gate_model,
    residual_model,
    triples: list[tuple[int, int, int]],
    split: str,
    seed: int,
    direction: str,
    true_index: dict,
    num_entities: int,
    top_k: int,
    include_target: bool,
    device: str,
    chunk_size: int,
    query_batch_size: int,
    has_img: torch.Tensor,
) -> list[dict]:
    triples_t = torch.tensor(triples, dtype=torch.long)
    all_entities = torch.arange(num_entities, dtype=torch.long)
    rows: list[dict] = []

    for q_start in range(0, triples_t.size(0), query_batch_size):
        q_end = min(triples_t.size(0), q_start + query_batch_size)
        q_cpu = triples_t[q_start:q_end]
        q = q_cpu.to(device)
        bq = q.size(0)

        top_gate_scores = torch.full((bq, top_k), float("-inf"), dtype=torch.float32, device=device)
        top_gate_entities = torch.full((bq, top_k), -1, dtype=torch.long, device=device)
        top_res_scores = torch.full((bq, top_k), float("-inf"), dtype=torch.float32, device=device)
        top_res_entities = torch.full((bq, top_k), -1, dtype=torch.long, device=device)

        h = q[:, 0]
        r = q[:, 1]
        t = q[:, 2]

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            cand = all_entities[start:end].to(device)
            c = cand.numel()
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

            gate_scores = gate_model.score(batch).view(bq, c)
            res_scores = residual_model.score(batch).view(bq, c)
            filter_scores_(gate_scores, q_cpu, start, direction, true_index)
            filter_scores_(res_scores, q_cpu, start, direction, true_index)

            cand_matrix = cand.unsqueeze(0).expand(bq, c)
            g_scores = torch.cat([top_gate_scores, gate_scores], dim=1)
            g_entities = torch.cat([top_gate_entities, cand_matrix], dim=1)
            top_gate_scores, idx = torch.topk(g_scores, k=top_k, dim=1)
            top_gate_entities = torch.gather(g_entities, 1, idx)

            r_scores = torch.cat([top_res_scores, res_scores], dim=1)
            r_entities = torch.cat([top_res_entities, cand_matrix], dim=1)
            top_res_scores, idx = torch.topk(r_scores, k=top_k, dim=1)
            top_res_entities = torch.gather(r_entities, 1, idx)

        top_gate_entities_cpu = top_gate_entities.detach().cpu()
        top_res_entities_cpu = top_res_entities.detach().cpu()

        for j in range(bq):
            h_id = int(q_cpu[j, 0].item())
            r_id = int(q_cpu[j, 1].item())
            t_id = int(q_cpu[j, 2].item())
            target_id = h_id if direction == "head" else t_id
            ordered_candidates: list[int] = []
            for eid in top_gate_entities_cpu[j].tolist() + top_res_entities_cpu[j].tolist():
                if eid >= 0 and eid not in ordered_candidates:
                    ordered_candidates.append(int(eid))
            if include_target and target_id not in ordered_candidates:
                ordered_candidates.append(target_id)

            candidate_tensor = torch.tensor(ordered_candidates, dtype=torch.long).unsqueeze(0)
            q_one = q_cpu[j : j + 1]
            gate_candidate_scores = score_candidates(gate_model, q_one, candidate_tensor, direction, device)[0]
            residual_candidate_scores = score_candidates(residual_model, q_one, candidate_tensor, direction, device)[0]

            gate_rank = {int(e): idx + 1 for idx, e in enumerate(top_gate_entities_cpu[j].tolist())}
            residual_rank = {int(e): idx + 1 for idx, e in enumerate(top_res_entities_cpu[j].tolist())}
            target_has_img = bool(has_img[target_id].item())
            query_id = make_query_id(split, seed, direction, r_id, h_id, t_id, target_id)

            for local_idx, candidate_id in enumerate(ordered_candidates):
                score_gate = float(gate_candidate_scores[local_idx].item())
                score_residual = float(residual_candidate_scores[local_idx].item())
                rows.append(
                    {
                        "query_id": query_id,
                        "seed": seed,
                        "split": split,
                        "direction": direction,
                        "relation_id": r_id,
                        "head_id": h_id,
                        "tail_id": t_id,
                        "observed_entity_id": t_id if direction == "head" else h_id,
                        "target_entity_id": target_id,
                        "candidate_entity_id": candidate_id,
                        "is_target": int(candidate_id == target_id),
                        "candidate_rank_gate": gate_rank.get(candidate_id),
                        "candidate_rank_residual": residual_rank.get(candidate_id),
                        "score_gate": score_gate,
                        "score_residual": score_residual,
                        "score_diff": score_gate - score_residual,
                        "score_mean": 0.5 * (score_gate + score_residual),
                        "score_max": max(score_gate, score_residual),
                        "in_gate_topk": int(candidate_id in gate_rank),
                        "in_residual_topk": int(candidate_id in residual_rank),
                        "target_regime": target_regime(direction, target_has_img),
                    }
                )
    return rows


def write_outputs(rows: list[dict], out_parquet: str | None, out_csv: str | None) -> None:
    if not out_parquet and not out_csv:
        raise ValueError("At least one of --out-parquet or --out-csv is required.")
    if out_parquet:
        import pandas as pd

        out_path = Path(out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for col in ("candidate_rank_gate", "candidate_rank_residual"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
        df.to_parquet(out_path, index=False)
        print(f"[OK] wrote parquet -> {out_path}")
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] wrote csv     -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    gate_run_dir = Path(args.gate_run_dir)
    gate_cfg = json.loads((gate_run_dir / "config_merged.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device, gate_cfg.get("system", {}).get("device", "cuda"))
    gate_cfg, gate_model, gate_num_entities = load_run(args.gate_run_dir, device)
    residual_cfg, residual_model, residual_num_entities = load_run(args.residual_run_dir, device)
    if gate_num_entities != residual_num_entities:
        raise RuntimeError(f"num_entities mismatch: gate={gate_num_entities}, residual={residual_num_entities}")

    seed = int(gate_cfg.get("system", {}).get("seed", 1))
    residual_seed = int(residual_cfg.get("system", {}).get("seed", seed))
    if residual_seed != seed:
        raise RuntimeError(f"seed mismatch: gate={seed}, residual={residual_seed}")

    triples = load_split_triples(gate_cfg, args.split)
    true_tails_idx, true_heads_idx = build_filtered_indexes(gate_cfg)
    ev_cfg = gate_cfg.get("evaluation", {})
    chunk_size = int(args.chunk_size or ev_cfg.get("chunk_size", 4096))
    query_batch_size = int(args.query_batch_size or ev_cfg.get("query_batch_size", 8))
    has_img = getattr(gate_model, "has_img", None)
    if has_img is None:
        raise RuntimeError("Gate model does not expose has_img.")
    has_img = has_img.detach().cpu().to(dtype=torch.bool)

    directions = ["head", "tail"] if args.direction == "both" else [args.direction]
    rows: list[dict] = []
    for direction in directions:
        true_index = true_heads_idx if direction == "head" else true_tails_idx
        rows.extend(
            export_direction(
                gate_model=gate_model,
                residual_model=residual_model,
                triples=triples,
                split=args.split,
                seed=seed,
                direction=direction,
                true_index=true_index,
                num_entities=gate_num_entities,
                top_k=args.top_k,
                include_target=args.include_target,
                device=device,
                chunk_size=chunk_size,
                query_batch_size=query_batch_size,
                has_img=has_img,
            )
        )

    rows.sort(key=lambda row: (row["query_id"], int(row["candidate_entity_id"])))
    write_outputs(rows, args.out_parquet, args.out_csv)

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        query_ids = {row["query_id"] for row in rows}
        summary = {
            "split": args.split,
            "seed": seed,
            "direction": args.direction,
            "top_k": args.top_k,
            "include_target": args.include_target,
            "n_rows": len(rows),
            "n_queries": len(query_ids),
            "avg_candidates_per_query": len(rows) / len(query_ids) if query_ids else 0.0,
            "gate_run_dir": str(args.gate_run_dir),
            "residual_run_dir": str(args.residual_run_dir),
            "device": device,
        }
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
