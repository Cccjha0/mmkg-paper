from __future__ import annotations

import torch


_EMPTY_LONG_CPU = torch.empty(0, dtype=torch.long)
# A dense bool mask uses one byte per query/entity pair. OpenBG-IMG uses less
# than 1 MiB at query_batch_size=32; retain the sparse path for larger datasets.
_DENSE_FILTER_MASK_MAX_BYTES = 256 * 1024 * 1024


def _score_tail(model, triples: torch.LongTensor) -> torch.Tensor:
    """Score tail-prediction triples, with a legacy ``score`` fallback."""
    score_tail = getattr(model, "score_tail", None)
    if score_tail is not None:
        return score_tail(triples)
    return model.score(triples)


def _score_head(model, triples: torch.LongTensor) -> torch.Tensor:
    """Score head-prediction triples, with a legacy ``score`` fallback."""
    score_head = getattr(model, "score_head", None)
    if score_head is not None:
        return score_head(triples)
    return model.score(triples)


def _metrics_from_ranks(ranks: torch.Tensor, ks=(1, 3, 10)) -> dict:
    ranks = ranks.to(dtype=torch.long).detach().cpu()
    count = int(ranks.numel())
    if count == 0:
        out = {"mrr": 0.0}
        for k in ks:
            out[f"hits@{k}"] = 0.0
        out["count"] = 0
        return out

    ranks_f = ranks.to(dtype=torch.float32)
    out = {
        "mrr": float((1.0 / ranks_f).mean().item()),
        "count": count,
    }
    for k in ks:
        out[f"hits@{k}"] = float((ranks <= k).float().mean().item())
    return out


def _split_metrics_from_ranks(ranks: torch.Tensor, mask: torch.Tensor, prefix: str, ks=(1, 3, 10)) -> dict:
    ranks = ranks.detach().cpu()
    mask = mask.detach().cpu().to(dtype=torch.bool)
    pos = _metrics_from_ranks(ranks[mask], ks=ks)
    neg = _metrics_from_ranks(ranks[~mask], ks=ks)

    out = {
        f"{prefix}_has_img_count": pos["count"],
        f"{prefix}_no_img_count": neg["count"],
    }
    for key, value in pos.items():
        if key == "count":
            continue
        out[f"{prefix}_has_img_{key}"] = value
    for key, value in neg.items():
        if key == "count":
            continue
        out[f"{prefix}_no_img_{key}"] = value
    return out


def _modality_metrics_from_ranks(
    ranks: torch.Tensor,
    has_text: torch.Tensor,
    has_img: torch.Tensor,
    prefix: str,
    ks=(1, 3, 10),
) -> dict:
    """Return target-side T/V availability subgroups for the general protocol."""
    ranks = ranks.detach().cpu()
    has_text = has_text.detach().cpu().bool()
    has_img = has_img.detach().cpu().bool()
    out: dict[str, float | int] = {}
    for text_flag in (0, 1):
        for img_flag in (0, 1):
            mask = (has_text == bool(text_flag)) & (has_img == bool(img_flag))
            metrics = _metrics_from_ranks(ranks[mask], ks=ks)
            tag = f"{prefix}_T{text_flag}V{img_flag}"
            out[f"{tag}_count"] = metrics.pop("count")
            for key, value in metrics.items():
                out[f"{tag}_{key}"] = value
    return out


def _prepare_index(mapping: dict) -> dict:
    """
    Convert filtering map to sorted CPU LongTensor values once.
    Input:
      mapping[key] -> set/list/tensor of true entity ids
    Output:
      dict[key] -> sorted LongTensor on CPU
    """
    out = {}
    for k, v in mapping.items():
        if isinstance(v, torch.Tensor):
            vt = v.detach().cpu().to(dtype=torch.long)
            if vt.numel() > 1:
                vt = torch.sort(vt).values
        else:
            vt = torch.tensor(sorted(v), dtype=torch.long) if len(v) > 0 else torch.empty(0, dtype=torch.long)
        out[k] = vt
    return out


def prepare_true_tails_index(true_tails: dict) -> dict:
    return _prepare_index(true_tails)


def prepare_true_heads_index(true_heads: dict) -> dict:
    return _prepare_index(true_heads)


def _prepare_eval_tensors(
        triples: torch.LongTensor,
        num_entities: int,
        entity_has_img: torch.Tensor | None,
        entity_has_text: torch.Tensor | None,
        device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.Tensor | None, torch.Tensor | None]:
    """Move immutable evaluation tensors once for both ranking directions."""
    triples_cpu = triples.detach().cpu()
    triples_gpu = triples_cpu.to(device)
    all_entities = torch.arange(num_entities, dtype=torch.long, device=device)
    entity_has_img_cpu = entity_has_img.detach().cpu() if entity_has_img is not None else None
    entity_has_text_cpu = entity_has_text.detach().cpu() if entity_has_text is not None else None
    return triples_cpu, triples_gpu, all_entities, entity_has_img_cpu, entity_has_text_cpu


def _build_dense_filter_mask(
        filt_excl_list: list[torch.LongTensor],
        target_entities: torch.LongTensor,
        num_entities: int,
        device: torch.device,
) -> torch.Tensor | None:
    """Build one exact filtered-entity mask for a complete query batch.

    Returns ``None`` when the dense mask would exceed the fixed memory budget;
    callers then use the semantically identical sparse chunk fallback.
    """
    batch_size = len(filt_excl_list)
    if batch_size * num_entities > _DENSE_FILTER_MASK_MAX_BYTES:
        return None

    filter_mask = torch.zeros((batch_size, num_entities), dtype=torch.bool, device=device)
    row_chunks = []
    col_chunks = []
    for row, filt_idx in enumerate(filt_excl_list):
        if filt_idx.numel() == 0:
            continue
        row_chunks.append(torch.full((filt_idx.numel(),), row, dtype=torch.long))
        col_chunks.append(filt_idx)
    if row_chunks:
        rows = torch.cat(row_chunks, dim=0).to(device)
        cols = torch.cat(col_chunks, dim=0).to(device)
        filter_mask[rows, cols] = True

    # The target fact must remain a valid candidate even though it is present
    # in the all-split true-fact index.
    target_rows = torch.arange(batch_size, dtype=torch.long, device=device)
    filter_mask[target_rows, target_entities] = False
    return filter_mask


def _apply_filter_mask(
        scores: torch.Tensor,
        *,
        dense_filter_mask: torch.Tensor | None,
        filt_excl_list: list[torch.LongTensor],
        start: int,
        end: int,
        device: torch.device,
) -> None:
    """Mask one candidate chunk using the dense path or exact sparse fallback."""
    if dense_filter_mask is not None:
        scores.masked_fill_(dense_filter_mask[:, start:end], float("-inf"))
        return

    row_chunks = []
    col_chunks = []
    for row, filt_idx in enumerate(filt_excl_list):
        if filt_idx.numel() == 0:
            continue
        left = int(torch.searchsorted(filt_idx, start, right=False).item())
        right = int(torch.searchsorted(filt_idx, end, right=False).item())
        local = filt_idx[left:right]
        if local.numel() == 0:
            continue
        row_chunks.append(torch.full((local.numel(),), row, dtype=torch.long))
        col_chunks.append(local - start)
    if row_chunks:
        rows = torch.cat(row_chunks, dim=0).to(device)
        cols = torch.cat(col_chunks, dim=0).to(device)
        scores[rows, cols] = float("-inf")


@torch.inference_mode()
def _filtered_tail_ranking_eval(
        model,
        triples: torch.LongTensor,
        true_tails: dict,
        num_entities: int,
        chunk_size: int = 10000,
        query_batch_size: int = 1,
        device: str = "cuda",
        ks=(1, 3, 10),
        entity_has_img: torch.Tensor | None = None,
        entity_has_text: torch.Tensor | None = None,
        _prepared_tensors=None,
):
    model.eval()
    device = torch.device(device)

    all_ranks = []
    all_target_has_img = []
    all_target_has_text = []

    if _prepared_tensors is None:
        _prepared_tensors = _prepare_eval_tensors(triples, num_entities, entity_has_img, entity_has_text, device)
    triples_cpu, triples_gpu, all_entities, entity_has_img_cpu, entity_has_text_cpu = _prepared_tensors

    if len(true_tails) > 0 and isinstance(next(iter(true_tails.values())), torch.Tensor):
        true_tails_t = true_tails
    else:
        true_tails_t = prepare_true_tails_index(true_tails)

    n = triples_cpu.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q_cpu = triples_cpu[q_start:q_end]
        q_gpu = triples_gpu[q_start:q_end]
        bq = q_cpu.size(0)

        h = q_gpu[:, 0]
        r = q_gpu[:, 1]
        t = q_gpu[:, 2]
        t_cpu = q_cpu[:, 2]

        target_scores = _score_tail(model, torch.stack([h, r, t], dim=1))
        target = target_scores.unsqueeze(1)

        filt_excl_list = []
        for j in range(bq):
            key = (int(q_cpu[j, 0].item()), int(q_cpu[j, 1].item()))
            filt_idx = true_tails_t.get(key, _EMPTY_LONG_CPU)
            if filt_idx.numel() > 0:
                filt_idx = filt_idx[filt_idx != int(t_cpu[j].item())]
            filt_excl_list.append(filt_idx)
        dense_filter_mask = _build_dense_filter_mask(
            filt_excl_list,
            target_entities=t,
            num_entities=num_entities,
            device=device,
        )

        greater = torch.zeros(bq, device=device, dtype=torch.long)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end]

            h_g = h.unsqueeze(1).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = cand.unsqueeze(0).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = _score_tail(model, batch).view(bq, c)

            _apply_filter_mask(
                scores,
                dense_filter_mask=dense_filter_mask,
                filt_excl_list=filt_excl_list,
                start=start,
                end=end,
                device=device,
            )

            greater += (scores > target).sum(dim=1)

        rank_tail = greater + 1

        all_ranks.append(rank_tail.detach().cpu())
        if entity_has_img_cpu is not None:
            all_target_has_img.append(entity_has_img_cpu[t_cpu])
        if entity_has_text_cpu is not None:
            all_target_has_text.append(entity_has_text_cpu[t_cpu])

    rank_tail_all = torch.cat(all_ranks, dim=0) if all_ranks else torch.empty(0, dtype=torch.long)
    out = _metrics_from_ranks(rank_tail_all, ks=ks)
    if entity_has_img is not None and all_target_has_img:
        target_has_img_all = torch.cat(all_target_has_img, dim=0)
        out.update(_split_metrics_from_ranks(rank_tail_all, target_has_img_all, prefix="tail", ks=ks))
    if entity_has_text is not None and entity_has_img is not None and all_target_has_text and all_target_has_img:
        out.update(
            _modality_metrics_from_ranks(
                rank_tail_all,
                torch.cat(all_target_has_text, dim=0),
                torch.cat(all_target_has_img, dim=0),
                prefix="tail",
                ks=ks,
            )
        )
    return out


@torch.inference_mode()
def _filtered_head_ranking_eval(
        model,
        triples: torch.LongTensor,
        true_heads: dict,
        num_entities: int,
        chunk_size: int = 10000,
        query_batch_size: int = 1,
        device: str = "cuda",
        ks=(1, 3, 10),
        entity_has_img: torch.Tensor | None = None,
        entity_has_text: torch.Tensor | None = None,
        _prepared_tensors=None,
):
    model.eval()
    device = torch.device(device)

    all_ranks = []
    all_target_has_img = []
    all_target_has_text = []

    if _prepared_tensors is None:
        _prepared_tensors = _prepare_eval_tensors(triples, num_entities, entity_has_img, entity_has_text, device)
    triples_cpu, triples_gpu, all_entities, entity_has_img_cpu, entity_has_text_cpu = _prepared_tensors

    if len(true_heads) > 0 and isinstance(next(iter(true_heads.values())), torch.Tensor):
        true_heads_t = true_heads
    else:
        true_heads_t = prepare_true_heads_index(true_heads)

    n = triples_cpu.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q_cpu = triples_cpu[q_start:q_end]
        q_gpu = triples_gpu[q_start:q_end]
        bq = q_cpu.size(0)

        h = q_gpu[:, 0]
        r = q_gpu[:, 1]
        t = q_gpu[:, 2]
        h_cpu = q_cpu[:, 0]

        target_scores = _score_head(model, torch.stack([h, r, t], dim=1))
        target = target_scores.unsqueeze(1)

        filt_excl_list = []
        for j in range(bq):
            key = (int(q_cpu[j, 1].item()), int(q_cpu[j, 2].item()))
            filt_idx = true_heads_t.get(key, _EMPTY_LONG_CPU)
            if filt_idx.numel() > 0:
                filt_idx = filt_idx[filt_idx != int(h_cpu[j].item())]
            filt_excl_list.append(filt_idx)
        dense_filter_mask = _build_dense_filter_mask(
            filt_excl_list,
            target_entities=h,
            num_entities=num_entities,
            device=device,
        )

        greater = torch.zeros(bq, device=device, dtype=torch.long)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end]

            h_g = cand.unsqueeze(0).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = t.unsqueeze(1).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = _score_head(model, batch).view(bq, c)

            _apply_filter_mask(
                scores,
                dense_filter_mask=dense_filter_mask,
                filt_excl_list=filt_excl_list,
                start=start,
                end=end,
                device=device,
            )

            greater += (scores > target).sum(dim=1)

        rank_head = greater + 1

        all_ranks.append(rank_head.detach().cpu())
        if entity_has_img_cpu is not None:
            all_target_has_img.append(entity_has_img_cpu[h_cpu])
        if entity_has_text_cpu is not None:
            all_target_has_text.append(entity_has_text_cpu[h_cpu])

    rank_head_all = torch.cat(all_ranks, dim=0) if all_ranks else torch.empty(0, dtype=torch.long)
    out = _metrics_from_ranks(rank_head_all, ks=ks)
    if entity_has_img is not None and all_target_has_img:
        target_has_img_all = torch.cat(all_target_has_img, dim=0)
        out.update(_split_metrics_from_ranks(rank_head_all, target_has_img_all, prefix="head", ks=ks))
    if entity_has_text is not None and entity_has_img is not None and all_target_has_text and all_target_has_img:
        out.update(
            _modality_metrics_from_ranks(
                rank_head_all,
                torch.cat(all_target_has_text, dim=0),
                torch.cat(all_target_has_img, dim=0),
                prefix="head",
                ks=ks,
            )
        )
    return out


@torch.inference_mode()
def filtered_ranking_eval(
        model,
        triples: torch.LongTensor,          # [N,3] on CPU
        true_tails: dict,
        true_heads: dict,
        num_entities: int,
        chunk_size: int = 10000,
        query_batch_size: int = 1,
        device: str = "cuda",
        ks=(1, 3, 10),
        direction: str = "both",
        entity_has_img: torch.Tensor | None = None,
        entity_has_text: torch.Tensor | None = None,
):
    """
    model.score(triples) -> scores (higher is better)
    Compute filtered ranks for tail/head prediction.
    direction:
      - "tail": evaluate (h, r, ?)
      - "head": evaluate (?, r, t)
      - "both": average head/tail metrics
    """
    direction = direction.lower()
    if direction not in {"tail", "head", "both"}:
        raise ValueError(f"Unsupported evaluation direction: {direction}")
    model.eval()
    prepare_eval_cache = getattr(model, "prepare_eval_cache", None)
    if prepare_eval_cache is not None:
        # Cache only deterministic clean entity representations. Rebuilding
        # here also prevents a stale cache after loading the best checkpoint.
        prepare_eval_cache()
    device_obj = torch.device(device)
    prepared_tensors = _prepare_eval_tensors(triples, num_entities, entity_has_img, entity_has_text, device_obj)
    if direction == "tail":
        return _filtered_tail_ranking_eval(
            model=model,
            triples=triples,
            true_tails=true_tails,
            num_entities=num_entities,
            chunk_size=chunk_size,
            query_batch_size=query_batch_size,
            device=device,
            ks=ks,
            entity_has_img=entity_has_img,
            entity_has_text=entity_has_text,
            _prepared_tensors=prepared_tensors,
        )
    if direction == "head":
        return _filtered_head_ranking_eval(
            model=model,
            triples=triples,
            true_heads=true_heads,
            num_entities=num_entities,
            chunk_size=chunk_size,
            query_batch_size=query_batch_size,
            device=device,
            ks=ks,
            entity_has_img=entity_has_img,
            entity_has_text=entity_has_text,
            _prepared_tensors=prepared_tensors,
        )
    tail_metrics = _filtered_tail_ranking_eval(
        model=model,
        triples=triples,
        true_tails=true_tails,
        num_entities=num_entities,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        device=device,
        ks=ks,
        entity_has_img=entity_has_img,
        entity_has_text=entity_has_text,
        _prepared_tensors=prepared_tensors,
    )
    head_metrics = _filtered_head_ranking_eval(
        model=model,
        triples=triples,
        true_heads=true_heads,
        num_entities=num_entities,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        device=device,
        ks=ks,
        entity_has_img=entity_has_img,
        entity_has_text=entity_has_text,
        _prepared_tensors=prepared_tensors,
    )

    out = {}
    for key in tail_metrics.keys():
        if key.startswith("tail_") or key.startswith("head_"):
            continue
        if key.endswith("_count"):
            continue
        out[key] = 0.5 * (tail_metrics[key] + head_metrics[key])
    out["tail_mrr"] = tail_metrics["mrr"]
    out["head_mrr"] = head_metrics["mrr"]
    general_modality_subgroups = entity_has_text is not None and entity_has_img is not None
    if entity_has_img is not None:
        out["tail_has_img_count"] = tail_metrics.get("tail_has_img_count", 0)
        out["tail_no_img_count"] = tail_metrics.get("tail_no_img_count", 0)
        out["head_has_img_count"] = head_metrics.get("head_has_img_count", 0)
        out["head_no_img_count"] = head_metrics.get("head_no_img_count", 0)
        if general_modality_subgroups:
            out["has_img_count"] = out["tail_has_img_count"] + out["head_has_img_count"]
            out["no_img_count"] = out["tail_no_img_count"] + out["head_no_img_count"]
        for suffix in ["mrr", "hits@1", "hits@3", "hits@10"]:
            tail_has_img = tail_metrics.get(f"tail_has_img_{suffix}", 0.0)
            head_has_img = head_metrics.get(f"head_has_img_{suffix}", 0.0)
            tail_no_img = tail_metrics.get(f"tail_no_img_{suffix}", 0.0)
            head_no_img = head_metrics.get(f"head_no_img_{suffix}", 0.0)
            if general_modality_subgroups:
                has_count = out["has_img_count"]
                no_count = out["no_img_count"]
                out[f"has_img_{suffix}"] = (
                    out["tail_has_img_count"] * tail_has_img + out["head_has_img_count"] * head_has_img
                ) / has_count if has_count else 0.0
                out[f"no_img_{suffix}"] = (
                    out["tail_no_img_count"] * tail_no_img + out["head_no_img_count"] * head_no_img
                ) / no_count if no_count else 0.0
                out[f"direction_balanced_has_img_{suffix}"] = 0.5 * (tail_has_img + head_has_img)
                out[f"direction_balanced_no_img_{suffix}"] = 0.5 * (tail_no_img + head_no_img)
            else:
                # Frozen OpenBG legacy semantics: these published diagnostics
                # remain the equal head/tail average used by existing results.
                out[f"has_img_{suffix}"] = 0.5 * (tail_has_img + head_has_img)
                out[f"no_img_{suffix}"] = 0.5 * (tail_no_img + head_no_img)
            out[f"tail_has_img_{suffix}"] = tail_metrics.get(f"tail_has_img_{suffix}", 0.0)
            out[f"tail_no_img_{suffix}"] = tail_metrics.get(f"tail_no_img_{suffix}", 0.0)
            out[f"head_has_img_{suffix}"] = head_metrics.get(f"head_has_img_{suffix}", 0.0)
            out[f"head_no_img_{suffix}"] = head_metrics.get(f"head_no_img_{suffix}", 0.0)
    if general_modality_subgroups:
        for text_flag in (0, 1):
            for img_flag in (0, 1):
                tag = f"T{text_flag}V{img_flag}"
                out[f"tail_{tag}_count"] = tail_metrics.get(f"tail_{tag}_count", 0)
                out[f"head_{tag}_count"] = head_metrics.get(f"head_{tag}_count", 0)
                out[f"{tag}_count"] = out[f"tail_{tag}_count"] + out[f"head_{tag}_count"]
                for suffix in ["mrr", "hits@1", "hits@3", "hits@10"]:
                    tail_value = tail_metrics.get(f"tail_{tag}_{suffix}", 0.0)
                    head_value = head_metrics.get(f"head_{tag}_{suffix}", 0.0)
                    out[f"tail_{tag}_{suffix}"] = tail_value
                    out[f"head_{tag}_{suffix}"] = head_value
                    count = out[f"{tag}_count"]
                    out[f"{tag}_{suffix}"] = (
                        out[f"tail_{tag}_count"] * tail_value + out[f"head_{tag}_count"] * head_value
                    ) / count if count else 0.0
                    out[f"direction_balanced_{tag}_{suffix}"] = 0.5 * (tail_value + head_value)
    return out
