import torch


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
):
    model.eval()
    device = torch.device(device)

    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    count = 0

    all_entities = torch.arange(num_entities, dtype=torch.long)
    neg_inf = float("-inf")

    if len(true_tails) > 0 and isinstance(next(iter(true_tails.values())), torch.Tensor):
        true_tails_t = true_tails
    else:
        true_tails_t = prepare_true_tails_index(true_tails)

    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]
        bq = q.size(0)

        h = q[:, 0].to(device)
        r = q[:, 1].to(device)
        t = q[:, 2].to(device)
        t_cpu = q[:, 2]

        target_scores = model.score(torch.stack([h, r, t], dim=1))
        target = target_scores.unsqueeze(1)

        filt_excl_list = []
        for j in range(bq):
            key = (int(q[j, 0].item()), int(q[j, 1].item()))
            filt_idx = true_tails_t.get(key, torch.empty(0, dtype=torch.long))
            if filt_idx.numel() > 0:
                filt_idx = filt_idx[filt_idx != int(t_cpu[j].item())]
            filt_excl_list.append(filt_idx)

        greater = torch.zeros(bq, device=device, dtype=torch.long)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device)

            h_g = h.unsqueeze(1).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = cand.unsqueeze(0).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = model.score(batch).view(bq, c)

            row_chunks = []
            col_chunks = []
            for j in range(bq):
                filt_idx = filt_excl_list[j]
                if filt_idx.numel() == 0:
                    continue
                l = int(torch.searchsorted(filt_idx, start, right=False).item())
                rr = int(torch.searchsorted(filt_idx, end, right=False).item())
                local = filt_idx[l:rr]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)

            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device)
                cols = torch.cat(col_chunks, dim=0).to(device)
                scores[rows, cols] = neg_inf

            greater += (scores > target).sum(dim=1)

        rank_tail = greater + 1

        mrr_sum += float((1.0 / rank_tail.float()).sum().item())
        for k in ks:
            hits[k] += int((rank_tail <= k).sum().item())
        count += bq

    mrr = mrr_sum / count
    out = {"mrr": mrr}
    for k in ks:
        out[f"hits@{k}"] = hits[k] / count
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
):
    model.eval()
    device = torch.device(device)

    mrr_sum = 0.0
    hits = {k: 0 for k in ks}
    count = 0

    all_entities = torch.arange(num_entities, dtype=torch.long)
    neg_inf = float("-inf")

    if len(true_heads) > 0 and isinstance(next(iter(true_heads.values())), torch.Tensor):
        true_heads_t = true_heads
    else:
        true_heads_t = prepare_true_heads_index(true_heads)

    n = triples.size(0)
    for q_start in range(0, n, query_batch_size):
        q_end = min(n, q_start + query_batch_size)
        q = triples[q_start:q_end]
        bq = q.size(0)

        h = q[:, 0].to(device)
        r = q[:, 1].to(device)
        t = q[:, 2].to(device)
        h_cpu = q[:, 0]

        target_scores = model.score(torch.stack([h, r, t], dim=1))
        target = target_scores.unsqueeze(1)

        filt_excl_list = []
        for j in range(bq):
            key = (int(q[j, 1].item()), int(q[j, 2].item()))
            filt_idx = true_heads_t.get(key, torch.empty(0, dtype=torch.long))
            if filt_idx.numel() > 0:
                filt_idx = filt_idx[filt_idx != int(h_cpu[j].item())]
            filt_excl_list.append(filt_idx)

        greater = torch.zeros(bq, device=device, dtype=torch.long)

        for start in range(0, num_entities, chunk_size):
            end = min(num_entities, start + chunk_size)
            c = end - start
            cand = all_entities[start:end].to(device)

            h_g = cand.unsqueeze(0).expand(bq, c)
            r_g = r.unsqueeze(1).expand(bq, c)
            t_g = t.unsqueeze(1).expand(bq, c)
            batch = torch.stack([h_g.reshape(-1), r_g.reshape(-1), t_g.reshape(-1)], dim=1)

            scores = model.score(batch).view(bq, c)

            row_chunks = []
            col_chunks = []
            for j in range(bq):
                filt_idx = filt_excl_list[j]
                if filt_idx.numel() == 0:
                    continue
                l = int(torch.searchsorted(filt_idx, start, right=False).item())
                rr = int(torch.searchsorted(filt_idx, end, right=False).item())
                local = filt_idx[l:rr]
                if local.numel() == 0:
                    continue
                row_chunks.append(torch.full((local.numel(),), j, dtype=torch.long))
                col_chunks.append(local - start)

            if row_chunks:
                rows = torch.cat(row_chunks, dim=0).to(device)
                cols = torch.cat(col_chunks, dim=0).to(device)
                scores[rows, cols] = neg_inf

            greater += (scores > target).sum(dim=1)

        rank_head = greater + 1

        mrr_sum += float((1.0 / rank_head.float()).sum().item())
        for k in ks:
            hits[k] += int((rank_head <= k).sum().item())
        count += bq

    mrr = mrr_sum / count
    out = {"mrr": mrr}
    for k in ks:
        out[f"hits@{k}"] = hits[k] / count
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
        )
    if direction != "both":
        raise ValueError(f"Unsupported evaluation direction: {direction}")

    tail_metrics = _filtered_tail_ranking_eval(
        model=model,
        triples=triples,
        true_tails=true_tails,
        num_entities=num_entities,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        device=device,
        ks=ks,
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
    )

    out = {}
    for key in tail_metrics.keys():
        out[key] = 0.5 * (tail_metrics[key] + head_metrics[key])
    out["tail_mrr"] = tail_metrics["mrr"]
    out["head_mrr"] = head_metrics["mrr"]
    return out
