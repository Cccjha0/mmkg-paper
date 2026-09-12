"""B10 rank diagnostics; evaluates frozen conventions, never changes weights."""
import torch

from ml.training.src.eval.filtered_ranking import _build_dense_filter_mask, _apply_filter_mask
from scripts.eval_heterogeneous_complementarity import (
    filtered_copy, query_zscore_with_reference, ranks_against_reference, endpoint_safe_mixed_ranks)


def shared_endpoint_ranks(raw, gold, queries, direction, facts, *, dense=True):
    """Use the shared training evaluator's gold-preserving filter/count path."""
    target = queries[:, 0 if direction=='head' else 2]
    exclusions=[]
    for query, gold_id in zip(queries, target):
        key = (int(query[1]),int(query[2])) if direction=='head' else (int(query[0]),int(query[1]))
        values=facts.get(key,torch.empty(0,dtype=torch.long))
        exclusions.append(values[values != gold_id])
    mask=_build_dense_filter_mask(exclusions,target,raw.shape[1],raw.device) if dense else None
    copied=raw.clone()
    _apply_filter_mask(copied,dense_filter_mask=mask,filt_excl_list=exclusions,start=0,end=raw.shape[1],device=raw.device)
    return (copied > gold[:,None]).sum(dim=1).long()+1


def compare_endpoints(raw, gold, queries, direction, facts):
    shared=shared_endpoint_ranks(raw,gold,queries,direction,facts)
    direct=ranks_against_reference(filtered_copy(raw,queries,direction,facts),gold)
    z, reference=query_zscore_with_reference(raw,gold)
    z=filtered_copy(z,queries,direction,facts)
    normalized=ranks_against_reference(z,reference)
    # An inactive non-finite expert must not contaminate the deployed endpoint.
    inactive=torch.full_like(z,float('nan')); inactive_gold=torch.full_like(gold,float('nan'))
    first=endpoint_safe_mixed_ranks(z,inactive,reference,inactive_gold,1.,direct,shared)
    second=endpoint_safe_mixed_ranks(inactive,z,inactive_gold,reference,0.,shared,direct)
    mismatch=normalized != shared
    counts=dict(rows=len(raw), export_vs_shared_mismatches=int((direct!=shared).sum()),
                alpha_one_vs_shared_mismatches=int((first!=shared).sum()),
                alpha_zero_vs_shared_mismatches=int((second!=shared).sum()),
                normalized_vs_raw_mismatches=int(mismatch.sum()),
                raw_nonfinite_rows=int(((~torch.isfinite(raw)).any(1)|~torch.isfinite(gold)).sum()),
                normalized_nonfinite_gold_rows=int((~torch.isfinite(reference)).sum()),
                raw_rr_sum=float((1/shared.double()).sum()), normalized_rr_sum=float((1/normalized.double()).sum()))
    return counts, shared, normalized
