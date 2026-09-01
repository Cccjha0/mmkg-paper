"""Sampling primitives reserved for recent baseline models.

The existing ``sampler.negative_sample`` intentionally remains unchanged for
all legacy models.  This module implements the OpenKE-style Bernoulli plus
filtered-negative policy used by configured recent baselines.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, NamedTuple

import torch


class RelationStatistics(NamedTuple):
    """Per-relation statistics for Bernoulli negative sampling."""

    tph: dict[int, float]
    hpt: dict[int, float]
    corrupt_head_probability: dict[int, float]


def build_relation_statistics(triples: list[tuple[int, int, int]]) -> RelationStatistics:
    """Compute OpenKE-style ``tph``, ``hpt``, and head-corruption probability."""
    triples_per_relation: dict[int, int] = defaultdict(int)
    heads_per_relation: dict[int, set[int]] = defaultdict(set)
    tails_per_relation: dict[int, set[int]] = defaultdict(set)
    for h, r, t in triples:
        triples_per_relation[r] += 1
        heads_per_relation[r].add(h)
        tails_per_relation[r].add(t)

    tph: dict[int, float] = {}
    hpt: dict[int, float] = {}
    probabilities: dict[int, float] = {}
    for relation, count in triples_per_relation.items():
        tph_value = count / len(heads_per_relation[relation])
        hpt_value = count / len(tails_per_relation[relation])
        tph[relation] = tph_value
        hpt[relation] = hpt_value
        probabilities[relation] = tph_value / (tph_value + hpt_value)
    return RelationStatistics(tph=tph, hpt=hpt, corrupt_head_probability=probabilities)


def _lookup_head_probabilities(
    relation_ids: torch.LongTensor,
    relation_stats: RelationStatistics | Mapping[str, Mapping[int, float]],
) -> torch.Tensor:
    if isinstance(relation_stats, RelationStatistics):
        probabilities = relation_stats.corrupt_head_probability
    else:
        probabilities = relation_stats["corrupt_head_probability"]
    values = [float(probabilities.get(int(relation_id), 0.5)) for relation_id in relation_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float32, device=relation_ids.device)


def _filter_candidate_batch(
    candidates: torch.LongTensor,
    *,
    forbidden: set[int],
    num_entities: int,
    max_attempts: int,
) -> torch.LongTensor:
    """Reject known answers for one query using batched random draws.

    Set membership remains on CPU, but random candidates are generated a whole
    query group at a time instead of issuing one ``torch.randint`` call per
    negative triple.
    """
    if candidates.numel() == 0:
        return candidates
    if len(forbidden) >= num_entities:
        raise ValueError("Cannot draw a filtered negative: every entity is a known true answer for this query.")

    result = candidates
    for attempt in range(max_attempts):
        invalid = torch.tensor(
            [candidate in forbidden for candidate in result.tolist()],
            dtype=torch.bool,
        )
        if not bool(invalid.any()):
            return result
        if attempt + 1 < max_attempts:
            result[invalid] = torch.randint(num_entities, (int(invalid.sum().item()),))

    fallback = next(candidate for candidate in range(num_entities) if candidate not in forbidden)
    result[invalid] = fallback
    return result


def bernoulli_filtered_negative_sample(
    pos: torch.LongTensor,
    num_entities: int,
    true_heads: Mapping[tuple[int, int], set[int]],
    true_tails: Mapping[tuple[int, int], set[int]],
    relation_stats: RelationStatistics | Mapping[str, Mapping[int, float]],
    neg_ratio: int,
    *,
    max_attempts: int = 64,
) -> torch.LongTensor:
    """Draw Bernoulli head/tail corruptions while excluding all known true facts.

    ``true_heads`` is indexed by ``(r, t)`` and ``true_tails`` by ``(h, r)``.
    The output has shape ``[len(pos) * neg_ratio, 3]`` and stays on ``pos``'s
    device.  It is intentionally not called by the legacy standard trainer.
    """
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"Expected pos with shape [batch, 3], got {tuple(pos.shape)}.")
    if num_entities <= 0:
        raise ValueError("num_entities must be positive.")
    if neg_ratio <= 0:
        raise ValueError("neg_ratio must be positive.")

    source_device = pos.device
    # The trainer now calls this sampler before H2D. ``copy=False`` therefore
    # avoids the former GPU -> CPU round trip while keeping direct GPU callers
    # backward compatible.
    pos_cpu = pos.detach().to(device="cpu", copy=False)
    triples = pos_cpu.tolist()
    head_probabilities = _lookup_head_probabilities(pos_cpu[:, 1], relation_stats)
    corrupt_head = torch.rand((pos_cpu.shape[0], neg_ratio)) < head_probabilities.unsqueeze(1)
    candidates = torch.randint(num_entities, (pos_cpu.shape[0], neg_ratio))
    negatives = pos_cpu.unsqueeze(1).expand(-1, neg_ratio, -1).clone()
    empty: set[int] = set()

    # One Python iteration per positive triple (not per generated negative).
    # The potentially large neg_ratio dimension is handled in batches. This
    # preserves the sampling distribution, although batching changes the exact
    # random-number draw order relative to the former serial implementation.
    for row_index, (h, r, t) in enumerate(triples):
        head_mask = corrupt_head[row_index]
        tail_mask = ~head_mask
        if bool(head_mask.any()):
            forbidden_heads = true_heads.get((r, t), empty)
            negatives[row_index, head_mask, 0] = _filter_candidate_batch(
                candidates[row_index, head_mask],
                forbidden=forbidden_heads,
                num_entities=num_entities,
                max_attempts=max_attempts,
            )
        if bool(tail_mask.any()):
            forbidden_tails = true_tails.get((h, r), empty)
            negatives[row_index, tail_mask, 2] = _filter_candidate_batch(
                candidates[row_index, tail_mask],
                forbidden=forbidden_tails,
                num_entities=num_entities,
                max_attempts=max_attempts,
            )

    neg_cpu = negatives.reshape(-1, 3)
    return neg_cpu.to(device=source_device)
