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


def _draw_filtered_entity(
    *,
    forbidden: set[int],
    num_entities: int,
    device: torch.device,
    max_attempts: int,
) -> int:
    if len(forbidden) >= num_entities:
        raise ValueError("Cannot draw a filtered negative: every entity is a known true answer for this query.")
    for _ in range(max_attempts):
        candidate = int(torch.randint(num_entities, (1,), device=device).item())
        if candidate not in forbidden:
            return candidate
    # Deterministic fallback guarantees a valid negative despite unlucky draws.
    for candidate in range(num_entities):
        if candidate not in forbidden:
            return candidate
    raise AssertionError("Checked non-saturated forbidden set but found no valid negative.")


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

    # Filtering uses Python sets keyed by integer ids.  Keep that work on CPU
    # in one batch to avoid a GPU synchronization for every sampled triple.
    neg_cpu = pos.detach().cpu().repeat_interleave(neg_ratio, dim=0).clone()
    head_probabilities = _lookup_head_probabilities(neg_cpu[:, 1], relation_stats)
    corrupt_head = torch.rand(neg_cpu.shape[0]) < head_probabilities
    device = torch.device("cpu")

    for row_index in range(neg_cpu.shape[0]):
        h, r, t = (int(value) for value in neg_cpu[row_index].tolist())
        if bool(corrupt_head[row_index].item()):
            neg_cpu[row_index, 0] = _draw_filtered_entity(
                forbidden=set(true_heads.get((r, t), set())),
                num_entities=num_entities,
                device=device,
                max_attempts=max_attempts,
            )
        else:
            neg_cpu[row_index, 2] = _draw_filtered_entity(
                forbidden=set(true_tails.get((h, r), set())),
                num_entities=num_entities,
                device=device,
                max_attempts=max_attempts,
            )
    return neg_cpu.to(device=pos.device)
