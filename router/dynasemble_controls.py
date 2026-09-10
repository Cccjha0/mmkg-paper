"""D01-D03 controls. Selector inputs contain distributions, never gold/filter facts."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn

VARIANTS = ("R1_source", "R2_init", "R3_softplus_fixed", "R3_softplus", "R4_swap")


class ControlSelector(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        # Exact upstream layer order: there is no ReLU after the first Linear.
        first = nn.Linear(4, 16)
        second = nn.Linear(16, 16)
        # Preserve upstream RNG consumption: initialize the hidden weight before
        # constructing the output Linear, not after building the whole Sequential.
        nn.init.uniform_(second.weight, 0., 2.)
        self.layers = nn.Sequential(first, second, nn.ReLU(), nn.Linear(16, 1))
        if variant != "R1_source":
            # Same output weights and initial effective ratio for the repaired arms.
            nn.init.uniform_(self.layers[3].weight, -1e-3, 1e-3)
            nn.init.constant_(self.layers[3].bias, 1. if variant == "R2_init" else math.log(math.expm1(1.)))

    def preactivation(self, features):
        return self.layers(features).squeeze(-1)

    def forward(self, features):
        z = self.preactivation(features)
        return torch.relu(z) if self.variant in ("R1_source", "R2_init") else torch.nn.functional.softplus(z)


def coefficients(weight, variant):
    """Return unnormalized score coefficients. Loss keeps the released margin scale."""
    one = torch.ones_like(weight)
    return (one, weight) if variant == "R4_swap" else (weight, one)


def training_candidates(a, b, ref_a, ref_b, queries, facts, count, generator):
    """Sample loss candidates AFTER full-distribution normalization/features.

    a/b are already normalized full candidate scores. Only this supervision
    branch receives queries/golds/facts. Sampling is with replacement.
    """
    left, right = [], []
    for i, (h, r, t, direction) in enumerate(queries):
        gold = int(t if direction == 1 else h)
        key = (int(h), int(r)) if direction == 1 else (int(r), int(t))
        forbidden = facts[int(direction)].get(key, set()) | {gold}
        eligible = np.setdiff1d(np.arange(a.shape[1]), np.fromiter(forbidden, dtype=np.int64))
        if not len(eligible):
            raise ValueError("No strict negatives remain")
        ids = eligible[generator.integers(0, len(eligible), size=count)]
        left.append(np.concatenate(([ref_a[i]], a[i, ids])))
        right.append(np.concatenate(([ref_b[i]], b[i, ids])))
    return np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)


def fact_sets(triples):
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        tails[(int(h), int(r))].add(int(t))
        heads[(int(r), int(t))].add(int(h))
    return {0: heads, 1: tails}


def grouped_folds(triples, folds=3, seed=20260910):
    """Relation-stratified deterministic original-triple folds, independent of model seed."""
    triples = np.asarray(triples)
    if len(set(map(tuple, triples.tolist()))) != len(triples):
        raise ValueError("Duplicate DEV triples")
    output = np.empty(len(triples), dtype=np.int64)
    for relation in sorted(set(triples[:, 1])):
        indices = np.flatnonzero(triples[:, 1] == relation).tolist()
        indices.sort(key=lambda i: hashlib.sha256(f"{seed}|{tuple(triples[i].tolist())}".encode()).digest())
        for position, index in enumerate(indices):
            output[index] = position % folds
    if len(set(output)) != folds:
        raise ValueError("Insufficient DEV groups for all folds")
    return output


def complete_order_equal(primary, mixed):
    """Exact weak-order equality, including ties, on the full supplied candidate set."""
    order = np.argsort(primary, kind="stable")
    dp = np.diff(primary[order])
    dm = np.diff(mixed[order])
    return bool(np.all(np.where(dp == 0, dm == 0, dm > 0)))


def choose_configuration(records):
    """Pool every predeclared fold/base/selector seed; never remove a failed seed."""
    groups = defaultdict(list)
    for row in records:
        if row["variant"] == "R3_softplus":
            groups[(row["learning_rate"], row["epoch"])].append(row)
    if not groups:
        raise ValueError("No DEV selection records")
    coverage = {tuple(sorted((r["base_seed"], r["selector_seed"], r["fold"]) for r in rows)) for rows in groups.values()}
    if len(coverage) != 1:
        raise ValueError("Unequal CV coverage; failed seeds cannot be dropped")
    scored = []
    for (lr, epoch), rows in groups.items():
        n = sum(r["n"] for r in rows)
        mean = sum(r["mrr"] * r["n"] for r in rows) / n
        if not np.isfinite(mean):
            raise ValueError("Non-finite DEV score; selection must stop, not omit the run")
        scored.append((mean, lr, epoch))
    mean, lr, epoch = min(scored, key=lambda x: (-x[0], x[2], x[1]))
    return {"learning_rate": lr, "epochs": epoch, "heldout_dev_mrr": mean}
