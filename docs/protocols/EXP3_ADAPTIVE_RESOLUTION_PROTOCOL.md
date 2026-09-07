# Experiment 3 — Granularity Ladder / Adaptive Resolution Audit

**Status:** frozen before the systematic run

**Date:** 2026-09-07

## Scope

This DEV-only experiment compares adaptation resolution. It does not develop a hierarchical method, LCB, conservative policy, or new query selector. Every learned action, degree boundary, and support choice is estimated strictly inside the applicable outer-training original-triple groups.

The six dataset/expert pairs, exact filtered per-query RR curves, 21-point alpha grid, three seeds, and head/tail queries are inherited unchanged from Experiments 1 and 2.

## Granularity ladder

- **L0 Global:** every held-out query uses the alpha selected on the current outer-training partition.
- **L1 Direction:** one outer-training action for head and one for tail.
- **L2 Relation:** one action per relation with at least 25 distinct outer-training original triples; otherwise fold Global.
- **L3 Relation × Direction:** one action per `(relation, direction)` with at least 25 distinct outer-training original triples; otherwise fold Global.
- **L4 Context Group:** one action per `(relation, direction, degree_bucket, known-side modality_state)`. The support threshold is selected from `{10,25,50,100}` using three-fold grouped CV entirely inside outer-train.
- **L5 Individual Query:** direct reuse of Experiment 2 X4 `nested_selected` strict-OOF actions. No model is trained in Experiment 3. X4 is frozen globally because it is the single information representation with 6/6 robust-positive pairs and the closest frozen preliminary-gate result.

L2/L3 support and all L4 support values count distinct original triples, not the six duplicated seed/direction observations. Every unsupported group falls back only to the current fold Global.

## Context definition

L4 uses the frozen TRAIN-only known-side degree feature `r3_train_observed_entity_frequency_log1p`. Quartile boundaries are fit separately within the current inner/outer training partition using unique `(original_triple_id, direction)` rows. Assignment uses `searchsorted(..., side="right")`.

Known-side modality state is the deterministic four-way state from the frozen inference-time fields `r3_observed_entity_has_text` and `r3_observed_entity_has_image`: none, text-only, image-only, or text-and-image. No target-side modality information is used.

## Nested protocol

The five outer folds are exactly those in Experiment 2 X4 OOF artifacts and are validated to keep every original triple intact. For each outer fold:

1. select Global alpha using outer-train RR only;
2. fit L1–L4 actions using outer-train groups only;
3. for L4 only, select `n_min` by three-fold grouped inner CV, with inner-specific Global, degree boundaries, and group actions fit on inner-train only;
4. evaluate actions once on outer-held-out queries.

Action ties prefer the fold Global, then the nearest alpha, then the smaller alpha. This is a deterministic audit rule, not a confidence fallback.

## Metrics and stability

For every dataset/pair/level, report training adaptive gain, outer-OOF MRR and gain, original-triple clustered bootstrap CI, train–OOF gap, negative/positive transfer ratios, changed rate, support coverage, available-headroom recovery, seed and direction slices, fold action-distribution MAD, and group-alpha foldwise MAD where a shared group identity exists.

A pair is robust-positive only when aggregate OOF gain and CI lower are positive, at least two of three seed gains are positive, and head/tail gains are not both negative.

## Route comparisons

“Best group” is the descriptive per-pair maximum OOF gain among L1–L4; it is used only for the frozen route audit and is not proposed as a deployable policy. Query-minus-group uncertainty uses paired original-triple clustered bootstrap.

For Route A's granularity-overfit condition, a pair exhibits the pattern when at least two of the three adjacent L1→L2→L3→L4 refinements increase training gain while OOF gain does not increase. “Most pairs” means at least 4/6.

Route B uses Experiment 2 X6 set-encoder OOF actions exactly as frozen. If neither Route A nor Route B satisfies every condition, the decision defaults to `ROUTE_C_LIMITS` as insufficient evidence.

## Operational invariants

- TEST access = 0
- outer-triple leakage = 0
- new query-selector training = 0
- hierarchical shrinkage / LCB / conservative policy = 0
- checkpoint modification = 0
- all direct source and output hashes are recorded
