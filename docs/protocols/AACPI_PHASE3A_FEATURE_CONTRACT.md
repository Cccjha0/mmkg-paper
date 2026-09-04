# AACPI Phase 3A Feature Contract

**Status:** frozen before systematic comparison
**Date:** 2026-09-04

All rankings use the full unfiltered candidate score vector and sort by descending score with ascending candidate ID as the deterministic tie breaker. For union-top-20 statistics, candidates are re-ranked within that union and displacements are divided by `max(|union| - 1, 1)`; correlations are Spearman correlations of those union ranks. The two cross-expert top-1-under-other ranks use full candidate ranks divided by `max(num_entities - 1, 1)`. Scores use the frozen query-wise z-score function.

## R0 — Base13 control

R0 contains the unchanged fields from `router.constants.QUERY_GEOMETRY_FIELDS`, followed by `delta_alpha` and `abs_delta_alpha`. It reproduces the Phase 2B representation contract.

## R1 — Base13 plus static cross-expert ranking geometry

R1 contains R0 plus:

- `r1_expert_top5_jaccard`, `r1_expert_top10_jaccard`, `r1_expert_top20_jaccard`;
- `r1_union20_rank_spearman`;
- `r1_union20_rank_displacement_mean`, `r1_union20_rank_displacement_median`, `r1_union20_rank_displacement_max`;
- `r1_expert_top1_same`;
- `r1_a_top1_rank_under_b`, `r1_b_top1_rank_under_a`.

These fields compare expert A and B before any action is selected. They never inspect a target.

## R2 — Base13 plus action-response geometry

R2 contains R0 plus:

- `r2_response_top5_jaccard`, `r2_response_top10_jaccard`, `r2_response_top20_jaccard`;
- `r2_response_union20_rank_spearman`;
- `r2_response_union20_rank_displacement_mean`, `r2_response_union20_rank_displacement_median`, `r2_response_union20_rank_displacement_max`;
- `r2_response_top1_changed`;
- `r2_action_top1_top2_margin`;
- `r2_response_top1_top2_margin_change`;
- `r2_response_top5_mean_score_change`;
- `r2_response_score_std_change`.

Each field compares the unfiltered candidate landscape at the frozen Global `alpha0` with the landscape at the row's frozen local action `alpha`.

Entering and leaving top-k fractions are not stored: for two sets of equal cardinality `k`, both are `(k-|intersection|)/k`, and Jaccard uniquely determines the same quantity. A separate concentration-change feature is not stored because the mean of each query-zscore expert vector, and therefore every convex mixture, is zero; top-five concentration change would duplicate `r2_response_top5_mean_score_change`.

Reference rows must have Jaccard 1, rank correlation 1, zero displacement, unchanged top-1, and zero changes in margin, top-five mean, and score standard deviation within numerical tolerance.

## R3 — R2 plus lightweight TRAIN-only query context

R3 contains R2 plus:

- `r3_train_relation_frequency_log1p`;
- `r3_train_observed_entity_frequency_log1p`;
- `r3_train_observed_entity_direction_frequency_log1p`;
- `r3_train_observed_entity_unique_relation_count_log1p`;
- `r3_observed_entity_has_text`;
- `r3_observed_entity_has_image`;
- `r3_train_relation_target_text_support`;
- `r3_train_relation_target_image_support`.

For tail prediction the observed entity is the head; for head prediction it is the tail. No feature describes the held-out target entity. Relation target-modality support is computed only from TRAIN triples, separately by prediction direction. Raw relation ID, relation one-hot encoding, learned relation embedding, DEV outcome summary, and TEST statistic are excluded. Prediction direction is already present in Base13 and is not duplicated.

## Family nesting and invariants

R1 and R2 are separate ablations: R2 does not include R1. R3 adds context to R2. Every feature is finite, has a single documented definition, and is computed before joining any target rank, RR, or advantage. The join key is `(query_id, alpha)`; after the join, the Phase 1 utility columns remain unchanged byte-for-byte in their source artifact.
