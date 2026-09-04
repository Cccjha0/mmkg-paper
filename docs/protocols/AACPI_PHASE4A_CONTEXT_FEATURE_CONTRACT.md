# AACPI Phase 4A Context Feature Contract

**Status:** frozen before systematic Phase 4A OOF evaluation

**Date:** 2026-09-05

## C0

C0 is the unchanged Phase 3A R3 representation and its exact existing OOF predictions. It contains Base13, R2 action-response geometry, and the eight frozen R3 lightweight context fields.

## C1 structural additions

For tail prediction, the known entity is the head and its role count is its TRAIN head count. For head prediction, the known entity is the tail and its role count is its TRAIN tail count. C0 already contains TRAIN relation frequency, known-entity total and role frequency, and unique-relation count; C1 reuses those fields without adding duplicate columns. Its new values are computed only from canonical TRAIN triples:

- `c1_known_entity_neighborhood_fraction`: unique one-hop TRAIN neighbors divided by `max(num_entities-1,1)`;
- `c1_relation_unique_heads_log1p`, `c1_relation_unique_tails_log1p`;
- `c1_relation_tails_per_head_log1p`: log1p(relation count / unique heads);
- `c1_relation_heads_per_tail_log1p`: log1p(relation count / unique tails);
- `c1_relation_head_tail_log_ratio`: `log((unique_heads+1)/(unique_tails+1))`;
- `c1_relation_known_role_diversity_log1p`: log1p unique heads for tail prediction and unique tails for head prediction.

These continuous statistics encode mapping regimes without relation IDs, one-hot features, or relation-specific models.

## C2 modality additions

- `c2_relation_known_image_support`, `c2_relation_known_text_support`: TRAIN-only support rates among relation heads for tail prediction and relation tails for head prediction;

C0 already contains the canonical known-side image/text presence masks and TRAIN-only target-role relation support rates. C2 inherits them without adding duplicate mask or linear-combination columns.

The processed bundles expose presence masks, but no canonical per-entity image-count or text-field-count asset. Phase 4A therefore does not infer counts from filesystem layout or raw source data. All three frozen experts use both modalities; duplicate expert-specific copies of the same masks are omitted. No target-side property of the current DEV answer is included.

## C3 frozen latent additions

For each seed and query, `z_A(q)` and `z_B(q)` are extracted from the already selected frozen checkpoints:

- **M-Hyper:** the output of `OpenBGMHyper._queries` immediately before its dot product with the candidate matrix. Head prediction first applies `reciprocal_head_triples`, exactly matching `ReciprocalHeadScoringMixin.score_head`.
- **NativE:** the known entity's relation-gated fused embedding from `get_batch_ent_multimodal_embs` and `get_joint_embeddings`, concatenated with `rel_embeddings(relation)`. Tail queries use the known head; head queries use the known tail.
- **AdaMF-MAT:** the known entity's fused embedding from `get_batch_ent_multimodal_embs` and `get_joint_embeddings`, concatenated with `rel_embeddings(relation)`.

All three tensors are candidate-independent and determined before the unknown answer is supplied. NativE/AdaMF-MAT use a decoder-input representation because their symmetric RotatE score has no single candidate-independent transformed vector for both directions.

Each expert's raw latent is independently centered and reduced to exactly 16 dimensions by a seeded Halko randomized-SVD PCA on unique query instances from the current training groups, with eight oversamples and two power iterations. Component signs are fixed by making each component's largest-absolute loading positive. C3 contains 64 fields: projected `z_A[16]`, projected `z_B[16]`, `z_A-z_B[16]`, and `abs(z_A-z_B)[16]`. The elementwise interaction is a frozen low-dimensional diagnostic interaction; no claim is made that raw expert coordinate systems are aligned.

PCA is re-fitted for each inner-training partition during hyperparameter selection and once on each outer-training partition for final OOF prediction. It uses no advantage target and no held-out rows.

## C4

C4 is the only combined family: C0 + all C1 additions + all C2 additions + all C3 latent fields.

All fields are finite and answer-agnostic. The utility columns remain supervision only. The original-triple group key and all historical representation definitions remain unchanged.
