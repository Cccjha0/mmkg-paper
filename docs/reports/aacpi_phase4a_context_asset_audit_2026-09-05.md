# AACPI Phase 4A Context Asset Audit

**Audit date:** 2026-09-05

**Scope:** frozen TRAIN/DEV assets and selected expert checkpoints only

## Structural context

Both dataset configs point to canonical processed bundles containing `train.tsv`, `valid.tsv`, entity/relation mappings, modality tensors, and a hash manifest. `train.tsv` is sufficient to compute known-side degree/frequency, direction-specific role frequency, unique neighbors, relation frequency, unique heads/tails, heads-per-tail, tails-per-head, and continuous mapping-regime statistics. These features do not require DEV labels.

The filtered-ranking evaluator also has broader answer filters, but those are not a legal structural source. The Phase 4A builder opens only canonical TRAIN for structural statistics and uses DEV only to identify query rows already present in the frozen utility/action-response asset. It rejects TEST-like input paths.

## Modality context

The processed bundles expose `has_img.pt` and `has_text.pt` for every entity. They support known-side image/text availability and TRAIN-only relation support priors. The current canonical bundle does not expose a stable per-entity image-count or text-field-completeness count, so Phase 4A freezes binary presence and two-modality completeness only. It will not reconstruct counts from raw directories.

For tail prediction the known side is the head; for head prediction it is the tail. Current-answer target masks are prohibited. A relation-level prior computed from TRAIN targets remains answer-agnostic, but it is already part of frozen R3; C2 adds known-role support rather than duplicating current-answer information. M-Hyper, NativE, and AdaMF-MAT all receive canonical image and text tensors, so redundant expert-specific mask copies add no information and are omitted.

## Frozen latent context

| Expert | Source module/function | Frozen tensor semantic | Raw dimension | Direction-dependent | Candidate-independent |
| --- | --- | --- | ---: | --- | --- |
| M-Hyper | `recent_baselines/mhyper.py: OpenBGMHyper._queries` | clean fused relation-transformed query immediately before candidate dot product | `8*rank = 1024` for rank 128 | yes; head uses reciprocal relation/query | yes |
| NativE | `native.py: get_batch_ent_multimodal_embs`, `get_joint_embeddings`, `rel_embeddings` | known-side relation-gated fused entity concatenated with relation | `3*d = 750` for d 250 | yes through known side; fusion is relation-gated | yes |
| AdaMF-MAT | `adamf_mat.py: get_batch_ent_multimodal_embs`, `get_joint_embeddings`, `rel_embeddings` | known-side fused entity concatenated with relation | MKG-W `3*d = 600` for d 200; DB15K `750` for d 250 | yes through known side | yes |

M-Hyper's `ReciprocalHeadScoringMixin` maps `(h,r,t)` to `(t,r^-1,h)` for head prediction, so extraction must use the same transform. NativE and AdaMF-MAT inherit the symmetric directional scorer; for them the known-side fused entity and relation are available before candidate scoring, while a full scored triple embedding would require the unknown candidate and is forbidden.

Each Phase 3A source manifest records the six selected checkpoint/config paths per pair and their SHA-256 hashes. Phase 4A reuses those exact paths, validates hashes before loading, calls `eval()` under inference mode, performs no optimizer step, and records checkpoint/config hashes in its latent manifest.

## Asset limitations and decisions

- Exact image counts and raw text-field completeness are unavailable as canonical processed fields; they are excluded.
- Raw latent dimensions differ across architectures and, for AdaMF-MAT, across datasets. They are never fed directly to the low-capacity regressor.
- Separate 16-dimensional training-fold PCA is frozen before results. It controls input size without supervised representation learning.
- Elementwise projected interactions are diagnostic and do not imply aligned raw latent bases.
- No audited source requires TEST, the correct DEV answer identity, its modality, rank, RR, or score.

**Audit decision:** the repository contains sufficient legal assets for C1-C4. Phase 4A implementation may proceed on DEV, subject to hash checks and fold-local preprocessing.
