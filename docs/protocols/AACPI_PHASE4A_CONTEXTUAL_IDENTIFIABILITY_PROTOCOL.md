# AACPI Phase 4A Contextual Identifiability Protocol

**Status:** frozen before the first systematic Phase 4A run

**Effective date:** 2026-09-05

**Evidence scope:** DEV-only information-sufficiency diagnosis

## Frozen research boundary

AACPI V2 and Phase 3A remain frozen **NO-GO**. Phase 4A is a new information-sufficiency hypothesis: it asks whether inference-time, answer-agnostic structural, modality, or frozen-expert latent query context contains sign information absent from score/action-response geometry. It does not modify AACPI V2, Phase 3A, Base13, R1/R2/R3, expert checkpoints, checkpoint selection, score normalization, filtered ranking, `alpha0`, the local action grid, or the utility target.

All work is DEV-only. Phase 4A runs no Advantage-Greedy or conservative policy, no `kappa/lambda/tau`, no LCB or fallback rule, and no TEST evaluation. MKG-W and DB15K TEST remain retrospective/secondary and ineligible for selection. A Phase 4A GO cannot authorize TEST or policy work; either requires a separately frozen Phase 4B protocol.

The only experimental variable is the legal query representation. No new score-only handcrafted family, larger MLP, relation-specific model, supervised encoder, expert fine-tuning, checkpoint reselection, or dataset-specific architecture is allowed.

## Legal information

Every context field must be available before the correct target is known. Correct-target identity, embedding, score, rank, RR, modality availability, and any statistic constructed from DEV outcomes are forbidden. Structural counts and modality priors use canonical TRAIN triples only. The all-split filtered graph is never used for context. Frozen latent extraction may read canonical TRAIN/DEV mappings and feature tensors plus the already selected checkpoints, but it may not backpropagate into an expert or open TEST.

## Frozen representation families

- **C0:** Phase 3A R3 OOF control. Its row-level predictions are copied with hash verification, not retrained; independent analysis requires a maximum absolute difference no larger than `1e-18` after CSV serialization.
- **C1:** C0 plus the frozen TRAIN-only structural fields in `AACPI_PHASE4A_CONTEXT_FEATURE_CONTRACT.md`.
- **C2:** C0 plus the frozen known-side modality and TRAIN-only relation modality fields in the contract.
- **C3:** C0 plus frozen expert query latents. Each expert is independently reduced to 16 dimensions by unsupervised PCA fitted on the current training groups; inputs are projected `z_A`, projected `z_B`, their difference, and absolute difference.
- **C4:** the single full combination C0 + C1 + C2 + C3.

No C5 or post-result feature variant may be added in Phase 4A.

## Estimator and nested evaluation

The estimator and search space are exactly those in `AACPI_V2_PHASE2_SEARCH_SPACE.yaml`: two hidden layers; widths `{32,64,128}`; Smooth-L1 beta `0.02`; negative weights `{1.0,1.5,2.0,3.0}`; Adam learning rates `{1e-4,3e-4,1e-3}`; 30 epochs; batch size 4096; identical selection rule and seeds. Only input dimension changes.

Outer CV uses five original-triple grouped folds. Inner selection uses three original-triple grouped folds inside each outer-training partition. All seeds, directions, and actions belonging to one original triple remain together. Each DEV query-action row has exactly one outer-held-out prediction.

For C3/C4, PCA dimension is fixed at 16 per expert and is not selected. The frozen solver is seeded Halko randomized SVD with eight oversamples and two power iterations. Inner validation uses PCA fitted only on that inner-training partition; final outer prediction uses PCA fitted only on outer-training groups. Feature standardization remains inside the unchanged regressor and is fitted on its current training partition. PCA never sees utility targets. Fold manifests record training-group hashes and preprocessing-array hashes.

## Frozen metrics

Primary rows exclude the reference action. Every pair and representation reports Spearman, positive-action AP and lift, harmful-action AP and lift, positive-versus-harmful AUROC, and top-predicted-decile actual mean U, `P(U>0)`, and `P(U<0)`. The same H1 metrics are reported by action, direction, seed, and supported relation. Positive-vs-rest, harmful-vs-rest, and positive-vs-harmful remain separate; nonzero-activity discrimination cannot substitute for sign separation.

Context increments are paired `Cx - C0` differences for sign AUROC, both AP lifts, top-decile U, and activity AP lift.

## Frozen gates

An H1-style pair passes only when Spearman, positive AP lift, harmful AP lift, and top-decile actual U are all strictly positive.

### Primary Gate

At least one of C1-C4 must satisfy all of the following:

1. at least 4/6 pairs pass H1;
2. both NativE + AdaMF-MAT pairs pass;
3. for MKG-W / NativE + AdaMF-MAT, both AP lifts and top-decile U are positive and sign AUROC exceeds C0;
4. the three Phase 3A prior-PASS pairs remain stable: at least two still pass H1 and none simultaneously has nonpositive top-decile U and nonpositive sign-AUROC lift over chance.

### Strong Recovery Gate

For MKG-W / NativE + AdaMF-MAT, sign AUROC must be at least `0.55`, top-decile U must be positive, at least 1/2 directions must pass H1, and at least 2/3 seeds must pass H1.

### Context Contribution Gate

A claimed recovery must improve positive-versus-harmful AUROC and top-decile actual utility over C0. Improvement confined to nonzero activity or plateau detection is insufficient.

Phase 4A is GO only when one frozen contextual representation passes the Primary, Strong Recovery, Context Contribution, and prior-PASS stability gates. Otherwise it is NO-GO. It is also an explicit NO-GO if every contextual representation leaves the critical pair with nonpositive top-decile U, sign AUROC in the approximately `0.50-0.53` range, or unchanged 0/2-direction and 0/3-seed support. A NO-GO stops additional score, structural, modality, and frozen-latent feature fishing.

## Artifact and integrity contract

Large raw context tables, latent tensors, and row-level OOF predictions live under ignored `outputs/aacpi/phase4a/{raw,latents,oof_raw}`. Git retains protocols, scripts, manifests, hashes, small summaries, figures, and the audit report. Every systematic run must certify: TEST rows/commands 0; policy evaluations 0; expert training 0; checkpoint reselection 0; outer group leakage 0; and C0 equality with Phase 3A R3 within the frozen `1e-18` CSV round-trip tolerance.
