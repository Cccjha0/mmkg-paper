# OpenBG-IMG heterogeneous complementarity protocol

## Research question

Do independently designed strong MMKGC models exhibit query-dependent
complementarity that is stable across seeds and recoverable without access to
the labeled answer at inference time?

The controlled Gate-only/Residual-only line remains a mechanism study. This
protocol adds an external-validity line using M-Hyper/AdaMF-MAT and
M-Hyper/NativE.

## Frozen first-stage methods

All score combinations use exact filtered full-entity rankings. M-Hyper is
expert A and receives weight alpha.

1. Equal RRF: `1/(60 + rank_A) + 1/(60 + rank_B)`. No DEV selection.
2. Query-zscore 0.5: normalize each expert over the filtered candidate
   distribution for the current query, then average. No DEV selection.
3. Global alpha: select one shared alpha from `0.00:0.05:1.00` on pooled
   three-seed DEV MRR.
4. Relation alpha: select one alpha per relation on pooled three-seed DEV MRR
   when support is at least 60 seed-query observations; otherwise use the
   global alpha.
5. Oracle: per-query maximum expert reciprocal rank. This is answer-aware and
   is only an upper bound.

Tie-breaking for equally good DEV alphas prefers the value closest to 0.5,
then the smaller alpha. TEST consumes only `selection.json` written by DEV.

## Information boundaries

- Equal RRF is rank-aware and answer-agnostic.
- Query-zscore and alpha policies are score-aware and answer-agnostic.
- Relation alpha is not metadata-only because it combines candidate scores.
- Oracle is answer-aware and is never a deployable method.

## Required reporting

Report pooled and per-seed MRR/Hits@K, delta from M-Hyper, and Oracle-gap
recovery. The audit must reproduce each fixed expert's stored DEV/TEST MRR to
within `5e-7`. Also report seed agreement for the sign of `RR_B - RR_A`.

DEV is exploratory policy selection. Do not interpret DEV-selected global or
relation alpha as final evidence. Run TEST only after reviewing and freezing
the two generated `selection.json` files.

## Decision rule

The strong heterogeneous claim is supported only if answer-agnostic gains are
positive and directionally consistent across both model pairs and seeds on
TEST. A large Oracle with weak or unstable simple-combination gains is evidence
of error diversity, not readily exploitable structured complementarity.
