# E02 fixed-setting component ablation

Recorded before evaluating these new contrasts. This is a retrospective, fixed-policy diagnostic after historical TEST inspection, not new confirmatory evidence or a parameter-selection exercise. Preserve prior tuned matched alternatives, corrected ADC, radius and gate results. Six pairs, three seeds, two directions, original grouped OOF DEV and frozen TEST; no model fits or scorer runs.

## Shared inputs and minimal contrast family

All variants share the same original balanced classifier, margin g, preference p=sigmoid(g), original anchor a, original beta b and threshold t for each fitted object, candidate scores/evaluator, clipping to [0,1], and 0.05 projection with ties toward a then the smaller weight. Query-soft is not treated as an isolated anchor ablation. Let Q denote this common clipping/projection and c=abs(2p-1). Rejected/invalid rows always use a.

| Variant | Accepted proposal | Sole setting change or stated compound map |
|---|---|---|
| ADC | a+b*tanh(g) | Original b,t |
| Center-0.5 | 0.5+b*tanh(g) | Change accepted proposal origin only; keep original fallback target and projection ties |
| Linear-p | a+b*(2p-1) | Change bounded response shape at same b,t; lower local slope is part of this mapping |
| Clip-g | a+b*clip(g,-1,1) | Change saturation shape at same origin,b,t; same derivative at zero before clipping/projection |
| Shrink-fixed | (1-b)*a+b*p | Set lambda=b once; same gate, no map-specific tuning; shrinkage is not a pure origin change |
| Radius-1 | a+tanh(g) | Replace b by 1, preserving origin, tanh and original gate |
| No-gate | a+b*tanh(g) | Replace t by 0, preserving origin, radius, mapping and projection |
| Global | a | Static context row, not an extra fitted model |

Center-0.5 removes the DEV-selected center from the **accepted proposal**, not from the entire algorithm: fallback and projection ties still reference a. It is a neutral-center sensitivity, not an optimized competing reference or proof that every reference is necessary. It equals ADC at a=0.5. Fixed-setting comparisons condition on original ADC-selected b,t and may favor that configuration; prior per-family 41-candidate DEV comparisons provide the separately tuned context. Shrink-fixed shares a nominal parameter b and displacement upper envelope, but its asymmetric shrinkage range and direction differ; it is not solely a tanh-shape change. No factorial grid of arbitrary combinations is added.

Repeated observable units are `(base seed, fitted fold, direction, relation, known entity)`. Use their mean exported margin to avoid first-gold-record representatives in the presence of known small numeric export drift, then compute all maps once per unit. Require exact agreement of projected original ADC, ungated and expanded-radius actions/outcomes with the original observation-level path before interpreting results. The new maps must not inspect gold, ranks, harm, winner labels or outcomes. Invalid feature cases remain guarded; the current reviewed rows must all be finite.

## Equal actual intervention counts

Compare ADC separately with each of six nonstatic variants. Start with each side's actual nonzero actions after its own declared gate and projection. In each `(seed, fold, observable repetition count)` stratum, retain `k=min(n_ADC_active_units,n_variant_active_units)` whole units independently and uniformly from each side's eligible pool. Every realization has exactly k nonzero units and the same observation count on both sides; do not require identical selected identities, signed directions or amplitudes. The static reference is the same a on rejection.

Report exact expected discrete-action outcomes using marginal probabilities k/n for active units, never ranking at an averaged alpha. Because intervention counts are fixed, conditional expected harm is expected harm count divided by the common intervention count. These are workload-conditioned thinning diagnostics, not new deployable policies. Pairwise budgets differ; record both original counts, common retained count and fractions retained for each side. If a stratum has no common eligible budget both sides contribute zero. A deterministic single random realization is used only to check integer quotas, not as a selected result. Preserve signed/magnitude caveats and existing Random-shape controls; equal count alone cannot isolate every mechanism.

## Outcomes and uncertainty

For all eight native policies report N, MRR, U versus original Global, actual intervention I, mean absolute movement D, harm over all and active observations, unconditional loss and gain; retain all 576 seed/direction cells. For ADC-minus-each-variant report native and matched differences, plus both sides' risk and movement metrics and retained counts. For matching all expectations are computed before bootstrapping; allocations/quotas are not rerun on the resampled workload.

Use 10,000 common original-triple percentile draws, carrying all six seed/direction observations. TEST PCG64 seeds 2026091212 (MKG-W) and 2026091213 (DB15K); DEV 2026091214 and 2026091215. Preserve sorted triple order and reconcile native TEST ADC-minus-Radius-1/No-gate with `query_pair_v1` points and intervals. Compute all twelve effect columns together per pair/split. Intervals condition on original models, folds and fixed workload allocations; no training/selection/new-workload uncertainty or multiplicity adjustment. A zero-inclusive interval is not equivalence. No method or configuration is selected using these results.

Completion: code/protocol/tests and frozen source hashes; original action/CI replay; all four component questions have explicit controls and an evidence table, including no advantage or adverse outcomes. No claim of universal component necessity, independent additive causal contributions, or unique tanh superiority is presupposed.
