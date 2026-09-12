# B17/B18: matched rejection controls and direct gate effects

Specified on 2026-09-12 before computing new control outcomes. This is a retrospective fixed-policy analysis of all six repaired pairs, not a new confirmatory test. No new fitting, scorer execution, TEST selection, direction reversal or best-random-seed reporting. Original OOF DEV policies and original TEST locks are retained.

## Query-consistent matched intervention

Replay the original margins, raw ungated projected proposals, full actions and RRs using the B15/B16 source identities. Raw here still includes clipping, the original radius, projection and endpoint convention. Only nonzero raw proposals are eligible for retention. A retained proposal keeps its existing weight; rejection returns to its existing anchor.

A rejection unit is the observable query `(seed, selector_fold, direction, relation, known_entity)`. TEST uses fold 0; DEV keeps original fold identities. All repetitions of the same observable query under the same fitted object share one decision. Never use target IDs, RR, labels or harm to set priorities, ties or budgets. The frequency of a query in the supplied workload is observable without its gold answers.

Match the full policy's retained-query count **within base seed × selector fold × observable-query repetition count**. Because every unit in a stratum has the same repetition count, every actual random realization and deterministic rule has exactly the same number of nonzero observation-level interventions as full ADC, including within each base seed/fold. This handles repeated queries without inconsistent gold-specific weights. Quotas come from full's gold-free actions, not outcomes. These are batch-level diagnostic controls conditional on the observed workload/full budget, not newly DEV-selected streaming policies or independent alternatives with separately selected budgets.

Freeze all four deterministic priorities and report all of them on DEV and TEST (no best rule selection):

- Head-first; tail-first: only prediction direction orders candidates.
- Low-A-gap: smaller unfiltered primary top-1/top-2 gap divided by its population score standard deviation.
- Gap-agree: larger normalized gap advantage for the expert favored by the sign of the proposed displacement. For upward moves use gap_A-gap_B; downward uses gap_B-gap_A.

Use gap/std with max(std,1e-12) in the denominator. Unit ties use a fixed pseudorandom ordering of canonically sorted observable keys (seed 20260927 + 2*pair_index + split_index). No gold/query-row ID participates. No new learner or hyperparameter search.

## Random controls and amplitude/direction confounding

Random-I retains a uniform random subset of whole query units of the fixed quota in each stratum. Report exact expected utility, harm frequency and mean loss: each unit has inclusion probability k/n, multiplying its original raw-action outcome; these are expectations of randomized gates, not the rank of a score mixed at an expected weight. All realizations have exactly full's intervention count.

Random-shape further stratifies by direction and **signed projected displacement in 0.05 steps**. Every realization additionally matches full's direction-by-displacement histogram, hence mean absolute movement, at each base seed/fold. This controls clipping, feasible direction and amplitude distribution together. Report how many eligible units lie in partially retained strata (0<k<n); when none do, the matched random gate equals full structurally. Because c and proposal magnitude are coupled, matching their distribution may remove nearly all freedom. Do not call this evidence of independent confidence information.

For each random control also retain the 2.5/97.5 percentiles of 2,000 random gate realizations on the observed workload (same fixed random seed rule, shape offset +100). These are randomization spread, not population confidence intervals. Never select a draw. Compute observed-weighted metrics using all original observations, including inactive proposals and rejected rows; matched I is not merely pre-clipping acceptance coverage. Check exact intervention budgets; for Random-shape check exact signed displacement/direction budgets as well.

## Paired uncertainty and B18

Report full-minus-each-control RR differences directly. For Random-I/shape compare full against the exact random-gate expectation, with randomization spread separately reported above. Report full-minus-no-fallback directly, holding margin model, anchor, beta, clipping and projection fixed and setting tau=0. No-fallback keeps raw zero proposals at the anchor. Preserve all 12 pair/split effects and 72 pair/split/seed/direction cells; if tau=0, full and no-fallback must agree exactly.

Use 2,000 original-triple bootstrap resamples, keeping all six seed/direction observations together, RNG 20260929 + 2*pair_index + split_index. Percentile 95% intervals condition on fitted states, OOF assignment, the fixed observed-workload control allocation and matched quotas. Do not refit/reallocate in the bootstrap; these intervals omit selection/training uncertainty and are not guarantees for a new workload or multiplicity-adjusted tests. Bootstrap the paired difference, not two separate intervals against Global.

Metrics: MRR and U=mean RR(action)-RR(anchor); I=actual nonzero weight fraction; D=mean absolute weight displacement; H=harm fraction (RR loss >1e-12); L=unconditional mean RR loss. For random rows use their exact expectations. Global remains the common zero-intervention reference. Unit/property tests cover exact count and shape matching, zero/full quotas, tau=0 identity, random expectation against enumeration, gold replacement and row-order invariance, and paired clustering.

Close the evidence gap honestly: distinguish a gate benefit beyond merely fewer interventions from any residual benefit after controlling amplitude and direction. If the latter is unsupported, narrow the mechanism claim. Preserve all favorable, null and adverse effects; local gate utility cannot establish universal necessity.
