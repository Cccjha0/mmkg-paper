# E05/E06: absolute group metrics and bounded benchmark claims

Recorded 2026-09-13 before computing the new grouped summaries. This is a
retrospective descriptive audit of the fixed six pairs, not a new validation
sample, policy fit, parameter search or causal group analysis.

## E05: fixed partitions and support

Use the SHA-256-bound information_boundary_v2 complementarity summaries, which
already contain absolute Global/ADC MRR and exact support by original relation
and direction. DEV refers to original grouped held-out predictions with their
fold-specific references; TEST refers to final DEV-locked policies. Reconcile
weighted aggregates to every pair/split and to the later scope_boundary_v1
outcome audit. Do not rank, filter or select relations by TEST utility.

1. Show head and tail separately on all six pairs and both splits, including
   Global MRR, ADC MRR, their difference, original triple count T and observation
   count N. N=3T for a fixed direction and N=6T when pooling both directions.
   Original triple counts are not unique observable query keys and do not prove
   independence. All MRR values pool the three paired seeds with equal row weight.
2. Derive relation frequencies ONLY from the effective base-model TRAIN triples
   in the earlier canonical split ledger. DB15K uses 71,300 effective training
   triples, excluding the 7,922 DEV holdout from the source 79,222; MKG-W uses
   34,196. Verify train counts, distinct triples, row-index completeness and
   the canonical train hash. Preserve all relation IDs including train-zero IDs.
3. Before new outcome aggregation fix four bins: train-zero, 1--99, 100--999,
   and >=1,000 triples. They describe training support, not relation semantics.
   Keep empty groups with undefined conditional means and zero contribution.
   Apply the same mapping to every pair, split, seed and direction of a dataset.
4. Independently choose the five relations with largest effective TRAIN counts,
   breaking count ties by ascending canonical relation ID. Assign ranks R1--R5
   in that order, and a disjoint Other group. Freeze the relation map before
   reading the outcome summaries. Report both signs and every selected relation,
   plus Other; no replacement if a result is sparse, zero or negative. Retain
   every single relation and relation x direction in the supplementary CSV.
5. Every group reports T, N, Global and ADC absolute MRR, delta, intervention,
   harm/benefit, mean gain/loss, and C_g=(N_g/N_all)*delta_g. Check sum C_g equals
   the pair delta before rounding. Quantify the TRAIN-top-five share of N,
   positive RR gain and RR loss separately; avoid percentages of near-zero net
   gain. Shares are descriptive and not an optimal concentration statistic.

No new confidence intervals or significance claims are introduced: the purpose
is to expose absolute difficulty, support and signed contribution. Previously
reported paired query intervals and seed variability retain their original
scope. Report all group definitions, effective relation counts and mapping IDs.
New synthetic tests cover train-only selection/tie rules, weighted aggregation,
negative-group preservation and empty-group denominators.

## E06: explicit scope decision

Choose the narrower claim: a retrospective two-benchmark study within the
reliable-primary regime and the six fixed architecture pairs. MKG-W is primary
retrospective evaluation; DB15K is secondary external evaluation with historical
TEST exposure; OpenBG-IMG is development/discovery only and is outside the
completed corrected rerun. These are not three independent confirmation sets.
The six pairs reuse two datasets and are not six independent benchmark samples.

Do not add a dataset or a held-out pair in this review, and do not claim broad
cross-benchmark robustness or independent replication from these results.
Stronger external validation remains an unperformed extension requiring a
traceable feature/policy/pair freeze before using an unexposed evaluation set.
The new group tables describe current heterogeneity and cannot substitute for
that validation. Keep the scope consistent in the abstract, setup, results,
discussion and conclusion. No server computation is needed for this choice.
