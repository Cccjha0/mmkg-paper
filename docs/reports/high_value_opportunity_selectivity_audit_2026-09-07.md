# Frozen High-Value Opportunity Selectivity Audit

Date: 2026-09-07

## Scope and frozen interpretation boundary

This DEV-only audit asks whether the already-frozen Experiment 2 X4 strict-OOF score ranks the seed-stable and high-Oracle-gain opportunities diagnosed by Experiment 6. It trains no model, introduces no feature or representation, and cannot change the frozen closure route.

The primary ranking scalar is the predicted advantage of the action already chosen by the X4 probe. Unselected rows fall back to their fold-specific Global action. Coverage is selected within each dataset/expert pair, with lexicographic query-ID tie-breaking.

## Primary 10% coverage results

| Pair | Realized gain (population) | Selected utility | Stable-2 enrich X4/random | Stable-2 gain capture | High-gain enrich X4/random | High-gain recall | Negative transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper+NativE | +0.003126 [+0.002083, +0.004185] | +0.031259 | 0.99×/0.89× | 22.2% | 2.01×/1.61× | 20.1% | 25.3% |
| MKG-W / M-Hyper+AdaMF | +0.005405 [+0.004275, +0.006587] | +0.054039 | 1.45×/1.07× | 30.6% | 1.98×/1.37× | 19.8% | 26.1% |
| MKG-W / NativE+AdaMF | +0.000732 [-0.000074, +0.001528] | +0.007320 | 0.70×/0.61× | 17.9% | 1.11×/1.19× | 11.1% | 22.8% |
| DB15K / M-Hyper+NativE | +0.002012 [+0.001369, +0.002679] | +0.020117 | 1.42×/1.16× | 21.0% | 1.61×/1.37× | 16.1% | 27.0% |
| DB15K / M-Hyper+AdaMF | +0.001049 [+0.000556, +0.001558] | +0.010489 | 1.42×/1.16× | 18.2% | 1.34×/1.24× | 13.4% | 24.2% |
| DB15K / NativE+AdaMF | +0.001485 [+0.001025, +0.001962] | +0.014848 | 1.35×/1.11× | 19.7% | 1.10×/1.17× | 11.0% | 18.3% |

## Descriptive synthesis

At 10% coverage, frozen X4 has positive selective population gain in 6/6 pairs, with clustered CI lower above zero in 5/6. Stable-2 enrichment exceeds its matched random mean in 6/6 pairs, and high-gain enrichment exceeds matched random in 4/6 pairs.

| Coverage | Positive gain | CI lower > 0 | Stable-2 enrich > random mean | High-gain enrich > random mean | Median negative transfer |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 6/6 | 5/6 | 6/6 | 6/6 | 24.7% |
| 5% | 6/6 | 5/6 | 6/6 | 5/6 | 24.1% |
| 10% | 6/6 | 5/6 | 6/6 | 4/6 | 24.7% |
| 20% | 6/6 | 6/6 | 6/6 | 4/6 | 24.6% |
| 30% | 6/6 | 6/6 | 6/6 | 2/6 | 25.8% |

The matched-random comparison preserves the selected count within every seed × direction × relation stratum. Statements of `> random` compare with its Monte Carlo mean; they are descriptive and are not a newly introduced significance gate.

The score therefore carries useful ranking information, but it is not a clean opportunity detector: median negative transfer remains about one quarter across the coverage ladder, and high-gain enrichment over matched random weakens as coverage expands.

These comparisons are diagnostics of a previously frozen score, not a newly developed selective policy. They do not reopen selector development or alter `FINAL_SELECTIVE_RARE_OPPORTUNITY`.

## Figures

1. [Selective realized gain](../../outputs/complementarity_identifiability/part2_selectivity_audit/figure_p2_1_selective_gain.svg)
2. [Stable-opportunity enrichment](../../outputs/complementarity_identifiability/part2_selectivity_audit/figure_p2_2_stable2_enrichment.svg)
3. [High-gain enrichment](../../outputs/complementarity_identifiability/part2_selectivity_audit/figure_p2_3_high_gain_enrichment.svg)

## Machine-readable outputs

- `selectivity_per_query.csv.gz`: frozen row scores, exact realized utilities, stable/high-gain targets, and coverage indicators.
- `coverage_metrics.csv`: pair × coverage estimates, clustered intervals, and matched-random differences.
- `bootstrap_ci.csv`: long-form original-triple clustered intervals.
- `matched_random_baseline.csv`: long-form matched-random summaries.
- `audit_manifest.json`: complete source/output hash inventory and operational audit.

## Operational audit

- TEST access = 0
- new ranker / selector = 0
- new representation / feature = 0
- checkpoint retraining / reselection = 0
- policy tuning = 0
- closure route change = 0

Frozen closure route retained: `FINAL_SELECTIVE_RARE_OPPORTUNITY`.
