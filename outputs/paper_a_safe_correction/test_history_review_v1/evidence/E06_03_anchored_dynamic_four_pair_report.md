# Anchored Dynamic: method and four-pair results draft

## Evidence boundary

MKG-W is the confirmatory dataset: the policy family was validated with grouped DEV cross-fitting, then refit and locked on full DEV before one TEST application. DB15K is reported as a secondary replication because its TEST split had been accessed by earlier score-ensemble experiments. No TEST outcome was used to alter the feature set, alpha grid, beta grid, fallback grid, model family, or pair-specific locked parameters.

## Methods draft

Each pair contains independently trained experts A (M-Hyper) and B (NativE or AdaMF-MAT). Candidate scores are normalized with a query-wise z-score before interpolation. A shared weight $\alpha_0$ is selected on DEV and serves as the static Global baseline. The dynamic policy uses 13 answer-agnostic score-geometry features: direction, each expert's top-1 score, top-5 mean, top-1/top-2 margin, score standard deviation, and four cross-expert differences. A balanced logistic regression predicts whether expert A has the larger reciprocal rank; tied training observations are excluded.

The applied mixture is $\alpha(q)=\operatorname{clip}(\alpha_0 + \beta \tanh(g(\phi(q))),0,1)$. Low-confidence or non-finite observations fall back to $\alpha_0$, and the continuous output is rounded to the nearest precomputed exact-ranking alpha in increments of 0.05. DEV evaluation uses five-fold grouped cross-fitting: all seeds and both prediction directions of an original triple remain in one fold. After the policy family passed this analysis, the model and pair-specific parameters were fitted on full DEV, serialized with hashes, and applied once to TEST.

## Main results

| Dataset | Expert B | DEV Δ QS | DEV Δ Anchored | TEST Δ QS | TEST Δ Anchored | TEST 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MKG-W | NativE | +0.006487 | +0.006617 | +0.006201 | +0.005872 | [+0.004708, +0.007035] |
| MKG-W | AdaMF-MAT | -0.007804 | +0.002382 | -0.007251 | +0.001984 | [+0.001370, +0.002598] |
| DB15K | NativE | -0.009461 | +0.001300 | -0.008946 | +0.001271 | [+0.000768, +0.001775] |
| DB15K | AdaMF-MAT | -0.026513 | +0.000450 | -0.026595 | +0.000714 | [+0.000339, +0.001089] |

Anchored Dynamic improves over Global alpha in 4/4 pairs, 12/12 pair-seed cells, and 8/8 pair-direction cells. All four paired original-triple 95% intervals exclude zero. Query-soft improves in only 1/4 TEST pairs.

On MKG-W with NativE, Query-soft and Anchored are statistically tied, while both improve over Global. With AdaMF-MAT, Query-soft degrades and Anchored remains positive. DB15K repeats this pattern for both expert pairs. The evidence therefore supports robustness across expert quality rather than a claim that Anchored is always the numerically best adaptive policy on every pair.

## Reporting guidance

The primary claim should be that anchoring converts pair-dependent dynamic behavior into stable improvements over a strong static mixture. Report the four pair-level effects and intervals rather than a row-count-weighted aggregate, because DB15K contains more queries and is a secondary replication. Keep Oracle as an answer-aware headroom diagnostic and Relation alpha as a secondary baseline. Query-soft is the no-anchor ablation.
