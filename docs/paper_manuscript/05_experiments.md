# Experiments and Discussion

## 1. Experimental Setup

All experiments are conducted on the current OpenBG-IMG `paper_split` under the unified protocol defined in the method section. Base-model checkpoints are selected on the development split, and final base-model metrics are reported on the test split using filtered ranking with `direction=both`.

The revised experiments are organized around four research questions:

- **RQ1:** Where does multimodal gain actually appear under the current OpenBG-IMG protocol?
- **RQ2:** Why does strict query-level clean routing remain limited?
- **RQ3:** Can score-aware candidate-level routing recover more deployable gain?
- **RQ4:** What explains the difference between clean candidate metadata and score-aware candidate routing?

The experiments preserve four evaluation lines:

1. **Official model-comparison line:** formal `test_metrics.json` outputs for the original base-model family.
2. **Strict query-level clean line:** metadata-only query-level routing under legal query-time features.
3. **Score-aware candidate-level line:** candidate-level routing using non-answer-aware expert score, confidence, and disagreement features after fixed expert scoring.
4. **Oracle / post-hoc line:** answer-aware upper bound only, not deployable.

Rows from these lines should not be mixed as if they came from the same information condition. In particular, the regression-based clean router is the strongest strict query-level clean result, whereas `CA-S2` is the strongest score-aware candidate-level result.

For significance testing, we report paired query-level bootstrap confidence intervals. Each compared pair is evaluated on the same shared test query instances, and reciprocal-rank differences are resampled with replacement to estimate the confidence interval of delta-MRR.

## 2. RQ1: Where Does Multimodal Gain Appear?

RQ1 asks whether multimodal gain is globally uniform or concentrated in protocol-shaped local regimes.

The current OpenBG-IMG `paper_split` is asymmetric. Head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. Therefore the meaningful target-side regimes are:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

`tail_has_img` should not be used as a meaningful final subgroup under the current test distribution.

The base-model and subgroup results support the bounded-gain diagnosis. Multimodal gain is real, but it is not globally reliable. Relative fusion advantage is most visible in image-supported head-side behavior, while global MRR remains dominated by the large structure-favorable `tail_no_img` region. This explains why always-on multimodal fusion does not automatically surpass the structural expert under the current protocol.

Relation-group evidence supports the same bounded-gain story. Relation context affects where multimodal evidence is useful, but coarse visual/non-visual grouping alone does not explain the whole result pattern. The paper should therefore avoid the oversimplified claim that all visually named relations automatically favor multimodal models.

The answer to RQ1 is therefore:

> Under the current OpenBG-IMG protocol, multimodal gain appears as a local and protocol-dependent effect rather than as a globally uniform advantage.

## 3. RQ2: Why Does Strict Query-level Clean Routing Remain Limited?

RQ2 evaluates strict metadata-only query-level routing. The key question is whether a deployable query-time router can recover bounded multimodal gain using only direction, relation-derived priors, and observed-side modality indicators.

The strict query-level clean results show that naive global-threshold routing is insufficient. A single global threshold reaches only `0.2939` MRR and does not outperform the legal clean rule baseline at `0.2943` MRR. This negative result is important: it shows that the difficulty is not only a lack of classifier capacity, but also the mismatch between one global decision boundary and the asymmetric gain structure induced by the protocol.

Direction-specific thresholding improves this result by assigning separate operating points to head-prediction and tail-prediction queries. It reaches `0.2974` MRR, improving over the clean rule by approximately `+0.0032` MRR with a strictly positive paired bootstrap interval. This shows that policy granularity matters.

Regression-based gain prediction further improves strict query-level clean routing to `0.2982` MRR. This is the strongest strict metadata-only query-level clean result. It improves over `Residual-only` by approximately `+0.0051` MRR and over the clean rule by approximately `+0.0039` MRR. The gain is statistically supported, but it remains modest relative to the Oracle upper bound.

The strict query-level clean routing evidence can be summarized as follows:

| Method | Routing level | Feature type | MRR | Interpretation |
|---|---|---|---:|---|
| `Residual-only` | fixed | structural | 0.2930 | fixed structural expert |
| `Clean rule` | query | strict clean | 0.2943 | conservative legal baseline |
| `Naive global clean router` | query | strict clean | 0.2939 | global threshold is too coarse |
| `Direction-specific threshold` | query | strict clean | 0.2974 | policy granularity helps |
| `Regression-based clean router` | query | strict clean | 0.2982 | strongest strict query-level clean result |
| `Oracle routing` | oracle | answer-aware | 0.3337 | post-hoc upper bound only |

The answer to RQ2 is therefore:

> Strict query-level clean routing is useful but limited. Metadata-only query features expose part of the multimodal-gain boundary, but not enough to recover most Oracle headroom.

## 4. RQ3: Can Score-aware Candidate-level Routing Recover More Gain?

RQ3 tests whether moving from query-level routing to candidate-level routing can recover more deployable gain. Unlike strict query-level clean routing, candidate-level routing predicts a soft fusion weight for each query-candidate pair. The strongest variant, `CA-S2`, uses non-answer-aware score-aware features derived from the fixed experts.

The main selective activation results are shown below.

| Method | Routing level | Feature type | Objective | MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs Residual | Delta vs E5 | Oracle gap recovered |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Residual-only` | fixed | structural | none | 0.2930 | 0.2328 | 0.3306 | 0.4031 | 0.0000 | -0.0051 | 0.0% |
| `Clean rule` | query | strict clean | rule | 0.2943 | 0.2332 | 0.3328 | 0.4052 | +0.0012 | -0.0039 | 3.0% |
| `Direction-specific threshold` | query | strict clean | threshold | 0.2974 | 0.2338 | 0.3342 | 0.4125 | +0.0044 | -0.0007 | 10.8% |
| `Regression-based clean router` | query | strict clean | delta-RR regression | 0.2982 | 0.2331 | 0.3361 | 0.4151 | +0.0051 | 0.0000 | 12.6% |
| `CA-S1 clean candidate` | candidate | clean candidate | pairwise ranking | 0.2503 ± 0.0054 | 0.1838 ± 0.0064 | 0.2864 ± 0.0052 | 0.3828 ± 0.0036 | -0.0427 | -0.0479 | -105.0% |
| `CA-S2 score-aware` | candidate | score-aware | pairwise ranking | **0.3142 ± 0.0010** | **0.2538 ± 0.0008** | **0.3603 ± 0.0016** | **0.4190 ± 0.0029** | **+0.0211** | **+0.0160** | **51.9%** |
| `CA-S3 clean + score` | candidate | clean + score-aware | pairwise ranking | 0.3135 ± 0.0010 | 0.2535 ± 0.0011 | 0.3584 ± 0.0021 | 0.4180 ± 0.0028 | +0.0204 | +0.0153 | 50.2% |
| `Oracle routing` | oracle | answer-aware | post-hoc max | 0.3337 | 0.2645 | 0.3775 | 0.4541 | +0.0407 | +0.0356 | 100.0% |

This table changes the main empirical message of the paper. The regression-based clean router remains the strongest strict query-level clean method, but it is no longer the strongest selective activation result. `CA-S2` reaches `0.3142 ± 0.0010` MRR, improving over `Residual-only` by `+0.0211` MRR and over E5 by `+0.0160` MRR. It recovers `51.9%` of the Oracle headroom, whereas E5 recovers only `12.6%`.

The answer to RQ3 is therefore:

> Yes. Score-aware candidate-level routing recovers substantially more deployable gain than strict query-level clean routing, because expert score patterns expose fine-grained candidate-level evidence that metadata-only query features cannot capture.

## 5. Significance Evidence

The paired bootstrap evidence supports the candidate-level result.

| Comparison | Baseline MRR | Candidate MRR | Delta MRR | 95% bootstrap CI | Paired queries |
|---|---:|---:|---:|---|---:|
| `CA-S1` vs `Residual-only` | 0.2930 | 0.2503 | -0.0427 | [-0.0452, -0.0404] | 60000 |
| `CA-S1` vs E5 regression clean router | 0.2982 | 0.2503 | -0.0479 | [-0.0503, -0.0455] | 60000 |
| `CA-S2` vs `Residual-only` | 0.2930 | 0.3142 | +0.0211 | [+0.0191, +0.0232] | 60000 |
| `CA-S2` vs E5 regression clean router | 0.2982 | 0.3142 | +0.0160 | [+0.0139, +0.0181] | 60000 |
| `CA-S3` vs `Residual-only` | 0.2930 | 0.3135 | +0.0204 | [+0.0184, +0.0226] | 60000 |
| `CA-S3` vs E5 regression clean router | 0.2982 | 0.3135 | +0.0153 | [+0.0132, +0.0174] | 60000 |
| `CA-S2` vs `CA-S3` | 0.3135 | 0.3142 | +0.0007 | [-0.0002, +0.0014] | 60000 |

The strongest supported claims are:

1. `CA-S2` significantly outperforms both `Residual-only` and E5.
2. `CA-S3` significantly outperforms both `Residual-only` and E5.
3. `CA-S2` does **not** significantly outperform `CA-S3`, because the confidence interval crosses zero.

Therefore the safe interpretation is:

> `CA-S2` is numerically the strongest and more parsimonious candidate-level router, while `CA-S3` performs similarly. Adding static clean candidate metadata to score-aware features does not provide a reliable additional gain.

## 6. RQ4: Clean Candidate Metadata vs Score-aware Candidate Features

RQ4 asks why candidate-level routing succeeds only for score-aware variants.

The feature ablation is clear. `CA-S1`, which uses clean candidate metadata alone, performs poorly overall. It reaches only `0.2503 ± 0.0054` MRR, far below `Residual-only` and E5. This negative result is important because it prevents an over-simple interpretation that candidate-level modality metadata alone solves the routing problem.

Subgroup results explain the failure more precisely.

| Method | Regime | Count/seed | Gate MRR | Residual MRR | Method MRR | Delta vs Residual |
|---|---|---:|---:|---:|---:|---:|
| `CA-S1 clean candidate` | `head_has_img` | 7048 | 0.0134 | 0.0017 ± 0.0001 | 0.0052 ± 0.0004 | +0.0034 ± 0.0003 |
| `CA-S1 clean candidate` | `head_no_img` | 2952 | 0.0123 | 0.0055 ± 0.0003 | 0.0180 ± 0.0013 | +0.0125 ± 0.0015 |
| `CA-S1 clean candidate` | `tail_no_img` | 10000 | 0.3945 | 0.5832 ± 0.0015 | 0.4917 ± 0.0103 | -0.0916 ± 0.0116 |
| `CA-S2 score-aware` | `head_has_img` | 7048 | 0.0134 | 0.0017 ± 0.0001 | 0.0032 ± 0.0002 | +0.0015 ± 0.0001 |
| `CA-S2 score-aware` | `head_no_img` | 2952 | 0.0123 | 0.0055 ± 0.0003 | 0.0081 ± 0.0004 | +0.0026 ± 0.0003 |
| `CA-S2 score-aware` | `tail_no_img` | 10000 | 0.3945 | 0.5832 ± 0.0015 | 0.6236 ± 0.0019 | +0.0404 ± 0.0025 |
| `CA-S3 clean + score` | `head_has_img` | 7048 | 0.0134 | 0.0017 ± 0.0001 | 0.0034 ± 0.0002 | +0.0017 ± 0.0001 |
| `CA-S3 clean + score` | `head_no_img` | 2952 | 0.0123 | 0.0055 ± 0.0003 | 0.0079 ± 0.0007 | +0.0024 ± 0.0004 |
| `CA-S3 clean + score` | `tail_no_img` | 10000 | 0.3945 | 0.5832 ± 0.0015 | 0.6222 ± 0.0016 | +0.0389 ± 0.0013 |

`CA-S1` helps some head-side regimes but fails badly in the dominant `tail_no_img` regime. Since `tail_no_img` accounts for half of all bidirectional test queries, this failure dominates the global metric. In contrast, `CA-S2` and `CA-S3` improve mainly in `tail_no_img`, where broad multimodal activation would normally be risky.

This shifts the interpretation of candidate-level routing. The main signal is not static image availability itself. The successful variants rely on expert score patterns, confidence, and disagreement. In other words, score-aware routing works because it observes how the two experts behave on candidate distributions, not merely whether a candidate has an image.

## 7. Behavior Analysis: Sparse Alpha and Regime-specific Correction

The alpha behavior provides the most direct explanation for why score-aware candidate routing works.

| Method | Mean alpha | Target alpha | Target/mean alpha | MRR gain vs Residual |
|---|---:|---:|---:|---:|
| `CA-S1 clean candidate` | 0.3811 ± 0.0000 | 0.2570 ± 0.0000 | 0.6744 | -0.0427 ± 0.0061 |
| `CA-S2 score-aware` | 0.0470 ± 0.0001 | 0.1964 ± 0.0084 | 4.1768 | +0.0211 ± 0.0012 |
| `CA-S3 clean + score` | 0.0480 ± 0.0004 | 0.2016 ± 0.0085 | 4.1997 | +0.0204 ± 0.0007 |

`CA-S2` and `CA-S3` assign low average fusion weight to most candidates, but much higher fusion weight to the target candidate in diagnostic analysis. This indicates sparse candidate-level correction rather than broad multimodal activation.

The tail-side behavior is especially revealing:

| Method | Regime | Mean alpha | Target alpha | Target/mean alpha | MRR gain vs Residual |
|---|---|---:|---:|---:|---:|
| `CA-S2 score-aware` | `tail_no_img` | 0.0054 ± 0.0001 | 0.3788 ± 0.0176 | 70.5009 | +0.0404 ± 0.0025 |
| `CA-S3 clean + score` | `tail_no_img` | 0.0070 ± 0.0008 | 0.3871 ± 0.0179 | 55.0089 | +0.0389 ± 0.0013 |

This pattern explains why score-aware candidate routing can improve the structure-dominant `tail_no_img` regime. The router does not broadly turn on fusion for tail-side candidate lists. Instead, it keeps fusion weight extremely low for most candidates while assigning much higher fusion weights to the correct target in evaluation diagnostics.

A necessary caveat is that target alpha is diagnostic only. It is computed after evaluation to interpret the learned behavior; it is not an input feature used by the router. The router itself uses non-answer-aware candidate-level features and expert score patterns.

## 8. Remaining Oracle Gap and Limitations

Despite the stronger candidate-level results, Oracle routing still reaches `0.3337` MRR, above `CA-S2` at `0.3142`. Therefore, `CA-S2` recovers `51.9%` of the Oracle headroom but does not close the Oracle gap.

This remaining gap is useful for interpretation. It shows that score-aware candidate-level routing exposes substantially more separability than strict query-level metadata-only routing, but still does not access all answer-aware information visible to Oracle selection.

The main limitations are:

1. **Protocol specificity.** The conclusions are tied to the current OpenBG-IMG `paper_split` and its role--modality asymmetry.
2. **Score-aware deployability cost.** `CA-S2` requires running both experts and computing candidate-level score features before routing.
3. **Not strict metadata-only clean.** `CA-S2` is deployable and non-answer-aware, but it is not a strict query-level clean router.
4. **Training/evaluation distinction.** Candidate-router training uses top-K hard candidate sets for efficiency, while final reporting uses full filtered ranking.
5. **Remaining Oracle gap.** Even the strongest candidate-level router remains below Oracle routing.
6. **Fixed expert pair.** The routing framework is evaluated only between `Gate-only` and `Residual-only` in the current paper.

## 9. Integrated Discussion

The combined evidence supports a revised paper story.

First, the OpenBG-IMG protocol creates role--modality asymmetry. This produces bounded multimodal gain rather than globally reliable multimodal superiority.

Second, strict query-level clean routing is useful but limited. Direction-specific thresholding and regression-based gain prediction provide statistically supported improvements, but their absolute gains remain modest.

Third, candidate metadata alone is not enough. `CA-S1` fails globally because it severely degrades the dominant `tail_no_img` regime.

Fourth, score-aware candidate-level routing is the strongest deployable selective activation strategy tested in this paper. `CA-S2` reaches `0.3142 ± 0.0010` MRR, significantly outperforming both `Residual-only` and E5, and recovering `51.9%` of the Oracle headroom.

Finally, alpha behavior suggests that the improvement comes from sparse candidate-level correction. The router does not globally activate multimodal fusion. Instead, it uses expert score patterns to assign higher fusion weight only to selected candidates, especially in the structure-dominant `tail_no_img` regime.

## 10. Section-Level Takeaway

The revised experiments section should communicate the following message:

> Under the current OpenBG-IMG protocol, multimodal gain is local and conditional. Strict query-level metadata-only routing recovers only modest gain. Score-aware candidate-level selective activation recovers substantially more Oracle headroom by using non-answer-aware expert score patterns for sparse candidate-level correction, while a clear remaining Oracle gap keeps the claim controlled.
