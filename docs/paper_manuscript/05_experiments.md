# Experiments and Discussion

## 1. Experimental Setup

All experiments are conducted on the current OpenBG-IMG `paper_split` under the unified protocol defined in the method section. Base-model checkpoints are selected on the development split, and final base-model metrics are reported on the test split using filtered ranking with `direction=both`.

For the strongest clean-policy comparisons, the current manuscript also reports paired significance evidence on the clean routing line. Each comparison is defined on the same shared set of test queries and is evaluated both seed-wise and query-wise. The main significance evidence uses paired query-level bootstrap resampling over reciprocal-rank differences, so the reported confidence intervals directly measure paired delta-MRR under the same query set.

This section is organized around three research questions:

- **RQ1:** Where does multimodal gain actually appear under the current OpenBG-IMG protocol?
- **RQ2:** Why does naive clean routing fail, and what kind of clean decision structure is actually required?
- **RQ3:** How should bounded multimodal gain be exploited effectively once it is known to be conditional?

The section must preserve the manuscript's three evaluation lines:

1. **Official model-comparison line:** formal `test_metrics.json` outputs for the original seven-model family.
2. **Clean routing line:** query-level exported expert outcomes with recomputed aggregation under legal query-time constraints.
3. **Post-hoc analysis line:** target-aware or confidence-rich analysis for offline separability only, not deployable clean claims.

Rows from these lines should not be mixed as if they came from the same aggregation basis.

## 2. RQ1: Where Does Multimodal Gain Appear?

RQ1 asks whether multimodal gain is globally uniform or concentrated in protocol-shaped local regimes.

The current OpenBG-IMG `paper_split` is asymmetric. Head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. Therefore the meaningful target-side regimes are:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

`tail_has_img` should not be used as a meaningful final subgroup under the current test distribution.

The current manuscript's key RQ1 interpretation is:

> Multimodal gain is real but bounded. It is strongest in image-supported head-side regimes, but this local benefit is diluted at the global level because a large part of the evaluation space is structure-favorable and image-unavailable.

Relation-group evidence supports the same bounded-gain story. Relation context affects where multimodal evidence is useful, but coarse visual/non-visual grouping alone does not explain the whole result pattern. The paper should therefore avoid the oversimplified claim that all visually named relations automatically favor multimodal models.

## 3. RQ2: Why Does Naive Clean Routing Fail?

RQ2 asks whether deployable selective activation can be solved by a single global clean threshold.

The current answer is **no**. Naive clean query-time routing with one global threshold does not outperform the simple legal clean rule baseline. This negative result is important because it shows that the weakness of clean routing is not explained solely by insufficient query-time observable signals. The original clean formulation is also too coarse for the asymmetric decision structure induced by the current protocol.

The current manuscript's key RQ2 conclusion is:

> The clean decision boundary is not homogeneous. Head-side and tail-side queries require different operating points, so a single global threshold is an inadequate policy structure.

## 4. Structured Clean Thresholding

The strongest structured threshold result in the current manuscript is direction-specific thresholding.

Instead of using one threshold `tau`, the policy uses separate thresholds for head-prediction and tail-prediction queries:

\[
\alpha(q)=
\begin{cases}
\mathbf{1}[p(q)>\tau_{head}], & q \text{ is a head-prediction query}, \\
\mathbf{1}[p(q)>\tau_{tail}], & q \text{ is a tail-prediction query}.
\end{cases}
\]

This policy remains clean because query direction is known at inference time.

The current manuscript reports that direction-specific thresholding raises clean routing performance to approximately `0.2974` MRR. This improves over the clean rule baseline and shows that policy granularity matters under the current split.

The safest wording is:

> Direction-specific thresholding demonstrates that clean routing is not fundamentally exhausted by the failure of the naive global-threshold router. The problem is not merely signal availability; it is also the granularity of the routing policy.

## 5. RQ3: How Should Bounded Gain Be Exploited?

RQ3 asks how conditional multimodal gain should be operationalized once it is known to be uneven.

The current manuscript's answer is that structured policy design should be combined with more target-aligned clean supervision. The original binary gain label is too coarse because it collapses the magnitude of expert advantage into a hard class. Regression-based gain prediction preserves more information about the difference between the fusion expert and the structural expert.

For each query `q`, the clean routing line defines the reciprocal-rank gain:

\[
\Delta(q)=RR_f(q)-RR_s(q).
\]

The regression router predicts:

\[
\widehat{\Delta}(q) \approx \Delta(q),
\]

using only legal clean query-time features. At test time, the predicted gain is converted into a routing decision through a development-selected threshold.

The current manuscript reports that regression-based gain prediction reaches approximately `0.2982` MRR, making it the strongest clean strategy. It improves over both the clean rule and `Residual-only`, with paired bootstrap confidence intervals strictly above zero.

The safest wording is:

> Target-aligned clean supervision further strengthens deployable routing. The gain is still modest in absolute value, but it is consistent and statistically supported under the clean routing line.

## 6. Remaining Oracle Gap

Even the best clean strategy remains below Oracle routing.

This gap should not be described as a failure of the paper. It is part of the paper's controlled claim. It shows that the current clean features recover only part of the separability visible to an Oracle or post-hoc selector.

The correct interpretation is:

> Structured clean routing improves over naive clean routing and fixed clean baselines, but legal query-time information still exposes only part of the multimodal-gain boundary. The remaining Oracle gap marks unresolved deployable headroom.

## 7. Integrated Discussion

The combined evidence across RQ1--RQ3 supports the current paper's main argument:

1. The OpenBG-IMG `paper_split` creates role--modality asymmetry.
2. This asymmetry produces bounded, local multimodal gain rather than globally reliable multimodal superiority.
3. Naive global-threshold clean routing is too coarse to exploit this gain.
4. Direction-specific thresholding improves clean routing by matching the protocol's asymmetric structure.
5. Regression-based gain prediction improves further by using a more target-aligned supervision form.
6. A gap to Oracle remains, so the result should be framed as partial deployable recovery rather than complete solution.

## 8. Section-Level Takeaway

The current experiments section should communicate the following message:

> Under the current OpenBG-IMG protocol, multimodal gain is local and conditional. A naive clean router cannot exploit it reliably, but structured clean routing and target-aligned clean supervision recover additional deployable gain while still leaving a visible Oracle gap.
