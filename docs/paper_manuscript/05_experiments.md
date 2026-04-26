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

## 2. Main Clean Routing Evidence Table

The clean routing line should include a compact main table before the detailed RQ2/RQ3 discussion. This table is important because it turns the paper's central clean-routing claim from narrative text into auditable evidence.

| Method | Evaluation line | MRR | Delta vs Clean Rule | Delta vs Residual-only | Paired bootstrap evidence |
|---|---|---:|---:|---:|---|
| `Residual-only` | Clean routing line | 0.2930 | -0.0012 | -- | fixed structural expert |
| `Clean rule` | Clean routing line | 0.2943 | -- | +0.0012 | legal rule baseline |
| `Naive global clean router` | Clean routing line | 0.2939 | -0.0004 | +0.0008 | does not beat clean rule |
| `Direction-specific threshold` | Clean routing line | 0.2974 | +0.0032 | +0.0044 | vs clean rule: 95% CI [+0.002532, +0.003786] |
| `Regression-based clean router` | Clean routing line | 0.2982 | +0.0039 | +0.0051 | vs clean rule: 95% CI [+0.003345, +0.004442]; vs Residual-only: 95% CI [+0.004438, +0.005844] |
| `Oracle routing` | Oracle / post-hoc upper bound | 0.3337 | +0.0395 | +0.0407 | upper-bound reference only |

The table should be interpreted strictly on the clean routing line. The fixed expert values in this table are routing-compatible recomputed values, not rows copied from the official model-comparison table. Oracle routing is included only to show remaining headroom and should not be described as deployable.

This table supports three key claims:

1. naive global-threshold clean routing is weaker than the legal clean rule;
2. direction-specific thresholding recovers a clear and statistically supported clean gain;
3. regression-based gain prediction is the strongest clean strategy but still remains far below Oracle.

## 3. RQ1: Where Does Multimodal Gain Appear?

RQ1 asks whether multimodal gain is globally uniform or concentrated in protocol-shaped local regimes.

The current OpenBG-IMG `paper_split` is asymmetric. Head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. Therefore the meaningful target-side regimes are:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

`tail_has_img` should not be used as a meaningful final subgroup under the current test distribution.

The current manuscript's key RQ1 interpretation is:

> Multimodal gain is real but bounded. It is strongest in image-supported head-side regimes, but this local benefit is diluted at the global level because a large part of the evaluation space is structure-favorable and image-unavailable.

Relation-group evidence supports the same bounded-gain story. Relation context affects where multimodal evidence is useful, but coarse visual/non-visual grouping alone does not explain the whole result pattern. The paper should therefore avoid the oversimplified claim that all visually named relations automatically favor multimodal models.

## 4. RQ2: Why Does Naive Clean Routing Fail?

RQ2 asks whether deployable selective activation can be solved by a single global clean threshold.

The current answer is **no**. Naive clean query-time routing with one global threshold reaches `0.2939` MRR and does not outperform the simple legal clean rule baseline at `0.2943` MRR. This negative result is important because it shows that the weakness of clean routing is not explained solely by insufficient query-time observable signals. The original clean formulation is also too coarse for the asymmetric decision structure induced by the current protocol.

The current manuscript's key RQ2 conclusion is:

> The clean decision boundary is not homogeneous. Head-side and tail-side queries require different operating points, so a single global threshold is an inadequate policy structure.

## 5. Structured Clean Thresholding

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

Direction-specific thresholding raises clean routing performance to `0.2974` MRR. This improves over the clean rule baseline by approximately `+0.0032` MRR, with a paired bootstrap 95% CI of `[+0.002532, +0.003786]`. This result shows that policy granularity matters under the current split.

The safest wording is:

> Direction-specific thresholding demonstrates that clean routing is not fundamentally exhausted by the failure of the naive global-threshold router. The problem is not merely signal availability; it is also the granularity of the routing policy.

## 6. RQ3: How Should Bounded Gain Be Exploited?

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

Regression-based gain prediction reaches `0.2982` MRR, making it the strongest clean strategy. It improves over the clean rule by approximately `+0.0039` MRR, with 95% CI `[+0.003345, +0.004442]`. It also improves over `Residual-only` by approximately `+0.0051` MRR, with 95% CI `[+0.004438, +0.005844]`.

The safest wording is:

> Target-aligned clean supervision further strengthens deployable routing. The gain is still modest in absolute value, but it is consistent and statistically supported under the clean routing line.

## 7. Remaining Oracle Gap

Even the best clean strategy remains below Oracle routing. The strongest clean method reaches `0.2982` MRR, whereas Oracle routing reaches `0.3337` MRR. This leaves a remaining gap of approximately `0.0356` MRR.

This gap should not be described as a failure of the paper. It is part of the paper's controlled claim. It shows that the current clean features recover only part of the separability visible to an Oracle or post-hoc selector.

The correct interpretation is:

> Structured clean routing improves over naive clean routing and fixed clean baselines, but legal query-time information still exposes only part of the multimodal-gain boundary. The remaining Oracle gap marks unresolved deployable headroom.

## 8. Integrated Discussion

The combined evidence across RQ1--RQ3 supports the current paper's main argument:

1. The OpenBG-IMG `paper_split` creates role--modality asymmetry.
2. This asymmetry produces bounded, local multimodal gain rather than globally reliable multimodal superiority.
3. Naive global-threshold clean routing is too coarse to exploit this gain.
4. Direction-specific thresholding improves clean routing by matching the protocol's asymmetric structure.
5. Regression-based gain prediction improves further by using a more target-aligned supervision form.
6. A gap to Oracle remains, so the result should be framed as partial deployable recovery rather than complete solution.

## 9. Section-Level Takeaway

The current experiments section should communicate the following message:

> Under the current OpenBG-IMG protocol, multimodal gain is local and conditional. A naive clean router cannot exploit it reliably, but structured clean routing and target-aligned clean supervision recover additional deployable gain while still leaving a visible Oracle gap.
