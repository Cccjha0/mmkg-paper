# Experiments and Discussion

## 1. Experimental Setup

All experiments are conducted on the current OpenBG-IMG `paper_split` under the unified protocol defined in the previous section. Base-model checkpoints are selected on the development split, and final paper-facing base-model metrics are reported on the test split using filtered ranking with `direction=both`. Unless otherwise stated, official main-result metrics are aggregated over three random seeds and reported as `mean ± std` when the source line supports that form.

This chapter is organized around three research questions. **RQ1** asks where multimodal gain actually appears under the current protocol. **RQ2** asks why naive clean routing fails and what kind of clean decision structure is actually required. **RQ3** asks how bounded multimodal gain should be exploited effectively once it is known to be conditional.

To answer these questions cleanly, we distinguish among three complementary evaluation lines. The **official model-comparison line** is based on the aggregated `test_metrics.json` files from completed paper-stage runs and is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`. The **clean routing line** is based on query-level exported expert predictions and unified recomputation under legal query-time constraints, and is used for `Gate-only`, `Residual-only`, Oracle routing, the clean rule baseline, naive global-threshold clean routing, structured threshold policies, and target-aligned clean supervision. The **post-hoc analysis line** is retained only for stronger offline separability analysis and upper-bound-style comparisons; it is not part of the deployable clean claim.

## 2. RQ1: Where does multimodal gain appear?

RQ1 asks where multimodal gain actually appears under the current OpenBG-IMG protocol. Our starting point is that the benchmark is not a neutral multimodal setting. Under the current paper split, target position is intrinsically entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. As a result, the bidirectional test space is partitioned into three meaningful target-side regimes, namely `head_has_img`, `head_no_img`, and `tail_no_img`, rather than into two symmetric prediction directions alone. Table 1 and Figure 1 make this asymmetry explicit and show that `tail_no_img` alone accounts for half of all bidirectional test queries, which already suggests that aggregate performance will be strongly shaped by a globally large structure-favorable region.

The subgroup results confirm that multimodal gain is not uniformly distributed across this evaluation space. In `head_has_img`, the ordering becomes `Gate-only > Full Model > Residual-only`, which identifies the clearest multimodal-favorable regime under the current protocol. In contrast, `Residual-only` remains decisively strongest in `tail_no_img`, the regime that is both structurally favorable and globally dominant in support. Interpreting these subgroup outcomes together with the regime counts helps explain the broader pattern of the paper: multimodal benefit is visible and meaningful, but it is concentrated in a local subset of queries rather than spread uniformly across the benchmark.

Relation-group evidence provides a second, but supporting, view of this bounded-gain structure. Across the retained coarse relation groups, the grouped ordering remains structure-dominant overall, indicating that relation dependence is real but still bounded. In other words, relation characteristics do affect where multimodal benefit is more plausible, yet they do not overturn the broader structure-dominant pattern established by the official main results. The retained-support summary further keeps this analysis auditable by reporting retained relation and query support after minimum-support filtering, so that the grouped trends are not driven by a handful of low-support cases. Taken together, these observations answer RQ1: under the current OpenBG-IMG protocol, multimodal gain appears in specific protocol-shaped local regions rather than as a globally reliable property of the benchmark.

## 3. RQ2: Why does naive clean routing fail, and what kind of clean decision structure is required?

RQ2 asks why multimodal evidence cannot be routed effectively by a naive global-threshold clean selector, and what kind of decision structure is actually needed under the current protocol.

### 3.1 Naive global-threshold clean routing is too coarse

A natural first attempt is a clean learned router with one global threshold `tau`. However, the clean routing results show that this formulation is too weak. On the clean routing line, the strongest fixed expert remains `Residual-only`, the clean rule baseline is slightly stronger still, and the naive global-threshold learned clean router does not reliably surpass that rule. This means that the main difficulty of deployable routing is not merely classifier capacity. Rather, it lies in the mismatch between a single global decision boundary and the strongly asymmetric gain structure induced by the protocol.

This result is important because it rules out an overly simple interpretation of clean routing. If the deployable decision problem could be reduced to one global threshold or one shallow modality shortcut, the original clean learned router should already have been sufficient. The fact that it is not already suggests that the decision structure is more heterogeneous than the naive formulation assumes.

### 3.2 Direction-specific thresholding provides the clearest structured clean gain

Once the clean policy is allowed to use direction-specific thresholds instead of one global threshold, performance increases substantially above the clean rule baseline. The strongest dual-threshold configuration reaches approximately `0.2974` MRR, improving over the clean rule (`0.2943`) by about `+0.00315`. This gain is not only numerically meaningful but also statistically clean: the bootstrap confidence interval for the comparison between the best direction-specific threshold policy and the clean rule is strictly positive, with mean delta MRR `+0.003154` and 95% CI `[+0.002532, +0.003786]`.

The strongest operating point is also highly interpretable. The best thresholds are strongly asymmetric, with a permissive head-side threshold and a conservative tail-side threshold, confirming that global thresholding is too coarse for the current protocol. Under the current split, head-side and tail-side queries do not operate under one homogeneous gain regime, so the routing policy should not be forced to do so either.

### 3.3 Supporting structured policies show that policy granularity matters more than naive heuristics

Additional structured-policy variants reinforce the same conclusion. Relation-prior bucket thresholding and the hybrid prior-first strategy both improve over the clean rule, but only modestly. Their role is therefore not to replace direction-specific thresholding as the strongest clean policy, but to show that policy granularity is genuinely part of the problem. Once the clean policy is allowed to adapt to structured protocol conditions rather than one global cutoff, performance systematically moves in the right direction.

By contrast, extremely conservative fallback policies do not recover the main gains. This negative result is also informative. It shows that the key issue is not simply to activate fusion less often, but to activate it under the right structural granularity.

Taken together, these results answer RQ2. The weakness of the original clean router does not mean that clean deployable routing is hopeless. Instead, it shows that the clean decision boundary is highly asymmetric and cannot be captured adequately by a single global threshold.

## 4. RQ3: How should bounded multimodal gain be exploited effectively?

RQ3 asks how bounded multimodal gain should be exploited once it is known to be conditional. The official model-comparison line establishes the motivating tension of the paper. Under the formal paper protocol, `Residual-only` is the strongest overall model, whereas `Full Model` remains the strongest multimodal variant within the internal family. This means that multimodal modeling is not ineffective, but its benefit is not strong enough to overturn structure-dominant performance at the global level. The benchmark therefore does not simply call for a larger always-on multimodal model. Instead, it presents a setting in which multimodal benefit is real but uneven and therefore unlikely to be handled optimally by one globally shared compromise representation.

### 4.1 Target-aligned clean supervision improves over both the clean rule and naive clean routing

The new clean routing results show that policy structure is only part of the story. Supervision form also matters. Replacing the coarse binary gain label with a more target-aligned objective yields a further improvement in clean routing. In particular, the strongest regression-based clean router reaches approximately `0.2982` MRR, outperforming both the clean rule and the direction-specific threshold policy.

This gain is statistically well supported. Relative to the clean rule, the best regression-based clean router achieves mean delta MRR `+0.003900` with 95% bootstrap CI `[+0.003345, +0.004442]`. Relative to `Residual-only`, it achieves mean delta MRR `+0.005133` with 95% bootstrap CI `[+0.004438, +0.005844]`. These results are strong enough to support the main clean claim of the paper: structured clean routing and target-aligned clean supervision do not merely tend to help, but consistently improve over the earlier clean baselines.

Ordinal gain modeling also improves over the binary baseline, although it remains below the regression formulation. This pattern is conceptually important. It suggests that preserving gain magnitude information is especially helpful under the current protocol, where the gain boundary is shallow, uneven, and strongly shaped by direction-specific asymmetry.

### 4.2 Calibration helps little; supervision redesign helps more

Probability calibration provides only limited gains relative to the stronger clean results above. This means that the weakness of the original clean router is not primarily a calibration issue. Likewise, delta sensitivity shows that the effectiveness of clean routing depends meaningfully on label definition: more permissive settings are easier, whereas `delta=0.01` remains substantially harder. This further supports the view that the earlier weak clean result was partly caused by the coarseness of the binary gain label itself rather than by query-time signal scarcity alone.

### 4.3 Remaining Oracle gap and the role of post-hoc analysis

Even after structured thresholding and stronger supervision, the best clean strategy remains substantially below Oracle routing. This means that the deployable gap is real, but its interpretation must be refined. The gap does not arise solely because clean query-time observable signals are useless. Rather, it persists because the recoverable decision boundary remains only partially visible under legal query-time information, even after policy and supervision are improved.

This also clarifies how the post-hoc analysis line should be interpreted. The post-hoc selector remains useful as an analysis tool because it reveals stronger offline separability than what can be realized in a deployable clean router. Once clean routing is strengthened, the remaining distance to post-hoc or Oracle performance becomes a more precise indicator of how much separability is still hidden from legal query-time observation.

The system-level conclusion is therefore no longer simply that bounded multimodal gain should be routed rather than always switched on. It is more specific: bounded gain should be exploited through **structured clean routing and more target-aligned clean supervision**, rather than through either always-on multimodal fusion or a naive global-threshold clean selector.

## 5. Integrated Discussion and Limitations

The combined evidence across RQ1–RQ3 supports a more refined conclusion than the earlier paper version. Under the current OpenBG-IMG protocol, multimodal gain is real but bounded, and a lightweight naive clean router is not sufficient to exploit it. However, this weakness should not be interpreted as evidence that deployable clean routing is impossible in principle. Instead, the new results show that the main bottleneck lies in the coarseness of policy granularity and supervision form. Once the clean policy is structured more appropriately and trained with a more target-aligned objective, deployable routing can recover additional gains beyond both the clean rule and the original global-threshold learned baseline.

At the same time, the current results do not show that multimodal fusion is globally superior to structural modeling in general, nor that clean routing fully solves the bounded-gain problem under the current protocol. The strongest clean strategies remain substantially below Oracle, and the resulting gains are still protocol-specific rather than universal. The findings therefore do not imply that the same thresholds, gain margins, or supervision targets should transfer unchanged to other benchmarks, data splits, or missing-modality patterns.

A further limitation is that some auxiliary supporting analyses remain less stable than the main MRR-based clean results. In particular, oracle-gap decomposition and some hit-based summary exports require further verification before they should be treated as final quantitative evidence. For this reason, the strongest claims of the present paper are grounded primarily in the protocol diagnosis, the clean routing comparison under MRR, and the updated bootstrap-supported comparisons for the strongest clean policies.

More broadly, however, the results suggest an important direction for future MMKGC research. In settings with incomplete or asymmetric visual support, deciding not only how to fuse modalities, but also how to structure the clean routing policy and how to define gain-aware supervision, may be just as important as designing a richer fusion operator.