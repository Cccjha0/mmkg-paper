# Gain-Boundary Analysis Draft

## 1. Motivation

The main result table shows a clear but incomplete empirical picture: the `Full Model` is stronger than simpler multimodal baselines, yet it still fails to surpass the stronger structure-heavy alternatives under the unified OpenBG-IMG protocol. This immediately raises a more informative question than “which model ranks first overall.” If multimodal information is genuinely useful, where does that usefulness appear? And if it is not globally dominant, what constrains its effect?

In this section, we answer these questions through two complementary analyses. First, we study image availability at the target side through a protocol-aware `has_img / no_img` subgroup evaluation. Second, we examine whether multimodal gain depends on coarse relation categories through relation-type grouped analysis and relation-level comparisons with minimum-support filtering. Together, these results show that multimodal gain is real, but it is not uniformly distributed across the task. Instead, it is bounded by target position, modality availability, and relation characteristics.

## 2. `has_img / no_img` Analysis

### 2.1 Why subgroup analysis is needed

Under the current `paper_split`, target position and image availability are not symmetric. Head-side targets can still be image-available for a substantial portion of test queries, whereas tail-side targets are effectively always `no_img`. As a result, the overall metric mixes together multiple regimes with very different multimodal conditions. If multimodal gain depends on whether the target actually has image support, then this dependency will be diluted or even obscured at the aggregate level.

To make this effect visible, we analyze the test set using the following protocol-aware subgroups:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define `tail_has_img` because it does not meaningfully exist under the current test distribution. This point is important for interpretation: the subgroup analysis is not intended as a symmetric head-tail decomposition, but as a protocol-aware way to isolate where image-supported prediction is actually possible.

### 2.2 Main subgroup result

The subgroup results show that multimodal usefulness is not globally uniform. At the overall level, the model ordering remains:

- `Residual-only > Full Model > Gate-only`

However, this overall ordering changes once image availability at the target side is taken into account. In `head_has_img`, the ordering becomes:

- `Gate-only > Full Model > Residual-only`

By contrast, in `head_no_img`, the ordering becomes:

- `Full Model > Gate-only > Residual-only`

Meanwhile, the globally dominant `tail_no_img` setting continues to favor stronger structural modeling and pulls the overall metric away from the multimodal-favorable regime.

### 2.3 Interpretation

These results establish the first layer of the gain boundary. Multimodal gain is not absent, but it is clearly localized. The most favorable regime appears when the prediction target is on the head side and has image support, which is exactly the setting in which visual evidence can plausibly participate in ranking the correct entity. Once image availability is removed from the target side, the relative advantage of multimodal modeling becomes weaker and the structural path regains importance.

This means that target position should not be interpreted as a purely formal evaluation dimension. Under the current protocol, it is tightly coupled with modality availability and therefore acts as a practical boundary on multimodal usefulness. In this sense, the `has_img / no_img` analysis already shows that the global main-result ranking is masking a more local and conditional gain pattern.

## 3. Relation-Type Analysis

### 3.1 Why relation grouping matters

Image availability alone does not fully determine whether multimodal information should help. Even when the target entity has image support, the usefulness of multimodal evidence may depend on the semantics of the queried relation. Some relations are more directly tied to local appearance, while others are governed more by product metadata, structured categorical regularities, or tail vocabulary predictability. To examine this dimension, we group relations into coarse analysis-oriented categories:

- `visual_relations`
- `weak_visual_relations`
- `ambiguous_material_relations`

This grouping is not intended as a definitive ontology of relation semantics. Its role is analytical: it allows us to test whether multimodal gain is relation-dependent, and whether such dependence supports a global or only local advantage.

### 3.2 Group-level result

At the grouped level, the ordering remains stable across all three relation groups:

- `Residual-only > Full Model > Gate-only`

This finding is already informative. If multimodal usefulness were simply equivalent to “visual relations favor multimodal models,” we would expect the multimodal models to dominate at least the coarse `visual_relations` group. But this is not what we observe. `Residual-only` remains clearly stronger than `Full Model` even on the visual group, and `Full Model` in fact obtains a higher grouped MRR on `weak_visual_relations` than on `visual_relations`.

Therefore, the grouped analysis does not support a naive “visual relations as a whole are the multimodal win zone” story. Relation dependence is present, but it is not reducible to a coarse visual-versus-nonvisual dichotomy.

### 3.3 Relation-level analysis with minimum support

To reduce noise from extremely small-support relations, we interpret relation-level results using the `MIN20` protocol, which retains only relations with at least 20 test triples. Under this filtered analysis, a more nuanced pattern appears.

On most retained relations, `Full Model` outperforms `Gate-only`. This is true across all three relation groups and shows that residual-enhanced multimodal modeling provides broad local improvements over pure gate fusion once relation support is large enough to be meaningful. In this sense, multimodal gain is not limited to one or two accidental outliers.

At the same time, `Full Model` surpasses `Residual-only` on only a minority of retained relations. This means that local multimodal benefit is real, but it does not translate into relation-group-level or global dominance over the stronger structural compensation path.

### 3.4 Interpretation

The relation-type analysis establishes the second layer of the gain boundary. Multimodal gain is relation-dependent, but not in the simplistic sense that visually named relations are automatically favorable to multimodal models. Instead, the evidence supports a more careful statement:

- the `Full Model` improves over `Gate-only` on many medium-support and high-support relations
- but these gains remain insufficient to overturn the stronger structural path in most retained relations and at the grouped level

Accordingly, the relation-type evidence supports bounded local gain rather than global multimodal superiority.

## 4. Synthesizing the Boundary

Taken together, the `has_img / no_img` and relation-type analyses reveal a consistent pattern. Multimodal gain is strongest when the prediction target is actually image-supported and the local query context is favorable to appearance-aware ranking. It becomes weaker when target-side image support disappears, when relation cues are less locally perceptible, or when the prediction setting becomes dominated by structure-heavy tail regularities.

This leads to a protocol-aware synthesis of the gain boundary:

- multimodal gain is **local**, because it is concentrated in specific subgroups and retained relations rather than distributed uniformly across the task
- multimodal gain is **conditional**, because it depends on target position, image availability, and relation characteristics
- multimodal gain is **bounded**, because it improves over simpler multimodal baselines broadly enough to be real, yet still fails to displace stronger structure-heavy competitors overall

Importantly, this section does not argue that multimodal information is weak in principle. Instead, it argues that under the current OpenBG-IMG protocol, multimodal usefulness is constrained by the interaction between query direction, missing-visual structure, and relation-level variation.

## 5. Why This Section Matters for the Paper

This section is the core analytical center of the paper. The main result table alone could be read too pessimistically, as if multimodal modeling simply fails. The gain-boundary analysis shows that such a reading is incomplete. At the same time, the present section also prevents an overly optimistic interpretation in which a few favorable examples or local subgroup wins are mistaken for a global advantage.

The correct conclusion lies between these extremes. Under the current protocol, multimodal gain is clearly visible, but only under identifiable boundary conditions. This is why the paper should be framed not as a claim of globally superior multimodal architecture, but as a systematic analysis of when multimodal information helps and why that help remains bounded.

## 6. Section-Level Takeaway

The most important message of this section should be:

> Under the current OpenBG-IMG protocol, multimodal gain is real but bounded. Its strongest benefits appear in favorable local regimes, especially when the prediction target is image-supported, yet these benefits are not strong enough to overturn the globally dominant structural compensation path.
