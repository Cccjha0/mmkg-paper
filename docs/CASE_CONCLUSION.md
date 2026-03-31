# Case Conclusion

## 1. Purpose

This document closes Section 8 by summarizing what the selected cases tell us about:

- which scenarios are more favorable to multimodal modeling
- which scenarios are more favorable to strong structural modeling

It is intended as the case-level bridge between the statistical findings in Sections 6-7 and the final paper-level judgment in Section 9.

Primary supporting files:

- `docs/CASE_ANALYSIS.md`
- `docs/CASE_ANALYSIS_INTERPRETATION.md`
- `docs/HAS_IMG_ANALYSIS.md`
- `docs/RELATION_TYPE_ANALYSIS.md`
- `docs/RELATION_AWARE_RESIDUAL_SUMMARY.md`

## 2. What The Success Cases Show

The refined success cases now share a clear pattern:

- they are all `head` queries
- they all have `target_has_img=True`
- the strongest main-text examples are concentrated in visually grounded relations

The clearest main-text success cases are:

- `佩戴方式`
- `裙长`
- `细分风格`

These cases indicate that multimodal gain is most convincing when all of the following are true:

- the prediction target is on the `head` side
- the target entity has image support
- the relation is plausibly supported by appearance cues

This aligns closely with the earlier `6.1` finding that the most favorable multimodal regime appears on the `head_has_img` side.

## 3. What The Failure Cases Show

The refined failure cases are just as consistent:

- they are all `tail` queries
- they all have `target_has_img=False`
- they include both clearly non-visual relations and relations that only look superficially visual

The clearest main-text failure cases are:

- `适用场景`
- `材质`
- `净含量`

These cases indicate that strong structural modeling is most favorable when:

- the prediction target is on the `tail` side
- the target entity lacks image support
- the relation is governed more by structural regularity, metadata patterns, or tail vocabulary predictability than by local visual evidence

This aligns with the earlier `6.1` and `7.2` findings that the global result remains dominated by the `tail_noimg` regime and by stronger residual preference under weak visual support.

## 4. Case-Level Boundary Summary

Taken together, the success and failure cases support a simple boundary pattern.

### 4.1 Scenarios More Favorable To Multimodal Modeling

Multimodal modeling is more favorable when:

- the query is `head`-side
- the target entity has image support
- the relation is appearance-related or at least locally perceptible
- the task can benefit from aligning image-backed product representation with relation-specific cues

This does not mean multimodal gain is universal within all visual relations.
It means the gain becomes most visible in local regimes where image evidence can plausibly participate in ranking the correct target.

### 4.2 Scenarios More Favorable To Strong Structural Modeling

Strong structural modeling is more favorable when:

- the query is `tail`-side
- the target entity has no image support
- the relation is dominated by scenario, material, metadata, or structured tail semantics
- the current split makes tail prediction especially structure-heavy

This also explains why even some visually named relations can still produce structure-dominant failures when the actual prediction setting is `tail_noimg`.

## 5. Final Section-8 Conclusion

The case study stage supports the following final conclusion:

- multimodal gain is real at the sample level
- but the gain is concentrated in a favorable local regime rather than distributed uniformly across the task
- strong structural modeling remains more robust in the globally dominant unfavorable regime

Therefore, the case evidence is fully consistent with the paper's gain-boundary narrative:

- multimodal modeling is best understood as locally helpful
- strong structural compensation remains globally dominant under the current OpenBG-IMG paper protocol

## 6. Paper-Facing Wording

Recommended short paragraph for the paper:

> The case study results mirror the broader subgroup and behavior analyses. Successful multimodal cases are concentrated in `head`-side queries whose target entities have image support, especially for visually grounded relations such as `佩戴方式`, `裙长`, and `细分风格`. In contrast, failure cases are concentrated in `tail`-side queries with image-missing targets, where structurally stronger prediction patterns dominate, as seen in relations such as `适用场景`, `材质`, and `净含量`. These sample-level observations reinforce the central conclusion of this work: multimodal gain is real but bounded, while strong structural compensation remains the dominant global driver under the current protocol.
