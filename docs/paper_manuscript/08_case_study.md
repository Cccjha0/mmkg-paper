# Case Study

## 1. Motivation

The previous sections established the gain boundary through aggregate evidence. Subgroup analysis showed that multimodal gain is strongest in image-supported head-side regimes, relation-type analysis showed that such gain is broad enough to be real but not broad enough to overturn stronger structural alternatives, and behavior analysis explained why fusion remains useful while residual compensation still dominates the final preference. A case study is needed to make this boundary visible at the individual-query level.

The role of this section is therefore not to replace the earlier analyses with anecdotal examples, but to validate them with carefully selected representative cases. To avoid cherry-picking, the final case set is selected under fixed rules from the completed paper-stage runs. The selected success cases require consistent `Full Model > Residual-only` superiority across seeds, while the selected failure cases require consistent `Residual-only > Full Model` superiority across seeds. The final main-text cases are also chosen to be semantically interpretable and aligned with the broader subgroup and behavior evidence.

## 2. Success Cases: When Multimodal Modeling Helps

### 2.1 Shared pattern of the selected success cases

The refined success cases exhibit a highly consistent pattern. They are all head-side queries, and all of them have `target_has_img=True`. This already matches the earlier `has_img / no_img` analysis, which identified `head_has_img` as the strongest local regime for multimodal benefit. In other words, the case analysis does not introduce a new story; it confirms the subgroup finding at the sample level.

Among the final candidates, the most suitable main-text success cases are:

- `佩戴方式`
- `裙长`
- `细分风格`

These relations are not identical, but they share an important property: the correct prediction can plausibly benefit from appearance-aware evidence. This makes them particularly useful as representative examples of bounded multimodal gain.

### 2.2 `佩戴方式`

The `佩戴方式` case is the cleanest multimodal-favorable example in the final selection. The query is a head-side prediction with image-supported target semantics, and the relation itself is visually grounded in a direct and intuitive way. Under this condition, the `Full Model` ranks the correct entity much more effectively than `Residual-only`, indicating that image-sensitive fusion contributes information that structural compensation alone does not fully capture.

This case is important because it exemplifies the most favorable regime identified throughout the paper: a query in which the target side is image-supported and the relation can plausibly use appearance cues. It therefore serves as a strong demonstration that multimodal gain is not merely hypothetical under the current protocol. It is real and visible when the query conditions are favorable.

### 2.3 `裙长`

The `裙长` case provides another direct example of local visual usefulness. Here again, the relation is closely tied to visible appearance, and the target is image-supported. The `Full Model` substantially outperforms `Residual-only`, showing that the multimodal path can supply ranking signal that remains difficult to replace with structural regularity alone.

This case is valuable because it reinforces that the multimodal benefit is not limited to one unusually easy or isolated query. Instead, it appears across multiple appearance-oriented relations when the protocol conditions are favorable. At the same time, the gain remains local rather than global, which is exactly the pattern argued throughout the paper.

### 2.4 `细分风格`

The `细分风格` case is slightly more abstract than raw visual attributes such as length or wearing style, yet it still remains plausible as an appearance-supported relation. The `Full Model` again outperforms `Residual-only`, suggesting that multimodal usefulness is not confined to only the most literal low-level visual cues. Instead, it can also extend to locally interpretable, style-oriented distinctions when image support is available on the target side.

This case is especially useful for the paper because it broadens the success narrative. It shows that the favorable multimodal regime is not restricted to a single trivial visual relation family. Rather, the model can extract useful multimodal information from a somewhat wider but still locally perceptible semantic region.

### 2.5 Interpretation of the success cases

Taken together, the selected success cases support the following interpretation:

- multimodal gain is most convincing when the query is head-side
- the prediction target has image support
- the relation is appearance-related or otherwise locally perceptible

These cases therefore serve as sample-level evidence for the paper's broader gain-boundary claim. They do not prove that multimodal models should dominate globally. Instead, they show that when the right conditions align, the multimodal path can supply useful signal that structure-only alternatives do not fully recover.

## 3. Failure Cases: When Strong Structural Modeling Remains Better

### 3.1 Shared pattern of the selected failure cases

The refined failure cases are just as consistent as the success cases, but in the opposite direction. They are all tail-side queries, and all of them have `target_has_img=False`. This directly matches the earlier subgroup and residual-behavior findings, which showed that the current protocol is globally dominated by a `tail_noimg` regime and that residual preference becomes especially important when image support is weak or missing.

Among the final candidates, the most suitable main-text failure cases are:

- `适用场景`
- `材质`
- `净含量`

These relations collectively demonstrate that strong structural modeling remains more reliable when the prediction target is image-missing and the query is governed more by structural or metadata regularities than by local visual evidence.

### 3.2 `适用场景`

The `适用场景` case is the clearest structure-dominant failure in the final set. It is a tail-side prediction with an image-missing target, and the relation itself is not one that should be expected to benefit strongly from local appearance cues. Under this condition, `Residual-only` ranks the correct entity extremely well while `Full Model` fails badly.

This case is important because it gives a concrete instance of the global pattern already visible in the main results. The problem is not that `Full Model` is universally weak; the problem is that in strongly structure-favorable settings, the residual path provides a much more reliable signal than the multimodal route.

### 3.3 `材质`

The `材质` case is especially informative because it looks, at first glance, like a relation that might benefit from visual information. Yet under the current protocol it still becomes a structure-favorable failure case. The reason is that the actual query setting is `tail_noimg`, meaning that whatever superficial visual intuition the relation name carries, the prediction target itself lacks image support and the model is pushed toward structure-heavy inference.

This is one of the most useful cases for the paper because it prevents an oversimplified interpretation of the results. It shows that even visually suggestive relations do not automatically become multimodal-favorable once the actual prediction condition is dominated by image-missing targets and structural tail semantics.

### 3.4 `净含量`

The `净含量` case is a cleaner metadata-like example. Here the relation is more naturally governed by structured categorical or specification-style regularity than by visual cues. `Residual-only` again outperforms `Full Model`, reinforcing the idea that strong structural compensation remains globally advantageous in the current task when the prediction setting is not visually favorable.

This case complements `适用场景` and `材质` by showing that structure dominance is not limited to one semantic type. It appears both in clearly non-visual metadata relations and in relations that only appear visually plausible at a superficial level.

### 3.5 Interpretation of the failure cases

Taken together, the failure cases support the following interpretation:

- strong structural modeling is most favorable when the query is tail-side
- the prediction target has no image support
- the relation is governed more by structural regularity, metadata patterns, or tail vocabulary predictability than by local visual evidence

This pattern is fully consistent with the earlier subgroup and behavior analyses. The failure cases are therefore not random errors; they are representative manifestations of the structure-dominant side of the gain boundary.

## 4. Case-Level Boundary Summary

The case study supports a simple but important conclusion. Successful multimodal cases and structure-favorable failures occupy different parts of the protocol space. The success cases cluster in a favorable regime characterized by `head + has_img + locally perceptible relation cues`. The failure cases cluster in an unfavorable regime characterized by `tail + no_img + structure-heavy prediction conditions`.

This is precisely the sample-level form of the boundary identified in the earlier analyses. The case study therefore confirms that the gain-boundary narrative is not only a property of aggregated metrics or derived behavioral statistics. It is also visible in concrete prediction instances.

## 5. Role of This Section in the Paper

The paper should use this section carefully. Its purpose is not to dramatize isolated wins and losses, but to make the earlier conclusions tangible. The case study should therefore be written as supporting evidence for three already established claims:

- multimodal gain is real
- multimodal gain is local and conditional
- strong structural compensation remains globally dominant under the current protocol

When framed this way, the case study becomes a natural final validation step rather than a detached appendix of examples.

## 6. Section-Level Takeaway

The most important message of this section should be:

> At the individual-query level, successful multimodal cases concentrate in `head + has_img` settings with locally perceptible relations, while strong structure-favorable failures concentrate in `tail + no_img` settings. This confirms the gain boundary identified by the subgroup, relation-type, and behavior analyses.
