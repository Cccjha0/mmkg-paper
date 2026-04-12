# Related Work Draft

## 1. Multimodal Knowledge Graph Completion

Knowledge graph completion has traditionally focused on learning from graph structure alone, where entity and relation embeddings are optimized to score candidate triples. With the growth of multimodal knowledge graphs, a large body of work has extended this setting by incorporating auxiliary entity information such as textual descriptions, visual features, and other modality-specific signals. The central motivation is that non-structural evidence can enrich entity representations and improve link prediction, especially when graph structure is sparse or semantically ambiguous.

Within MMKGC, prior work has shown that multimodal information can be useful in several ways. Text features can inject semantic context that is difficult to recover from graph topology alone, while image features can provide appearance cues that are especially relevant for visually grounded entities and relations. These methods collectively establish the core premise of MMKGC: external modalities can complement graph structure and, under suitable conditions, improve completion quality beyond what purely structural models can achieve.

However, much of the MMKGC literature is primarily oriented toward improving architecture design or demonstrating aggregate benchmark gains. As a result, the dominant question has often been how to incorporate more modalities more effectively, rather than under what conditions those modalities actually contribute useful signal. Our work builds on this literature but shifts emphasis from architecture-first improvement to protocol-aware diagnosis. Instead of asking whether multimodal information is useful in principle, we ask when it helps under incomplete visual support and why its gain may remain bounded.

## 2. Adaptive and Relation-Aware Fusion

A major line of MMKGC research studies how to combine structural and non-structural representations more effectively. Simple early-fusion strategies concatenate or aggregate modality-specific features before scoring, while more advanced methods use learned gates, attention, or relation-aware transformations to modulate the contribution of each modality. These designs are motivated by the observation that different relations may rely on different kinds of evidence, and that a fixed fusion rule is often too rigid for heterogeneous knowledge graphs.

Relation-aware and adaptive fusion approaches are particularly relevant to this work. Their underlying intuition is that multimodal usefulness is context-dependent: the same entity image may be highly informative for one relation and nearly irrelevant for another. This family of methods therefore attempts to condition fusion behavior on the predicted relation or query context, allowing the model to select or weight modalities dynamically rather than uniformly.

Our `Gate-only` and `Full Model` settings are conceptually related to this line of work, since both implement relation-sensitive multimodal interaction. However, the present paper is not primarily a proposal for a stronger relation-aware fusion mechanism. Instead, we use this model family as an analysis instrument. By comparing `Gate-only`, `Residual-only`, and `Full Model` under a unified protocol, we study not only whether relation-aware fusion helps, but also how much it helps relative to stronger structural compensation and under which conditions that help survives.

## 3. Missing Modality, Modality Imbalance, and Robustness

Another closely related research direction concerns robustness under incomplete or unreliable modalities. In many real-world settings, entities do not have uniformly available text or images, and even when multimodal features exist, their quality may vary substantially. This has motivated work on missing-modality learning, modality dropout, incomplete multimodal representation learning, and robust fusion under noisy inputs. The broader lesson from this literature is that multimodal systems should not assume perfectly balanced or fully available auxiliary information.

This perspective is especially important for product-oriented KGs, where image availability may be highly uneven across entity subsets or prediction directions. Under such conditions, the contribution of multimodal information is likely to depend not only on relation semantics, but also on where in the query the target entity appears and whether that target is actually image-supported. In other words, missing-modality effects are not merely a nuisance variable; they can fundamentally shape which modeling path becomes dominant.

Our work is aligned with this robustness-oriented perspective, but again differs in emphasis. Rather than proposing a new missing-modality training algorithm, we use the current OpenBG-IMG protocol as an empirical setting in which image availability is already asymmetric. This allows us to ask a more diagnostic question: how do modality availability, target position, relation type, and branch competition jointly determine whether multimodal gain appears?

## 4. Structural Baselines and the Persistence of Structure-Dominant Performance

Classical structural KGC models remain an important reference point for any MMKGC study. Even when multimodal information is available, strong structural baselines can remain highly competitive, especially if the graph already contains regularities that are easy to exploit. This is one reason that multimodal gains should not be evaluated only in isolation or only relative to weak baselines; they must also be judged against structure-heavy alternatives that may still dominate overall.

This issue is central to our findings. Under the current OpenBG-IMG protocol, `ComplEx` emerges as a competitive structural baseline, while `Residual-only` is the strongest overall model in our internal family. By contrast, `TuckER`, although a classical structural reference model, does not emerge as a competitive structural baseline under the current split and protocol. This distinction matters for writing and interpretation: the main empirical tension in our paper is not between `Full Model` and a weak reference system, but between local multimodal gains and globally stronger structure-dominant alternatives.

Our work therefore treats structural baselines not as incidental comparison points, but as part of the explanation itself. The paper does not merely ask whether multimodal fusion helps; it also asks why stronger structural compensation continues to dominate under missing-visual conditions.

## 5. Position of This Paper

The present work is related to all three strands above, but its contribution is different in kind. It is not primarily a new-fusion paper, because our central claim is not that a proposed multimodal architecture is globally strongest. It is not primarily a missing-modality method paper, because our goal is not to introduce a new robustness algorithm. And it is not simply a benchmark comparison paper, because the key contribution lies in explaining the pattern of gains and failures rather than only reporting model rankings.

Instead, this paper is best understood as a gain-boundary analysis of MMKGC under missing-visual conditions. We use a unified OpenBG-IMG protocol to show that multimodal gain is real but bounded, and we trace that boundary through four linked forms of evidence: subgroup analysis, relation-type analysis, behavior analysis, and case study. In this sense, our main contribution is diagnostic and explanatory: we identify when multimodal information helps, why it remains local, and why stronger structural compensation continues to dominate globally under the current protocol.
