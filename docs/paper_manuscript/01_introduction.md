# Introduction

Knowledge graph completion (KGC) aims to infer missing facts from observed triples. In multimodal knowledge graph completion (MMKGC), auxiliary modalities such as text and images can enrich entity representations beyond pure graph structure. In product-oriented knowledge graphs, this naturally creates the expectation that multimodal models should outperform structure-only alternatives.

Under incomplete visual support, however, multimodal usefulness is not uniformly available. This issue is especially important in OpenBG-IMG under the current paper protocol, where target position is entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. The protocol therefore induces a role--modality asymmetry rather than a neutral multimodal setting. Under such a protocol-shaped missing-modality setting, the central question is no longer how to fuse more modalities, but when multimodal evidence is worth activating.

Our prior analysis shows that under the unified OpenBG-IMG protocol with filtered ranking, `direction=both`, and three-seed reporting, the strongest overall model is not the most complete multimodal architecture. Instead, `Residual-only` achieves the best global performance, while `Full Model` still retains value within the internal multimodal family. This indicates that multimodal gain exists, but it is uneven across the evaluation space rather than globally reliable.

These findings substantially narrow the problem. The empirical issue is no longer whether multimodal information can ever help, but whether it should be activated for the current query. If gain is conditional rather than uniformly reliable, then always-on fusion is no longer the most natural strategy, and multimodal activation itself becomes a query-level decision problem.

We therefore propose a lightweight gain-threshold routing framework that turns bounded multimodal gain into a query-level selective-activation problem between a fusion expert and a structural expert. Concretely, the framework routes each query between `Gate-only` as a fusion expert and `Residual-only` as a structural expert through hard-threshold selection. The method is not intended as a universally stronger multimodal architecture; rather, it operationalizes the bounded-gain finding under the current OpenBG-IMG setting.

Experiments support this reformulation from two complementary evaluation lines. Under the official model-comparison line, `Residual-only` remains the strongest fixed model overall, while `Full Model` is the strongest multimodal variant within the internal family. Under the unified query-level routing line, the best learned router improves over fixed-expert baselines and the rule-based router. Subgroup, threshold-scan, and ablation evidence further suggest that this gain cannot be reduced to a simple image-availability shortcut.

We therefore make a protocol-specific rather than universal claim: the contribution of this paper is not global multimodal superiority, but more effective use of multimodal evidence through selective activation under the current OpenBG-IMG setting.

In summary, this paper makes the following contributions:

1. We establish a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, causing multimodal benefit to remain local rather than globally uniform.
2. We propose a gain-threshold routing framework for selective multimodal activation that converts bounded multimodal gain into a query-level decision problem and routes each query between a fusion expert (`Gate-only`) and a structural expert (`Residual-only`) through hard-threshold selection.
3. We demonstrate that selective activation is effective and non-trivial under the unified query-level routing line: the best learned router improves over fixed-expert baselines, and ablation evidence shows that the gain cannot be reduced to a simple image-availability heuristic.
