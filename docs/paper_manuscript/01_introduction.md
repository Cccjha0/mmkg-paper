# Introduction

Knowledge graph completion (KGC) aims to infer missing facts from observed triples. In multimodal knowledge graph completion (MMKGC), auxiliary modalities such as text and images can enrich entity representations beyond pure graph structure. In product-oriented knowledge graphs, this naturally creates the expectation that multimodal models should outperform structure-only alternatives.

Under incomplete visual support, however, multimodal usefulness is not uniformly available. This issue is especially important in OpenBG-IMG under the current paper protocol, where target position is entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. The protocol therefore induces a role--modality asymmetry rather than a neutral multimodal setting. Under such a protocol-shaped missing-modality setting, the core issue is no longer simply how to fuse more modalities, but whether multimodal evidence yields reliable gain at all, and if so, how that gain should be used.

Our prior analysis already reveals a strong tension under the unified OpenBG-IMG protocol with filtered ranking, `direction=both`, and three-seed reporting. The strongest overall model is not the most complete multimodal architecture. Instead, `Residual-only` achieves the best global performance, while `Full Model` still retains value within the internal multimodal family. This means that multimodal gain exists, but it is uneven across the evaluation space rather than globally reliable.

These observations motivate the total problem addressed in this paper:

> Under the current OpenBG-IMG protocol, multimodal gain is not globally reliable. How can this bounded gain be understood more precisely and exploited more effectively?

Rather than treating this as a single undifferentiated question, we decompose it into three progressively narrower research questions.

**RQ1. Where does multimodal gain actually appear under the current OpenBG-IMG protocol?**  
This question asks whether multimodal benefit is globally uniform or instead concentrated in specific target-side regimes and relation conditions.

**RQ2. When should multimodal evidence be activated rather than deferred to structural fallback?**  
This question asks whether activation can be decided by a simple modality-availability heuristic, or whether it requires a richer query-level decision boundary.

**RQ3. How should bounded multimodal gain be exploited effectively once it is known to be conditional?**  
This question asks what system-level response is appropriate under heterogeneous gain: always-on multimodal enhancement, or selective activation between fusion and structural experts.

This decomposition also determines the overall logic of the paper. RQ1 is diagnostic: it identifies where bounded multimodal gain appears. RQ2 is decisional: it asks what signals determine whether fusion is worth activating for the current query. RQ3 is operational: it turns bounded gain into a selective-activation mechanism that can be evaluated directly.

To address these questions, we propose a lightweight gain-threshold routing framework that turns bounded multimodal gain into a query-level selective-activation problem between a fusion expert and a structural expert. Concretely, the framework routes each query between `Gate-only` as a fusion expert and `Residual-only` as a structural expert through hard-threshold selection. The method is not intended as a universally stronger multimodal architecture; rather, it operationalizes the bounded-gain finding under the current OpenBG-IMG setting.

The experiments are organized accordingly. For RQ1, protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. For RQ2, rule-based routing, feature ablation, threshold scanning, delta sensitivity, and interpretability analysis show that activation cannot be reduced to a simple image-availability shortcut. For RQ3, the routing-compatible comparison shows that selective activation improves over fixed-expert baselines and heuristic routing, while Oracle routing still leaves visible headroom.

The strongest claim of the paper is therefore controlled rather than universal. We do not claim that multimodal fusion is globally superior to structural modeling in general. Instead, we argue that under the current OpenBG-IMG protocol, multimodal gain is conditional, and that this bounded gain can be used more effectively through selective activation than through fixed, always-on multimodal use.

In summary, this paper makes the following contributions:

1. **Protocol-aware bounded-gain diagnosis.** We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
2. **Query-level gain decision formulation.** We reformulate bounded multimodal gain as a query-level gain prediction problem and show that deciding whether fusion should be activated requires more than a simple image-availability heuristic.
3. **Selective-activation routing framework.** We propose and validate a dual-expert gain-threshold routing framework that converts bounded gain into effective multimodal use under a routing-compatible evaluation basis.
