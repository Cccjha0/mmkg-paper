# Introduction

Knowledge graph completion (KGC) aims to infer missing facts from observed triples. In multimodal knowledge graph completion (MMKGC), auxiliary modalities such as text and images can enrich entity representations beyond pure graph structure. In product-oriented knowledge graphs, this naturally creates the expectation that multimodal models should outperform structure-only alternatives.

Under incomplete visual support, however, multimodal usefulness is not uniformly available. This issue is especially important in OpenBG-IMG under the current paper protocol, where target position is entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. The protocol therefore induces a role--modality asymmetry rather than a neutral multimodal setting. Under such a protocol-shaped missing-modality setting, the core issue is no longer simply how to fuse more modalities, but whether multimodal evidence yields reliable gain at all, and if so, how that gain should be used under clean query-time constraints.

Our prior analysis already reveals a strong tension under the unified OpenBG-IMG protocol with filtered ranking, `direction=both`, and three-seed reporting. The strongest overall model is not the most complete multimodal architecture. Instead, `Residual-only` achieves the best global performance, while `Full Model` still retains value within the internal multimodal family. This means that multimodal gain exists, but it is uneven across the evaluation space rather than globally reliable.

These observations motivate the total problem addressed in this paper:

> Under the current OpenBG-IMG protocol, multimodal gain is not globally reliable. How can this bounded gain be understood more precisely and exploited more effectively under clean, deployable routing constraints?

Rather than treating this as a single undifferentiated question, we decompose it into three progressively narrower research questions.

**RQ1. Where does multimodal gain actually appear under the current OpenBG-IMG protocol?**  
This question asks whether multimodal benefit is globally uniform or instead concentrated in specific target-side regimes and relation conditions.

**RQ2. Why does naive clean routing fail, and what kind of clean decision structure is actually required?**  
This question asks whether activation can be decided by a single global threshold or a shallow modality-availability heuristic, or whether it instead requires a more structured policy under protocol-shaped asymmetry.

**RQ3. How should bounded multimodal gain be exploited effectively once it is known to be conditional?**  
This question asks what operational response is appropriate under heterogeneous gain: a naive global clean router, a more structured clean policy, or richer clean supervision aligned with gain magnitude.

This decomposition also determines the overall logic of the paper. RQ1 is diagnostic: it identifies where bounded multimodal gain appears. RQ2 is structural: it asks what kind of clean decision boundary is needed under asymmetric protocol conditions. RQ3 is operational: it turns bounded gain into a selective-activation mechanism that can be evaluated directly.

To address these questions, we move from always-on multimodal fusion to query-level selective activation between a fusion expert and a structural expert. We begin with a naive global-threshold clean routing formulation as a minimal deployable baseline. We then evaluate stronger clean formulations, especially direction-specific thresholding and more target-aligned supervision such as regression-style gain prediction. The method is not intended as a universally stronger multimodal architecture; rather, it operationalizes the bounded-gain finding under the current OpenBG-IMG setting and tests how much of that gain can actually be recovered under clean query-time constraints.

The experiments are organized accordingly. For RQ1, protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. For RQ2, the comparison between a naive global-threshold clean router and structured clean policies shows that the weakness of the original clean formulation comes not only from signal scarcity, but also from overly coarse policy granularity. For RQ3, target-aligned clean supervision further improves over the clean rule baseline, while Oracle routing still leaves visible headroom.

The strongest claim of the paper is therefore controlled rather than universal. We do not claim that multimodal fusion is globally superior to structural modeling in general. Nor do we claim that clean routing fully solves the bounded-gain problem under the current protocol. Instead, we argue that under the current OpenBG-IMG protocol, the weakness of clean routing is not solely caused by insufficient query-time observable signals. It also stems from overly coarse global thresholding and binary gain supervision. Once the clean policy is structured more appropriately and trained with a more target-aligned objective, deployable multimodal gain can be recovered more effectively, although a substantial gap to Oracle still remains.

In summary, this paper makes the following contributions:

1. **Protocol-aware bounded-gain diagnosis.** We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
2. **Negative result on naive clean routing.** We show that naive clean query-time routing based on a single global threshold is insufficient and can underperform a simple legal rule-based selector, indicating that deployable routing cannot be characterized adequately by a coarse global decision boundary.
3. **Positive result on structured clean routing.** We demonstrate that clean routing is not fundamentally exhausted: direction-specific thresholding and more target-aligned supervision, especially regression-style gain modeling, consistently outperform the clean rule baseline and recover additional deployable gains.
4. **Remaining deployable gap.** We further show that even these stronger clean strategies remain substantially below Oracle, revealing a persistent gap between deployable separability and oracle-level or post-hoc separability.
