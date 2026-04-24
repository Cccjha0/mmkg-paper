# Introduction

Knowledge graph completion (KGC) aims to infer missing facts from observed triples. In multimodal knowledge graph completion (MMKGC), auxiliary modalities such as text and images can enrich entity representations beyond pure graph structure. In product-oriented knowledge graphs, this naturally creates the expectation that multimodal models should outperform structure-only alternatives.

Under incomplete visual support, however, multimodal usefulness is not uniformly available. This issue is especially important in OpenBG-IMG under the current paper protocol, where target position is entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. The protocol therefore induces a role--modality asymmetry rather than a neutral multimodal setting. Under such a protocol-shaped missing-modality setting, the core issue is no longer simply how to fuse more modalities, but whether multimodal evidence yields reliable gain at all, and if so, how that gain should be activated under clean query-time constraints.

Our prior analysis reveals a strong empirical tension under the unified OpenBG-IMG protocol with filtered ranking, `direction=both`, and three-seed reporting. The strongest overall model is not the most complete multimodal architecture: `Residual-only` achieves the best global performance, while `Full Model` remains useful within the internal multimodal family. This indicates that multimodal gain exists, but it is uneven across the evaluation space rather than globally reliable.

These observations motivate the central problem of this paper:

> Under the current OpenBG-IMG protocol, multimodal gain is bounded and protocol-dependent. How can this gain be diagnosed more precisely and exploited more effectively under clean, deployable routing constraints?

We decompose this problem into three research questions.

**RQ1. Where does multimodal gain actually appear?**  
We first examine whether multimodal benefit is globally uniform or concentrated in specific target-side regimes and relation conditions.

**RQ2. Why does naive clean routing fail?**  
We then test whether a single global threshold is sufficient for clean selective activation, or whether the protocol requires a more structured decision policy.

**RQ3. How should bounded gain be exploited once it is known to be conditional?**  
Finally, we study whether target-aligned clean supervision can recover more deployable gain than coarse binary routing.

To address these questions, we move from always-on multimodal fusion to query-level selective activation between a fusion expert and a structural expert. We begin with a naive global-threshold clean router as a minimal deployable baseline, then evaluate stronger clean formulations, including direction-specific thresholding and regression-style gain prediction. The method is not intended as a universally stronger multimodal architecture. Instead, it operationalizes the bounded-gain finding and tests how much of that gain can be recovered under legal query-time constraints.

The experiments follow the same progression. Protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. Clean routing experiments show that a naive global threshold is too coarse, while direction-specific thresholding improves over the clean rule baseline. Target-aligned regression supervision further provides the strongest clean result, although a visible gap to Oracle routing remains.

The paper's claim is therefore controlled: under the current OpenBG-IMG protocol, clean routing fails not only because legal query-time signals are limited, but also because a global threshold and binary gain labels are too coarse. Once the policy is structured more appropriately and trained with a more target-aligned objective, deployable multimodal gain can be recovered more effectively, while the remaining Oracle gap marks unresolved headroom.

In summary, this paper makes the following contributions:

1. **Protocol-aware bounded-gain diagnosis.** We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
2. **Negative result on naive clean routing.** We show that naive clean query-time routing based on a single global threshold is insufficient and can underperform a simple legal rule-based selector, indicating that deployable routing cannot be characterized adequately by a coarse global decision boundary.
3. **Positive result on structured clean routing.** We demonstrate that clean routing is not fundamentally exhausted: direction-specific thresholding and more target-aligned supervision, especially regression-style gain modeling, consistently outperform the clean rule baseline and recover additional deployable gains.
4. **Remaining deployable gap.** We further show that even these stronger clean strategies remain substantially below Oracle, revealing a persistent gap between deployable separability and oracle-level or post-hoc separability.
