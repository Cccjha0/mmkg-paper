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

In summary, this paper makes three linked contributions, each corresponding to a fine-grained problem, a proposed response, and an experimental validation:

| Fine-grained problem | Proposed response | Validation |
|---|---|---|
| The current protocol creates role--modality asymmetry. | Protocol-aware bounded-gain diagnosis. | Target-side subgroup and relation-group analysis. |
| Naive clean routing is too coarse. | Direction-specific structured thresholding. | Clean routing comparison and paired bootstrap evidence. |
| Binary gain supervision is too coarse. | Regression-based gain prediction. | Strongest clean MRR and paired delta-MRR confidence intervals. |

More specifically, our contributions are:

1. **Protocol-aware bounded-gain diagnosis.** We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
2. **Structured clean routing for selective activation.** We show that naive clean query-time routing with a single global threshold is insufficient, and that direction-specific thresholding provides a stronger clean policy under the asymmetric protocol.
3. **Target-aligned clean supervision.** We further show that regression-based gain prediction recovers the strongest deployable clean result, outperforming both the clean rule and `Residual-only` with positive paired bootstrap evidence.

These contributions do not imply that clean routing fully closes the gap to Oracle routing. Instead, the remaining Oracle gap is treated as a discussion and limitation result: structured clean routing recovers part of the bounded multimodal gain, while oracle-level separability still marks unresolved deployable headroom.
