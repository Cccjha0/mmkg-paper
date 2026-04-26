# Introduction

Knowledge graph completion (KGC) aims to infer missing facts from observed triples. In multimodal knowledge graph completion (MMKGC), auxiliary modalities such as text and images can enrich entity representations beyond pure graph structure. In product-oriented knowledge graphs, this naturally creates the expectation that multimodal models should outperform structure-only alternatives.

Under incomplete visual support, however, multimodal usefulness is not uniformly available. This issue is especially important in OpenBG-IMG under the current paper protocol, where target position is entangled with modality availability: head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable. The protocol therefore induces a role--modality asymmetry rather than a neutral multimodal setting. Under such a protocol-shaped missing-modality setting, the core issue is no longer simply how to fuse more modalities, but whether multimodal evidence yields reliable gain at all, and if so, how that gain should be activated under deployable routing constraints.

Our prior analysis reveals a strong empirical tension under the unified OpenBG-IMG protocol with filtered ranking, `direction=both`, and three-seed reporting. The strongest overall model is not the most complete multimodal architecture: `Residual-only` achieves the best global performance, while `Full Model` remains useful within the internal multimodal family. This indicates that multimodal gain exists, but it is uneven across the evaluation space rather than globally reliable.

These observations motivate the central problem of this paper:

> Under the current OpenBG-IMG protocol, multimodal gain is bounded and protocol-dependent. How can this gain be diagnosed more precisely and exploited more effectively under deployable selective-activation constraints?

We decompose this problem into four research questions.

**RQ1. Where does multimodal gain actually appear?**  
We first examine whether multimodal benefit is globally uniform or concentrated in specific target-side regimes and relation conditions.

**RQ2. Why does naive strict query-level clean routing fail?**  
We then test whether a single global metadata-only threshold is sufficient for strict clean selective activation, or whether the protocol requires a more structured query-level decision policy.

**RQ3. How far can structured query-level clean routing go?**  
We study whether direction-specific thresholding and target-aligned regression supervision can recover more gain than coarse binary query-level routing while remaining strict metadata-only clean.

**RQ4. Can score-aware candidate-level routing recover more deployable gain?**  
Finally, we test whether moving selective activation from the query level to the candidate level, and allowing non-answer-aware expert score signals, can recover substantially more of the Oracle headroom.

To address these questions, we proceed in two stages. Stage 1 studies strict metadata-only query-level routing between a fusion expert and a structural expert. It begins with a naive global-threshold router, then evaluates direction-specific thresholding and regression-based gain prediction. Stage 2 moves selective activation to the candidate level. Instead of making one hard expert decision for the whole query, the candidate-level router learns a soft mixing weight for each query-candidate pair using non-answer-aware expert score, confidence, and disagreement features.

The experiments follow this progression. Protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. Strict query-level clean routing provides statistically supported but modest improvements: the strongest regression-based clean router reaches `0.2982` MRR. The new score-aware candidate-level result is substantially stronger. `CA-S2` reaches `0.3142 ± 0.0010` MRR, improves over the strongest query-level clean router by `+0.0160` MRR, and recovers `51.9%` of the Oracle headroom.

The paper's claim is therefore controlled but stronger than a purely clean-routing claim. Under the current OpenBG-IMG protocol, strict query-level metadata-only routing is limited because its signals and decision granularity are too coarse. Score-aware candidate-level routing recovers more gain by using expert score patterns for sparse candidate-level correction. However, the strongest score-aware router still remains below Oracle routing, so the result should be understood as partial recovery of bounded multimodal gain rather than full closure of the Oracle gap.

In summary, this paper makes four linked contributions, each corresponding to a fine-grained problem, a proposed response, and an experimental validation:

| Fine-grained problem | Proposed response | Validation |
|---|---|---|
| The current protocol creates role--modality asymmetry. | Protocol-aware bounded-gain diagnosis. | Target-side subgroup and relation-group analysis. |
| Naive query-level clean routing is too coarse. | Direction-specific query-level clean routing. | Clean routing comparison and paired bootstrap evidence. |
| Binary query-level supervision is too coarse. | Regression-based clean gain prediction. | Best strict clean result: `0.2982` MRR. |
| Query-level metadata remains insufficient. | Score-aware candidate-level soft routing. | `0.3142` MRR, `+0.0160` over E5, and `51.9%` Oracle headroom recovery. |

More specifically, our contributions are:

1. **Protocol-aware bounded-gain diagnosis.** We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
2. **Structured strict query-level clean routing.** We show that naive clean query-time routing with a single global threshold is insufficient, and that direction-specific thresholding provides a stronger strict clean policy under the asymmetric protocol.
3. **Target-aligned strict clean supervision.** We show that regression-based gain prediction is the strongest strict metadata-only query-level clean policy, but that its gain remains modest relative to Oracle headroom.
4. **Score-aware candidate-level selective activation.** We introduce candidate-level soft expert weighting using non-answer-aware score-aware signals and show that it substantially outperforms strict query-level clean routing, with `CA-S2` reaching `0.3142 ± 0.0010` MRR and recovering `51.9%` of the Oracle headroom.

These contributions do not imply that score-aware candidate routing fully closes the gap to Oracle routing. Instead, the remaining Oracle gap is treated as a discussion and limitation result: strict query-level clean routing recovers a small but reliable amount of bounded gain, score-aware candidate-level routing recovers substantially more, and oracle-level separability still marks unresolved deployable headroom.
