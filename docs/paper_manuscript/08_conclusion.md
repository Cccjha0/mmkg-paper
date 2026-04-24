# Conclusion

This paper studies multimodal knowledge graph completion on OpenBG-IMG under a more precise question than the standard "does multimodal information help?" framing. Under the current protocol, the answer is not a simple yes or no. Multimodal evidence can help, but it is not globally reliable. Its usefulness depends on target position, image availability, relation context, and the competition between a fusion expert and a structurally stronger fallback expert.

The first contribution of the paper is diagnostic. The current OpenBG-IMG `paper_split` induces a clear role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This creates three meaningful target-side regimes--`head_has_img`, `head_no_img`, and `tail_no_img`--and explains why aggregate bidirectional evaluation does not represent a neutral multimodal setting. Under this protocol, multimodal gain is real but bounded: it is strongest in favorable image-supported head-side regimes, but remains limited at the global level.

The second contribution is a negative result on naive clean routing. A single global-threshold clean router does not outperform a simple legal clean rule baseline. This shows that the bottleneck is not only insufficient query-time observable signal. The original clean formulation is also too coarse for the asymmetric decision structure induced by the protocol.

The third contribution is a positive result on structured clean routing. Direction-specific thresholding improves clean routing performance to approximately `0.2974` MRR, showing that head-side and tail-side queries require different operating points. More target-aligned clean supervision further strengthens the result: regression-based gain prediction reaches approximately `0.2982` MRR and becomes the strongest clean strategy. It improves over both the clean rule and `Residual-only`, with paired bootstrap confidence intervals strictly above zero.

These improvements do not close the gap to Oracle routing. This remaining gap is important because it marks the difference between deployable clean separability and oracle-level or post-hoc separability. The paper therefore does not claim that clean routing fully solves the bounded-gain problem, nor that multimodal fusion is universally superior to structural modeling.

The final claim is deliberately controlled:

> Under the current OpenBG-IMG protocol, deployable multimodal gain can be recovered more effectively when the routing policy is structured appropriately and trained with a more target-aligned objective. However, the recovered gain remains partial, and the best clean strategy still leaves visible headroom to Oracle routing.

In this sense, the contribution of the paper is a protocol-aware transition from **bounded multimodal gain diagnosis** to **structured clean routing**. The results suggest that future MMKGC research should focus not only on how to fuse modalities, but also on when multimodal evidence should be activated under heterogeneous and incomplete modality conditions.
