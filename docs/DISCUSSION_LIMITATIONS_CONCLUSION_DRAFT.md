# Discussion, Limitations, and Conclusion Draft

## 1. Discussion

### 1.1 Why multimodal gain is not globally dominant

The completed experiments show that multimodal information is useful under the current OpenBG-IMG protocol, but not uniformly useful. This point is crucial for interpreting the final ranking. The `Full Model` is clearly stronger than simpler multimodal baselines such as `Gate-only` and `Early Fusion`, which means that multimodal fusion is not failing in a trivial sense. At the same time, the strongest overall model remains `Residual-only`, and `ComplEx` also outperforms the `Full Model` globally. The right question is therefore not why multimodal information provides no value, but why its value is insufficient to become dominant at the task level.

The earlier analyses suggest that the answer lies in the interaction of three factors. First, multimodal gain depends on the target-side modality condition. The strongest favorable regime appears in `head_has_img`, whereas the current protocol is globally dominated by `tail_noimg`. Second, multimodal gain depends on relation context. The `Full Model` improves over `Gate-only` on most retained relations, but this local benefit does not scale into grouped-level dominance over `Residual-only`. Third, branch preference inside the `Full Model` remains residual-dominant, especially when image support is weak. Together, these observations explain why multimodal usefulness can be real without becoming the global driver of performance.

In this sense, the central challenge of MMKGC under missing-visual conditions is not merely to add more multimodal information or design more expressive fusion. It is to understand whether the prediction setting actually allows that information to matter. Under the current protocol, the strongest obstacle to global multimodal dominance is not the absence of local multimodal signal, but the fact that stronger structural compensation remains more reliable in the globally dominant regimes.

### 1.2 Why this is not evidence against multimodal modeling

It is important not to overinterpret the global dominance of `Residual-only` as evidence that multimodal modeling is unnecessary. Such a conclusion would be inconsistent with multiple parts of the empirical record. The subgroup results show that multimodal-favorable regimes exist. The relation-type analysis shows that `Full Model` broadly improves over `Gate-only` when relation support is sufficient. The behavior analysis shows that the fusion branch is active, adaptive, and nontrivial. Finally, the case study shows that some prediction instances are much better explained by multimodal reasoning than by pure structural compensation.

What the current study shows is therefore not that multimodal information fails, but that its contribution is bounded by protocol conditions. This is a more informative conclusion than either naïve optimism (“multimodal models should always win”) or naïve pessimism (“multimodal information is useless here”). The evidence supports a middle position: multimodal gain is real, but it appears in a favorable local regime rather than uniformly across the task.

This interpretation is also valuable for future method design. If multimodal usefulness is conditional rather than universal, then the next generation of MMKGC models should focus not only on richer fusion mechanisms, but also on identifying when multimodal information should be trusted, when structural fallback should dominate, and how these two paths should interact under uneven modality support.

### 1.3 Implications for future MMKGC research

The current findings suggest at least three broader implications for MMKGC research. First, future work should pay more attention to protocol-aware evaluation. The current paper shows that split design, target position, and modality availability can jointly shape the observed strength of multimodal methods. Without making these factors explicit, global metrics alone can be misleading.

Second, multimodal gains should be evaluated relative to strong structural competitors, not only relative to weak multimodal baselines. In our results, the most informative tension is not between `Full Model` and `Early Fusion`, but between local multimodal gains and globally stronger structure-heavy alternatives such as `Residual-only` and `ComplEx`. This suggests that future MMKGC studies should preserve strong structural comparison points rather than treating them as mere formal baselines.

Third, branch interaction deserves more attention as an object of study in its own right. Our results show that a multimodal model can learn meaningful fusion while still remaining residual-dominant overall. This means that branch competition is not just an implementation detail; it is part of the task-level explanation. Future work may benefit from modeling not only how to build better fusion, but also how to regulate branch preference under missing-modality conditions.

## 2. Limitations

### 2.1 Protocol dependence

The most important limitation of this study is that the core findings are protocol-aware rather than universal. The gain-boundary conclusion is derived under the current OpenBG-IMG `paper_split`, unified train/dev/test setup, and `direction=both` evaluation. Under a different split or a more symmetric modality distribution, the relative strength of multimodal fusion and structural compensation could change. Therefore, the present conclusions should be understood as an empirical diagnosis of the current OpenBG-IMG protocol, not as a universal law of MMKGC.

### 2.2 Target-position and modality asymmetry

A second limitation is the asymmetry between target position and image availability. In the current test distribution, head-side targets can still be image-supported, whereas tail-side targets are effectively `no_img`. This asymmetry is central to the paper's interpretation, but it also limits how broadly the findings can be generalized. For example, the conclusion that multimodal gain is strongest in `head_has_img` should not be read as a universal superiority of head-side prediction. It should be read as a finding specific to the present protocol, where target position and modality availability are entangled.

### 2.3 Baseline optimization scope

Another limitation concerns baseline optimization. All models are evaluated under a unified protocol, which is a strength for fairness and comparability, but it also means that not every classical baseline has been individually over-tuned for its own best-case performance. This issue is most relevant for models such as `TuckER`, which are included as classical structural references but do not emerge as competitive under the current protocol. Accordingly, the current `TuckER` result should be interpreted as a unified-protocol reference outcome rather than a definitive statement about its absolute upper bound.

### 2.4 Coarseness of relation grouping and case interpretation

The relation-type grouping used in this work is intentionally coarse and analysis-oriented. It is designed to test whether multimodal gain behaves differently across broad relation categories, not to provide a complete or unique semantic ontology of relations. Similarly, the case study is intended to validate the broader findings at the sample level rather than stand alone as the main evidence base. Future work could refine relation taxonomy and expand case reasoning with more fine-grained semantic annotation.

## 3. Conclusion

This paper studies multimodal knowledge graph completion on OpenBG-IMG under missing-visual conditions and asks a question that is more precise than the standard “does multimodal information help?” Under the current protocol, the answer is yes, but only under identifiable boundary conditions. The `Full Model` consistently improves over simpler multimodal baselines, showing that multimodal fusion is meaningful. At the same time, stronger structural compensation, especially in the form of `Residual-only`, remains globally dominant.

Through subgroup analysis, relation-type analysis, behavior analysis, and case study, we show that multimodal gain is local, conditional, and bounded. The strongest favorable regime appears when the prediction target is image-supported and lies on the head side. By contrast, the globally dominant unfavorable regime is characterized by `tail_noimg` prediction and stronger structural regularity. Behavior analysis further explains this pattern by showing that the fusion path is active and relation-aware, but final branch preference remains residual-dominant, especially when image support is weak.

The main contribution of this work is therefore not a claim of globally superior multimodal architecture. Instead, it is a systematic gain-boundary analysis of MMKGC under incomplete visual support. We identify when multimodal information helps, why those gains remain local, and why stronger structural compensation continues to dominate under the current OpenBG-IMG protocol. We hope this protocol-aware perspective encourages future MMKGC research to focus not only on stronger fusion designs, but also on the conditions under which multimodal information can genuinely matter.
