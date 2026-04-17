# Limitations

## 1. Protocol Specificity

The most important limitation of this study is that its strongest claims are **protocol-aware rather than universal**. The central findings are derived under the current OpenBG-IMG `paper_split`, unified train/dev/test workflow, filtered ranking, and `direction=both` evaluation. Under a different split, a different modality distribution, or a more symmetric target-side condition, the relative strength of multimodal fusion, structural compensation, and selective routing could change.

This limitation is not incidental. The entire argument of the paper begins from the observation that the current protocol creates a meaningful but asymmetric missing-visual setting. As a result, the paper should not claim that bounded multimodal gain or gain-threshold routing has exactly the same form in all MMKGC settings. The strongest defensible statement is narrower: under the present OpenBG-IMG protocol, multimodal gain is conditional and query-dependent, and selective activation is an effective way to exploit it.

## 2. Entanglement Between Target Position and Modality Availability

A second limitation is the entanglement between target position and image availability. In the current test distribution, head-side targets can still be image-supported, whereas tail-side targets are effectively `no_img`. This asymmetry is analytically useful because it reveals the gain boundary clearly, but it also restricts generalization.

In particular, the conclusion that multimodal gain is strongest in `head_has_img` should not be read as a universal superiority of head-side prediction or as a claim that tail-side prediction is inherently structure-only in multimodal KGC. It is a finding tied to the present protocol, where target position and modality availability are coupled. A more symmetric benchmark could lead to a different distribution of local multimodal benefit.

## 3. Fixed-Expert Routing Scope

The proposed method is intentionally lightweight and controlled. We fix two experts—`Gate-only` and `Residual-only`—and learn a query-level router on top of them. This is a strength for interpretability, because it isolates the contribution of selective activation itself. However, it is also a limitation.

The current results do not show that the chosen expert pair is uniquely optimal, nor do they show that fixed-expert hard-threshold routing exhausts the design space. Other possibilities remain open, including:

- routing with different expert families
- routing that includes `ComplEx` or other structural references
- soft routing rather than hard thresholding
- end-to-end co-adaptive training of experts and router
- representation-level rather than query-level routing

The present paper does not attempt to cover these alternatives. Its goal is narrower: to show that bounded multimodal gain can already be operationalized effectively with a simple, interpretable routing framework.

## 4. Evaluation-Dialect Separation

Another limitation is that the paper necessarily relies on two complementary evaluation dialects.

The official seven-model comparison uses the paper-facing `test_metrics.json` aggregation line. The routing experiments use a unified query-level recomputed line, because Oracle, rule-based routing, and learned routing all depend on query-level exported expert outcomes. This separation is methodologically justified and necessary for fairness inside the routing experiments, but it also introduces additional complexity into presentation and interpretation.

Although the paper makes this distinction explicit, readers must still interpret some tables with care. In particular, the fixed experts in the routing comparison are not simply copied from the official main-result table; they are recomputed under the routing-compatible line. This is the correct choice for internal consistency, but it also means the paper cannot be read as if every number came from one single aggregation pipeline.

## 5. Coarseness of Relation-Group and Feature Design

The relation-group analysis in this paper is deliberately coarse and analysis-oriented. The groups such as `visual_relations`, `weak_visual_relations`, and `ambiguous_material_relations` are useful for testing whether multimodal gain varies with broad relation characteristics, but they do not define a complete semantic ontology of relations. Likewise, the router feature sets are designed to test protocol-aware selective activation in a controlled way, not to exhaust every possible feature family.

As a result, the current paper should not be interpreted as offering a final taxonomy of multimodal-favorable relation types or a complete characterization of all useful router features. Instead, it provides a first operational decomposition showing that target regime, relation priors, and expert-confidence signals already contain enough information to support effective selective activation under the current protocol.

## 6. No Claim of Universal Multimodal Superiority

Finally, the present work does not claim that multimodal fusion is globally superior to structural modeling in MMKGC. In fact, one of the paper's central empirical findings is the opposite: under the current protocol, strong structural fallback remains indispensable, and the strongest fixed model is still structure-heavy.

This is a limitation in the sense that the paper does not deliver a universally dominant multimodal architecture. But it is also part of the paper's honesty and positioning. The contribution is not to "win MMKGC" in the most general sense. It is to identify a bounded-gain regime clearly and then show that once this regime is treated as a routing problem, selective activation can outperform always-on multimodal use under the present setting.

## 7. Section-Level Takeaway

The main message of this section is:

> The conclusions of this paper are intentionally controlled. They are strongest as protocol-aware statements about OpenBG-IMG under the current split and evaluation setting. The proposed gain-threshold router demonstrates that bounded multimodal gain can be exploited effectively through selective activation, but it does not claim to be the final or universal solution to MMKGC under all conditions.
