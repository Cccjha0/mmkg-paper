# Limitations

## Status

This file is now a **reference-only limitations note**.

The current latest manuscript is `docs/paper/manuscript_main.tex`. In the latest TeX structure, limitations are integrated into the `Experiments and Discussion` / concluding discussion rather than maintained as an independent active Chapter 7.

The content below is aligned with the current manuscript and may be reused when a separate limitations section is needed.

## 1. Protocol Specificity

The strongest claims of this study are protocol-aware rather than universal. The findings are derived under the current OpenBG-IMG `paper_split`, unified train/dev/test workflow, filtered ranking, and `direction=both` evaluation.

Under a different split, a different modality distribution, or a more symmetric target-side condition, the relative strength of multimodal fusion, structural compensation, and clean routing could change. The paper should therefore avoid claiming that the same bounded-gain structure or the same routing policy will transfer unchanged to all MMKGC settings.

The safest claim is:

> Under the current OpenBG-IMG protocol, multimodal gain is conditional and query-dependent, and structured clean routing can recover additional deployable gain beyond naive clean routing and fixed clean baselines.

## 2. Role--Modality Entanglement

The current test distribution entangles target position with modality availability. Head-side targets may still be image-supported, whereas tail-side targets are effectively image-unavailable.

This asymmetry is analytically useful because it exposes the gain boundary clearly, but it also restricts generalization. The conclusion that multimodal gain is strongest in `head_has_img` should not be interpreted as a universal superiority of head-side prediction. It is a finding tied to the present split and evaluation protocol.

## 3. Clean Feature Limitation

The main deployable routing claims are intentionally restricted to clean query-time features such as:

- direction;
- relation identity;
- development-derived relation priors;
- observed-side modality indicators.

This makes the claim deployable, but it also limits the amount of separability the router can recover. The remaining gap to Oracle routing indicates that some useful selection information is not visible under legal query-time constraints.

## 4. Fixed-Expert Routing Scope

The current routing framework uses a fixed fusion expert and a fixed structural expert:

- fusion expert: `Gate-only`;
- structural expert: `Residual-only`.

This design is useful for diagnosis and interpretability, but it does not exhaust the routing design space. Future work could explore:

- different expert pairs;
- including external structural baselines as routing candidates;
- soft routing instead of hard thresholding;
- end-to-end co-training of experts and router;
- representation-level routing instead of query-level expert selection.

The current paper should therefore present structured clean routing as a controlled and interpretable method, not as the final universal solution to MMKGC under missing visual support.

## 5. Evaluation-Line Separation

The paper relies on multiple evaluation lines:

1. official model-comparison line;
2. clean routing line;
3. post-hoc analysis line.

This separation is necessary because the official seven-model comparison and the routing comparisons answer different questions under different aggregation bases. However, it also increases presentation complexity.

Readers must not compare rows across these lines as if they came from one identical evaluation pipeline. This is especially important when interpreting fixed experts inside routing tables, which are recomputed on a routing-compatible query-level basis.

## 6. No Claim of Universal Multimodal Superiority

The present work does not claim that multimodal fusion is globally superior to structural modeling in MMKGC. In fact, one of the paper's central empirical observations is that strong structural fallback remains indispensable under the current protocol.

The contribution is narrower and more defensible:

> The paper identifies bounded multimodal gain under a specific protocol and shows that structured clean routing plus target-aligned supervision can exploit part of that gain more effectively than naive clean routing.

## 7. Section-Level Takeaway

The limitations should be summarized as follows:

> The conclusions are intentionally controlled. Structured clean routing improves deployable use of bounded multimodal gain under the current OpenBG-IMG protocol, but it remains protocol-specific, fixed-expert in scope, and below Oracle routing.
