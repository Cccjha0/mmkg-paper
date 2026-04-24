# Analysis and Discussion

## Status

This file is now a **reference-only analysis note**.

The current latest manuscript is `docs/paper/manuscript_main.tex`, whose active structure is:

1. Introduction
2. Related Work
3. Task Setting and Method
4. Experiments and Discussion
5. Conclusion

Therefore this file is no longer an independent active Section 6. Its useful discussion points have been absorbed into the current `Experiments and Discussion` section of the TeX manuscript and into the synchronized support file `05_experiments.md`.

## Current aligned interpretation

The current manuscript no longer uses the older routing claim that the best learned router reaches `0.3160` MRR with `XGBoost + delta=0.01 + tau=0.7` as the main result.

Instead, the current aligned result line is:

- naive single-threshold clean routing is insufficient and does not outperform the legal clean rule baseline;
- direction-specific thresholding reaches approximately `0.2974` MRR;
- regression-based gain prediction reaches approximately `0.2982` MRR and is the strongest clean strategy;
- the best clean strategy improves over both the clean rule and `Residual-only`, with paired bootstrap confidence intervals strictly above zero;
- the best clean strategy remains below Oracle routing.

## Discussion points that remain reusable

The following ideas from the older discussion remain compatible with the current paper, provided they are rewritten using the current result line:

1. **Bounded gain is a routing problem.**  
   The earlier bounded-gain diagnosis motivates selective activation because multimodal evidence is locally useful but not globally reliable.

2. **Always-on fusion is mismatched to heterogeneous gain.**  
   A single always-on multimodal representation cannot cleanly separate multimodal-favorable and structure-favorable query regions.

3. **Rule-like heuristics are insufficient.**  
   Simple clean rules help only up to a point. The current results show that structured clean policies and target-aligned supervision are needed for stronger deployable gains.

4. **Policy granularity matters.**  
   Direction-specific thresholding works better than a single global threshold because head-side and tail-side queries operate under different gain regimes.

5. **Supervision form matters.**  
   Regression-based gain prediction outperforms coarse binary gain supervision because it preserves more information about the magnitude of expert advantage.

6. **Oracle gap remains.**  
   The best clean strategy recovers only part of the available headroom, so the paper should avoid claiming that clean routing fully solves the problem.

## Points that should not be reused without revalidation

Do not reuse the following older statements as current paper claims:

- `XGBoost + delta=0.01 + tau=0.7` is the final best router.
- the best learned router reaches `0.3160` MRR.
- learned routing clearly beats the original always-on `Full Model` on a directly comparable single line, unless the exact evaluation basis is reintroduced and scoped.
- expert-confidence features are part of the current deployable clean-router claim, unless they are explicitly included in the latest clean-feature definition.

## Current section-level takeaway

The discussion should now be framed as follows:

> The success of the current method does not overturn the bounded-gain diagnosis; it refines it. Under the current OpenBG-IMG protocol, multimodal gain is real but conditional. Naive clean routing is too coarse, while direction-specific thresholding and regression-based gain prediction recover more deployable gain under clean constraints, though a visible Oracle gap remains.
