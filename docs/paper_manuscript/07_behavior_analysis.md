# Retired Reference: Behavior Analysis

## Status

This file is an older behavior-analysis draft from the previous analysis-oriented manuscript structure. It is now **reference-only**.

The current latest manuscript is:

- `docs/paper/manuscript_main.tex`

The current paper no longer maintains a standalone behavior-analysis chapter. Mechanism-level interpretation should be folded into `Experiments and Discussion` only when it directly supports the current routing argument.

## Reusable aligned idea

The following ideas may still be useful as background explanation:

- multimodal fusion can be active and locally useful;
- structural compensation remains globally important under the current protocol;
- weak or missing visual support makes structural fallback more reliable;
- this motivates selective activation rather than always-on fusion.

## Current routing-centered interpretation

In the current manuscript, behavior analysis should serve the routing story:

> Because multimodal usefulness is conditional, the system should decide when to activate the fusion expert and when to fall back to the structural expert.

The current paper's strongest quantitative claims should come from the clean routing line, especially:

- direction-specific thresholding: approximately `0.2974` MRR;
- regression-based gain prediction: approximately `0.2982` MRR;
- paired bootstrap evidence against the clean rule and `Residual-only`;
- remaining gap to Oracle routing.

## Do not use this old file as an active section

Do not restore this file as a standalone chapter unless the manuscript is deliberately expanded again.
