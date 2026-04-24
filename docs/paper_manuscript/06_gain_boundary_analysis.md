# Retired Reference: Gain-Boundary Analysis

## Status

This file is an older draft from the previous analysis-oriented manuscript structure. It is now **reference-only**.

The current latest manuscript is:

- `docs/paper/manuscript_main.tex`

The active bounded-gain narrative is now integrated into:

- Introduction;
- Task Setting and Method;
- Experiments and Discussion.

## Reusable aligned idea

The following idea remains central to the current paper:

> Under the current OpenBG-IMG protocol, multimodal gain is real but bounded. It is strongest in image-supported head-side regimes, but it does not become globally reliable because the evaluation space contains large structure-favorable and image-unavailable regimes.

## Current connection to routing

In the current manuscript, bounded-gain analysis is not the final contribution by itself. It motivates the clean routing method:

- bounded gain explains why always-on multimodal fusion is insufficient;
- naive clean routing fails because one global threshold is too coarse;
- structured clean routing and regression-based gain prediction recover more deployable gain.

## Do not use this old file as an active section

Do not restore this file as an independent Section 6 unless the paper is explicitly expanded again. In the current five-part TeX structure, its content belongs inside the RQ1 discussion of `Experiments and Discussion`.
