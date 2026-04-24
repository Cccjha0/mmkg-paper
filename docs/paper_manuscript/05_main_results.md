# Retired Reference: Main Results

## Status

This file is an older draft from the previous analysis-oriented manuscript structure. It is now **reference-only**.

The current latest manuscript is:

- `docs/paper/manuscript_main.tex`

The active results narrative is now maintained in:

- `docs/paper_manuscript/05_experiments.md`
- `docs/paper/manuscript_main.tex`

## Current aligned result framing

The current paper should not be framed as a simple leaderboard paper. The main results are used to establish a bounded-gain and clean-routing argument:

1. The current OpenBG-IMG protocol creates role--modality asymmetry.
2. Multimodal gain is real but not globally reliable.
3. Naive single-threshold clean routing is too coarse.
4. Direction-specific thresholding improves clean routing to approximately `0.2974` MRR.
5. Regression-based gain prediction improves clean routing further to approximately `0.2982` MRR.
6. The best clean strategy improves over both the clean rule and `Residual-only`, but remains below Oracle routing.

## Do not use this old file as the current results section

Do not copy the previous seven-model ranking narrative directly into the current manuscript without checking the evaluation line. The current results section must separate:

- official model-comparison line;
- clean routing line;
- post-hoc analysis line.
