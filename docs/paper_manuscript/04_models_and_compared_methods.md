# Retired Reference: Models and Compared Methods

## Status

This file is an older draft from the previous analysis-oriented manuscript structure. It is now **reference-only**.

The current latest manuscript is:

- `docs/paper/manuscript_main.tex`

In the current manuscript, model descriptions are not maintained as a standalone active chapter. The relevant content has been absorbed into:

- `Task Setting and Method`
- `Experiments and Discussion`

## Current aligned role of the model family

The current paper uses the model family mainly to support a protocol-aware routing argument:

- `Gate-only` is the fusion expert in the clean routing line.
- `Residual-only` is the structural expert in the clean routing line.
- `Full Model` remains useful as evidence that always-on multimodal fusion is not globally sufficient under the current protocol.
- `ComplEx` and `TuckER` remain part of the official model-comparison context, but they should not be mixed directly with clean routing rows unless the evaluation line is explicitly stated.

## Current framing to preserve

The model comparison should support the following claim:

> The paper is not claiming universal multimodal superiority. Instead, it shows that multimodal gain is conditional and that selective activation between a fusion expert and a structural expert can recover deployable gain under clean constraints.

## Do not use this old file as an active section

Do not copy this file into the current manuscript as Section 4. The current Section 4 is `Experiments and Discussion` in `docs/paper/manuscript_main.tex`.
