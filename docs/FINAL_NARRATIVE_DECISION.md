# Final Narrative Decision

## 1. Purpose

This document closes Section 9 of the experiment execution plan and records the final paper-facing narrative choice for the current OpenBG-IMG study.

The key question is no longer whether the project has enough experiments. It is whether the completed evidence supports:

- a method-improvement story centered on making the `Full Model` globally strongest
- or an analysis-driven story centered on when multimodal gain is real, and why it remains bounded

## 2. Final Decision

The paper should formally adopt the following core narrative:

**multimodal gain in OpenBG-IMG is real but bounded; the central contribution of this work is to explain when multimodal information helps and why it does not become globally dominant under the current protocol**

This corresponds to Section `9.2` in `docs/EXPERIMENT_EXECUTION_TODO.md`.

## 3. Why `9.2` Is The Right Path

The current evidence consistently matches the `9.2` condition:

- `Full Model` is not globally stronger than `Residual-only`
- but multimodal gain is still visible in meaningful local regimes

This means the project should not be framed as:

- "the multimodal full model is globally best"

and also should not be framed as:

- "multimodal information is basically useless on this dataset"

The evidence instead supports a third position:

- multimodal information contributes real local improvements
- but the current task remains globally dominated by stronger structural compensation

## 4. Evidence Chain

### 4.1 Main Results

The main-result ranking remains:

1. `Residual-only`
2. `ComplEx`
3. `Full Model`
4. `Gate-only`
5. `Early Fusion`
6. `Text-only`
7. `TuckER`

Implication:

- `Full Model` improves over simpler multimodal baselines
- but does not overturn the stronger structure-heavy path

### 4.2 `has_img / no_img`

The strongest local multimodal-favorable regime appears in `head_has_img`.

At the same time:

- overall performance is still pulled back by the structurally stronger `tail_noimg` setting

Implication:

- multimodal gain is sensitive to target position and modality availability

### 4.3 Relation Type

Relation-type analysis shows:

- grouped results still favor `Residual-only`
- `Full Model` beats `Gate-only` on most medium-support and high-support relations
- `Full Model` only beats `Residual-only` on a minority of retained relations

Implication:

- multimodal gain is broad enough to be real
- but not strong enough to define the whole task globally

### 4.4 Behavior Analysis

Behavior analysis shows:

- gate is relation-aware, but not relation-dominant
- residual contribution increases under weaker image support
- the final branch relation is best described as `residual-dominant asymmetric complementarity`

Implication:

- fusion is useful
- but the model still trusts residual structure compensation more in the final decision path

### 4.5 Case Analysis

Case analysis now gives sample-level confirmation:

- successful multimodal cases concentrate in `head + has_img`
- strong structure-favorable failures concentrate in `tail + no_img`

Implication:

- the gain boundary is not only a metric-level phenomenon
- it is also visible at the individual-query level

## 5. Why `9.1` Is Not The Current Narrative

Section `9.1` would require a story in which:

- `Full Model` is stably better than `Residual-only` on key subsets
- behavior analysis supports a more optimistic complementarity story

The second half is partly true:

- fusion and residual are indeed complementary

But the first half is not true strongly enough:

- `Full Model` still does not beat `Residual-only` broadly enough to justify a method-improvement-led framing

So `9.1` is not the best fit for the current paper.

## 6. Why `9.3` Is Also Not The Current Narrative

Section `9.3` would require a story in which:

- `Residual-only` dominates almost everything
- multimodal local gain is also not obvious

That is also not what the evidence shows.

We already have:

- favorable `head_has_img` subgroup behavior
- broad `Full Model > Gate-only` relation-level gains
- multiple convincing success cases

So the current result is not "multimodal gain is absent."
It is "multimodal gain exists, but under clear boundary conditions."

## 7. Final Paper Position

The final paper position should therefore be:

- analysis-driven as the primary identity
- method-improvement as a secondary component

The strongest paper-facing question is:

- when does multimodal information help MMKGC under missing-visual conditions, and why does that help remain bounded?

This is more faithful to the evidence than asking:

- how to make the multimodal full model globally win

## 8. Recommended One-Paragraph Narrative

Recommended concise narrative for the paper introduction or conclusion:

> Under the current OpenBG-IMG protocol, multimodal gain is real but not globally dominant. The `Full Model` consistently improves over simpler multimodal baselines and achieves stable local gains in favorable regimes, especially when the prediction target is image-available and appears on the head side. However, grouped relation analysis, behavior analysis, and case studies all show that the task remains globally dominated by stronger structural compensation, particularly in `tail_noimg` settings. We therefore frame the contribution of this work not as a globally superior multimodal architecture, but as a systematic analysis of the boundary conditions under which multimodal information helps knowledge graph completion.

## 9. Outcome For The Todo

The execution plan can now record the following:

- Section `9.2` is the formally adopted path
- the paper narrative is finalized as a gain-boundary analysis
- the minimum pre-writing checklist is complete
