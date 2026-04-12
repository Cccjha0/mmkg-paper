# Paper Detailed Outline

## 1. Purpose

This document expands the current paper structure into a paragraph-level writing outline.

It is intended to answer a practical drafting question:

- if we start writing now, what should each section and paragraph do?

This outline is based on the finalized project narrative:

- multimodal gain in OpenBG-IMG is real but bounded
- the paper is analysis-driven first, method-improvement second

Primary narrative references:

- [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)
- [PAPER_WRITING_GUIDE.md](/E:/learn/R&D/mmkg-project-research/docs/PAPER_WRITING_GUIDE.md)

## 2. Title Direction

The title should reflect gain boundary rather than global model superiority.

Recommended title direction:

- emphasize missing-visual conditions
- emphasize boundary / conditions / when multimodal helps
- avoid claiming a universally better multimodal architecture

Possible title templates:

- `When Does Multimodal Information Help MMKGC Under Missing-Visual Conditions?`
- `Understanding the Boundary of Multimodal Gain in Product Knowledge Graph Completion`
- `Multimodal Gain Is Real but Bounded: A Study on OpenBG-IMG Under Missing-Visual Conditions`

## 3. Abstract

### Paragraph 1: Problem Setup

This paragraph should introduce:

- MMKGC as the task setting
- the practical problem of missing or uneven visual availability
- the mismatch between multimodal promise and real-world incomplete-modality conditions

Suggested content:

- Prior MMKGC work often assumes multimodal information is broadly beneficial.
- In realistic product KGs, image support is incomplete and uneven.
- This raises a more precise question: when does multimodal information actually help?

### Paragraph 2: What We Do

This paragraph should summarize:

- unified comparison of structural, fusion-only, residual-only, and full multimodal models
- subgroup analysis
- relation-type analysis
- behavior analysis
- case analysis

Suggested content:

- We evaluate a family of multimodal and structural models under a unified OpenBG-IMG protocol.
- We analyze gain boundary from subgroup, relation, mechanism, and case-study perspectives.

### Paragraph 3: Main Findings

This paragraph should report the most important findings:

- `Residual-only` is globally strongest
- `Full Model` improves over simpler multimodal baselines
- strongest multimodal-favorable regime appears in `head_has_img`
- grouped relation analysis does not show global multimodal dominance

Suggested content:

- The most complete multimodal model improves over gate-only fusion but does not surpass the stronger residual-based structural path overall.
- The clearest favorable regime appears when the target is image-available and predicted on the head side.

### Paragraph 4: Final Conclusion

This paragraph should close with the paper identity:

- multimodal gain is real
- multimodal gain is bounded
- current contribution is diagnostic rather than purely architectural

Suggested content:

- We conclude that multimodal gain is local, conditional, and bounded under the current protocol.
- The paper therefore contributes a gain-boundary analysis rather than a globally superior multimodal architecture.

## 4. Introduction

### Paragraph 1: Broad Motivation

Goal:

- motivate MMKGC broadly

Should mention:

- KGC benefits from richer entity information
- multimodal signals appear promising in product-oriented KGs

### Paragraph 2: Realistic Difficulty

Goal:

- introduce missing-modality realism

Should mention:

- image support is incomplete
- modality availability is uneven across prediction settings
- real gains may therefore be non-uniform

### Paragraph 3: Core Tension

Goal:

- establish the empirical tension that drives the paper

Should mention:

- more complete multimodal models are not automatically globally strongest
- stronger structural alternatives may still dominate overall

### Paragraph 4: Research Question

Goal:

- state the actual question of the paper

Use language like:

- when does multimodal information help?
- why is that gain bounded under the current protocol?

### Paragraph 5: High-Level Method

Goal:

- describe the study design at a high level

Should mention:

- unified protocol
- compared model family
- subgroup and relation analyses
- behavior and case analyses

### Paragraph 6: Main Findings

Goal:

- summarize the main empirical message

Should mention:

- `Residual-only` remains globally strongest
- `Full Model` still improves over simpler multimodal baselines
- local multimodal gain is most visible in favorable regimes

### Paragraph 7: Contributions

Recommended flat contribution list:

1. We provide a unified empirical comparison of structural and multimodal model families on OpenBG-IMG.
2. We show that multimodal gain is bounded by target position, image availability, and relation characteristics.
3. We explain the mechanism of this boundary through gate, residual, and branch-preference analysis.
4. We support the statistical findings with case-level evidence.

## 5. Related Work

### Paragraph 1: MMKGC Foundations

Goal:

- define the field this paper belongs to

Should mention:

- multimodal entity representation
- link prediction with image and text support

### Paragraph 2: Fusion-Oriented MMKGC

Goal:

- summarize fusion-based methods

Should mention:

- early fusion
- adaptive fusion
- relation-aware fusion

### Paragraph 3: Missing-Modality and Robustness

Goal:

- connect to incomplete modality research

Should mention:

- missing modality
- modality imbalance
- noisy or weak modality support

### Paragraph 4: Gap in Existing Work

Goal:

- explain what is missing in current literature

Should mention:

- many works study how to fuse
- fewer works carefully analyze when the fused signal is actually useful under asymmetric modality support

### Paragraph 5: Our Position

Goal:

- position this paper clearly

Should mention:

- this is not mainly a new module paper
- this is a gain-boundary analysis paper under missing-visual conditions

## 6. Task Setting and Experimental Protocol

### Paragraph 1: Dataset and Task

Goal:

- define OpenBG-IMG and the link prediction task

### Paragraph 2: `paper_split`

Goal:

- explain split design and why it matters

Should mention:

- head targets can be image-available
- tail targets are effectively `no_img`
- this asymmetry is central to interpreting later results

### Paragraph 3: Evaluation Protocol

Goal:

- define the reporting protocol

Should mention:

- train/dev/test
- `best.ckpt`
- filtered ranking
- `direction=both`
- 3 seeds

### Paragraph 4: Why Unified Protocol Matters

Goal:

- justify comparability across models

Should mention:

- same split
- same selection rule
- same evaluation standard

### Paragraph 5: Protocol Limitation

Goal:

- make limitation explicit before analysis begins

Should mention:

- gain-boundary findings are protocol-aware
- they should not be over-generalized beyond this setting

## 7. Models and Compared Methods

### Paragraph 1: Internal Model Family Overview

Goal:

- present the five internal models as an organized family

### Paragraph 2: `Text-only` and `Early Fusion`

Goal:

- define the simplest references

Should mention:

- lower-bound single-modality text reference
- direct multimodal fusion reference

### Paragraph 3: `Gate-only`

Goal:

- explain relation-aware fusion without residual support

### Paragraph 4: `Residual-only`

Goal:

- explain the structure-heavy compensation path

Should mention:

- no gate fusion branch
- strong internal competitor under current protocol

### Paragraph 5: `Full Model`

Goal:

- define the most complete multimodal variant

Should mention:

- combines fusion and residual paths
- used as the central object of analysis rather than assumed winner

### Paragraph 6: Structural Baselines

Goal:

- describe `ComplEx` and `TuckER`

Should mention:

- `ComplEx` is the competitive structural baseline in current results
- `TuckER` remains an important classical reference baseline
- under the current protocol, `TuckER` does not emerge as a competitive structural baseline

## 8. Main Results

### Paragraph 1: Main Results Table Introduction

Goal:

- explain what is reported

Should mention:

- test metrics
- `mean ± std`
- 7-model comparison

### Paragraph 2: Global Ranking

Goal:

- state the headline result

Should mention:

- `Residual-only > ComplEx > Full Model > Gate-only > Early Fusion > Text-only > TuckER`

### Paragraph 3: Positive Result for `Full Model`

Goal:

- state what `Full Model` does achieve

Should mention:

- clear improvement over `Gate-only`
- clear improvement over simpler multimodal baselines

### Paragraph 4: Negative Result for `Full Model`

Goal:

- state what it does not achieve

Should mention:

- fails to surpass `Residual-only`
- also trails `ComplEx`

### Paragraph 5: Why This Is Not the End of the Story

Goal:

- transition into analysis

Should mention:

- global metrics alone do not tell us where multimodal gain appears
- the rest of the paper investigates the boundary conditions behind this gap

## 9. Gain-Boundary Analysis

### 9.1 `has_img / no_img`

#### Paragraph 1: Motivation

Goal:

- explain why modality-aware subgrouping matters

#### Paragraph 2: Subgroup Definition

Goal:

- define `head_has_img`, `head_no_img`, `tail_no_img`

#### Paragraph 3: Main Finding

Goal:

- report the key subgroup result

Should mention:

- strongest multimodal-favorable regime appears in `head_has_img`
- overall still dominated by `tail_no_img`

#### Paragraph 4: Interpretation

Goal:

- state what this means

Should mention:

- multimodal gain depends on target position and modality availability

### 9.2 Relation Type

#### Paragraph 1: Motivation

Goal:

- explain why relation grouping is needed

#### Paragraph 2: Coarse Group Definition

Goal:

- introduce `visual_relations`, `weak_visual_relations`, `ambiguous_material_relations`

#### Paragraph 3: Group-Level Finding

Goal:

- report grouped ranking

Should mention:

- grouped ordering remains `Residual-only > Full Model > Gate-only`

#### Paragraph 4: Relation-Level `MIN20` Finding

Goal:

- highlight retained-relation evidence

Should mention:

- `Full Model` beats `Gate-only` on most retained relations
- `Full Model` beats `Residual-only` on only a minority

#### Paragraph 5: Interpretation

Goal:

- reject oversimplified visual-group story

Should mention:

- relation dependence is real
- but not equivalent to “visual relations globally favor multimodal models”

### 9.3 Gain-Boundary Synthesis

#### Paragraph 1: Integrating `has_img` and relation type

Goal:

- merge the two analytical views

#### Paragraph 2: Final Section Conclusion

Goal:

- state the main paper claim clearly

Use language like:

- multimodal gain is local
- multimodal gain is conditional
- multimodal gain is bounded

## 10. Behavior Analysis

### 10.1 Section Opening

Goal:

- explain that this section addresses mechanism rather than headline result

### 10.2 Gate Behavior

#### Paragraph 1: Why Gate Matters

Goal:

- explain why gate is worth analyzing

#### Paragraph 2: Global Gate Statistics

Goal:

- summarize mean/std evidence

#### Paragraph 3: Relation-Aware Gate Variation

Goal:

- show that gate changes across relation contexts

#### Paragraph 4: Interpretation

Goal:

- state:
  - `gate is relation-aware, but not relation-dominant`

### 10.3 Residual Behavior

#### Paragraph 1: Why Residual Matters

Goal:

- connect to residual dominance hypothesis

#### Paragraph 2: Global Residual Statistics

Goal:

- summarize `residual_scale` and mix preference

#### Paragraph 3: Subgroup-Aware Residual Shift

Goal:

- show stronger residual reliance under weaker image support

#### Paragraph 4: Interpretation

Goal:

- explain why residual becomes stronger in weak-visual regimes

### 10.4 Fusion vs Residual

#### Paragraph 1: Branch Competition

Goal:

- explain why branch interaction matters

#### Paragraph 2: Evidence of Residual Dominance

Goal:

- summarize long-term branch preference

#### Paragraph 3: Evidence That Fusion Still Helps

Goal:

- avoid the false conclusion that fusion is dead

#### Paragraph 4: Final Judgment

Goal:

- state:
  - `residual-dominant asymmetric complementarity`

### 10.5 Section Close

Goal:

- connect behavior findings back to main result

Should mention:

- behavior analysis explains why `Full Model` improves over `Gate-only`
- but still rarely surpasses `Residual-only`

## 11. Case Study

### 11.1 Section Opening

Goal:

- remind the reader that case studies support, rather than replace, the earlier analyses

### 11.2 Success Cases

#### Paragraph 1: Selection Principle

Goal:

- state that cases are selected by fixed rules, not anecdotal preference

#### Paragraph 2: Main-Text Success Cases

Goal:

- present:
  - `佩戴方式`
  - `裙长`
  - `细分风格`

#### Paragraph 3: Shared Pattern

Goal:

- explain the common structure:
  - `head`
  - `target_has_img=True`
  - locally perceptible relation cues

### 11.3 Failure Cases

#### Paragraph 1: Main-Text Failure Cases

Goal:

- present:
  - `适用场景`
  - `材质`
  - `净含量`

#### Paragraph 2: Shared Pattern

Goal:

- explain the common structure:
  - `tail`
  - `target_has_img=False`
  - stronger structural or metadata regularity

### 11.4 Case-Level Boundary Summary

Goal:

- state that the case evidence matches the statistical and behavior evidence

## 12. Discussion

### Paragraph 1: Why Multimodal Gain Is Not Globally Dominant

Goal:

- interpret the overall phenomenon

### Paragraph 2: Why This Does Not Mean Multimodal Information Is Useless

Goal:

- defend the positive side of the story

### Paragraph 3: Implications for MMKGC

Goal:

- discuss what future work should learn from this

Suggested direction:

- future work should model when multimodal cues matter, not merely assume they should always help

## 13. Limitations

### Paragraph 1: Protocol Dependence

Goal:

- state dependence on current split and protocol

### Paragraph 2: Modality Asymmetry

Goal:

- state the head/tail asymmetry clearly

### Paragraph 3: Baseline Optimization Scope

Goal:

- mention that some classical baselines such as `TuckER` were not individually over-tuned

### Paragraph 4: Relation Grouping Coarseness

Goal:

- mention that current relation grouping is intentionally coarse and analysis-oriented

## 14. Conclusion

### Paragraph 1: Restate the Core Finding

Goal:

- restate:
  - multimodal gain is real but bounded

### Paragraph 2: Restate the Mechanism

Goal:

- summarize:
  - favorable regime for multimodal gain
  - global dominance of structural compensation

### Paragraph 3: Final Paper Position

Goal:

- close with the paper identity

Suggested emphasis:

- this work contributes a systematic gain-boundary analysis
- not a claim of universally superior multimodal architecture

## 15. Minimal Figure and Table Plan

### Main Text

1. Main results table
2. `has_img / no_img` subgroup table
3. Relation-type grouped or `MIN20` table
4. Compact behavior table or figure
5. Case-study summary table

### Optional

1. Protocol asymmetry figure
2. Additional relation-level appendix tables
3. Appendix case examples

## 16. Recommended Drafting Order

Write in this order:

1. Main Results
2. Gain-Boundary Analysis
3. Behavior Analysis
4. Case Study
5. Related Work
6. Introduction
7. Discussion
8. Limitations
9. Conclusion

Reason:

- the results and analysis sections are already the most mature
- introduction should be written after the evidence wording is stable
