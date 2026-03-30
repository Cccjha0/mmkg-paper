# Gain Boundary Judgment

## 1. Purpose

This document consolidates the current evidence from subgroup analysis and residual-dominance diagnosis in order to answer two paper-facing questions:

- whether multimodal gain is only locally effective
- whether the paper can now formally adopt a "gain boundary" narrative

The goal here is not to introduce a new experimental setting, but to synthesize the existing results from:

- `docs/HAS_IMG_ANALYSIS.md`
- `docs/RELATION_TYPE_ANALYSIS.md`
- `docs/FULL_MODEL_DIAGNOSIS_TODO.md`

## 2. Evidence From `has_img / no_img`

The current `paper_split` is asymmetric:

- head target entities are partly image-available
- tail target entities are entirely `no_img`

Under this setting, the key observations are:

- overall ranking remains `Residual-only > Full Model > Gate-only`
- in `head_has_img`, the ranking becomes `Gate-only > Full Model > Residual-only`
- in `head_no_img`, the ranking becomes `Full Model > Gate-only > Residual-only`
- in `tail_no_img`, `Residual-only` is strongly dominant and pulls back the overall result

Interpretation:

- multimodal gain is not globally visible at the overall-metric level
- it appears in specific target-position and modality-availability conditions
- the strongest direct evidence of local multimodal usefulness appears in `head_has_img`

## 3. Evidence From Relation-Type Analysis

The coarse relation grouping and `MIN20` filtered relation-level analysis together give a second layer of evidence.

### 3.1 Group-level result

Across `visual_relations`, `weak_visual_relations`, and `ambiguous_material_relations`, the focused 3-model grouped ranking remains:

1. `Residual-only`
2. `Full Model`
3. `Gate-only`

This means the current relation grouping does not support a simple claim that multimodal models dominate a whole visual-relation subset.

### 3.2 Relation-level result with minimum support

After filtering to relations with `test triples >= 20`:

- `Full Model > Gate-only` on most retained relations
- `Full Model > Residual-only` only on a small minority of retained relations

Current retained-count summary:

| Group | `Full Model > Gate-only` | `Full Model > Residual-only` |
|---|---:|---:|
| `visual_relations` | `22 / 24` | `4 / 24` |
| `weak_visual_relations` | `15 / 18` | `4 / 18` |
| `ambiguous_material_relations` | `9 / 9` | `1 / 9` |

Interpretation:

- residual-enhanced multimodal modeling provides broad local gains over pure gate fusion
- but these gains are not strong enough to displace the stronger structural compensation path overall
- therefore, relation-level evidence supports "bounded local gain" rather than "global multimodal superiority"

## 4. Evidence From Residual-Dominance Diagnosis

The residual-dominance diagnosis adds the mechanism-facing background:

- residual dominance is real, but it appears mainly at the final representation mixing level rather than purely as a gradient shortcut
- delayed-residual, weaker-residual, and stronger-residual-regularization variants did not surpass the default `Full Model`
- this weakens the story that the current result is merely a training bug or a trivial optimization artifact
- the stronger explanation is that, under the current dataset and protocol, the structural compensation path itself is better aligned with the task bias

Interpretation:

- `Residual-only` is not winning by accident
- `Full Model` is learning useful multimodal information, but that information is not strong enough to overturn the structural preference of the task

## 5. Final Judgment

### 5.1 Is multimodal gain only locally effective?

Yes.

The current evidence supports the statement that multimodal gain is local, conditional, and not globally dominant.

It is local in at least three senses:

- target-position-local:
  strongest evidence appears in `head_has_img`, not in the whole test set
- modality-availability-local:
  gains depend on whether the prediction target has image support
- relation-local:
  `Full Model` improves over `Gate-only` on many supported relations, but does not translate that into group-level or global dominance over `Residual-only`

### 5.2 Can the paper adopt a "gain boundary" core narrative?

Yes.

The current evidence is strong enough to support the following paper-facing direction:

- multimodal gain in OpenBG-IMG is real but bounded
- its usefulness depends on target position, modality availability, and relation characteristics
- strong structural compensation remains the dominant global driver under the current protocol

This is better aligned with the results than either of the following stories:

- "the multimodal full model is globally best"
- "simple residual suppression would be sufficient to make the multimodal full model globally strongest"

## 6. Recommended Writing Position

Recommended one-paragraph paper-facing summary:

> Under the current OpenBG-IMG paper protocol, multimodal gain is not globally consistent but instead appears under clear boundary conditions. In `has_img / no_img` analysis, the strongest local gains emerge when the prediction target is image-available, especially in the head direction, while the overall metric remains dominated by the structurally stronger `tail_no_img` setting. In relation-type analysis, the `Full Model` improves over `Gate-only` on most medium-support and high-support relations, yet still fails to surpass `Residual-only` at the grouped level and on most retained relations. Combined with residual-dominance diagnosis, these results support a gain-boundary narrative: multimodal information is useful, but its benefit is local, conditional, and constrained by the stronger structural bias of the current task.

## 7. Outcome For The Todo

Based on the current evidence, `6.3` can be marked complete with the following decisions:

- multimodal gain should be judged as locally effective rather than globally effective
- the paper can formally adopt "gain boundary" as the core narrative for the current stage
