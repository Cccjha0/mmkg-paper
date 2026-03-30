# Relation Type Analysis

## 1. Purpose

This document records the current paper-facing interpretation of relation-type grouped evaluation under the unified OpenBG-IMG protocol.

The goal is not only to compare grouped test performance, but also to judge whether multimodal gain:

- is relation-dependent
- is globally reliable or only locally observable
- can support a "gain boundary" narrative together with `has_img / no_img` analysis

## 2. Protocol

- dataset split: `paper_split`
- selection/reporting: `best.ckpt` on the same `test` split as the main experiments
- evaluation: filtered ranking
- direction: `both`
- relation grouping file: `docs/relation_type_groups_draft.json`

Current grouped outputs:

- full 7-model grouped summary:
  - `docs/RELATION_TYPE_SUMMARY_ALL.md`
  - `docs/relation_type_summary_all.json`
- focused 3-model grouped summary with relation-level support filter:
  - `docs/RELATION_TYPE_SUMMARY_MIN20.md`
  - `docs/relation_type_summary_min20.json`

## 3. Group-Level Result

The current coarse grouping contains:

- `visual_relations`
- `weak_visual_relations`
- `ambiguous_material_relations`

Using the focused 3-model comparison (`Gate-only`, `Full Model`, `Residual-only`), the group-level ordering is stable across all three groups:

1. `Residual-only`
2. `Full Model`
3. `Gate-only`

Observed group-level MRR:

| Group | Gate-only | Full Model | Residual-only |
|---|---:|---:|---:|
| `visual_relations` | 0.1350 | 0.1826 | 0.2691 |
| `weak_visual_relations` | 0.2194 | 0.2252 | 0.3197 |
| `ambiguous_material_relations` | 0.1337 | 0.1948 | 0.2842 |

Current group-level interpretation:

- `Full Model` is consistently better than `Gate-only` at the grouped level.
- `Residual-only` remains clearly stronger than `Full Model` in all three groups.
- The current coarse grouping does not support a strong claim that "visual relations are naturally where multimodal models win."
- In fact, `Full Model` has higher grouped MRR on `weak_visual_relations` than on `visual_relations`, which weakens a simple "visual group advantage" story.

## 4. Relation-Level Result With Minimum Support Filter

To reduce noise from extremely small-support relations, relation-level interpretation is based on the `MIN20` version:

- only relations with `test triples >= 20` are kept in the per-relation table

Retained relation counts:

| Group | Kept Relations | Total Relations In Test |
|---|---:|---:|
| `visual_relations` | 24 | 28 |
| `weak_visual_relations` | 18 | 37 |
| `ambiguous_material_relations` | 9 | 15 |

### 4.1 `Full Model` vs `Gate-only`

Among retained relations:

- `visual_relations`: `Full Model > Gate-only` on `22 / 24` relations
- `weak_visual_relations`: `Full Model > Gate-only` on `15 / 18` relations
- `ambiguous_material_relations`: `Full Model > Gate-only` on `9 / 9` relations

Interpretation:

- residual-enhanced multimodal modeling is not just helping on one or two accidental relations
- compared with pure gate fusion, `Full Model` shows broad relation-level improvement once relation support is large enough to be worth interpreting

### 4.2 `Full Model` vs `Residual-only`

Among retained relations:

- `visual_relations`: `Full Model > Residual-only` on `4 / 24` relations
- `weak_visual_relations`: `Full Model > Residual-only` on `4 / 18` relations
- `ambiguous_material_relations`: `Full Model > Residual-only` on `1 / 9` relations

Interpretation:

- `Full Model` does achieve local wins over `Residual-only`, but these wins are limited
- the relation-level evidence does not overturn the group-level conclusion that `Residual-only` is still the stronger overall model
- therefore, multimodal benefit appears to be real but bounded

## 5. Current Paper-Facing Conclusion

The current `6.2 relation type` result supports the following interpretation:

- multimodal gain is relation-dependent, but not in the simple sense that "visual relations as a whole favor multimodal models"
- `Full Model` consistently improves over `Gate-only` on most medium-support and high-support relations, so the residual-enhanced multimodal path is meaningful
- however, `Residual-only` still dominates at the group level and on most retained relations, which means multimodal gain is not globally strong enough to define the main story by itself
- together with `has_img / no_img`, the safer narrative is that multimodal benefit is local, conditional, and bounded by target position, modality availability, and relation characteristics

## 6. Suggested Use In Writing

Recommended wording for the paper:

- relation-type grouped evaluation does not show a grouped-level advantage for multimodal models on the coarse visual relation subset
- however, after filtering out very low-support relations, the `Full Model` outperforms `Gate-only` on most retained relations, indicating that residual-enhanced multimodal fusion provides stable local gains
- these gains remain insufficient to surpass the stronger structural compensation path overall, which supports a "gain boundary" rather than a "global multimodal superiority" conclusion
