# Fusion vs Residual Analysis

## 1. Purpose

This note closes `7.3 Fusion vs residual` by combining three evidence sources:

- training-time behavior summary: `docs/BEHAVIOR_SUMMARY.md`
- relation-aware gate summary: `docs/RELATION_AWARE_GATE_SUMMARY.md`
- relation-aware residual summary: `docs/RELATION_AWARE_RESIDUAL_SUMMARY.md`

The goal is not to ask whether fusion exists at all, but to judge:

- which branch dominates the final mixed representation
- whether fusion is still doing useful work
- whether the relation between the two branches is better described as complementarity or competition

## 2. Key Evidence

### 2.1 Mix weights show long-term residual preference

At the best epoch of the 3-seed `Full Model` runs:

- `mix_w_fusion ≈ 0.2049 ± 0.0402`
- `mix_w_residual ≈ 0.7951 ± 0.0402`

From first eval to best epoch:

- `mix_w_fusion` continues to decrease
- `mix_w_residual` continues to increase

This is the strongest direct evidence that the final mixed representation is persistently biased toward the residual branch.

### 2.2 Fusion is not dead

Even under residual-dominant mixing, fusion remains active:

- `grad_fusion` and `grad_projection` stay non-zero throughout training
- relation-aware gate values are stable and systematically different across relation groups
- `Full Model` still beats `Gate-only` on most medium-support relations in the completed `6.2` analysis

The `MIN20` relation-level summary already showed:

- `visual_relations`: `Full Model > Gate-only` on `22 / 24`
- `weak_visual_relations`: `Full Model > Gate-only` on `15 / 18`
- `ambiguous_material_relations`: `Full Model > Gate-only` on `9 / 9`

So fusion is not a dead branch or pure optimization artifact. It contributes real local gains over `Gate-only`.

### 2.3 Residual dependence rises when image support is weak

Relation-aware residual analysis shows that, inside `Full Model`, head-side residual contribution increases on `head_noimg`:

- `visual_relations`: `effective residual norm 0.7462 > 0.6776`
- `weak_visual_relations`: `0.7976 > 0.6909`
- `ambiguous_material_relations`: `0.7504 > 0.6812`

The head-side `residual / fused ratio` also rises on `head_noimg`:

- `visual_relations`: `0.4461 > 0.4212`
- `weak_visual_relations`: `0.4733 > 0.4279`
- `ambiguous_material_relations`: `0.4490 > 0.4232`

This indicates that the residual branch is not only globally preferred by mix weights; it is also used more aggressively when the target head lacks image support.

### 2.4 Fusion still carries substantial magnitude

Residual dominance does not mean the fused branch becomes numerically negligible.

For the `Full Model`, relation-aware subgroup summaries show:

- on head-side slices, `effective fused norm` is still clearly larger than `effective residual norm`
- on tail-side slices, the two become much closer, and several relations approach parity

This matters because it means the observed dominance is mainly a matter of **selection pressure and branch preference**, not complete suppression of fused features.

## 3. Interpretation

The evidence supports three linked judgments.

### 3.1 Who dominates?

Residual dominates the final branch preference.

This is shown most directly by:

- `mix_w_residual ≈ 0.80`
- increasing residual preference over training
- stronger residual reliance on `head_noimg`

So the final representation is not balanced. It is residual-biased.

### 3.2 Is fusion being permanently suppressed?

Partly yes, but not absolutely.

If “suppressed” means:

- lower final mixture weight
- weaker contribution in the final decision path
- inability to overturn residual dominance at overall evaluation level

then the answer is yes.

But if “suppressed” means:

- no gradients
- no relation-aware behavior
- no measurable performance contribution

then the answer is no.

Fusion still learns, still varies with relation context, and still produces the consistent local improvement from `Gate-only` to `Full Model`.

### 3.3 Complementarity or competition?

The best description is:

**asymmetric complementarity under residual-dominant competition**

More concretely:

- there is complementarity, because adding fusion to residual pathways does improve many medium-support relations relative to `Gate-only`
- there is competition, because the final model keeps assigning more trust to residual and does not let fusion become the dominant route
- the competition is asymmetric, because residual usually keeps the upper hand in the final mixed representation

## 4. Final Judgment

`7.3` can be closed with the following paper-facing wording:

- Fusion is useful but not globally dominant.
- Residual is the preferred fallback and final decision branch, especially when image support is weak.
- The relation between the two branches is not pure redundancy and not pure cooperation; it is better described as **residual-dominant, asymmetric complementarity**.
- This also explains the main empirical pattern of the paper:
  - `Full Model` consistently improves over `Gate-only`
  - but still rarely surpasses `Residual-only`

## 5. Implication for the Paper Narrative

This result further strengthens the current paper storyline:

- the main question is not “how to make fusion win globally”
- the stronger question is “when does fusion add enough useful signal to matter, and when does the model fall back to residual structure compensation”

That framing is aligned with the already completed gain-boundary narrative.
