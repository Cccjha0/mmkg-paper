# Gate Relation Judgment

## 1. Purpose

This note closes the remaining question in `7.1 Gate`:

- does gate really change with relation context?
- if yes, is that change large enough to explain the main performance pattern by itself?

The judgment here is based on:

- `docs/BEHAVIOR_SUMMARY.md`
- `docs/RELATION_AWARE_GATE_SUMMARY.md`
- `docs/FUSION_VS_RESIDUAL_ANALYSIS.md`

## 2. What Is Already Established

Training-time diagnostics already showed that gate is not a fixed scalar:

- `Gate-only` best-epoch `g_mean_all ≈ 0.5346 ± 0.0107`
- `Full Model` best-epoch `g_mean_all ≈ 0.4288 ± 0.0058`
- both models show `g_mean_img < g_mean_noimg`

That was enough to show that gate responds to modality availability, but not yet enough to prove relation dependence.

## 3. Relation-Aware Evidence

The relation-aware gate summary uses real test `(entity, relation)` pairs rather than random relation sampling. Under that stricter setting, gate still shows systematic changes.

### 3.1 Group-level variation exists

For `Gate-only`, group-level gate mean changes across relation groups:

- `visual_relations`: `0.5481 ± 0.0104`
- `weak_visual_relations`: `0.5540 ± 0.0073`
- `ambiguous_material_relations`: `0.5603 ± 0.0112`

For `Full Model`, the same pattern also changes by group:

- `visual_relations`: `0.4719 ± 0.0056`
- `weak_visual_relations`: `0.4955 ± 0.0073`
- `ambiguous_material_relations`: `0.4814 ± 0.0040`

So gate is not relation-invariant. It does respond to relation-group context.

### 3.2 Per-relation variation also exists

Inside the same group, different relations produce visibly different gate levels.

Examples from `visual_relations` in `Full Model`:

- `rel_0009` (`裤长`): `0.3885 ± 0.0119`
- `rel_0044` (`闭合方式`): `0.4922 ± 0.0069`
- `rel_0134` (`基础风格`): `0.4800 ± 0.0156`

Examples from `weak_visual_relations` in `Full Model`:

- `rel_0016` (`关联场景`): `0.5585 ± 0.0128`
- `rel_0022` (`是否精酿`): `0.4726 ± 0.0070`
- `rel_0104` (`包装方式`): `0.5242 ± 0.0046`

These differences are too large to call gate “basically constant”.

### 3.3 The direction is stable across models

Another robust pattern is that `Full Model` gate is consistently lower than `Gate-only` across all three relation groups.

- `visual_relations`: `0.4719 < 0.5481`
- `weak_visual_relations`: `0.4955 < 0.5540`
- `ambiguous_material_relations`: `0.4814 < 0.5603`

This suggests that once residual compensation is available, the model does not merely copy the gate behavior of `Gate-only`; it systematically shifts gate downward.

## 4. Why Gate Alone Does Not Explain The Main Results

Even though gate clearly varies with relation context, the current evidence does **not** support the stronger claim that gate variation alone explains the paper's main performance pattern.

There are three reasons.

### 4.1 Group-level gate differences are real but moderate

The group-level gate means move, but not in a way that directly mirrors the performance ranking from `6.2`.

For example:

- `Full Model` gate mean is highest on `weak_visual_relations`
- but `6.2` did not show that gate-sensitive groups become globally favorable enough for `Full Model` to surpass `Residual-only`

So the direction is informative, but not sufficient as a full explanation.

### 4.2 The stronger behavioral shift comes from residual preference

`7.2` and `7.3` showed that:

- `mix_w_residual ≈ 0.7951`
- residual preference grows during training
- residual contribution rises further on `head_noimg`

Compared with those signals, gate variation looks more like a useful adjustment mechanism than the main controlling force.

### 4.3 Performance evidence points to “helpful but secondary”

The completed `6.2 MIN20` analysis already showed:

- `Full Model` beats `Gate-only` on most medium-support relations
- but still rarely beats `Residual-only`

If gate were the dominant explanation, we would expect stronger alignment between relation-aware gate shifts and final ranking reversals. That pattern is not what we observe.

## 5. Final Judgment

`7.1` can be closed with the following statement:

- Gate does truly vary with relation context.
- This variation is visible at both relation-group and per-relation levels.
- However, the magnitude and direction of gate variation are not enough to explain the main empirical pattern on their own.
- The more complete explanation is:
  - gate provides relation-aware modulation
  - residual provides the dominant fallback and final preference

So the best short-form summary is:

**gate is relation-aware, but not relation-dominant**

## 6. Paper-Facing Implication

This means the paper should not claim that the model's main advantage comes from a highly selective gate that perfectly adapts to relation type.

A more accurate wording is:

- the model does learn relation-aware gating behavior
- but the final behavior is governed by a larger residual-dominant mechanism
- therefore, gate contributes useful local modulation rather than serving as the single decisive explanation for multimodal gains
