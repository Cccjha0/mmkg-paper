# Case Analysis Interpretation

## 1. Purpose

This note turns `docs/CASE_ANALYSIS.md` into paper-facing guidance:

- which cases are strong enough for the main paper body
- which cases are better placed in an appendix or supplementary material
- how these cases connect back to the gain-boundary storyline

Primary reference files:

- `docs/CASE_ANALYSIS.md`
- `docs/case_analysis.json`

## 2. Overall Reading

The refined case selection is now well aligned with the paper narrative.

- All selected success cases are `head` queries with `target_has_img=True`.
- All selected failure cases are `tail` queries with `target_has_img=False`.
- Success cases therefore reflect the favorable regime already identified in `6.1`.
- Failure cases reflect the structure-dominant regime already identified in `6.1`, `6.3`, and `7.2`.

This means the case study section can now be used as concrete sample-level evidence for the broader statistical conclusions, rather than as isolated anecdotes.

## 3. Main-Paper Success Cases

The following success cases are the best candidates for the main paper body.

### 3.1 Recommended Main-Text Success Cases

1. `佩戴方式` (`visual_relations`)
- Why keep it: this is the cleanest multimodal-favorable case.
- Evidence shape: `head` prediction, `target_has_img=True`, relation is visually grounded, and `Full Model` clearly outperforms `Residual-only`.
- Suggested role in paper: strongest example that image-grounded cues can produce real local gain.

2. `裙长` (`visual_relations`)
- Why keep it: directly tied to visible clothing appearance.
- Evidence shape: visually grounded relation with a clear rank gap in favor of `Full Model`.
- Suggested role in paper: supports the claim that some appearance attributes are better captured when multimodal cues are available.

3. `细分风格` (`visual_relations`)
- Why keep it: still appearance-related, but slightly more abstract than raw color or length.
- Evidence shape: `Full Model` remains clearly better than `Residual-only`, while the relation is still plausibly image-supported.
- Suggested role in paper: useful as a bridge case showing that multimodal gain is not limited to only the most obvious low-level visual attributes.

## 4. Appendix Success Cases

The following success cases are still useful, but are better treated as appendix-level or supplementary evidence.

### 4.1 Recommended Appendix Success Cases

1. `品牌` (`weak_visual_relations`)
- Why not lead with it: this is a very strong gain case, but it is not an ideal flagship example for a visual-benefit claim.
- Best interpretation: mixed-cue regime; likely reflects a combination of product appearance, product family clustering, and textual/semantic regularities.

2. `地市` (`weak_visual_relations`)
- Why not lead with it: the improvement is real, but the relation is not visually grounded enough for a clean main-text multimodal story.
- Best interpretation: useful for showing that local gain can extend beyond pure visual attributes.

3. `是否精酿` (`weak_visual_relations`)
- Why not lead with it: also better read as a mixed-cue case than as a pure image case.
- Best interpretation: good support for the paper's broader claim that multimodal gain is local and conditional, not exclusively tied to one narrow relation family.

## 5. Main-Paper Failure Cases

The following failure cases are the strongest candidates for the main paper body.

### 5.1 Recommended Main-Text Failure Cases

1. `适用场景` (`weak_visual_relations`)
- Why keep it: this is an excellent structure-dominant failure case.
- Evidence shape: `tail` prediction, `target_has_img=False`, and `Residual-only` achieves rank 1 while `Full Model` fails badly.
- Suggested role in paper: strongest example for why the overall task remains dominated by structural regularity in the unfavorable regime.

2. `材质` (`ambiguous_material_relations`)
- Why keep it: material-related relations often look visually tempting, but in this split they still favor the structure-heavy model.
- Suggested role in paper: strong counterexample showing that not every semantically “visual-looking” relation becomes multimodal-favorable in practice.

3. `净含量` (`weak_visual_relations`)
- Why keep it: clean non-visual, metadata-like failure case.
- Suggested role in paper: helps explain why structural pathways remain globally strong in product KG completion.

## 6. Appendix Failure Cases

The following failure cases are still useful, but are better suited to appendix discussion.

### 6.1 Recommended Appendix Failure Cases

1. `裤长` (`visual_relations`)
- Why appendix: interesting because the relation name looks visual, yet the failure happens on a `tail_noimg` query.
- Best interpretation: useful as a nuanced example that even visually named relations can be structure-dominant once the target side is image-missing.

2. `色泽` (`visual_relations`)
- Why appendix: same value as a “visually named but structurally dominated” example, but slightly more redundant if `裤长` is already shown.

3. `设计元素` (`visual_relations`)
- Why appendix: good for reinforcing the same boundary pattern without overloading the main text with too many similar visual-relation failures.

## 7. Paper-Facing Interpretation

The case analysis now supports the following paper-facing message:

- Multimodal gain is real, but it is local rather than global.
- The clearest success cases appear when the predicted target is on the `head` side and has image support.
- The clearest failure cases appear when the predicted target is on the `tail` side and lacks image support.
- Therefore, sample-level evidence is consistent with the broader gain-boundary conclusion developed in Sections 6 and 7.

## 8. Suggested Writing Pattern

If the paper body only has room for a compact case-study subsection, the most balanced set is:

- Success:
  - `佩戴方式`
  - `裙长`
  - `细分风格`
- Failure:
  - `适用场景`
  - `材质`
  - `净含量`

If space is tighter, reduce to:

- Success:
  - `佩戴方式`
  - `裙长`
- Failure:
  - `适用场景`
  - `材质`

This smaller set is already enough to show both sides of the gain boundary.
