# Manuscript Structure Alignment

## 1. Purpose

This document aligns the new gain-threshold paper manuscript with the current repository layout and with `docs/paper/manuscript_main.tex`. Its role is to make the drafting flow explicit:

- which manuscript files are now **canonical**
- which older files should be treated as **retired from the main paper flow**
- how the final eight-section paper structure maps onto the current repository
- how `manuscript_main.tex` should be interpreted during the transition

This alignment is necessary because the repository currently contains both the earlier analysis-oriented manuscript and the new analysis-driven method version.

## 2. Final Target Paper Structure

The target paper structure is:

1. Introduction
2. Related Work
3. Task Setting and Protocol
4. Method: Gain-Threshold Routing for Selective Multimodal Activation
5. Experiments
6. Analysis and Discussion
7. Limitations
8. Conclusion

This is the canonical structure for the new gain-threshold version.

## 3. Canonical Manuscript Files (Keep as Mainline)

The following files should now be treated as the **main drafting sources** for the paper:

| Final Section | Canonical File | Status |
|---|---|---|
| 1. Introduction | `docs/paper_manuscript/01_introduction.md` | keep |
| 2. Related Work | `docs/paper_manuscript/02_related_work.md` | keep |
| 3. Task Setting and Protocol | `docs/paper_manuscript/03_task_setting_and_protocol.md` | keep |
| 4. Method | `docs/paper_manuscript/04_method_gain_threshold_routing.md` | keep |
| 5. Experiments | `docs/paper_manuscript/05_experiments.md` | keep |
| 6. Analysis and Discussion | `docs/paper_manuscript/06_analysis_and_discussion.md` | keep |
| 7. Limitations | `docs/paper_manuscript/07_limitations.md` | keep |
| 8. Conclusion | `docs/paper_manuscript/08_conclusion.md` | keep |

These eight files define the new paper and should be used as the source of truth for future editing.

## 4. Older Section Files: Keep or Retire

The repository also contains earlier manuscript files from the analysis-oriented version. These should now be handled as follows.

### 4.1 Keep as supporting reference, but retire from the main manuscript flow

| File | Previous Role | New Role |
|---|---|---|
| `docs/paper_manuscript/04_models_and_compared_methods.md` | old Section 4 | supporting reference for model-family wording; retired from main flow |
| `docs/paper_manuscript/05_main_results.md` | old Section 5 | supporting reference for official seven-model tension; retired from main flow |
| `docs/paper_manuscript/06_gain_boundary_analysis.md` | old Section 6 | supporting reference for subgroup / relation-group interpretation; retired from main flow |
| `docs/paper_manuscript/07_behavior_analysis.md` | old Section 7 | supporting reference for branch-level interpretation; retired from main flow |
| `docs/paper_manuscript/09_discussion_limitations_conclusion.md` | old merged ending | supporting reference only; replaced by new Sections 6, 7, and 8 |

These files should not be deleted yet, because they still contain reusable language and analysis logic. However, they should no longer be treated as the active manuscript sequence.

### 4.2 Keep as evidence and result-source files, not manuscript sections

The following files remain important, but they should be treated as **evidence assets** rather than narrative sections:

- `docs/MAIN_RESULTS_SUMMARY.md`
- `docs/main_results_summary.json`
- `docs/RELATION_TYPE_ANALYSIS.md`
- `outputs/router/eval/main_results_table.md`
- `outputs/router/eval/subgroup_results_table.md`
- `outputs/router/eval/feature_ablation.md`
- `outputs/router/eval/first_round_takeaways.md`
- `outputs/router/eval/final_results_manifest.md`

These files support the new paper, especially Sections 5 and 6, but they are not part of the main manuscript directory sequence.

## 5. Final Mapping to `manuscript_main.tex`

At the moment, `docs/paper/manuscript_main.tex` still contains the earlier inline manuscript body. It should therefore be understood as a **legacy integrated TeX draft** rather than the final source-of-truth text for the new paper.

During the current transition stage, the correct mapping is:

| `manuscript_main.tex` legacy block | Replace with canonical source |
|---|---|
| Introduction | `01_introduction.md` |
| Related Work | `02_related_work.md` |
| Task Setting and Protocol | `03_task_setting_and_protocol.md` |
| Models and Compared Methods | `04_method_gain_threshold_routing.md` |
| Main Results + Gain-Boundary Analysis | `05_experiments.md` |
| Behavior Analysis + Discussion | `06_analysis_and_discussion.md` |
| Discussion, Limitations, and Conclusion | split into `07_limitations.md` and `08_conclusion.md` |

This means that the current TeX file should no longer be read as the authoritative paper body. The authoritative paper body now lives in the Markdown manuscript sequence under `docs/paper_manuscript/`.

## 6. Practical Interpretation of the Transition

The manuscript is currently in a **two-layer state**:

### Layer A: Canonical drafting layer

This is the real paper-writing layer and consists of:

- `01_introduction.md`
- `02_related_work.md`
- `03_task_setting_and_protocol.md`
- `04_method_gain_threshold_routing.md`
- `05_experiments.md`
- `06_analysis_and_discussion.md`
- `07_limitations.md`
- `08_conclusion.md`

### Layer B: Legacy integrated layer

This is the old all-in-one TeX paper and the earlier manuscript files. It is still useful for:

- preserving earlier wording
- keeping old figure/table placeholders visible
- maintaining a single compiled document entry point

But it is no longer the best place to continue substantive writing.

## 7. Recommended Next Integration Step

The next integration step should be performed in two passes.

### Pass 1: Structural synchronization

- keep `manuscript_main.tex` as the single TeX entry point
- update its title, running title, and top-level structure note to reflect the new paper identity
- treat the Markdown manuscript files as canonical section sources

### Pass 2: Content migration

- replace the old inline TeX section bodies one by one with the new section content
- start from Section 1 and continue in order
- once a section is migrated, the corresponding older manuscript file should remain only as archived support

## 8. Final Status Judgment

The manuscript is now structurally aligned enough to continue cleanly.

- The new paper structure is defined.
- The canonical section files are in place.
- The old section files have identifiable archival roles.
- The TeX entry point is now understood as a legacy integration layer awaiting staged migration.

## 9. Section-Level Takeaway

The main message of this alignment is:

> The new gain-threshold paper is now defined by the eight-section manuscript sequence under `docs/paper_manuscript/`. Older section files remain as reference material, while `manuscript_main.tex` should be treated as a legacy integrated entry point to be updated gradually rather than as the current source of truth.
