# Manuscript Structure Alignment

## 1. Current Source of Truth

The current latest manuscript is:

- `docs/paper/manuscript_main.tex`

The `docs/paper_manuscript/` directory is now a synchronized support area. It contains section-level notes, migration markers, and retired drafts, but it should not be treated as a separate manuscript that overrides the TeX file.

## 2. Current Active Paper Structure

The latest paper structure is:

1. Introduction
2. Related Work
3. Task Setting and Method
4. Experiments and Discussion
5. Conclusion

This is the structure used by `docs/paper/manuscript_main.tex`.

## 3. Active Support Files

The following Markdown files are aligned with the current TeX structure and may be used as support drafts:

| Current TeX section | Support file | Status |
|---|---|---|
| Introduction | `01_introduction.md` | active support |
| Related Work | `02_related_work.md` | active support |
| Task Setting and Method | `03_task_setting_and_protocol.md` | active support |
| Experiments and Discussion | `05_experiments.md` | active support |
| Conclusion | `08_conclusion.md` | active support |

`04_method_gain_threshold_routing.md` is retained only as a migration marker because its method content has already been integrated into `03_task_setting_and_protocol.md` and the TeX manuscript.

## 4. Retired Reference Files

The following files are retained as writing-history or reference material only:

| File | Current role |
|---|---|
| `04_models_and_compared_methods.md` | old model-family wording reference |
| `05_main_results.md` | old official-results wording reference |
| `06_gain_boundary_analysis.md` | old subgroup / relation-boundary wording reference |
| `06_analysis_and_discussion.md` | reference-only discussion note; updated to current result line |
| `07_behavior_analysis.md` | old branch-behavior wording reference |
| `07_limitations.md` | reference-only limitations note |
| `08_case_study.md` | old case-study wording reference |
| `09_discussion_limitations_conclusion.md` | old merged ending reference |

These files should not be used as the active chapter sequence unless they are explicitly rewritten and reintroduced.

## 5. Current Result Line to Preserve

The current manuscript result line is:

- the OpenBG-IMG `paper_split` induces role--modality asymmetry;
- the meaningful target-side regimes are `head_has_img`, `head_no_img`, and `tail_no_img`;
- naive single-threshold clean routing is insufficient and does not outperform the clean rule baseline;
- direction-specific thresholding reaches approximately `0.2974` MRR;
- regression-based gain prediction reaches approximately `0.2982` MRR and is the strongest clean strategy;
- the strongest clean strategy improves over both the clean rule and `Residual-only` with paired bootstrap support;
- the strongest clean strategy remains below Oracle routing.

The older `0.3160 MRR / XGBoost + delta=0.01 + tau=0.7` result should not be treated as the current paper claim.

## 6. Practical Editing Rule

For future edits:

1. update `docs/paper/manuscript_main.tex` first;
2. update only the relevant Markdown support file afterward;
3. do not copy content from retired drafts into the TeX manuscript unless it has been checked against the current result line;
4. keep evaluation lines separated: official model-comparison line, clean routing line, and post-hoc analysis line.

## 7. Section-Level Takeaway

The repository is now aligned around a single authoritative manuscript:

> `docs/paper/manuscript_main.tex` is the latest paper. The Markdown files in `docs/paper_manuscript/` are synchronized support notes and archived drafts, with the current clean-routing result line centered on `0.2974` direction-specific thresholding and `0.2982` regression-based gain prediction.
