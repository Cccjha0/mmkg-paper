# Paper Manuscript Directory

## Current source of truth

`docs/paper/manuscript_main.tex` is the current latest paper manuscript.

The files in this directory are now synchronized support drafts. They should be read as section-level working notes that mirror the current TeX manuscript, not as a separate competing manuscript version.

## Current paper structure

The latest manuscript uses the following main structure:

1. Introduction
2. Related Work
3. Task Setting and Method
4. Experiments and Discussion
5. Conclusion

The previous eight- or nine-part draft structure has been retired from the active paper flow.

## Active aligned files

| Current manuscript part | Aligned Markdown file | Status |
|---|---|---|
| Introduction | `01_introduction.md` | active support draft |
| Related Work | `02_related_work.md` | active support draft |
| Task Setting and Method | `03_task_setting_and_protocol.md` | active support draft; includes clean routing method |
| Experiments and Discussion | `05_experiments.md` | active support draft; updated to structured clean routing results |
| Conclusion | `08_conclusion.md` | active support draft |

`04_method_gain_threshold_routing.md` is retained only as a migration marker because the method content has already been integrated into Chapter 3.

## Retired / reference-only files

The following files are older drafts from the previous analysis-oriented structure. They are kept only as reference material and should not be used as the active paper sequence:

- `04_models_and_compared_methods.md`
- `05_main_results.md`
- `06_gain_boundary_analysis.md`
- `06_analysis_and_discussion.md`
- `07_behavior_analysis.md`
- `07_limitations.md`
- `08_case_study.md`
- `09_discussion_limitations_conclusion.md`
- `09_structure_alignment.md`

These files have been marked or revised so that they no longer contradict the current TeX manuscript.

## Current result line to preserve

When editing this manuscript, use the latest clean-routing line from `manuscript_main.tex`:

- naive single-threshold clean routing is insufficient and does not beat the clean rule baseline;
- direction-specific thresholding reaches approximately `0.2974` MRR;
- regression-based clean gain prediction reaches approximately `0.2982` MRR;
- the best clean strategy improves over both the clean rule and `Residual-only` under paired bootstrap evidence;
- the best clean strategy still remains below Oracle routing.

Do not reintroduce the older `0.3160 MRR / XGBoost + delta=0.01 + tau=0.7` claim into the current manuscript unless it is explicitly revalidated and redefined as a separate non-current experiment line.

## Editing rule

For future writing, edit `docs/paper/manuscript_main.tex` first. Update the Markdown support drafts only when they are needed as section-level notes or planning documents.
