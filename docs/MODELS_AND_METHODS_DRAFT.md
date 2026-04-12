# Models and Compared Methods Draft

## 1. Overview

To study multimodal gain under missing-visual conditions, we compare a family of internal multimodal models together with classical structural baselines. The goal of this comparison is not only to identify which model obtains the best overall score, but also to isolate the roles of text, fusion, residual compensation, and structural modeling under a unified protocol. For this reason, the compared methods are organized so that each model corresponds to a distinct design choice rather than being treated as an unrelated standalone system.

Our internal model family consists of five variants: `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, and `Full Model`. These variants make it possible to separate the effects of textual grounding, direct multimodal fusion, relation-aware fusion, structural compensation, and the interaction between fusion and residual branches. In addition, we include `ComplEx` and `TuckER` as structural reference baselines under the same train/dev/test pipeline and evaluation protocol.

## 2. Internal Model Family

### 2.1 Text-only

`Text-only` is the simplest semantic reference model in our internal family. It uses graph structure together with text-derived entity representations, but does not include image-based fusion. Its role in the comparison is to answer a minimal question: how much can be achieved with non-visual semantic support before adding image information and explicit multimodal interaction?

This model is important because it separates the effect of multimodality from the effect of semantic enrichment more generally. If a multimodal model only performs slightly better than `Text-only`, then the apparent gain may not come from visual information itself. Conversely, a consistent gap over `Text-only` indicates that image-aware modeling is contributing beyond text semantics alone.

### 2.2 Early Fusion

`Early Fusion` is the most direct multimodal baseline in our internal family. It combines text and image representations in a simple manner before downstream scoring, without relation-aware gating or a separate residual compensation path. Its purpose is to provide a straightforward multimodal reference and to test whether direct feature combination is already sufficient to obtain useful gains.

This model serves as an important bridge between single-modality and more structured multimodal variants. If `Early Fusion` performs substantially below later models, then improvements cannot be attributed to multimodality alone; they must also depend on how multimodal signals are combined and controlled.

### 2.3 Gate-only

`Gate-only` introduces relation-aware multimodal fusion without an explicit residual compensation branch. Conceptually, this model asks whether adaptive fusion conditioned on relation context is enough to capture the local usefulness of multimodal information. Compared with `Early Fusion`, it tests whether context-sensitive weighting is more effective than direct fusion under heterogeneous relation semantics.

In the current project, `Gate-only` is especially important because it acts as the cleanest comparison target for understanding what the `Full Model` gains from adding residual structure compensation. If the `Full Model` consistently improves over `Gate-only`, then the interaction between fusion and residual pathways is doing meaningful work beyond relation-aware fusion by itself.

### 2.4 Residual-only

`Residual-only` removes the multimodal gate-fusion path and retains only the structural compensation branch. This model is critical because it represents a strong structure-heavy alternative inside the same implementation family. Unlike external classical baselines, it is directly aligned with the design logic of the `Full Model`, which makes it the most informative competitor for diagnosing why the complete multimodal architecture does or does not succeed.

Empirically, `Residual-only` is also the strongest model in the internal family under the current protocol. For that reason, it should not be viewed merely as an ablation. It is a central comparison target that reveals how much of the final performance pattern can be explained by strong structural compensation alone.

### 2.5 Full Model

`Full Model` is the most complete multimodal variant in our study. It includes both relation-aware multimodal fusion and a residual compensation branch, allowing the model to combine fused multimodal evidence with structure-heavy fallback information. In design terms, this is the model that has access to the richest set of signals and the most flexible decision path.

Crucially, however, `Full Model` should not be framed in the paper as the assumed winner. Its main role is analytical rather than rhetorical. We use it as the central object through which to ask two questions: first, whether fusion and residual pathways are genuinely complementary; and second, under what conditions the multimodal path contributes enough signal to matter. In this sense, `Full Model` is best understood as the main diagnostic model of the paper, not simply as the method to be promoted.

## 3. Structural Baselines

### 3.1 ComplEx

`ComplEx` is included as a classical structural baseline under the same unified protocol. In the final results, it emerges as a genuinely competitive structural model and is therefore one of the strongest global comparison targets for the `Full Model`. This is important for interpretation: the main empirical tension of the paper is not only between multimodal models and a weak baseline, but also between the complete multimodal model and a strong classical structural alternative.

Because `ComplEx` performs competitively under the current protocol, it should be treated in writing as a true structural competitor rather than a routine reference model. It helps establish that the challenge faced by `Full Model` is not only to beat simpler multimodal variants, but also to justify multimodal complexity relative to a strong structure-only approach.

### 3.2 TuckER

`TuckER` is also included as a classical structural reference baseline under the same train/dev/test and evaluation setup. The reason for including it is methodological completeness: it is a well-known structural KGC model and remains a reasonable reference point for readers familiar with the KGC literature.

At the same time, the paper must be careful in how `TuckER` is described. Under the current OpenBG-IMG protocol, `TuckER` does not emerge as a competitive structural baseline in the resulting rankings. This is a statement about the present protocol and data distribution, not a universal statement about the model itself. We therefore keep `TuckER` in the comparison as a classical structural reference, but do not frame it as one of the strongest empirical competitors in the current study.

## 4. Comparison Logic

The compared methods are designed to support a layered analysis rather than a single leaderboard claim.

First, `Text-only` and `Early Fusion` provide minimal references for semantic enrichment and direct multimodal combination. Second, `Gate-only` isolates relation-aware fusion without structural residual support. Third, `Residual-only` isolates the structure-heavy compensation path. Finally, `Full Model` combines both mechanisms and allows us to study their interaction. `ComplEx` and `TuckER` provide classical structural reference points outside the internal family.

This comparison logic is central to the paper's contribution. Without it, we could only ask whether one model scores higher than another. With it, we can ask a more informative sequence of questions:

- Is multimodal information useful beyond text-only semantics?
- Is relation-aware fusion better than direct fusion?
- Is strong structural compensation still stronger than multimodal fusion?
- When both are available, does the model behave cooperatively or does one path dominate the other?

These are exactly the questions needed for a gain-boundary analysis.

## 5. Writing Guidance for This Section

When writing the final paper, this section should avoid two common mistakes.

The first mistake is to present the model family as if all variants exist only to justify `Full Model`. That is not the current evidence-based story. In this project, some variants, especially `Residual-only`, are not merely stepping stones but empirically important endpoints in their own right.

The second mistake is to flatten all structural baselines into a single category. Under the current protocol, `ComplEx` is a competitive structural baseline, while `TuckER` is better described as a classical structural reference model that does not emerge as competitive in the final results. Preserving this distinction will make the later result discussion more accurate and more defensible.

Overall, the safest framing is:

- `Full Model` is the most complete multimodal architecture
- `Residual-only` is the strongest internal structural competitor
- `ComplEx` is the strongest external structural baseline under the current protocol
- `TuckER` is a classical structural reference baseline, but not a competitive one in the present setting

## 6. Section-Level Takeaway

The most important message of this section should be:

> The compared methods are organized to isolate the roles of semantic enrichment, multimodal fusion, structural compensation, and branch interaction. Under the current protocol, the central analytical comparison is not simply whether `Full Model` ranks highest, but how its multimodal and residual components behave relative to both internal and classical structural competitors.
