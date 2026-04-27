# Remaining Gaps Execution Plan

This document converts the remaining review-risk analysis into a directly executable plan for the current `mmkg-paper` repository.

The goal is **not** to expand the paper into a new full experimental campaign. The goal is to close the most likely reviewer gaps while preserving the current positioning:

> Protocol-aware bounded-gain diagnosis + deployable selective activation under explicit information boundaries.

Current main evidence already in the paper:

- Main `paper_split` target-side regime counts.
- Subgroup MRR by target-side image availability.
- Relation-group supporting analysis.
- Clean routing progression: clean rule -> direction-specific threshold -> E5 regression clean router.
- Candidate router results: CA-S2 strongest score-aware candidate router.
- Appendix split robustness check on `appendix_split_seed20260427`.
- Appendix D now supports qualitative trend stability.

Therefore, this plan focuses on four remaining issues:

1. Prove CA-S2 is not merely a trivial score ensemble.
2. Make the non-answer-aware boundary visually explicit.
3. Refine the mechanism narrative from "image available -> fusion helps" to "score-aware sparse candidate-level correction".
4. Add only a lightweight protocol sanity check if time allows.

---

## 0. Decision Summary

| Gap | Decision | Priority | Main action |
|---|---|---:|---|
| Stronger protocol audit | Partially needed | P3 | Add lightweight appendix sanity only if time allows |
| CA-S2 vs simple score ensemble | Needed | P1 | Add minimal ensemble baselines |
| CA-S2 non-answer-aware boundary | Needed | P1 | Add Leakage Checklist table |
| Cautious head_has_img mechanism | Needed | P1 | Rewrite mechanism wording, no new experiment |

Do **not** do:

- Full new protocol audit section.
- Relation-level image-coverage heatmap unless it is nearly free.
- Full logistic/XGBoost stacking family.
- Relation-specific interpolation with one alpha per relation.
- Full seven-model rerun.
- New official main split.

---

# Phase 1: Add Information-boundary / Leakage Checklist Table

## 1.1 Goal

Make the deployability boundary explicit:

- Strict query-level clean router uses only metadata-level query-time signals.
- CA-S2 uses non-answer-aware candidate score signals after fixed expert scoring.
- Oracle/post-hoc analysis can use answer-aware signals but is not deployable.

This is a high-impact, low-cost paper edit.

## 1.2 Target manuscript location

File:

```text
docs/paper/manuscript_main.tex
```

Recommended location:

After the paragraph in Section 3.4 / 3.5 that states CA-S2/CA-S3 do not use hidden target, reciprocal rank, target image label, target-aware regime label, or answer-aware expert outcomes.

Good insertion location:

- End of `3.4 Candidate-level Score-aware Selective Activation`, before `3.5 Router Training Details`; or
- Early in `3.5 Router Training Details`, after the query-level feature and observed-side modality feature paragraphs.

## 1.3 Add this LaTeX table

```latex
\begin{table}[t]
\centering
\caption{Information boundary of the routing settings. A check mark means that the signal is available to that setting at inference or analysis time.}
\label{tab:information_boundary}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{0.38\textwidth}ccc}
\toprule
Signal & Strict clean query router & CA-S2 score-aware router & Oracle / post-hoc \\
\midrule
Query direction & \checkmark & \checkmark & \checkmark \\
Relation priors estimated from development data & \checkmark & \checkmark & \checkmark \\
Observed-side modality indicators & \checkmark & \checkmark & \checkmark \\
Candidate scores from fixed experts & -- & \checkmark & \checkmark \\
Candidate rank, confidence, or disagreement from fixed experts & -- & \checkmark & \checkmark \\
Static candidate modality metadata & -- & -- & \checkmark \\
Hidden target entity identity & -- & -- & \checkmark \\
Target-side image availability & -- & -- & \checkmark \\
Target-side regime label & -- & -- & \checkmark \\
Correct-target reciprocal rank & -- & -- & \checkmark \\
Test labels during router training & -- & -- & -- \\
Answer-aware post-hoc outcomes & -- & -- & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

### Important note about check marks

If `\checkmark` is not available, add this to the preamble:

```latex
\usepackage{amssymb}
```

The current manuscript already loads `amsmath,amssymb`, so `\checkmark` should work.

## 1.4 Text to add after the table

```latex
\Cref{tab:information_boundary} summarizes the information boundary used throughout the paper. The strict clean line excludes all expert score signals and relies only on query-time metadata. CA-S2 is less restrictive but remains non-answer-aware: it may use scores, ranks, confidence, and disagreement signals produced by fixed experts for candidate entities, but it never uses the hidden target identity, target-side image label, target-aware regime label, reciprocal rank, or test labels during router training. Oracle and post-hoc analyses are therefore reported only as upper-bound or diagnostic references rather than deployable routing evidence.
```

## 1.5 Acceptance checks

```text
[ ] Table compiles successfully.
[ ] Table appears before the main results, ideally in Method.
[ ] The text explicitly says CA-S2 is non-answer-aware.
[ ] The text explicitly says Oracle/post-hoc is not deployable evidence.
[ ] No row suggests CA-S2 uses target image availability or reciprocal rank.
```

---

# Phase 2: Refine Mechanism Narrative

## 2.1 Goal

Avoid the weak interpretation:

> Images exist, therefore fusion helps.

Replace it with the stronger and more data-consistent interpretation:

> `head_has_img` reveals relative fusion preference, but not high absolute solvability. The strongest deployable gain comes from CA-S2's sparse candidate-level correction, especially in the dominant `tail_no_img` regime.

## 2.2 Target locations

File:

```text
docs/paper/manuscript_main.tex
```

Likely locations:

1. Abstract
2. RQ1 discussion
3. RQ4 discussion
4. Integrated Discussion
5. Conclusion

## 2.3 Abstract change

Current abstract phrase likely says:

```latex
relative multimodal advantage is most visible in image-supported head-side regimes but remains limited globally
```

This is acceptable, but can be made slightly safer.

Recommended replacement:

```latex
relative fusion preference is most visible in image-supported head-side regimes, although this does not imply high absolute solvability and the global gain remains bounded
```

Full sentence version:

```latex
This creates a bounded-gain structure in which relative fusion preference is most visible in image-supported head-side regimes, although this does not imply high absolute solvability and the global gain remains bounded.
```

## 2.4 RQ1 wording patch

Find the RQ1 paragraph discussing `head_has_img`.

Add or ensure the following sentence exists:

```latex
Thus, \texttt{head\_has\_img} should be interpreted as a regime with relative fusion preference rather than as a regime that is easy to solve in absolute terms.
```

## 2.5 RQ4 wording patch

Find the paragraph after Table 18 / alpha diagnostics where CA-S2 and CA-S3 are discussed.

Add:

```latex
This means that the best deployable gain does not come from blindly activating fusion whenever images are available. Instead, CA-S2 behaves as a sparse candidate-level correction mechanism: most candidate scores remain structure-dominant, while fusion influence increases selectively for candidates whose fixed-expert score patterns indicate potential benefit.
```

## 2.6 Integrated Discussion patch

In Section 4.6, after the paragraph explaining E5/CA-S1/CA-S2/CA-S3, add:

```latex
This mechanism also changes how the bounded-gain diagnosis should be read. The key finding is not simply that multimodal evidence helps whenever target-side images exist. Rather, image-supported head-side targets reveal relative fusion preference, while the strongest deployable improvement is obtained by score-aware sparse correction at candidate level, especially in the globally dominant \texttt{tail\_no\_img} regime.
```

## 2.7 Acceptance checks

```text
[ ] No sentence implies that image-supported regimes are solved by fusion.
[ ] The phrase "relative fusion preference" appears at least once.
[ ] The phrase "sparse candidate-level correction" or equivalent appears in RQ4/Discussion.
[ ] CA-S2 is not described as blindly activating fusion.
[ ] The mechanism is consistent with alpha diagnostics.
```

---

# Phase 3: Add Simple Score Ensemble Baselines

## 3.1 Goal

Close the strongest method-level reviewer gap:

> Is CA-S2 just a trivial ensemble of Gate-only and Residual-only scores?

The paper currently proves:

```text
CA-S2 > E5 strict clean query router
```

But it should also prove:

```text
CA-S2 > simple score ensemble baselines
```

## 3.2 Minimal required baselines

Run these three baselines only:

| Baseline | Granularity | Purpose |
|---|---|---|
| Global score interpolation | global | Basic two-expert score ensemble |
| Direction-specific score interpolation | direction | Tests whether head/tail-specific alpha is enough |
| Query-level soft score weighting | query | Tests whether query-level soft granularity is enough |

Optional if cheap:

| Baseline | Granularity | Purpose |
|---|---|---|
| Rank averaging / reciprocal rank fusion | rank-level | Tests whether simple rank fusion explains gains |

Do **not** add a large stacking family unless absolutely necessary.

## 3.3 Expected input data

Current repo already has candidate-router infrastructure and intermediate ignore rules for appendix runs. Existing likely data sources:

```text
outputs/candidate_router/scores/*.parquet
outputs/candidate_router/features/*.parquet
outputs/candidate_router/eval/tables/candidate_router_main_results.csv
outputs/candidate_router/eval/tables/candidate_router_main_results.md
```

For appendix split, intermediates are intentionally ignored:

```text
outputs/appendix_seed20260427/candidate_router/scores/*.parquet
outputs/appendix_seed20260427/candidate_router/features/*.parquet
outputs/appendix_seed20260427/router/shared_targets/*.parquet
outputs/appendix_seed20260427/router/shared_targets_*/*.parquet
```

For the main paper result, use the main `paper_split` score exports, not appendix split, unless the goal is only appendix robustness.

## 3.4 New script to implement

Create:

```text
scripts/eval_score_ensemble_baselines.py
```

Purpose:

Evaluate simple Gate-only / Residual-only score-ensemble baselines on the same full-ranking basis as CA-S2.

## 3.5 Required script behavior

The script should support:

```bash
python scripts/eval_score_ensemble_baselines.py ^
  --split test ^
  --score-dir outputs/candidate_router/scores ^
  --output-dir outputs/score_ensemble/eval ^
  --alphas 0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0
```

If the current score export format does not use `score-dir`, adapt to the actual existing files, but keep output names below.

## 3.6 Required baselines

### B1: Global score interpolation

Formula:

```text
s_mix(q,e) = alpha * s_gate(q,e) + (1 - alpha) * s_residual(q,e)
```

Selection:

- Sweep alpha on dev.
- Pick alpha with highest dev MRR.
- Report test MRR / Hits@1 / Hits@3 / Hits@10 using that alpha.

Output row name:

```text
Global score interpolation
```

### B2: Direction-specific score interpolation

Formula:

```text
if query direction == head:
    s_mix = alpha_head * s_gate + (1 - alpha_head) * s_residual
else:
    s_mix = alpha_tail * s_gate + (1 - alpha_tail) * s_residual
```

Selection:

- Sweep alpha_head and alpha_tail on dev.
- Pick pair with highest dev MRR.
- Report test metrics.

Output row name:

```text
Direction-specific score interpolation
```

### B3: Query-level soft score weighting

Preferred version:

```text
s_mix(q,e) = alpha(q) * s_gate(q,e) + (1 - alpha(q)) * s_residual(q,e)
```

Where `alpha(q)` is predicted from deployable query-level score-distribution features, not from target labels.

If implementing a learned query-level soft model is too costly, implement a fallback version:

```text
Relation-prior-bucket soft interpolation
```

But label it honestly.

Output row names:

```text
Query-level soft score weighting
```

or fallback:

```text
Relation-bucket score interpolation
```

### Optional B4: Rank averaging / RRF

If cheap, implement:

```text
rank_avg(q,e) = 0.5 * rank_gate(q,e) + 0.5 * rank_residual(q,e)
```

or reciprocal rank fusion:

```text
rrf(q,e) = 1 / (k + rank_gate(q,e)) + 1 / (k + rank_residual(q,e))
```

Recommended `k=60` if using RRF.

Output row name:

```text
Reciprocal rank fusion
```

## 3.7 Required output files

Create:

```text
outputs/score_ensemble/eval/score_ensemble_baselines.csv
outputs/score_ensemble/eval/score_ensemble_baselines.json
outputs/score_ensemble/eval/score_ensemble_baselines.md
```

Then create paper table:

```text
docs/paper_tables/table_score_ensemble_baselines.tex
```

## 3.8 Required output columns

CSV/MD should include:

```text
method
level
granularity
selected_on
alpha_policy
mrr
hits1
hits3
hits10
delta_vs_residual
delta_vs_e5
delta_vs_ca_s2
notes
```

Example expected table format:

| Method | Level | Granularity | MRR | Delta vs E5 | Delta vs CA-S2 |
|---|---|---|---:|---:|---:|
| Global score interpolation | ensemble | global | TBD | TBD | TBD |
| Direction-specific score interpolation | ensemble | direction | TBD | TBD | TBD |
| Query-level soft score weighting | ensemble | query | TBD | TBD | TBD |
| CA-S2 score-aware candidate router | router | candidate | 0.3142 | +0.0160 | -- |

## 3.9 How to use in the paper

If ensemble baselines are clearly below CA-S2:

Add to RQ4 after Table 15 or after Table 16:

```latex
To rule out the possibility that CA-S2 merely behaves as a trivial score ensemble, we further compare it against simple Gate-only/Residual-only score-combination baselines, including global score interpolation, direction-specific interpolation, and query-level soft weighting. These baselines use the same fixed expert scores but do not learn candidate-level routing weights. As shown in \Cref{tab:score_ensemble_baselines}, CA-S2 remains stronger, indicating that the gain is not explained by fixed or query-level score averaging alone.
```

Insert table:

```latex
\input{paper_tables/table_score_ensemble_baselines.tex}
```

Path note:

- If compiling inside `docs/paper`, use `\input{paper_tables/table_score_ensemble_baselines.tex}` only if the file is under `docs/paper/paper_tables/`.
- If the file remains under `docs/paper_tables/`, use `\input{../paper_tables/table_score_ensemble_baselines.tex}`.

Current repository uses `docs/paper_tables/` for many table files, so recommended path is:

```latex
\input{../paper_tables/table_score_ensemble_baselines.tex}
```

## 3.10 If a simple ensemble is close to or beats CA-S2

Do **not** hide it.

Interpret honestly:

```latex
The score-ensemble baselines show that a substantial portion of the CA-S2 gain comes from exposing fixed-expert score information. CA-S2 remains useful when it improves over these baselines; if a simple score ensemble is competitive, the result should be interpreted as evidence that deployable expert score patterns, rather than metadata-only query features, are the primary source of recoverable gain.
```

If global/direction-specific ensemble beats CA-S2, revise claim:

```text
Score-aware expert combination is the key, while CA-S2 is one candidate-level implementation.
```

This would weaken method novelty but strengthen honesty.

## 3.11 Acceptance checks

```text
[ ] At least global score interpolation is implemented.
[ ] Direction-specific score interpolation is implemented.
[ ] Query-level soft weighting or a clearly labeled fallback is implemented.
[ ] Dev selection and test reporting are separated.
[ ] Baselines use no hidden target, no target image label, no reciprocal rank as features.
[ ] Results are reported in one compact table.
[ ] The paper explicitly says these baselines test whether CA-S2 is merely a simple score ensemble.
```

---

# Phase 4: Lightweight Protocol Audit / Degree-bias Sanity Check

## 4.1 Goal

Defend against the reviewer question:

> Is Residual-only strong in `tail_no_img` because visual information is missing, or simply because tail targets have higher graph degree and are easier for structure models?

The goal is **not** to prove degree has no effect. The goal is to show that the role--modality asymmetry is not reducible to an unexamined degree artifact.

## 4.2 Recommended scope

Do only a lightweight appendix sanity check.

Recommended output:

```text
docs/protocol_audit/degree_by_target_regime.json
docs/protocol_audit/table_degree_by_target_regime.tex
docs/protocol_audit/protocol_audit_summary.md
```

Optional:

```text
docs/protocol_audit/split_image_availability_summary.json
docs/protocol_audit/table_split_image_availability.tex
```

## 4.3 New script to implement

Create:

```text
scripts/build_protocol_audit_sanity.py
```

## 4.4 Required input paths

Use main split only:

```text
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_train.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_dev.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_test.tsv
data/cache/openbg_img/has_img.pt
```

Optional appendix split support:

```text
data/datasets/openbg_img/appendix_split_seed20260427/OpenBG-IMG_paper_train.tsv
data/datasets/openbg_img/appendix_split_seed20260427/OpenBG-IMG_paper_dev.tsv
data/datasets/openbg_img/appendix_split_seed20260427/OpenBG-IMG_paper_test.tsv
```

## 4.5 Degree definition

Use undirected entity degree from training triples only:

```text
degree(e) = count of training triples where e appears as head or tail
```

Reason:

- Training graph degree is available before evaluation.
- It avoids using test-edge information to compute degree.

Also compute log-degree:

```text
log_degree = log(1 + degree)
```

## 4.6 Target-regime degree table

For each test triple `(h, r, t)`:

Head prediction target:

```text
regime = head_has_img if has_img[h] else head_no_img
target = h
```

Tail prediction target:

```text
regime = tail_has_img if has_img[t] else tail_no_img
target = t
```

For each regime, compute:

```text
count
mean_degree
median_degree
q25_degree
q75_degree
mean_log_degree
median_log_degree
```

Output table:

| Regime | #Queries | Mean degree | Median degree | Q25 | Q75 | Mean log-degree |
|---|---:|---:|---:|---:|---:|---:|
| head_has_img | ... | ... | ... | ... | ... | ... |
| head_no_img | ... | ... | ... | ... | ... | ... |
| tail_no_img | ... | ... | ... | ... | ... | ... |

## 4.7 Optional split image availability summary

For train/dev/test, compute:

```text
head_has_img_ratio
head_no_img_ratio
tail_has_img_ratio
tail_no_img_ratio
unique_entity_image_coverage
```

This can be an appendix table only.

## 4.8 Optional correlation analysis

Only do this if the data is already easy to join:

```text
corr(log_degree, has_img)
corr(log_degree, rr_gain)
corr(has_img, rr_gain)
```

Where:

```text
rr_gain = RR_gate - RR_residual
```

Do not overinterpret correlations. Use as sanity only.

## 4.9 Paper wording if added

Appendix text:

```latex
\section{Additional Protocol Sanity Checks}
\label{app:protocol_sanity}

We add a lightweight degree-based sanity check to examine whether the target-side role--modality pattern is reducible to an unexamined degree artifact. Entity degree is computed only from the training graph, using the number of train triples in which an entity appears as head or tail. The results are reported by test-time target-side regime.

\input{../protocol_audit/table_degree_by_target_regime.tex}

This analysis is not intended to remove all structural confounding from the benchmark. Instead, it verifies that the paper's role--modality interpretation is not made without inspecting the most direct structural alternative explanation. The main conclusions remain protocol-aware: degree and modality availability may both contribute to the bounded-gain structure, while the routing experiments directly test how much deployable gain can be recovered under this protocol.
```

## 4.10 Acceptance checks

```text
[ ] Degree is computed from train only.
[ ] Table is in appendix, not main results.
[ ] Text does not claim degree bias is fully eliminated.
[ ] Text says degree and modality may both contribute.
[ ] The main claim remains protocol-aware.
```

---

# Phase 5: What Not To Do

## 5.1 Do not add a full protocol audit section

Avoid adding a new main section with:

- train/valid/test image availability distribution,
- head/tail target coverage,
- relation heatmap,
- degree bias,
- RR gain correlation,
- relation-level image coverage.

Reason:

The paper will become too broad. The current main contribution is selective activation under protocol-aware bounded gain, not a full benchmark audit paper.

## 5.2 Do not add a relation-level image coverage heatmap unless nearly free

Reason:

- It may be visually complex.
- It may require relation-by-relation explanation.
- It is redundant with current relation-group analysis unless it reveals a clean pattern.

## 5.3 Do not add full stacking-family experiments

Avoid a large table with many new classifiers:

```text
logistic stacking
XGBoost stacking
MLP stacking
calibrated stacking
relation-specific stacking
```

Reason:

This can blur the method line and make CA-S2 look like one point in an endless stacking sweep.

If one stacking baseline is added, label it clearly as optional sanity, not a new main family.

## 5.4 Do not make relation-specific interpolation the main baseline

Per-relation alpha can overfit low-support relations. If relation conditioning is needed, prefer relation-bucket interpolation with support constraints.

---

# Phase 6: Suggested Execution Order

## Step 1: Paper-only low-cost edits

```text
[ ] Add Information Boundary / Leakage Checklist table.
[ ] Add text explaining CA-S2 is non-answer-aware but score-aware.
[ ] Refine head_has_img wording to "relative fusion preference, not high absolute solvability".
[ ] Add sparse candidate-level correction wording in RQ4 / Integrated Discussion.
```

Expected time: low.

Expected value: high.

## Step 2: Score ensemble baselines

```text
[ ] Implement scripts/eval_score_ensemble_baselines.py.
[ ] Run global score interpolation.
[ ] Run direction-specific score interpolation.
[ ] Run query-level soft weighting or relation-bucket fallback.
[ ] Generate outputs/score_ensemble/eval/score_ensemble_baselines.*.
[ ] Generate docs/paper_tables/table_score_ensemble_baselines.tex.
[ ] Add one compact paragraph and table to RQ4 or Appendix C.
```

Expected time: medium.

Expected value: very high.

## Step 3: Optional protocol sanity

```text
[ ] Implement scripts/build_protocol_audit_sanity.py.
[ ] Compute degree by target-side regime.
[ ] Generate docs/protocol_audit/table_degree_by_target_regime.tex.
[ ] Add appendix paragraph only if results are interpretable.
```

Expected time: low to medium.

Expected value: medium.

## Step 4: Final consistency pass

```text
[ ] Ensure main numbers remain unchanged.
[ ] Ensure appendix split remains appendix-only.
[ ] Ensure new ensemble baselines are not mixed with official model-comparison line.
[ ] Ensure CA-S2 is not described as answer-aware.
[ ] Ensure no overclaiming such as "fully robust", "universal", or "image availability solves prediction".
```

---

# Phase 7: Final Target State

After executing this plan, the paper should be able to answer these likely reviewer objections:

## Reviewer objection 1

> The role--modality asymmetry may be a split artifact.

Response:

- Main test regime counts.
- Appendix split robustness check.
- Optional protocol sanity table.

## Reviewer objection 2

> CA-S2 may just be a simple score ensemble.

Response:

- Global score interpolation baseline.
- Direction-specific interpolation baseline.
- Query-level soft weighting baseline.
- CA-S2 remains stronger or the claim is adjusted honestly.

## Reviewer objection 3

> CA-S2 may leak answer-aware information.

Response:

- Information-boundary table.
- Explicit separation among strict clean, score-aware non-answer-aware, and Oracle/post-hoc settings.

## Reviewer objection 4

> The paper overstates image-supported regimes.

Response:

- Revised wording: `head_has_img` indicates relative fusion preference, not high absolute solvability.
- Mechanism: sparse candidate-level correction, especially in `tail_no_img`.

---

# Phase 8: Minimal Version If Time Is Limited

If time is tight, execute only:

```text
[ ] Add Leakage Checklist table.
[ ] Add sparse correction wording.
[ ] Add global + direction-specific score interpolation baselines.
```

This minimal version closes the highest-risk gaps.

Do not spend time on:

```text
[ ] relation heatmap
[ ] full protocol audit
[ ] full stacking family
[ ] seven-model rerun
```
