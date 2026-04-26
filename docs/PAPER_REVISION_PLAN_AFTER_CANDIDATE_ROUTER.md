# Paper Revision Plan after Candidate-aware / Score-aware Router Experiments

## 0. Purpose of This Revision Plan

This document defines the next major manuscript revision after the completion of the candidate-aware and score-aware router experiments.

The previous manuscript line was centered on:

> protocol-aware bounded-gain diagnosis -> structured clean query-level routing -> regression-based clean router.

The latest experiments change the strength and direction of the paper. The strongest new result is no longer the query-level regression clean router. Instead, the new evidence shows that **score-aware candidate-level selective activation** substantially outperforms all query-level clean routing variants.

This revision plan explains how to update the paper without overstating the result or mixing evaluation lines.

---

## 1. Updated Core Claim

### 1.1 Old Core Claim

The current manuscript still mainly claims:

> Under the current OpenBG-IMG protocol, multimodal gain is bounded. Naive clean routing is too coarse, while direction-specific thresholding and regression-based gain prediction recover a modest but statistically supported amount of deployable clean gain.

This claim remains true, but it is now incomplete.

### 1.2 New Core Claim

The revised paper should claim:

> Under the current OpenBG-IMG protocol, strict query-level clean routing recovers only modest gain because metadata-only query signals are too coarse. When routing is moved to the candidate level and allowed to use deployable score-aware expert signals, selective activation recovers substantially more of the Oracle headroom. The strongest score-aware candidate router reaches 0.3142 MRR, outperforming the strongest query-level clean router by +0.0160 MRR and recovering 51.9% of the Oracle headroom.

### 1.3 Safe Wording

Use:

- **score-aware candidate-level selective activation**
- **deployable non-answer-aware score-based routing**
- **candidate-level soft expert weighting**
- **expert score-distribution / confidence / disagreement signals**
- **partial recovery of Oracle headroom**

Avoid:

- `CA-S2 is a clean router`
- `CA-S2 closes the Oracle gap`
- `candidate modality metadata is the main reason for the gain`
- `CA-S2 significantly outperforms CA-S3`
- `multimodal fusion is globally superior`

---

## 2. Updated Evidence Summary

## 2.1 Main Result

Latest table:

| Method | Routing level | Feature type | MRR | Delta vs Residual | Delta vs E5 | Oracle gap recovered |
|---|---|---|---:|---:|---:|---:|
| Residual-only | fixed | structural | 0.2930 | 0.0000 | -0.0051 | 0.0% |
| Regression-based clean router | query | strict clean | 0.2982 | +0.0051 | 0.0000 | 12.6% |
| CA-S1 clean candidate | candidate | clean candidate | 0.2503 ± 0.0054 | -0.0427 | -0.0479 | -105.0% |
| CA-S2 score-aware | candidate | score-aware | **0.3142 ± 0.0010** | **+0.0211** | **+0.0160** | **51.9%** |
| CA-S3 clean + score | candidate | clean + score-aware | 0.3135 ± 0.0010 | +0.0204 | +0.0153 | 50.2% |
| Oracle routing | oracle | answer-aware | 0.3337 | +0.0407 | +0.0356 | 100.0% |

### Interpretation

- E5 remains the strongest **strict query-level clean** result.
- CA-S2 is the strongest **score-aware candidate-level** result.
- CA-S3 is statistically similar to CA-S2, but adding static clean candidate metadata does not provide reliable additional gain.
- CA-S1 shows that candidate metadata alone is insufficient and globally harmful.

---

## 2.2 Significance Evidence

Latest paired bootstrap evidence:

| Comparison | Delta MRR | 95% CI |
|---|---:|---|
| CA-S1 vs Residual-only | -0.0427 | [-0.0452, -0.0404] |
| CA-S1 vs E5 | -0.0479 | [-0.0503, -0.0455] |
| CA-S2 vs Residual-only | +0.0211 | [+0.0191, +0.0232] |
| CA-S2 vs E5 | +0.0160 | [+0.0139, +0.0181] |
| CA-S3 vs Residual-only | +0.0204 | [+0.0184, +0.0226] |
| CA-S3 vs E5 | +0.0153 | [+0.0132, +0.0174] |
| CA-S2 vs CA-S3 | +0.0007 | [-0.0002, +0.0014] |

### Interpretation

- CA-S2 and CA-S3 both significantly outperform E5 and Residual-only.
- CA-S2 does **not** significantly outperform CA-S3 because the CI crosses zero.
- The paper should describe CA-S2 and CA-S3 as similarly strong score-aware candidate routers, with CA-S2 being the numerically best and more parsimonious variant.

---

## 2.3 Subgroup Evidence

Latest subgroup pattern:

| Method | Regime | Delta vs Residual |
|---|---|---:|
| CA-S1 | head_has_img | +0.0034 |
| CA-S1 | head_no_img | +0.0125 |
| CA-S1 | tail_no_img | -0.0916 |
| CA-S2 | head_has_img | +0.0015 |
| CA-S2 | head_no_img | +0.0026 |
| CA-S2 | tail_no_img | +0.0404 |
| CA-S3 | head_has_img | +0.0017 |
| CA-S3 | head_no_img | +0.0024 |
| CA-S3 | tail_no_img | +0.0389 |

### Interpretation

The previous story that multimodal gain is most visible in `head_has_img` remains true for raw fusion behavior, but the new router gain mainly comes from `tail_no_img`.

The revised story should say:

> Static candidate modality features help some head-side regimes but fail badly in the dominant tail-side regime. Score-aware candidate routing instead improves the structure-dominant tail_no_img regime, suggesting that the key signal is not image availability itself, but expert score patterns and candidate-level confidence/disagreement.

---

## 2.4 Alpha Behavior Evidence

Overall alpha behavior:

| Method | Mean alpha | Target alpha | Target/mean alpha | MRR gain vs Residual |
|---|---:|---:|---:|---:|
| CA-S1 | 0.3811 | 0.2570 | 0.6744 | -0.0427 |
| CA-S2 | 0.0470 | 0.1964 | 4.1768 | +0.0211 |
| CA-S3 | 0.0480 | 0.2016 | 4.1997 | +0.0204 |

Key tail-side alpha behavior:

| Method | Regime | Mean alpha | Target alpha | Target/mean alpha | MRR gain vs Residual |
|---|---|---:|---:|---:|---:|
| CA-S2 | tail_no_img | 0.0054 | 0.3788 | 70.5009 | +0.0404 |
| CA-S3 | tail_no_img | 0.0070 | 0.3871 | 55.0089 | +0.0389 |

### Interpretation

CA-S2/CA-S3 perform sparse candidate-level correction:

- they assign very low average fusion weight to most candidates;
- they assign much higher fusion weight to target candidates in evaluation diagnostics;
- the largest improvement occurs in `tail_no_img`, where broad multimodal activation would normally be risky.

Important caveat:

> Target alpha is an evaluation-time diagnostic, not a feature used by the router.

---

## 3. Revised Paper Structure

Recommended active paper structure:

```text
1. Introduction
2. Related Work
3. Task Setting and Method
   3.1 Task setting and protocol
   3.2 Query-level clean selective activation
   3.3 Candidate-level score-aware selective activation
   3.4 Training objectives and legality constraints
   3.5 Evaluation lines and bootstrap testing
4. Experiments and Discussion
   4.1 Experimental setup
   4.2 RQ1: Where does multimodal gain appear?
   4.3 RQ2: Why does strict query-level clean routing remain limited?
   4.4 RQ3: Can score-aware candidate routing recover more gain?
   4.5 Ablation: clean candidate vs score-aware candidate features
   4.6 Behavior analysis: sparse alpha and regime-specific correction
   4.7 Remaining Oracle gap and limitations
5. Conclusion
```

---

## 4. Section-by-section Revision Plan

# 4.1 Abstract

## Current Problem

The current abstract stops at regression-based clean routing and still says the contribution is mainly a transition from bounded-gain diagnosis to structured clean routing.

## Required Changes

Rewrite the abstract into four movements:

1. Protocol-aware bounded-gain diagnosis.
2. Strict query-level clean routing gives modest gain.
3. Score-aware candidate-level routing substantially improves performance.
4. Oracle gap remains, so the claim is still controlled.

## Key Numbers to Include

- E5 regression clean router: `0.2982 MRR`
- CA-S2 score-aware candidate router: `0.3142 ± 0.0010 MRR`
- Delta vs E5: `+0.0160 MRR`, 95% CI `[+0.0139, +0.0181]`
- Oracle headroom recovered: `51.9%`

## Suggested Abstract Claim

> We further show that the limitation of strict query-level clean routing is not the end of selective activation. A score-aware candidate-level router that uses non-answer-aware expert score signals reaches 0.3142 MRR, outperforming the strongest query-level clean router by +0.0160 MRR and recovering 51.9% of the Oracle headroom.

---

# 4.2 Introduction

## Current Problem

The Introduction currently defines only three research questions and all three point toward clean query-level routing. The new experiments require a fourth question or a revised RQ3.

## Required Changes

### Option A: Four RQs

Recommended.

```text
RQ1. Where does multimodal gain actually appear?
RQ2. Why does naive strict clean routing fail?
RQ3. How far can structured query-level clean routing go?
RQ4. Can candidate-level score-aware routing recover more deployable gain?
```

### Option B: Keep Three RQs

Less recommended.

Make RQ3 broader:

```text
RQ3. How should bounded gain be exploited once query-level clean routing is known to be limited?
```

Then include both E5 and CA-S2 under RQ3.

## Contribution Table Update

Replace old three-row problem-response-validation table with:

| Fine-grained problem | Proposed response | Validation |
|---|---|---|
| Protocol creates role--modality asymmetry | Protocol-aware bounded-gain diagnosis | Target-side subgroup and relation analysis |
| Naive clean routing is too coarse | Direction-specific query-level clean routing | Clean routing comparison and bootstrap CI |
| Binary query-level supervision is too coarse | Regression-based clean gain prediction | Best strict clean result, 0.2982 MRR |
| Query-level metadata remains insufficient | Score-aware candidate-level soft routing | 0.3142 MRR, +0.0160 over E5, 51.9% Oracle recovery |

## Contribution List Update

Replace current contribution 3 with two separate contributions:

3. **Target-aligned strict clean routing.** We show that regression-based gain prediction is the strongest strict query-level clean policy, but its gain remains modest.
4. **Score-aware candidate-level selective activation.** We introduce candidate-level soft routing using non-answer-aware expert score signals and show that it substantially recovers more Oracle headroom than query-level clean routing.

---

# 4.3 Related Work

## Required Additions

Add or expand one subsection:

```text
2.3 Selective Activation, Expert Routing, and Score-aware Model Combination
```

Current related work probably already covers selective activation and MoE-style routing. It now needs one extra angle:

- score-based ensembling / stacking;
- learning-to-rank with multiple scorers;
- confidence-aware routing;
- candidate-level re-ranking.

## Key Positioning

The paper is not proposing a large-scale MoE model. It is a small, analysis-driven, candidate-level expert-weighting method under MMKGC protocol asymmetry.

Use wording:

> Unlike generic ensemble or MoE systems, our router is motivated by a protocol-aware bounded-gain diagnosis and is evaluated under strict separation between metadata-only clean routing, score-aware deployable routing, and Oracle answer-aware routing.

---

# 4.4 Method Section

The method section requires the largest rewrite.

## 4.4.1 Keep Existing Query-level Clean Routing

Keep the current sections on:

- clean legality constraint;
- naive global-threshold routing;
- direction-specific thresholding;
- regression-based gain prediction.

But reposition them as:

> Stage 1: strict metadata-only query-level routing.

Do not present E5 as the final method anymore. Present it as the strongest strict clean baseline.

---

## 4.4.2 Add New Subsection: Candidate-level Score-aware Selective Activation

Suggested subsection title:

```text
3.3 Candidate-level Score-aware Selective Activation
```

### Core Equations

Define fixed experts:

```math
s_f(q,e), \quad s_s(q,e)
```

where:

- `s_f` is Gate-only;
- `s_s` is Residual-only.

Define candidate-level router:

```math
\alpha(q,e)=\sigma(g_\theta(x_q, x_e, z_{q,e}))
```

where:

- `x_q` = query-level legal features;
- `x_e` = candidate metadata features;
- `z_{q,e}` = score-aware features from fixed experts.

Final score:

```math
s_{mix}(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e)
```

### Feature Groups

Define three candidate router variants:

| Variant | Features | Purpose |
|---|---|---|
| CA-S1 | clean candidate metadata | test static candidate modality |
| CA-S2 | score-aware features | test expert score/confidence signals |
| CA-S3 | clean + score-aware | combined feature set |

### Legality Explanation

Write explicitly:

> CA-S2 and CA-S3 are not strict metadata-only clean routers. They are score-aware deployable routers. They use expert scores and ranks computed for all candidates after running the fixed experts, but they do not use the hidden target, reciprocal rank, target image label, or answer-aware expert outcomes.

---

## 4.4.3 Add Pairwise Training Objective

Add pairwise ranking objective:

```math
\mathcal{L}_{pair}=-\log \sigma(s_{mix}(q,e^+)-s_{mix}(q,e^-))
```

Explain negative sampling:

- hard negatives from Gate-only top-K;
- hard negatives from Residual-only top-K;
- random negatives from union top-K.

Mention:

- training uses top-K hard candidate sets for efficiency;
- final CA-S1/CA-S2/CA-S3 reporting uses full filtered ranking over all entities.

---

## 4.4.4 Update Evaluation Lines

The paper now needs four evaluation lines:

| Evaluation line | Used for | Deployability |
|---|---|---|
| Official model-comparison line | original base models | standard model eval |
| Strict query-level clean line | clean rule, global threshold, direction-specific, E5 | metadata-only deployable |
| Score-aware candidate line | CA-S1/CA-S2/CA-S3 | non-answer-aware deployable after expert scoring |
| Oracle/post-hoc line | Oracle routing | not deployable |

Important:

- Do not mix strict clean line and score-aware candidate line as if both are the same clean setting.
- Use CA-S2 as the strongest score-aware candidate result.
- Use E5 as the strongest strict query-level clean result.

---

# 4.5 Experiments Section

The current `05_experiments.md` must be heavily revised.

## 4.5.1 New Experiment Flow

Recommended flow:

```text
4.1 Setup and evaluation lines
4.2 Base model comparison and bounded-gain diagnosis
4.3 Strict query-level clean routing results
4.4 Candidate-level score-aware routing results
4.5 Feature ablation: CA-S1 vs CA-S2 vs CA-S3
4.6 Alpha behavior and sparse candidate correction
4.7 Oracle gap and limitations
```

---

## 4.5.2 Main Result Table

Add main table from:

```text
outputs/candidate_router/eval/tables/candidate_router_main_results.md
```

In the paper, this should replace the old table where E5 was the strongest final result.

Recommended caption:

> Main selective activation results under the OpenBG-IMG paper protocol. E5 denotes the strongest strict query-level clean router, while CA-S1/CA-S2/CA-S3 are candidate-level routers evaluated under full filtered ranking. CA-S2 is score-aware but non-answer-aware; Oracle routing is a post-hoc upper bound.

---

## 4.5.3 Significance Table

Add significance table from:

```text
outputs/candidate_router/eval/tables/candidate_router_significance.md
```

In the discussion, emphasize:

- CA-S2 significantly outperforms E5 and Residual-only.
- CA-S3 significantly outperforms E5 and Residual-only.
- CA-S2 does not significantly outperform CA-S3.

Do not overclaim CA-S2 > CA-S3.

---

## 4.5.4 Subgroup Table

Add subgroup table from:

```text
outputs/candidate_router/eval/tables/candidate_router_subgroup_results.md
```

Interpretation:

- CA-S1 helps head-side regimes but fails strongly in `tail_no_img`.
- CA-S2/CA-S3 improve mainly through `tail_no_img`.
- This shifts the method explanation away from static image availability and toward expert score patterns.

---

## 4.5.5 Alpha Behavior Table

Add alpha behavior table from:

```text
outputs/candidate_router/eval/tables/candidate_router_alpha_overall.md
outputs/candidate_router/eval/tables/candidate_router_alpha_behavior.md
```

Interpretation:

- CA-S2 and CA-S3 have low mean alpha but high target alpha.
- In `tail_no_img`, CA-S2 has mean alpha `0.0054` and target alpha `0.3788`.
- This supports sparse candidate-level correction.

Important caveat:

> target alpha is diagnostic only; it is computed after evaluation to understand behavior.

---

# 4.6 Discussion Section

The discussion should be rewritten around four points.

## 4.6.1 Query-level clean routing is useful but limited

Say:

> E5 shows that strict query-level clean routing is statistically reliable but modest. This means query-level metadata signals expose part of the multimodal-gain boundary, but not enough to recover most Oracle headroom.

## 4.6.2 Candidate metadata alone is insufficient

Say:

> CA-S1 fails globally despite helping head-side regimes, because it severely degrades the dominant tail_no_img regime. Static modality availability cannot by itself decide expert usefulness.

## 4.6.3 Score-aware candidate routing is the main improvement

Say:

> CA-S2 reaches 0.3142 MRR and recovers 51.9% of Oracle headroom. This shows that expert score patterns provide deployable signals that are invisible to strict metadata-only clean routing.

## 4.6.4 Sparse alpha behavior explains the gain

Say:

> The router does not broadly activate fusion. It keeps average fusion weights low while assigning higher fusion weights to the target candidates in diagnostic analysis, especially in tail_no_img. This suggests sparse candidate-level correction rather than global multimodal activation.

---

# 4.7 Limitations Section

Update limitations with these points:

1. **Protocol specificity.** The conclusion is tied to the current OpenBG-IMG split and role--modality asymmetry.
2. **Score-aware deployability cost.** CA-S2 requires running both experts and computing candidate scores before routing.
3. **Not strict clean metadata-only.** CA-S2 is deployable but not strict query-level clean.
4. **Training/evaluation distinction.** Router training uses top-K hard candidate sets, while final reporting uses full filtered ranking.
5. **Remaining Oracle gap.** Even CA-S2 at 0.3142 remains below Oracle 0.3337.
6. **Fixed expert pair.** The router is tested only between Gate-only and Residual-only.

---

# 4.8 Conclusion

The conclusion should no longer end with regression-based clean routing as the final positive result.

## Required Final Message

Use this structure:

1. Bounded multimodal gain exists under OpenBG-IMG protocol asymmetry.
2. Strict query-level clean routing recovers modest but reliable gain.
3. Candidate-level score-aware routing recovers substantially more gain.
4. The strongest score-aware candidate router reaches 0.3142 MRR and recovers 51.9% of Oracle headroom.
5. Oracle gap remains, so the work is a partial but meaningful step.

## Suggested Final Paragraph

> Overall, the results suggest that the main deployable bottleneck is not simply whether multimodal evidence is available, but whether the system can observe sufficiently fine-grained expert behavior at inference time. Under the current OpenBG-IMG protocol, strict query-level metadata-only routing recovers only modest gain, whereas score-aware candidate-level routing recovers substantially more of the Oracle headroom. Future MMKGC research should therefore study not only richer multimodal fusion, but also candidate-level selective activation and deployable expert-confidence signals under incomplete modality support.

---

## 5. Tables and Figures to Update

## 5.1 Main Tables

Required tables:

1. Dataset/protocol statistics table.
2. Official base model comparison table.
3. Strict query-level clean routing table.
4. Candidate router main result table.
5. Candidate router significance table.
6. Candidate router subgroup table.
7. Alpha behavior table.

## 5.2 Recommended Figures

### Figure 1: Protocol role--modality asymmetry

Keep existing figure.

### Figure 2: Selective activation progression

Recommended visual flow:

```text
Always-on fusion
    -> query-level clean routing
    -> score-aware candidate-level routing
    -> Oracle upper bound
```

### Figure 3: Oracle headroom recovery

Bar chart:

```text
Residual-only
E5 regression clean
CA-S1
CA-S2
CA-S3
Oracle
```

### Figure 4: Alpha behavior by regime

Plot:

- mean alpha;
- target alpha;
- grouped by method and target regime.

Use this only if space allows.

---

## 6. Detailed Editing Checklist

## 6.1 Abstract

- [ ] Replace old final claim about structured clean routing as the endpoint.
- [ ] Add CA-S2 result: `0.3142 ± 0.0010 MRR`.
- [ ] Mention `+0.0160` over E5 and `51.9%` Oracle headroom recovery.
- [ ] Distinguish strict clean routing from score-aware candidate routing.

## 6.2 Introduction

- [ ] Revise central problem to include query-level clean limitation.
- [ ] Add RQ4 or broaden RQ3.
- [ ] Replace contribution table with four-row version.
- [ ] Add contribution for score-aware candidate routing.
- [ ] Reduce wording that implies E5 is the final strongest method.

## 6.3 Related Work

- [ ] Add candidate-level re-ranking / score-aware routing discussion.
- [ ] Clarify difference from generic MoE.
- [ ] Clarify difference from post-hoc Oracle selection.

## 6.4 Method

- [ ] Keep query-level clean routing as Stage 1.
- [ ] Add candidate-level score-aware selective activation as Stage 2.
- [ ] Add alpha equation.
- [ ] Add mixed score equation.
- [ ] Add pairwise ranking loss.
- [ ] Define CA-S1 / CA-S2 / CA-S3.
- [ ] Add legality boundary for score-aware deployable routing.
- [ ] Add training on top-K hard candidates and final full-ranking evaluation.

## 6.5 Experiments

- [ ] Reorganize around strict clean vs score-aware candidate lines.
- [ ] Keep E5 as strongest strict clean result.
- [ ] Add CA-S2 as strongest score-aware candidate result.
- [ ] Add CA-S1 negative result.
- [ ] Add CA-S2 vs CA-S3 non-significant difference.
- [ ] Add subgroup interpretation.
- [ ] Add alpha behavior analysis.

## 6.6 Discussion

- [ ] Explain why E5 was modest.
- [ ] Explain why CA-S1 fails globally.
- [ ] Explain why score-aware features dominate.
- [ ] Explain sparse candidate-level correction.
- [ ] Discuss remaining Oracle gap.

## 6.7 Conclusion

- [ ] Update final takeaway from structured clean routing to score-aware candidate routing.
- [ ] Keep claim controlled and protocol-aware.
- [ ] Mention remaining Oracle gap.

---

## 7. Revised Claim Hierarchy

Use this hierarchy to avoid overclaiming.

### Strong Claim

> Score-aware candidate-level routing substantially outperforms strict query-level clean routing and recovers about half of the Oracle headroom.

Supported by:

- CA-S2 MRR `0.3142 ± 0.0010`;
- delta vs E5 `+0.0160`;
- CI `[+0.0139, +0.0181]`;
- Oracle headroom recovered `51.9%`.

### Medium Claim

> Candidate-level routing is useful only when it has score-aware expert evidence.

Supported by:

- CA-S1 fails overall;
- CA-S2 succeeds;
- CA-S3 does not reliably improve over CA-S2.

### Careful Claim

> The gain mainly comes from sparse correction in the tail_no_img regime.

Supported by:

- CA-S2 tail_no_img delta `+0.0404`;
- alpha behavior: low mean alpha, high target alpha.

### Claim to Avoid

> Static candidate modality features solve routing.

Contradicted by CA-S1.

### Claim to Avoid

> CA-S2 is a clean router.

Incorrect. It is score-aware deployable, not strict metadata-only clean.

---

## 8. Recommended New Paper Title Options

Current title may still work if it emphasizes clean routing too much, but the new results justify a title update.

Options:

1. **Protocol-aware Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Support**
2. **Score-aware Candidate-level Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Support**
3. **From Bounded Multimodal Gain to Score-aware Candidate Routing in Multimodal Knowledge Graph Completion**
4. **Recovering Bounded Multimodal Gain with Score-aware Candidate Routing under Protocol-shaped Missing Visual Support**

Recommended title:

> **Recovering Bounded Multimodal Gain with Score-aware Candidate Routing under Incomplete Visual Support**

This title reflects the new strongest contribution more directly.

---

## 9. Suggested Rewrite Order

Recommended order of actual editing:

```text
Step 1. Rewrite Method section first.
Step 2. Rewrite Experiments section around new tables.
Step 3. Rewrite Introduction contributions and RQs.
Step 4. Rewrite Abstract.
Step 5. Rewrite Conclusion and Limitations.
Step 6. Update Related Work.
Step 7. Final pass for terminology consistency.
```

Reason:

- Method and Experiments determine the true claim.
- Introduction and Abstract should only be finalized after the result line is stable.

---

## 10. Terminology Consistency Table

| Old term | New recommended term | Reason |
|---|---|---|
| best clean router | strongest strict query-level clean router | avoid mixing with CA-S2 |
| clean routing result | strict metadata-only clean routing result | more precise |
| CA-S3 full-ranking | CA-S3 clean+score candidate router | clearer ablation name |
| candidate-aware router | candidate-level selective activation | more academic |
| score-aware router | score-aware candidate-level router | core contribution |
| Oracle gap | remaining answer-aware headroom | clearer limitation |

---

## 11. Final One-sentence Revision Direction

The revised paper should be rewritten around the following sentence:

> Under protocol-shaped incomplete visual support, strict query-level clean routing recovers only modest bounded multimodal gain, while score-aware candidate-level selective activation substantially improves performance by using non-answer-aware expert score patterns for sparse candidate-level correction.
