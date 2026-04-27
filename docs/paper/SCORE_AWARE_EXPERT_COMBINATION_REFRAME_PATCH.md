# Patch: Reframe the Paper Around Score-aware Expert Combination

This document is a directly executable manuscript-revision patch based on the latest full-ranking score-ensemble results.

The key new result is:

| Method | MRR | Delta vs CA-S2 |
|---|---:|---:|
| Global score interpolation | 0.3408 | +0.0266 |
| Direction-specific score interpolation | 0.3421 | +0.0279 |
| Relation-specific score interpolation | 0.3428 | +0.0286 |
| Query-level soft score weighting | 0.3055 | -0.0087 |
| CA-S2 score-aware candidate router | 0.3142 | -- |

Therefore, the paper should no longer frame CA-S2 as the final strongest method. The strongest interpretation is now:

> Strict metadata-only clean routing recovers only modest gain. Once non-answer-aware fixed-expert score information is exposed, score-aware expert combination recovers much larger gain. CA-S2 is a learned candidate-level score-aware implementation, but simple full-ranking score interpolation is even stronger. The main bottleneck is the information boundary, not router architecture complexity.

---

## 0. New Main Positioning

### Old positioning

```text
bounded-gain diagnosis
-> strict clean routing modest gain
-> CA-S2 candidate-level router recovers much more gain
-> CA-S2 is the main method
```

### New positioning

```text
bounded-gain diagnosis
-> strict metadata-only clean routing modest gain
-> learned CA-S2 shows score-aware candidate routing is stronger
-> simple full-ranking score interpolation is even stronger
-> main finding: score-aware expert-combination signals, not router complexity, are the key bottleneck
```

### Use these three evidence levels consistently

```text
1. Strict metadata-only clean routing
   Uses query metadata only. Does not use candidate scores.

2. Score-aware non-answer-aware expert combination
   Uses fixed-expert scores / ranks / confidence / disagreement.
   Does not use hidden target, target image availability, reciprocal rank, or test labels.

3. Answer-aware / hard-selection Oracle
   Uses answer-aware outcomes for post-hoc analysis only.
   It is an upper bound for hard expert selection, not for score interpolation.
```

---

## 1. Title Patch

### Current title

```latex
Protocol-Aware Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Support
```

### Recommended title

```latex
Protocol-Aware Score-aware Expert Combination for Multimodal Knowledge Graph Completion under Incomplete Visual Support
```

### Conservative alternative

```latex
Protocol-Aware Selective Activation and Score-aware Expert Combination for Multimodal Knowledge Graph Completion under Incomplete Visual Support
```

### Recommendation

Use the first title if the manuscript is now being reframed around score-aware expert combination. Use the conservative alternative only if you want to preserve continuity with older drafts.

---

## 2. Abstract Patch

### Reason

The current abstract still treats CA-S2 as the main final result. This is no longer accurate because simple full-ranking score interpolation reaches up to 0.3428 MRR, which is higher than CA-S2 at 0.3142.

### Replace the abstract with

```latex
\begin{abstract}
Multimodal knowledge graph completion (MMKGC) aims to improve link prediction by incorporating auxiliary modalities such as text and images, yet in product-oriented knowledge graphs visual support is often incomplete and multimodal evidence does not help uniformly. We study this problem on OpenBG-IMG under a unified filtered-ranking protocol with \texttt{direction=both} and show that the current split induces role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This creates a bounded-gain structure in which relative fusion preference is most visible in image-supported head-side regimes, although this does not imply high absolute solvability and the global gain remains bounded. We first show that strict metadata-only query-level clean routing recovers only modest gain: direction-specific thresholding reaches 0.2974 MRR, and regression-based gain prediction reaches 0.2982 MRR with positive paired bootstrap evidence. We then show that this limitation is not the end of deployable expert combination. Once fixed-expert score information is exposed, score-aware non-answer-aware combination becomes substantially stronger: a learned candidate-level router reaches 0.3142 $\pm$ 0.0010 MRR, while simple full-ranking score interpolation reaches up to 0.3428 MRR. These results sharpen the main bottleneck: strict query-level metadata is too coarse, whereas fixed-expert score patterns expose much more usable deployable signal. The strongest gain does not require a more complex router; rather, it shows that the key boundary is whether score-aware expert-combination information is available without leaking answer-aware target information.
\end{abstract}
```

### Key effect

This abstract:

- Keeps the bounded-gain diagnosis.
- Keeps E5 clean routing.
- Keeps CA-S2 as a learned candidate-level result.
- Promotes simple score interpolation to the strongest score-aware result.
- Makes the new main claim about the information boundary.

---

## 3. Keywords Patch

### Current

```latex
\textbf{Keywords:} multimodal knowledge graph completion; selective activation; score-aware routing; candidate-level routing; incomplete visual support; protocol-aware evaluation
```

### Replace with

```latex
\textbf{Keywords:} multimodal knowledge graph completion; score-aware expert combination; selective activation; score interpolation; incomplete visual support; protocol-aware evaluation
```

---

## 4. Introduction: RQ4 Patch

### Find current RQ4

```latex
\textbf{RQ4. Can candidate-level score-aware routing recover more deployable gain?} Finally, we move from query-level hard selection to candidate-level soft expert weighting using non-answer-aware expert score signals.
```

### Replace with

```latex
\textbf{RQ4. Can score-aware expert combination recover more deployable gain?} Finally, we move beyond strict metadata-only query-level routing and allow deployable non-answer-aware access to fixed-expert score patterns. We test both learned candidate-level routing and simple full-ranking score interpolation to determine whether the main bottleneck is router complexity or the score-aware information boundary itself.
```

---

## 5. Introduction: Experimental Story Patch

### Find sentence like

```latex
Candidate-level score-aware routing then recovers substantially more headroom, with CA-S2 reaching 0.3142 MRR and remaining statistically comparable to the combined clean-plus-score variant CA-S3.
```

### Replace with

```latex
Score-aware expert combination then recovers substantially more gain than strict clean routing. A learned candidate-level router, CA-S2, reaches 0.3142 MRR and remains statistically comparable to the combined clean-plus-score variant CA-S3. More importantly, simple full-ranking score interpolation reaches up to 0.3428 MRR, showing that the key source of recoverable gain is not necessarily router complexity, but access to non-answer-aware fixed-expert score patterns.
```

---

## 6. Introduction: Main Claim Patch

### Find sentence like

```latex
Larger deployable gain requires candidate-level score-aware routing, while the remaining Oracle gap still marks unresolved headroom.
```

### Replace with

```latex
Larger deployable gain requires moving beyond metadata-only query features to score-aware non-answer-aware expert combination. The strongest results further show that a complex learned router is not strictly necessary: simple score interpolation can outperform the learned CA-S2 router, indicating that the primary bottleneck is the information boundary rather than router architecture complexity.
```

---

## 7. Problem--Response--Validation Table Patch

### Current fourth row

```latex
Query-level metadata remains insufficient. & Score-aware candidate-level soft routing. & 0.3142 MRR, +0.0160 over E5, and 51.9\% Oracle recovery. \\
```

### Replace with

```latex
Query-level metadata remains insufficient. & Score-aware non-answer-aware expert combination. & CA-S2 reaches 0.3142 MRR, while full-ranking score interpolation reaches up to 0.3428 MRR. \\
```

---

## 8. Contribution 4 Patch

### Current contribution 4

```latex
\item \textbf{Score-aware candidate-level selective activation.} We introduce candidate-level soft routing using non-answer-aware expert score signals and show that it recovers substantially more Oracle headroom than strict query-level clean routing.
```

### Replace with

```latex
\item \textbf{Score-aware expert combination under deployable information boundaries.} We show that exposing non-answer-aware fixed-expert score information substantially changes the recoverable-gain boundary. CA-S2 demonstrates that learned candidate-level score-aware routing improves over strict metadata-only clean routing, while simple full-ranking score interpolation further reveals that much of the gain comes from score-aware expert combination itself rather than router complexity.
```

### Also replace the contribution-summary sentence

Current sentence may look like:

```latex
strict clean routing recovers part of the bounded multimodal gain, score-aware candidate routing recovers substantially more, and oracle-level separability still marks unresolved deployable headroom.
```

Replace with:

```latex
strict clean routing recovers part of the bounded multimodal gain, score-aware expert combination recovers substantially more, and answer-aware Oracle analyses remain diagnostic references rather than deployable evidence. Since score interpolation can create rankings not identical to either fixed expert, the hard-selection Oracle should not be interpreted as an upper bound for score-level combination.
```

---

## 9. Related Work Patch

### Find

```latex
The new candidate-level routing line adds a second deployable but less restrictive setting.
```

### Replace with

```latex
The new score-aware expert-combination line adds a second deployable but less restrictive setting.
```

### Find

```latex
We then extend this diagnosis into selective activation under explicit information boundaries, moving from strict query-level clean routing to score-aware candidate-level routing.
```

### Replace with

```latex
We then extend this diagnosis into selective activation under explicit information boundaries, moving from strict metadata-only query-level routing to score-aware non-answer-aware expert combination.
```

---

## 10. Method Section Title Patch

### Current

```latex
\subsection{Candidate-level Score-aware Selective Activation}
```

### Replace with

```latex
\subsection{Score-aware Expert Combination}
```

---

## 11. Method Section Opening Patch

### Replace the opening of Section 3.4 with

```latex
Strict query-level clean routing deliberately excludes all candidate scores. This keeps the setting maximally clean, but it also makes the router coarse: it must decide whether fusion helps before seeing how either expert scores the candidate set. We therefore add a second deployable but less restrictive line: score-aware non-answer-aware expert combination. In this line, both fixed experts are first evaluated over candidate entities, and the combination policy may use their score patterns without using the hidden target, target image availability, reciprocal rank, or test labels during router training.

This line includes two families. The first family consists of simple score-interpolation baselines that combine the fixed Gate-only and Residual-only scores with global, direction-specific, or relation-specific weights selected on the development split. The second family consists of learned candidate-level score-aware routers, CA-S2 and CA-S3, which predict candidate-specific fusion weights from non-answer-aware score, rank, confidence, and disagreement features. This design allows us to test whether the main gain comes from learning a more complex router or from exposing fixed-expert score information at all.
```

---

## 12. Add Score Interpolation Formula

### Insert before the learned CA-S2 formula

```latex
For simple score interpolation, the mixed score is
\[
s_{\text{mix}}(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e),
\]
where $s_f$ and $s_s$ denote the fixed fusion and structural expert scores. In the global variant, $\alpha(q)=\alpha$ is a single scalar selected on the development split. In the direction-specific variant, $\alpha(q)=\alpha_{\text{head}}$ for head prediction and $\alpha(q)=\alpha_{\text{tail}}$ for tail prediction. In the relation-specific variant, $\alpha(q)=\alpha_r$ is selected per relation when development support is sufficient, with low-support relations falling back to the global weight.

For the learned candidate-level router, the fusion weight becomes candidate-specific:
\[
\alpha(q,e)=\sigma(g_\theta(x_q,x_e,z_{q,e})).
\]
```

---

## 13. Information Boundary Table Patch

### Caption change

Current phrase:

```latex
A check mark means that the signal is available to that setting at inference or analysis time.
```

Replace with:

```latex
A check mark means that the signal is used or permitted by that setting under the reported evidence line.
```

### Static candidate metadata row

If current row is:

```latex
Static candidate modality metadata & -- & -- & \checkmark \\
```

Replace with:

```latex
Static candidate modality metadata & -- & optional in CA-S3 & \checkmark \\
```

Reason: CA-S2 does not use static candidate metadata, but CA-S3 does.

---

## 14. RQ4 Title Patch

### Current

```latex
\subsection{RQ4: Can candidate-level score-aware routing recover more deployable gain?}
```

### Replace with

```latex
\subsection{RQ4: Can score-aware expert combination recover more deployable gain?}
```

---

## 15. RQ4 First Paragraph Patch

### Current paragraph may read

```latex
RQ4 asks whether the limitation of strict query-level clean routing is caused by the selective-activation idea itself, or by the coarseness of metadata-only query-level signals. The candidate-level score-aware line answers this question by allowing the router to observe non-answer-aware expert score patterns after both fixed experts have scored the candidate set.
```

### Replace with

```latex
RQ4 asks whether the limitation of strict query-level clean routing is caused by the selective-activation idea itself, or by the coarseness of metadata-only query-level signals. The score-aware expert-combination line answers this question by allowing deployable non-answer-aware access to fixed-expert score patterns after both experts have scored the candidate set. This line includes both learned candidate-level routers and simple score-interpolation baselines, so it tests whether recoverable gain depends on router complexity or on the availability of score-aware expert-combination information.
```

---

## 16. RQ4 Result Structure

### Recommended structure

```text
1. Table: learned candidate routers show CA-S2/CA-S3 beat E5.
2. Bootstrap table: CA-S2/CA-S3 improvements over E5 are supported.
3. Score-interpolation table: simple full-ranking interpolation is even stronger.
4. Interpretation: RQ4 is about score-aware expert combination, not CA-S2 superiority.
```

---

## 17. RQ4 Text Before Score-ensemble Table

### Insert before `\input{../paper_tables/table_score_ensemble_baselines.tex}`

```latex
The learned candidate routers show that moving beyond strict metadata-only clean routing is beneficial. However, they are not the end of the story. To test whether the gain is due to candidate-level router complexity or to the availability of fixed-expert score information itself, we further evaluate simple full-ranking score-interpolation baselines using the same Gate-only and Residual-only expert scores. Their policies are selected on the development split and evaluated on the test split under full filtered ranking.
```

### Table input

```latex
\input{../paper_tables/table_score_ensemble_baselines.tex}
```

### Text after table

```latex
\Cref{tab:score_ensemble_baselines} shows that simple score interpolation is not merely a weak sanity baseline. Global interpolation already reaches 0.3408 MRR, direction-specific interpolation reaches 0.3421 MRR, and relation-specific interpolation reaches 0.3428 MRR, all outperforming CA-S2. This result changes the interpretation of RQ4. CA-S2 remains useful as a learned candidate-level implementation showing that score-aware routing is much stronger than strict clean query-level routing. Nevertheless, the strongest evidence indicates that much of the recoverable gain comes from the fixed-expert score-combination basis itself rather than from a more complex learned router.
```

---

## 18. Explain Why Score Interpolation Can Exceed Oracle

### Add after the RQ4 score-interpolation paragraph

```latex
This does not contradict the hard-selection Oracle routing result. The Oracle row is an upper bound for hard query-level expert selection between the two fixed experts. Score interpolation creates a new mixed ranking that is not identical to either expert ranking, and therefore it is not bounded by the hard-selection Oracle.
```

### Global wording rule

Replace generic:

```text
Oracle routing
Oracle upper bound
Oracle headroom
Oracle recovery
```

with:

```text
hard-selection Oracle routing
hard-selection Oracle upper bound
hard-selection Oracle headroom
hard-selection Oracle recovery
```

when the context is the query-level Oracle that selects between fixed experts.

---

## 19. Table 16 Note Patch

If the candidate-router main table note says:

```latex
Oracle routing is answer-aware and is shown only as an upper bound.
```

Replace with:

```latex
Oracle routing is answer-aware and is shown only as an upper bound for hard expert selection, not for score-level interpolation.
```

Do not merge score-interpolation rows into the candidate-router main table. Keep them in a separate table.

---

## 20. Score-ensemble Table Patch

### Replace `docs/paper_tables/table_score_ensemble_baselines.tex` with

```latex
\begin{table}[t]
\centering
\small
\caption{Full-ranking score-aware expert-combination baselines. Ensemble baselines use fixed \texttt{Gate-only} and \texttt{Residual-only} scores, select interpolation policies on the development split, and are evaluated on the test split. CA-S2 is included as the learned candidate-level score-aware router for comparison.}
\label{tab:score_ensemble_baselines}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{0.30\textwidth}p{0.12\textwidth}p{0.13\textwidth}ccc}
\toprule
Method & Level & Granularity & MRR & $\Delta$ vs E5 & $\Delta$ vs CA-S2 \\
\midrule
Global score interpolation & ensemble & global & 0.3408 & +0.0426 & +0.0266 \\
Direction-specific score interpolation & ensemble & direction & 0.3421 & +0.0439 & +0.0279 \\
Relation-specific score interpolation & ensemble & relation & 0.3428 & +0.0446 & +0.0286 \\
Query-level soft score weighting & ensemble & query & 0.3055 & +0.0073 & -0.0087 \\
CA-S2 score-aware candidate router & router & candidate & 0.3142 & +0.0160 & -- \\
\bottomrule
\end{tabular}
\vspace{0.4ex}
\caption*{\footnotesize These results show that the strongest deployable score-aware gains come from exposing fixed-expert score-combination information itself. CA-S2 should therefore be interpreted as a learned score-aware routing implementation, not as the uniquely strongest score-aware combiner.}
\end{table}
```

If a mirrored table exists under `docs/paper/figures/table_score_ensemble_baselines.tex`, update it too.

---

## 21. Integrated Discussion Patch

### Find paragraph like

```latex
E5 shows that strict query-level clean routing is statistically reliable but modest. CA-S1 shows that static candidate metadata alone is insufficient...
```

### Replace with

```latex
E5 shows that strict metadata-only query-level routing is statistically reliable but modest. CA-S1 shows that static candidate metadata alone is insufficient, because it severely damages the dominant \texttt{tail\_no\_img} regime. CA-S2 and CA-S3 show that learned score-aware candidate-level routing is much stronger than strict clean routing. The full-ranking interpolation results further sharpen this conclusion: global, direction-specific, and relation-specific score interpolation outperform CA-S2, indicating that the dominant source of recoverable gain is not the complexity of the learned router, but the availability of fixed-expert score-combination information.
```

### Add after sparse correction paragraph

```latex
In this sense, CA-S2 provides an interpretable learned route into the score-aware setting, while the interpolation baselines reveal that a simple combination rule can be even more effective once the information boundary is relaxed from metadata-only to score-aware.
```

---

## 22. Limitations Patch

### Add a new limitation paragraph

```latex
Fourth, the full-ranking score-interpolation baselines show that CA-S2 is not the uniquely strongest score-aware combiner. This is a clarifying limitation of the learned-router component rather than a contradiction of the main analysis. The result indicates that once fixed-expert score information is available, simple interpolation can recover more gain than the learned candidate-level router. Therefore, the strongest claim of the paper is about the score-aware information boundary and expert-combination basis, not about the architectural superiority of CA-S2.
```

Then renumber or adjust the later limitation paragraphs if they are ordinally numbered.

---

## 23. Conclusion Patch

### Replace methodological conclusion paragraph with

```latex
Methodologically, the main lesson is that strict metadata-only query-level routing is too weak to exploit bounded multimodal gain effectively, but selective activation is not exhausted. Direction-specific thresholding and regression-based gain prediction recover modest but statistically reliable clean gains. Once fixed-expert score information is exposed, however, deployable expert combination becomes much stronger. CA-S2 reaches 0.3142 MRR and improves substantially over E5, but simple full-ranking score interpolation reaches up to 0.3428 MRR. This shows that the strongest deployable gain comes from score-aware expert-combination information itself rather than from the complexity of the learned router.
```

### Replace final-message paragraph with

```latex
The final message is therefore focused: under the current OpenBG-IMG protocol, the weakness of strict query-level clean routing is not solely caused by insufficient model capacity. It also stems from overly coarse global thresholding, binary gain supervision, and metadata-only query-level observability. Once routing is moved beyond metadata-only signals and allowed to use non-answer-aware fixed-expert score patterns, deployable multimodal gain can be recovered much more effectively. The unexpectedly strong score-interpolation baselines further show that future work should not only design more complex routers, but also study simple, stable, and well-calibrated expert-combination rules under explicit information boundaries.
```

---

## 24. Global Search-and-Replace Guidance

Do not mechanically replace all occurrences. Review each occurrence manually.

### Search

```text
candidate-level score-aware routing
```

Usually replace with:

```text
score-aware expert combination
```

But keep `learned candidate-level score-aware routing` when referring specifically to CA-S2/CA-S3.

### Search

```text
CA-S2 is the strongest
CA-S2 as the strongest
numerically strongest
main improvement comes from CA-S2
```

Replace with safer wording:

```text
CA-S2 is the strongest learned candidate-level router, but not the strongest score-aware combiner.
```

or:

```text
CA-S2 demonstrates the value of learned score-aware routing, while score interpolation shows that the larger gain comes from fixed-expert score-combination information.
```

### Search

```text
Oracle upper bound
Oracle headroom
Oracle recovery
```

Replace with:

```text
hard-selection Oracle upper bound
hard-selection Oracle headroom
hard-selection Oracle recovery
```

Add clarification on first use:

```text
not an upper bound for score-level interpolation
```

---

## 25. Recommended Main Tables After Reframing

Keep these tables in the main paper:

```text
Official model-comparison line
Main clean routing comparison
Clean routing ablation
Learned candidate-level routers + hard-selection Oracle
Candidate-router bootstrap significance
Full-ranking score interpolation baselines
```

Do not merge interpolation rows into the candidate-router table, because the hard-selection Oracle recovery column does not apply to score interpolation.

---

## 26. New One-sentence Thesis

Use this sentence as the final guiding claim:

```latex
The main bottleneck is not simply how to design a stronger router, but which deployable information boundary the router or combiner is allowed to access: metadata-only query features recover only modest gain, whereas non-answer-aware fixed-expert score patterns enable substantially stronger expert combination, even with simple interpolation.
```

This sentence can go near the end of the Introduction or the Integrated Discussion.

---

## 27. Minimal Required Patch If Time Is Limited

If time is limited, only do these six edits:

```text
[ ] Rewrite the abstract around score-aware expert combination.
[ ] Change RQ4 to score-aware expert combination.
[ ] Change Contribution 4 to score-aware expert combination.
[ ] In RQ4, explicitly state that interpolation > CA-S2.
[ ] Explain that hard-selection Oracle is not an upper bound for score interpolation.
[ ] Rewrite the conclusion around information boundary rather than CA-S2 superiority.
```

Everything else is refinement.

---

## 28. What Not To Do

Do not try to hide the interpolation result.

Do not write:

```text
CA-S2 is the best deployable score-aware method.
```

Do not imply:

```text
0.3428 is comparable to hard-selection Oracle recovery.
```

Do not merge score interpolation into the Oracle recovery table.

Do not rerun large new model families just to make CA-S2 look stronger.

The honest and stronger paper is now an information-boundary paper:

> Strict clean metadata is too coarse; score-aware non-answer-aware expert combination is the key; learned CA-S2 is useful, but simple interpolation reveals the central mechanism more clearly.

---

## 29. Final Expected Positioning

After this patch, the paper should be understood as:

> A protocol-aware analysis showing that OpenBG-IMG contains bounded multimodal gain under role--modality asymmetry. Strict metadata-only clean routing recovers only modest gain. Once the information boundary is relaxed to non-answer-aware fixed-expert score patterns, deployable expert combination becomes much stronger. Surprisingly, simple full-ranking interpolation outperforms the learned CA-S2 router, shifting the main conclusion from router architecture to score-aware information access.
