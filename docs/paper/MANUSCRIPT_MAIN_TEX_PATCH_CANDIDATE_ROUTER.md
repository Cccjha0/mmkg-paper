# Manuscript Main TeX Patch: Candidate-level Score-aware Routing

This patch document is intended for the active manuscript file:

- `docs/paper/manuscript_main.tex`

It does **not** modify the legacy `docs/paper_manuscript/` drafts.

The purpose of this patch is to update the current manuscript after the candidate-aware / score-aware router experiments. The old endpoint was the regression-based strict clean router. The revised endpoint is score-aware candidate-level selective activation.

---

## 1. Replace Abstract

Replace the current abstract with the following:

```tex
\begin{abstract}
Multimodal knowledge graph completion (MMKGC) aims to improve link prediction by incorporating auxiliary modalities such as text and images, yet in product-oriented knowledge graphs visual support is often incomplete and multimodal evidence does not help uniformly. We study this problem on OpenBG-IMG under a unified filtered-ranking protocol with \texttt{direction=both} and show that the current split induces role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This creates a bounded-gain structure in which multimodal advantage is local rather than globally reliable. We first show that strict metadata-only query-level clean routing recovers only modest gain: direction-specific thresholding reaches 0.2974 MRR, and regression-based gain prediction reaches 0.2982 MRR. These results are statistically supported but remain far below Oracle routing, indicating that query-level clean signals are too coarse to expose most of the available expert-separation headroom. We then move selective activation to the candidate level. A score-aware candidate-level router that uses non-answer-aware expert score, confidence, and disagreement signals reaches 0.3142 $\pm$ 0.0010 MRR, outperforming the strongest query-level clean router by +0.0160 MRR with a 95\% paired bootstrap confidence interval of [+0.0139, +0.0181], and recovering 51.9\% of the Oracle headroom. Further ablations show that clean candidate metadata alone is insufficient, whereas score-aware features support sparse candidate-level correction, especially in the dominant \texttt{tail\_no\_img} regime. Overall, the paper provides a protocol-aware transition from bounded multimodal-gain diagnosis to score-aware candidate-level selective activation under incomplete visual support.
\end{abstract}
```

Update keywords to:

```tex
\textbf{Keywords:} multimodal knowledge graph completion; selective activation; score-aware routing; candidate-level routing; incomplete visual support; protocol-aware evaluation
```

---

## 2. Update Introduction: central problem and RQs

Replace the current central problem block and three RQs with:

```tex
\begin{quote}
Under the current OpenBG-IMG protocol, multimodal gain is bounded and protocol-dependent. How can this gain be diagnosed more precisely and exploited more effectively under deployable selective-activation constraints?
\end{quote}

We decompose this problem into four research questions.

\textbf{RQ1. Where does multimodal gain actually appear?} We first examine whether multimodal benefit is globally uniform or concentrated in specific target-side regimes and relation conditions.

\textbf{RQ2. Why does naive strict query-level clean routing fail?} We then test whether a single global metadata-only threshold is sufficient for strict clean selective activation, or whether the protocol requires a more structured query-level decision policy.

\textbf{RQ3. How far can structured query-level clean routing go?} We study whether direction-specific thresholding and target-aligned regression supervision can recover more gain than coarse binary query-level routing while remaining strict metadata-only clean.

\textbf{RQ4. Can score-aware candidate-level routing recover more deployable gain?} Finally, we test whether moving selective activation from the query level to the candidate level, and allowing non-answer-aware expert score signals, can recover substantially more of the Oracle headroom.
```

Then replace the following method-introduction paragraphs with:

```tex
To address these questions, we proceed in two stages. Stage 1 studies strict metadata-only query-level routing between a fusion expert and a structural expert. It begins with a naive global-threshold router, then evaluates direction-specific thresholding and regression-based gain prediction. Stage 2 moves selective activation to the candidate level. Instead of making one hard expert decision for the whole query, the candidate-level router learns a soft mixing weight for each query-candidate pair using non-answer-aware expert score, confidence, and disagreement features.

The experiments follow this progression. Protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. Strict query-level clean routing provides statistically supported but modest improvements: the strongest regression-based clean router reaches 0.2982 MRR. The new score-aware candidate-level result is substantially stronger. \texttt{CA-S2} reaches 0.3142 $\pm$ 0.0010 MRR, improves over the strongest query-level clean router by +0.0160 MRR, and recovers 51.9\% of the Oracle headroom.

The paper's claim is therefore controlled but stronger than a purely clean-routing claim. Under the current OpenBG-IMG protocol, strict query-level metadata-only routing is limited because its signals and decision granularity are too coarse. Score-aware candidate-level routing recovers more gain by using expert score patterns for sparse candidate-level correction. However, the strongest score-aware router still remains below Oracle routing, so the result should be understood as partial recovery of bounded multimodal gain rather than full closure of the Oracle gap.
```

---

## 3. Replace Contribution Table

Replace the current three-row contribution table with:

```tex
\begin{table}[H]
\centering
\caption{Problem--response--validation structure of this paper.}
\label{tab:problem_response_validation}
\small
\begin{tabular}{p{0.28\textwidth}p{0.30\textwidth}p{0.32\textwidth}}
\toprule
Fine-grained problem & Proposed response & Validation \\
\midrule
The current protocol creates role--modality asymmetry. & Protocol-aware bounded-gain diagnosis. & Target-side subgroup and relation-group analysis. \\
Naive query-level clean routing is too coarse. & Direction-specific query-level clean routing. & Clean routing comparison and paired bootstrap evidence. \\
Binary query-level supervision is too coarse. & Regression-based clean gain prediction. & Best strict clean result: 0.2982 MRR. \\
Query-level metadata remains insufficient. & Score-aware candidate-level soft routing. & 0.3142 MRR, +0.0160 over E5, and 51.9\% Oracle headroom recovery. \\
\bottomrule
\end{tabular}
\end{table}
```

Replace the contribution list with:

```tex
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Protocol-aware bounded-gain diagnosis.} We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
    \item \textbf{Structured strict query-level clean routing.} We show that naive clean query-time routing with a single global threshold is insufficient, and that direction-specific thresholding provides a stronger strict clean policy under the asymmetric protocol.
    \item \textbf{Target-aligned strict clean supervision.} We show that regression-based gain prediction is the strongest strict metadata-only query-level clean policy, but that its gain remains modest relative to Oracle headroom.
    \item \textbf{Score-aware candidate-level selective activation.} We introduce candidate-level soft expert weighting using non-answer-aware score-aware signals and show that it substantially outperforms strict query-level clean routing, with \texttt{CA-S2} reaching 0.3142 $\pm$ 0.0010 MRR and recovering 51.9\% of the Oracle headroom.
\end{enumerate}
```

---

## 4. Update Related Work Subsection 2.3

Rename subsection 2.3 to:

```tex
\subsection{Selective Activation, Expert Routing, and Score-aware Model Combination}
```

Add this paragraph near the end of the subsection:

```tex
The candidate-level part of our work is also related to score-based model combination, confidence-aware reranking, and stacked expert selection. However, our setting differs from generic ensembling in two ways. First, the routing problem is motivated by a protocol-aware bounded-gain diagnosis under incomplete visual support rather than by model scaling alone. Second, we explicitly separate strict metadata-only query-level clean routing, score-aware non-answer-aware candidate routing, and Oracle answer-aware routing. This separation allows us to test how much expert-separation headroom is visible under different deployability constraints.
```

Replace the final sentence of Related Work with:

```tex
We then extend this diagnosis into a method contribution by first testing strict query-level clean selective activation and then showing that score-aware candidate-level routing can recover substantially more deployable gain.
```

---

## 5. Update Method Section Structure

Rename the current method subsections into the following active structure:

```tex
\section{Task Setting and Method}
\subsection{Task Setting and Protocol}
\subsection{Stage 1: Strict Query-level Clean Selective Activation}
\subsection{Stage 2: Score-aware Candidate-level Selective Activation}
\subsection{Training Objectives and Legality Constraints}
\subsection{Evaluation Lines and Bootstrap Testing}
```

### 5.1 Update Evaluation Lines Table

Replace the current three-line evaluation table with:

```tex
\begin{table}[t]
\centering
\caption{Evaluation lines used in this paper. Rows from different lines answer different questions under different information conditions and should not be mixed as if they shared the same deployability basis.}
\label{tab:evaluation_lines}
\small
\begin{tabular}{p{0.24\textwidth}p{0.40\textwidth}p{0.24\textwidth}}
\toprule
Evaluation line & Used for & Deployability \\
\midrule
Official model-comparison line & Original base models under formal \texttt{test\_metrics.json} outputs & Standard model evaluation \\
Strict query-level clean line & Clean rule, global threshold, direction-specific thresholding, regression clean router & Metadata-only deployable \\
Score-aware candidate-level line & \texttt{CA-S1}, \texttt{CA-S2}, \texttt{CA-S3} & Deployable after expert scoring; non-answer-aware but not strict metadata-only clean \\
Oracle / post-hoc line & Oracle routing and answer-aware upper bounds & Not deployable \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.2 Add Candidate-level Routing Formulation

Insert after the query-level clean routing formulation:

```tex
\subsection{Stage 2: Score-aware Candidate-level Selective Activation}

Stage 1 is intentionally strict, but it also exposes a limitation: metadata-only query-level signals recover only modest gain. Stage 2 therefore moves selective activation from the query level to the candidate level. Instead of selecting one expert for the whole query, the router predicts a soft fusion weight for each query-candidate pair.

The candidate-level router is defined as
\[
\alpha(q,e)=\sigma(g_\theta(x_q,x_e,z_{q,e})),
\]
where $x_q$ denotes query-level legal features, $x_e$ denotes candidate-level metadata features, and $z_{q,e}$ denotes non-answer-aware score-aware features derived from the fixed experts. The mixed candidate score is
\[
s_{\text{mix}}(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e).
\]

We evaluate three candidate-level router variants. \texttt{CA-S1} uses clean candidate metadata and tests whether static modality metadata is sufficient. \texttt{CA-S2} uses score-aware expert features and tests whether expert score, confidence, and disagreement signals provide deployable routing evidence. \texttt{CA-S3} combines clean candidate metadata with score-aware features.

\texttt{CA-S2} and \texttt{CA-S3} are not strict metadata-only clean routers. They are score-aware deployable candidate-level routers: they may use expert scores, score margins, confidence-like statistics, and disagreement patterns computed after the fixed experts have scored candidates. They remain non-answer-aware because they do not use the hidden target label, reciprocal rank of the correct target, Oracle expert choice, or any post-hoc information requiring knowledge of which candidate is correct.
```

### 5.3 Add Pairwise Training Objective

Insert into the training subsection:

```tex
Candidate-level routers are trained with a pairwise ranking objective. For a query $q$, positive candidate $e^+$, and negative candidate $e^-$, the loss is
\[
\mathcal{L}_{pair}=-\log \sigma\left(s_{\text{mix}}(q,e^+)-s_{\text{mix}}(q,e^-)\right).
\]
For efficiency, training uses top-$K$ candidate sets. Negative candidates are sampled from hard negatives in the \texttt{Gate-only} top-$K$ list, hard negatives in the \texttt{Residual-only} top-$K$ list, and random negatives from the union candidate pool. Final reporting is stricter than the training approximation: \texttt{CA-S1}, \texttt{CA-S2}, and \texttt{CA-S3} are evaluated under full filtered ranking over all candidate entities.
```

---

## 6. Update Experiments Section

The revised Experiments section should follow this order:

```tex
\subsection{Experimental Setup}
\subsection{RQ1: Where does multimodal gain appear?}
\subsection{RQ2: Why does strict query-level clean routing remain limited?}
\subsection{RQ3: Can score-aware candidate-level routing recover more gain?}
\subsection{Feature Ablation: Clean Candidate vs Score-aware Candidate Features}
\subsection{Behavior Analysis: Sparse Alpha and Regime-specific Correction}
\subsection{Remaining Oracle Gap and Limitations}
```

### 6.1 Add Main Candidate Router Table

Insert the following table as the main selective-activation result:

```tex
\begin{table}[t]
\centering
\caption{Main selective activation results under the OpenBG-IMG paper protocol. E5 denotes the strongest strict query-level clean router, while \texttt{CA-S1}/\texttt{CA-S2}/\texttt{CA-S3} are candidate-level routers evaluated under full filtered ranking. \texttt{CA-S2} is score-aware but non-answer-aware; Oracle routing is a post-hoc upper bound.}
\label{tab:candidate_router_main}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{0.22\textwidth}p{0.12\textwidth}p{0.15\textwidth}rrrr}
\toprule
Method & Routing level & Feature type & MRR & $\Delta$ vs Residual & $\Delta$ vs E5 & Oracle gap recovered \\
\midrule
\texttt{Residual-only} & fixed & structural & 0.2930 & 0.0000 & -0.0051 & 0.0\% \\
E5 regression clean & query & strict clean & 0.2982 & +0.0051 & 0.0000 & 12.6\% \\
\texttt{CA-S1} clean candidate & candidate & clean candidate & 0.2503 $\pm$ 0.0054 & -0.0427 & -0.0479 & -105.0\% \\
\texttt{CA-S2} score-aware & candidate & score-aware & \textbf{0.3142 $\pm$ 0.0010} & \textbf{+0.0211} & \textbf{+0.0160} & \textbf{51.9\%} \\
\texttt{CA-S3} clean + score & candidate & clean + score & 0.3135 $\pm$ 0.0010 & +0.0204 & +0.0153 & 50.2\% \\
Oracle routing & oracle & answer-aware & 0.3337 & +0.0407 & +0.0356 & 100.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

### 6.2 Add Significance Table

```tex
\begin{table}[t]
\centering
\caption{Paired bootstrap significance for candidate-level routing. All comparisons use the same paired test query instances.}
\label{tab:candidate_router_significance}
\small
\begin{tabular}{lrrr}
\toprule
Comparison & $\Delta$MRR & 95\% bootstrap CI & Paired queries \\
\midrule
\texttt{CA-S1} vs \texttt{Residual-only} & -0.0427 & [-0.0452, -0.0404] & 60{,}000 \\
\texttt{CA-S1} vs E5 & -0.0479 & [-0.0503, -0.0455] & 60{,}000 \\
\texttt{CA-S2} vs \texttt{Residual-only} & +0.0211 & [+0.0191, +0.0232] & 60{,}000 \\
\texttt{CA-S2} vs E5 & +0.0160 & [+0.0139, +0.0181] & 60{,}000 \\
\texttt{CA-S3} vs \texttt{Residual-only} & +0.0204 & [+0.0184, +0.0226] & 60{,}000 \\
\texttt{CA-S3} vs E5 & +0.0153 & [+0.0132, +0.0174] & 60{,}000 \\
\texttt{CA-S2} vs \texttt{CA-S3} & +0.0007 & [-0.0002, +0.0014] & 60{,}000 \\
\bottomrule
\end{tabular}
\end{table}
```

Use the following interpretation:

```tex
\texttt{CA-S2} and \texttt{CA-S3} both significantly outperform \texttt{Residual-only} and E5. However, \texttt{CA-S2} does not significantly outperform \texttt{CA-S3}, because the confidence interval crosses zero. Therefore, \texttt{CA-S2} should be interpreted as the numerically strongest and more parsimonious score-aware candidate router, not as a statistically superior variant over \texttt{CA-S3}.
```

### 6.3 Add Subgroup Interpretation

Use this paragraph after the subgroup table:

```tex
The subgroup results shift the interpretation of routing gain. \texttt{CA-S1} helps some head-side regimes but fails badly in the dominant \texttt{tail\_no\_img} regime, with a delta of -0.0916 against \texttt{Residual-only}. In contrast, \texttt{CA-S2} and \texttt{CA-S3} improve mainly in \texttt{tail\_no\_img}, with deltas of +0.0404 and +0.0389, respectively. This shows that the main useful signal is not static image availability itself, but score-aware expert behavior over candidate distributions.
```

### 6.4 Add Alpha Behavior Interpretation

Use this paragraph after the alpha table:

```tex
The alpha behavior suggests sparse candidate-level correction. \texttt{CA-S2} has a low mean alpha of 0.0470 but a much higher target alpha of 0.1964. In \texttt{tail\_no\_img}, this pattern becomes more extreme: mean alpha is only 0.0054, while target alpha reaches 0.3788. This indicates that the router does not broadly activate fusion. Instead, it keeps fusion weights low for most candidates while assigning substantially higher fusion weights to target candidates in diagnostic analysis. Target alpha is an evaluation-time diagnostic only and is not used as a router feature.
```

---

## 7. Replace Conclusion

Replace the current conclusion with:

```tex
\section{Conclusion}
This paper studies multimodal knowledge graph completion on OpenBG-IMG under a more precise question than the standard ``does multimodal information help?'' framing. Under the current protocol, multimodal evidence can help, but it is not globally reliable. Its usefulness depends on target position, image availability, relation context, and the competition between a fusion expert and a structurally stronger fallback expert.

The first contribution is diagnostic. The current OpenBG-IMG \texttt{paper\_split} induces role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This creates three meaningful target-side regimes and explains why aggregate bidirectional evaluation is not a neutral multimodal setting.

The second contribution is strict query-level clean routing analysis. A single global-threshold clean router does not outperform a legal clean rule baseline, showing that query-level metadata-only routing is too coarse. Direction-specific thresholding and regression-based gain prediction improve this strict clean line, with the strongest query-level clean router reaching 0.2982 MRR, but the recovered gain remains modest.

The third and strongest method contribution is score-aware candidate-level selective activation. Moving routing from the query level to the candidate level allows the system to use non-answer-aware expert score, confidence, and disagreement patterns after fixed expert scoring. The best score-aware candidate router, \texttt{CA-S2}, reaches 0.3142 $\pm$ 0.0010 MRR, outperforming the strongest strict query-level clean router by +0.0160 MRR and recovering 51.9\% of the Oracle headroom.

The feature and behavior analyses refine this conclusion. Static clean candidate metadata alone is insufficient: \texttt{CA-S1} fails globally because it severely degrades the dominant \texttt{tail\_no\_img} regime. By contrast, \texttt{CA-S2} and \texttt{CA-S3} improve mainly through sparse candidate-level correction in \texttt{tail\_no\_img}. Their average fusion weights remain low, while diagnostic target alpha is much higher, indicating that the router does not globally turn on multimodal fusion but instead selectively applies fusion where expert score patterns support it.

These improvements do not close the gap to Oracle routing. Oracle routing still reaches 0.3337 MRR, above the best score-aware candidate router. This remaining gap marks the difference between deployable non-answer-aware score-based routing and answer-aware upper-bound selection.

Overall, the results suggest that the main deployable bottleneck is not simply whether multimodal evidence is available, but whether the system can observe sufficiently fine-grained expert behavior at inference time. Future MMKGC research should therefore study not only richer multimodal fusion, but also candidate-level selective activation and deployable expert-confidence signals under incomplete modality support.
```

---

## 8. Terminology Guardrails

Use:

- `strict query-level clean router`
- `strict metadata-only clean routing`
- `score-aware candidate-level router`
- `deployable non-answer-aware score-based routing`
- `candidate-level soft expert weighting`
- `sparse candidate-level correction`

Avoid:

- `CA-S2 is a clean router`
- `CA-S2 closes the Oracle gap`
- `candidate modality metadata is the main reason for the gain`
- `CA-S2 significantly outperforms CA-S3`
- `multimodal fusion is globally superior`
