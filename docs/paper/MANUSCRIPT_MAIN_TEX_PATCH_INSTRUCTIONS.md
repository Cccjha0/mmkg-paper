# Patch Instructions for `docs/paper/manuscript_main.tex`

## Scope

This file is a temporary patch companion for the active manuscript file:

- `docs/paper/manuscript_main.tex`

It is intended to avoid editing the wrong source. The repository also contains older or split drafting materials, but the active integrated paper file is the LaTeX manuscript above. The goal of this patch note is to provide exact replacement guidance for the strongest-claim sections before directly overwriting the main TeX file.

The new paper narrative must be:

1. **naive global-threshold clean routing is weak**;
2. **structured clean routing is stronger**;
3. **regression-style clean supervision is strongest on the clean line**;
4. **the strongest clean strategy still remains substantially below Oracle**.

---

## 1. Replace the entire abstract

### Replace current abstract with:

```tex
\begin{abstract}
Multimodal knowledge graph completion (MMKGC) aims to improve link prediction by incorporating auxiliary modalities such as text and images, yet in product-oriented knowledge graphs visual support is often incomplete and multimodal evidence does not help uniformly. We study this problem on OpenBG-IMG under a unified protocol with filtered ranking, \texttt{direction=both}, and three-seed reporting, and show that the current split induces a role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This reveals a bounded-gain structure in which multimodal benefit is strongest in image-supported head-side regimes but remains limited at the global level.

We first examine naive clean query-time routing under a single global threshold and find that this formulation does not outperform a simple legal clean rule baseline. This shows that the weakness of clean routing is not explained solely by insufficient query-time observable signals. Instead, the original clean formulation is also too coarse to capture the asymmetric decision structure induced by the protocol.

We therefore evaluate stronger clean routing designs. Direction-specific thresholding consistently improves over the clean rule baseline, showing that head-side and tail-side queries require different operating points under the current split. More target-aligned clean supervision further strengthens deployable routing: regression-based gain prediction yields the strongest clean result and consistently improves over both the clean rule and \texttt{Residual-only}, with bootstrap confidence intervals strictly above zero. These findings show that the bottleneck of clean routing lies not only in signal availability, but also in policy granularity and supervision form.

At the same time, the best clean strategy still remains substantially below Oracle routing. The contribution of this work is therefore not a universally dominant multimodal architecture, but a protocol-aware transition from bounded-gain diagnosis to stronger structured clean routing. Under the current OpenBG-IMG protocol, deployable multimodal gain can be recovered more effectively once the routing policy is structured appropriately and trained with a more target-aligned objective, although a substantial deployable gap still remains.
\end{abstract}
```

---

## 2. Replace the core introduction block from the problem statement through contributions

### Replace the current introduction block beginning at:

```tex
These observations motivate the total problem addressed in this paper:
```

### and ending at the contributions list with the following block:

```tex
These observations motivate the total problem addressed in this paper:

\begin{quote}
Under the current OpenBG-IMG protocol, multimodal gain is not globally reliable. How can this bounded gain be understood more precisely and exploited more effectively under clean, deployable routing constraints?
\end{quote}

Rather than treating this as a single undifferentiated question, we decompose it into three progressively narrower research questions.

\textbf{RQ1. Where does multimodal gain actually appear under the current OpenBG-IMG protocol?} This question asks whether multimodal benefit is globally uniform or instead concentrated in specific target-side regimes and relation conditions.

\textbf{RQ2. Why does naive clean routing fail, and what kind of clean decision structure is actually required?} This question asks whether activation can be decided by a single global threshold or a shallow modality-availability heuristic, or whether it instead requires a more structured policy under protocol-shaped asymmetry.

\textbf{RQ3. How should bounded multimodal gain be exploited effectively once it is known to be conditional?} This question asks what operational response is appropriate under heterogeneous gain: a naive global clean router, a more structured clean policy, or richer clean supervision aligned with gain magnitude.

This decomposition also determines the overall logic of the paper. RQ1 is diagnostic: it identifies where bounded multimodal gain appears. RQ2 is structural: it asks what kind of clean decision boundary is needed under asymmetric protocol conditions. RQ3 is operational: it turns bounded gain into a selective-activation mechanism that can be evaluated directly.

To address these questions, we move from always-on multimodal fusion to clean query-level selective activation between a fusion expert and a structural expert. We begin with a naive global-threshold clean routing formulation as a minimal deployable baseline. We then evaluate stronger clean formulations, especially direction-specific thresholding and more target-aligned supervision such as regression-style gain prediction. The method is not intended as a universally stronger multimodal architecture; rather, it operationalizes the bounded-gain finding under the current OpenBG-IMG setting and tests how much of that gain can actually be recovered under clean query-time constraints.

The experiments are organized accordingly. For RQ1, protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. For RQ2, the comparison between a naive global-threshold clean router and structured clean policies shows that the weakness of the original clean formulation comes not only from signal scarcity, but also from overly coarse policy granularity. For RQ3, target-aligned clean supervision further improves over the clean rule baseline, while Oracle routing still leaves visible headroom.

The strongest claim of the paper is therefore controlled rather than universal. We do not claim that multimodal fusion is globally superior to structural modeling in general. Nor do we claim that clean routing fully solves the bounded-gain problem under the current protocol. Instead, we argue that under the current OpenBG-IMG protocol, the weakness of clean routing is not solely caused by insufficient query-time observable signals. It also stems from overly coarse global thresholding and binary gain supervision. Once the clean policy is structured more appropriately and trained with a more target-aligned objective, deployable multimodal gain can be recovered more effectively, although a substantial gap to Oracle still remains.

In summary, this paper makes the following contributions:
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Protocol-aware bounded-gain diagnosis.} We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
    \item \textbf{Negative result on naive clean routing.} We show that naive clean query-time routing based on a single global threshold is insufficient and can underperform a simple legal rule-based selector, indicating that deployable routing cannot be characterized adequately by a coarse global decision boundary.
    \item \textbf{Positive result on structured clean routing.} We demonstrate that clean routing is not fundamentally exhausted: direction-specific thresholding and more target-aligned supervision, especially regression-style gain modeling, consistently outperform the clean rule baseline and recover additional deployable gains.
    \item \textbf{Remaining deployable gap.} We further show that even these stronger clean strategies remain substantially below Oracle, revealing a persistent gap between deployable separability and oracle-level or post-hoc separability.
\end{enumerate}
```

---

## 3. Retitle and rewrite the method section

### Current subsection title to replace:

```tex
\subsection{Gain-Threshold Routing Framework}
```

### Replace with:

```tex
\subsection{Clean Routing Formulations for Selective Activation}
```

### Replace the current subsection text under that title with this new structure:

```tex
The previous sections establish a protocol-aware empirical tension. Under the current OpenBG-IMG setting, multimodal gain is visible but not globally reliable. \texttt{Full Model} improves over weaker multimodal baselines, yet stronger structural alternatives---especially \texttt{Residual-only}, and in the official model-comparison line also \texttt{ComplEx}---remain superior at the global level. This shifts the central problem away from merely constructing a richer multimodal encoder. Instead, the problem becomes whether multimodal fusion should be activated for the current query at all, and if so, how that decision should be made under clean query-time legality.

We therefore move from always-on multimodal fusion to \textbf{clean query-level selective activation}. The central idea is simple: if multimodal gain is conditional rather than uniformly available, then the system should predict when fusion is likely to help and use a stronger structural fallback otherwise. To operationalize this idea, we study query-level expert selection between a fusion expert and a structural expert.

We define two fixed experts:
\begin{itemize}
    \item \textbf{Fusion expert:} \texttt{Gate-only}
    \item \textbf{Structural expert:} \texttt{Residual-only}
\end{itemize}

This choice is deliberate. \texttt{Gate-only} represents relation-aware multimodal fusion without an explicit structural compensation branch. \texttt{Residual-only} represents structure-heavy compensation within the same internal model family. Together, they are the clearest diagnostic endpoints in the current framework and provide a clean basis for selective routing.

\paragraph{Clean legality constraint.} The key methodological boundary of this paper is that the main deployable routing claims must remain \emph{clean}. A clean router may use only information that is available at query time, including query direction, relation identity and development-derived relation priors, and query-observable modality indicators from the observed side of the query. By contrast, the deployable clean router may not use hidden target-side information, target-side image labels, target-aware regimes, correct-target scores, or any other signals that require the true missing entity or post-hoc expert outcomes at inference time.

\paragraph{Naive global-threshold clean routing.} The simplest clean routing formulation uses a learned probability $p(q)$ and a single global threshold $\tau$. Let $x_q$ denote a clean feature vector available at query time. The router predicts
\[
p(q)=P(y=1\mid x_q),
\]
where $y=1$ means that the fusion expert should be selected. At inference time, we define
\[
\alpha(q)=\mathbf{1}[p(q)>\tau],
\]
and the final score becomes
\[
s_{\text{final}}(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e).
\]
This formulation serves as a minimal deployable clean baseline rather than the final structured policy proposed by the paper.

\paragraph{Structured clean threshold policies.} The new results of this paper show that the weakness of the naive clean router is not solely due to signal scarcity. It is also caused by policy granularity. Under the present protocol, head-side and tail-side queries operate under different gain regimes, so a single global threshold is too rigid. We therefore study a stronger family of \textbf{structured clean threshold policies}. The most important member is \textbf{direction-specific thresholding}, which replaces the single global threshold with two query-direction-dependent thresholds:
\[
\alpha(q)=
\begin{cases}
\mathbf{1}[p(q)>\tau_{head}], & q \text{ is a head-prediction query}, \\
\mathbf{1}[p(q)>\tau_{tail}], & q \text{ is a tail-prediction query}.
\end{cases}
\]
We also study optional bucketized thresholding variants, such as relation-prior buckets or query-observable groupings, as supporting structured-policy extensions.

\paragraph{Target-aligned supervision for clean routing.} The second major refinement concerns the training target. In the original gain-threshold formulation, supervision is derived from a coarse binary gain label. For each query $q$, let $RR_f(q)$ and $RR_s(q)$ denote the reciprocal ranks of the correct target under the fusion and structural experts. The gain difference is
\[
\Delta(q)=RR_f(q)-RR_s(q).
\]
The original binary label is then defined as
\[
y(q)=\mathbf{1}[\Delta(q)>\delta].
\]
To better align supervision with the actual expert difference, we study stronger clean targets: (1) the original binary gain label as a naive baseline, (2) a regression target that predicts $\Delta(q)$ directly, and (3) ordinal gain buckets that partition $\Delta(q)$ into multiple ordered intervals.
```

---

## 4. Replace router feature table with clean-routing framing

### The current feature table is built around target-aware / confidence-rich `FULL` routing.

For the active paper, replace the old interpretation around the table with this framing paragraph:

```tex
A naive router could easily degenerate into a shallow heuristic such as ``use fusion whenever the query looks multimodal-friendly.'' To prevent this, we separate three layers of routing evidence. The deployable clean line uses only legal query-time features such as \texttt{direction}, relation identity and development-derived relation priors, and observed-side modality indicators. Stronger target-aware or confidence-rich selectors are retained only as analysis tools for post-hoc separability and are not part of the main clean claims.
```

### Then add a new compact clean feature table before or instead of the old FULL-table:

```tex
\begin{table}[t]
\centering
\caption{Clean router feature families used in the deployable routing line.}
\label{tab:clean_router_feature_sets}
\small
\begin{tabular}{p{0.10\textwidth}p{0.78\textwidth}}
\toprule
Set & Features \\
\midrule
\texttt{C1} & \texttt{direction} \\
\texttt{C2} & \texttt{direction}, \texttt{relation\_gain\_prior}, \texttt{relation\_fusion\_win\_rate}, \texttt{relation\_support}, \texttt{relation\_is\_visual\_prior} \\
\texttt{C3} & \texttt{C2} + \texttt{observed\_has\_img}, \texttt{observed\_text\_img\_cosine}, \texttt{observed\_img\_missing\_replaced} \\
\texttt{C4} & \texttt{C3} + \texttt{relation\_id} \\
\bottomrule
\end{tabular}
\end{table}
```

### Then follow with this explanatory paragraph:

```tex
These features are intentionally limited. They do not attempt to reconstruct the hidden target, and they do not use target-aware or answer-aware confidence signals. Their purpose is to test how much of bounded multimodal gain can be recovered under a truly deployable query-time constraint.
```

---

## 5. Rewrite the experiments chapter opening and RQ framing

### Replace the current opening of `\section{Experiments and Discussion}` through the RQ framing with:

```tex
\section{Experiments and Discussion}
\subsection{Experimental Setup}
All experiments are conducted on the current OpenBG-IMG \texttt{paper\_split} under the unified protocol defined in the previous section. Base-model checkpoints are selected on the development split, and final base-model metrics are reported on the test split using filtered ranking with \texttt{direction=both}. Unless otherwise stated, official main-result metrics are aggregated over three random seeds and reported as mean $\pm$ standard deviation when the source line supports that form.

This section is organized around three research questions. \textbf{RQ1} asks where multimodal gain actually appears under the current protocol. \textbf{RQ2} asks why naive clean routing fails and what kind of clean decision structure is actually required. \textbf{RQ3} asks how bounded multimodal gain should be exploited effectively once it is known to be conditional.

To answer these questions cleanly, we distinguish among three complementary evaluation lines. The \textbf{official model-comparison line} is used for the seven-model comparison among \texttt{Text-only}, \texttt{Early Fusion}, \texttt{Gate-only}, \texttt{Residual-only}, \texttt{Full Model}, \texttt{ComplEx}, and \texttt{TuckER}. The \textbf{clean routing line} is based on query-level exported expert predictions and unified recomputation under legal query-time constraints, and is used for \texttt{Gate-only}, \texttt{Residual-only}, Oracle routing, the clean rule baseline, naive global-threshold clean routing, structured threshold policies, and target-aligned clean supervision. The \textbf{post-hoc analysis line} is retained only for stronger offline separability analysis and upper-bound-style comparisons; it is not part of the deployable clean claim.
```

---

## 6. Add a new experiments interpretation block after the clean results tables

### Insert a new subsection block with the following wording:

```tex
\subsection{Why naive clean routing fails, and what stronger clean routing requires}

A natural first attempt is a clean learned router with one global threshold $\tau$. However, the clean routing results show that this formulation is too weak. On the clean routing line, the strongest fixed expert remains \texttt{Residual-only}, the clean rule baseline is slightly stronger still, and the naive global-threshold learned clean router does not reliably surpass that rule. This means that the main difficulty of deployable routing is not merely classifier capacity. Rather, it lies in the mismatch between a single global decision boundary and the strongly asymmetric gain structure induced by the protocol.

Once the clean policy is allowed to use direction-specific thresholds instead of one global threshold, performance increases substantially above the clean rule baseline. The strongest dual-threshold configuration reaches approximately \texttt{0.2974} MRR, improving over the clean rule (\texttt{0.2943}) by about \texttt{+0.00315}. This gain is also statistically clean: the bootstrap confidence interval for the comparison between the best direction-specific threshold policy and the clean rule is strictly positive, with mean delta MRR \texttt{+0.003154} and 95\% CI [\texttt{+0.002532}, \texttt{+0.003786}].

The strongest operating point is highly interpretable. The best thresholds are strongly asymmetric, with a permissive head-side threshold and a conservative tail-side threshold, confirming that global thresholding is too coarse for the current protocol. Under the current split, head-side and tail-side queries do not operate under one homogeneous gain regime, so the routing policy should not be forced to do so either.
```

---

## 7. Add a new supervision results block

### Insert after the structured-threshold discussion:

```tex
\subsection{Target-aligned supervision improves over both the clean rule and naive clean routing}

The new clean routing results show that policy structure is only part of the story. Supervision form also matters. Replacing the coarse binary gain label with a more target-aligned objective yields a further improvement in clean routing. In particular, the strongest regression-based clean router reaches approximately \texttt{0.2982} MRR, outperforming both the clean rule and the direction-specific threshold policy.

This gain is statistically well supported. Relative to the clean rule, the best regression-based clean router achieves mean delta MRR \texttt{+0.003900} with 95\% bootstrap CI [\texttt{+0.003345}, \texttt{+0.004442}]. Relative to \texttt{Residual-only}, it achieves mean delta MRR \texttt{+0.005133} with 95\% bootstrap CI [\texttt{+0.004438}, \texttt{+0.005844}]. These results are strong enough to support the main clean claim of the paper: structured clean routing and target-aligned clean supervision do not merely tend to help, but consistently improve over the earlier clean baselines.

Ordinal gain modeling also improves over the binary baseline, although it remains below the regression formulation. This pattern is conceptually important. It suggests that preserving gain magnitude information is especially helpful under the current protocol, where the gain boundary is shallow, uneven, and strongly shaped by direction-specific asymmetry.
```

---

## 8. Replace the old strongest-claim conclusion sentence in discussion

### Replace any old sentence similar to:

```tex
selective activation is a more effective operational response than fixed always-on multimodal use
```

### with this stronger but bounded version:

```tex
The combined evidence supports a more refined conclusion than the earlier paper version. Under the current OpenBG-IMG protocol, multimodal gain is real but bounded, and a lightweight naive clean router is not sufficient to exploit it. However, this weakness should not be interpreted as evidence that deployable clean routing is impossible in principle. Instead, the new results show that the main bottleneck lies in the coarseness of policy granularity and supervision form. Once the clean policy is structured more appropriately and trained with a more target-aligned objective, deployable routing can recover additional gains beyond both the clean rule and the original global-threshold learned baseline. At the same time, the strongest clean strategies remain substantially below Oracle, so the resulting gains are still protocol-specific rather than universal.
```

---

## 9. Add a limitations caveat about still-unverified auxiliary analyses

### Append to limitations or final discussion:

```tex
A further limitation is that some auxiliary supporting analyses remain less stable than the main MRR-based clean results. In particular, oracle-gap decomposition and some hit-based summary exports require further verification before they should be treated as final quantitative evidence. For this reason, the strongest claims of the present paper are grounded primarily in the protocol diagnosis, the clean routing comparison under MRR, and the updated bootstrap-supported comparisons for the strongest clean policies.
```

---

## 10. Minimum main-table rewrite target

When rewriting the main clean comparison table, the new row order should be:

1. `Residual-only`
2. `Clean rule`
3. `Naive global clean learned router (best global-threshold C4)`
4. `Direction-specific dual-threshold clean policy (E1)`
5. `Regression-based clean router (E5)`
6. `Oracle`

This ordering is crucial because it visually communicates the new main story:

- naive clean is weak,
- structured clean is stronger,
- regression clean is strongest,
- but Oracle gap remains.
