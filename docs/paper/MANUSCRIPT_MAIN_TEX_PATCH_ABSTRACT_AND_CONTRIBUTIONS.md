# Patch Instructions: One-Paragraph Abstract and Three-Part Contributions

Target file:

- `docs/paper/manuscript_main.tex`

Purpose:

This patch addresses two revision priorities:

1. compress the abstract into a single paragraph while preserving the key results;
2. restructure the contribution section into a clearer three-part problem--response--validation logic, moving the Oracle gap from a standalone contribution to a discussion/limitation result.

---

## 1. Replace the current abstract with this one-paragraph version

Recommended replacement range:

Replace everything between:

```tex
\begin{abstract}
```

and:

```tex
\end{abstract}
```

with:

```tex
\begin{abstract}
Multimodal knowledge graph completion (MMKGC) aims to improve link prediction by incorporating auxiliary modalities such as text and images, yet in product-oriented knowledge graphs visual support is often incomplete and multimodal evidence does not help uniformly. We study this problem on OpenBG-IMG under a unified filtered-ranking protocol with \texttt{direction=both} and show that the current split induces role--modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. This creates a bounded-gain structure in which multimodal benefit is strongest in image-supported head-side regimes but remains limited globally. We first show that naive clean query-time routing with a single global threshold does not outperform a simple legal clean rule baseline, indicating that clean routing is limited not only by query-time signal availability but also by policy granularity. We then evaluate stronger clean routing designs. Direction-specific thresholding raises clean routing performance to 0.2974 MRR. Regression-based gain prediction further improves the clean result to 0.2982 MRR, outperforming both the clean rule and \texttt{Residual-only} with strictly positive paired bootstrap confidence intervals. These gains recover only part of the Oracle headroom: the best clean strategy remains below Oracle routing. Overall, the paper provides a protocol-aware transition from bounded multimodal-gain diagnosis to structured clean selective activation, showing that deployable multimodal gain can be recovered more effectively when the routing policy is structured appropriately and trained with a target-aligned objective.
\end{abstract}
```

---

## 2. Replace the contribution block with the three-part structure

Recommended replacement range:

Replace the current block starting at:

```tex
In summary, this paper makes the following contributions:
```

and ending after the current `enumerate` contribution list.

Replacement block:

```tex
In summary, this paper makes three linked contributions, each corresponding to a fine-grained problem, a proposed response, and an experimental validation.

\begin{table}[t]
\centering
\caption{Problem--response--validation structure of this paper.}
\label{tab:problem_response_validation}
\small
\begin{tabular}{p{0.30\textwidth}p{0.30\textwidth}p{0.30\textwidth}}
\toprule
Fine-grained problem & Proposed response & Validation \\
\midrule
The current protocol creates role--modality asymmetry. & Protocol-aware bounded-gain diagnosis. & Target-side subgroup and relation-group analysis. \\
Naive clean routing is too coarse. & Direction-specific structured thresholding. & Clean routing comparison and paired bootstrap evidence. \\
Binary gain supervision is too coarse. & Regression-based gain prediction. & Strongest clean MRR and paired $\Delta$MRR confidence intervals. \\
\bottomrule
\end{tabular}
\end{table}

More specifically, our contributions are:
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Protocol-aware bounded-gain diagnosis.} We identify a protocol-aware bounded-gain structure on OpenBG-IMG by showing that role--modality asymmetry entangles target position with image availability, so multimodal benefit remains local rather than globally uniform.
    \item \textbf{Structured clean routing for selective activation.} We show that naive clean query-time routing with a single global threshold is insufficient, and that direction-specific thresholding provides a stronger clean policy under the asymmetric protocol.
    \item \textbf{Target-aligned clean supervision.} We further show that regression-based gain prediction recovers the strongest deployable clean result, outperforming both the clean rule and \texttt{Residual-only} with positive paired bootstrap evidence.
\end{enumerate}

These contributions do not imply that clean routing fully closes the gap to Oracle routing. Instead, the remaining Oracle gap is treated as a discussion and limitation result: structured clean routing recovers part of the bounded multimodal gain, while oracle-level separability still marks unresolved deployable headroom.
```

---

## 3. Notes for integration

- Keep the Oracle gap in the abstract and discussion, because it is an important boundary result.
- Do not list Oracle gap as a fourth contribution.
- If the new problem--response--validation table feels too heavy in the Introduction, it can be converted into a compact inline sentence, but the table form matches the supervisor's requested problem-solution-validation structure most clearly.

---

## 4. Patch status

This patch directly addresses:

- one-paragraph abstract compression;
- retention of key result values: 0.2974 and 0.2982 MRR;
- clearer three-problem / three-solution / three-validation contribution framing;
- moving the Oracle gap from contribution to limitation/discussion framing.
