# Patch Instructions: Clean Routing Table and Router Training Details

Target file:

- `docs/paper/manuscript_main.tex`

Purpose:

This patch implements the first three high-priority fixes from the latest revision checklist:

1. add a main clean routing evidence table;
2. make the official / clean / post-hoc evaluation-line distinction visible around result tables;
3. add router training and reproducibility details to the Method section.

The repository contains a long TeX manuscript, and connector output may truncate long files. Therefore this patch is written as exact insertion blocks rather than an unsafe full-file replacement.

---

## 1. Insert the main clean routing evidence table

Recommended location:

Insert after the opening paragraphs of `\section{Experiments and Discussion}` / `\subsection{Experimental Setup}`, after the paragraph that defines the three evaluation lines and before the RQ1 results discussion.

```tex
\begin{table}[t]
\centering
\caption{Main clean routing comparison on the clean routing line. Fixed experts in this table are recomputed on the routing-compatible query-level basis and should not be mixed with the official model-comparison line. Oracle routing is included only as a post-hoc upper-bound reference and is not a deployable clean method.}
\label{tab:main_clean_routing}
\small
\begin{tabular}{lcccc}
\toprule
Method & MRR & $\Delta$ vs. clean rule & $\Delta$ vs. \texttt{Residual-only} & Evidence \\
\midrule
\texttt{Residual-only} & 0.2930 & -0.0012 & -- & fixed structural expert \\
Clean rule & 0.2943 & -- & +0.0012 & legal rule baseline \\
Naive global clean router & 0.2939 & -0.0004 & +0.0008 & does not beat clean rule \\
Direction-specific threshold & 0.2974 & +0.0032 & +0.0044 & CI vs. rule: [0.002532, 0.003786] \\
Regression-based clean router & \textbf{0.2982} & \textbf{+0.0039} & \textbf{+0.0051} & CI vs. rule: [0.003345, 0.004442]; CI vs. residual: [0.004438, 0.005844] \\
Oracle routing & 0.3337 & +0.0395 & +0.0407 & post-hoc upper bound \\
\bottomrule
\end{tabular}
\end{table}
```

Immediately after the table, insert:

```tex
\noindent\textbf{Interpretation.} \Cref{tab:main_clean_routing} is reported strictly on the clean routing line. The table supports three conclusions. First, naive global-threshold clean routing is weaker than the legal clean rule. Second, direction-specific thresholding recovers a statistically supported clean gain, showing that policy granularity matters. Third, regression-based gain prediction is the strongest clean strategy, but it still remains substantially below Oracle routing.
```

---

## 2. Add evaluation-line reminders to result table captions

Whenever a result table is added or revised, use one of the following caption prefixes.

### Official model-comparison table

```tex
\caption{Official model-comparison line. Results in this table are based on formal \texttt{test\_metrics.json} outputs and are used only to compare the original model family under the paper protocol. They should not be directly mixed with clean-routing-line rows.}
```

### Clean routing table

```tex
\caption{Clean routing line. Results in this table are recomputed from query-level expert outcomes under legal query-time constraints. Fixed expert rows are routing-compatible recomputed values, not copied from the official model-comparison line.}
```

### Post-hoc or Oracle analysis table

```tex
\caption{Post-hoc analysis line. These results use oracle-style or analysis-only information to quantify remaining separability and should not be treated as deployable clean-routing performance.}
```

---

## 3. Add `Router Training Details` to Method

Recommended location:

Insert after the current `Clean Router Features, Training, and Evaluation Basis` subsection and before `\section{Experiments and Discussion}`.

```tex
\subsection{Router Training Details}

\paragraph{Query-level feature and label construction.}
Router training is performed on query-level feature tables constructed from the fixed fusion and structural experts. For every development query $q$, we record the reciprocal ranks produced by the fusion expert and the structural expert, and compute the expert difference:
\[
\Delta(q)=RR_f(q)-RR_s(q).
\]
This development-side expert difference is used only to create router supervision. It is not used as a test-time feature. At test time, the router receives only clean query-time features and predicts whether the fusion expert should be activated.

Relation-level priors are also computed from the development split only. For a relation $r$, \texttt{relation\_gain\_prior} denotes the average development-side reciprocal-rank gain of the fusion expert over the structural expert for that relation. \texttt{relation\_fusion\_win\_rate} denotes the proportion of development queries for which the fusion expert outperforms the structural expert. \texttt{relation\_support} records the number of development queries available for that relation, and \texttt{relation\_is\_visual\_prior} is a development-derived indicator of whether the relation tends to behave as visually favorable under the clean feature construction. These relation priors are then attached to test queries by relation identity without using test labels or test expert outcomes.

Observed-side modality features are computed only from the known side of the query. For tail prediction $(h,r,?)$, the observed side is the head entity $h$; for head prediction $(?,r,t)$, the observed side is the tail entity $t$. The clean router therefore never uses the hidden target entity, target-side image availability, or target-aware subgroup labels.

\paragraph{Router model families.}
The clean routing experiments evaluate three families of routing models. The first is the legal clean rule baseline, which activates the fusion expert only under conservative query-observable conditions. The second is naive global-threshold learned routing, where classifiers such as logistic regression and XGBoost output a fusion-selection probability under a single threshold. The third family contains the stronger clean policies studied in this paper: direction-specific thresholding, optional bucketized thresholding, regression-based gain prediction, and ordinal gain modeling.

For binary routing, the classifier is trained to predict whether $\Delta(q)$ exceeds a development-defined gain margin $\delta$. For regression-based routing, the model predicts the scalar target $\Delta(q)$ directly, using a standard squared-error regression objective in the implemented regressor family. For ordinal routing, the target is formed by partitioning $\Delta(q)$ into ordered gain buckets, and the final routing decision is derived from whether the predicted bucket indicates positive expected fusion gain.

\paragraph{Threshold selection and test protocol.}
All routing hyperparameters are selected on the development split only. This includes the gain margin $\delta$, the single global threshold $\tau$, direction-specific thresholds $\tau_{head}$ and $\tau_{tail}$, bucketized thresholds, and the regression decision threshold $\theta$. After these settings are selected, the trained router is applied once to the test split for final reporting.

This protocol is important because the routing task is vulnerable to leakage. The test split is never used to fit the router, construct relation priors, tune thresholds, choose the gain margin, or select the router family. Test labels and test-side expert outcomes are used only after routing decisions have been made, in order to compute filtered ranking metrics and paired significance results.

\paragraph{Bootstrap evaluation.}
The main significance evidence is computed on the clean routing line using paired query-level bootstrap. For each pair of compared methods, we compute reciprocal-rank differences on the same set of test query instances, resample these paired query outcomes with replacement, and recompute the MRR difference on each bootstrap sample. The reported confidence intervals are taken from the empirical bootstrap distribution of paired $\Delta$MRR.

This paired design matters because all clean routing methods are evaluated on the same test query set. The bootstrap intervals therefore quantify whether one clean policy improves over another under matched query conditions, rather than comparing unrelated aggregate scores.
```

---

## 4. Replace or supplement the RQ2 interpretation with exact evidence

Recommended insertion inside the RQ2 clean-routing discussion:

```tex
A natural first attempt is a clean learned router with one global threshold $\tau$. However, \Cref{tab:main_clean_routing} shows that this formulation is too weak. The naive global clean router reaches 0.2939 MRR, below the clean rule baseline at 0.2943 MRR. This means that the main difficulty of deployable routing is not merely classifier capacity. Rather, it lies in the mismatch between a single global decision boundary and the asymmetric gain structure induced by the protocol.
```

---

## 5. Replace or supplement the RQ3 interpretation with exact evidence

Recommended insertion inside the RQ3 target-aligned supervision discussion:

```tex
Direction-specific thresholding improves the clean result to 0.2974 MRR, corresponding to a gain of approximately +0.0032 over the clean rule baseline. Regression-based gain prediction further improves the clean result to 0.2982 MRR. Its improvement over the clean rule is +0.0039 MRR with a 95\% paired bootstrap confidence interval of [0.003345, 0.004442]. Its improvement over \texttt{Residual-only} is +0.0051 MRR with a 95\% paired bootstrap confidence interval of [0.004438, 0.005844].
```

---

## 6. Add the Oracle-gap sentence

Recommended insertion near the end of the experiments/discussion section:

```tex
Despite these improvements, the best clean strategy remains far below Oracle routing. The strongest clean method reaches 0.2982 MRR, whereas Oracle routing reaches 0.3337 MRR, leaving approximately 0.0356 MRR of unresolved deployable headroom. This remaining gap supports the paper's controlled claim: structured clean routing recovers part, but not all, of the separability visible under oracle-style selection.
```

---

## 7. Patch status

This patch directly addresses:

- Priority 1: tabled experimental evidence and evidence closure;
- Priority 2: explicit evaluation-line distinction;
- Priority 3: router training and reproducibility details.

Later patches can address:

- compressing the abstract;
- compressing the introduction;
- reducing citation-list density in Related Work;
- restructuring contributions into problem--solution--validation form.
