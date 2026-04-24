# Patch Instructions: Compressed Introduction and Router Training Details

Target file:

- `docs/paper/manuscript_main.tex`

Purpose:

This patch addresses two revision priorities:

1. make the Method section more reproducible by specifying router models, threshold selection, regression training, relation-prior construction, and bootstrap evaluation;
2. compress the Introduction so that the RQ framing remains clear without repeatedly restating the same bounded-gain and clean-routing claims.

---

## 1. Replace the Introduction problem/RQ block with this compressed version

Recommended replacement range:

Start at the paragraph beginning:

```tex
These observations motivate the total problem addressed in this paper:
```

End immediately before:

```tex
In summary, this paper makes the following contributions:
```

Replacement block:

```tex
These observations motivate the central problem of this paper:

\begin{quote}
Under the current OpenBG-IMG protocol, multimodal gain is bounded and protocol-dependent. How can this gain be diagnosed more precisely and exploited more effectively under clean, deployable routing constraints?
\end{quote}

We decompose this problem into three research questions.

\textbf{RQ1. Where does multimodal gain actually appear?} We first examine whether multimodal benefit is globally uniform or concentrated in specific target-side regimes and relation conditions.

\textbf{RQ2. Why does naive clean routing fail?} We then test whether a single global threshold is sufficient for clean selective activation, or whether the protocol requires a more structured decision policy.

\textbf{RQ3. How should bounded gain be exploited once it is known to be conditional?} Finally, we study whether target-aligned clean supervision can recover more deployable gain than coarse binary routing.

To address these questions, we move from always-on multimodal fusion to query-level selective activation between a fusion expert and a structural expert. We begin with a naive global-threshold clean router as a minimal deployable baseline, then evaluate stronger clean formulations, including direction-specific thresholding and regression-style gain prediction. The method is not intended as a universally stronger multimodal architecture. Instead, it operationalizes the bounded-gain finding and tests how much of that gain can be recovered under legal query-time constraints.

The experiments follow the same progression. Protocol-aware subgroup and relation-group analyses show that multimodal gain is local, regime-dependent, and bounded. Clean routing experiments show that a naive global threshold is too coarse, while direction-specific thresholding improves over the clean rule baseline. Target-aligned regression supervision further provides the strongest clean result, although a visible gap to Oracle routing remains.

The paper's claim is therefore controlled: under the current OpenBG-IMG protocol, clean routing fails not only because legal query-time signals are limited, but also because a global threshold and binary gain labels are too coarse. Once the policy is structured more appropriately and trained with a more target-aligned objective, deployable multimodal gain can be recovered more effectively, while the remaining Oracle gap marks unresolved headroom.
```

Why this replacement helps:

- keeps RQ1--RQ3;
- removes repeated explanations after each RQ;
- keeps the controlled claim but makes it less defensive;
- preserves the paper's logic: diagnosis -> clean routing failure -> structured routing and target-aligned supervision.

---

## 2. Replace or refine `Router Training Details` with this stronger version

Recommended replacement range:

Replace the current `\subsection{Router Training Details}` block, or insert this block after `\subsection{Clean Router Features, Training, and Evaluation Basis}` if the subsection is not yet present.

```tex
\subsection{Router Training Details}

\paragraph{Query-level feature and label construction.}
Router training is performed on query-level feature tables constructed from the fixed fusion and structural experts. For every development query $q$, we record the filtered reciprocal ranks produced by the fusion expert and the structural expert, and compute
\[
\Delta(q)=RR_f(q)-RR_s(q).
\]
This development-side expert difference is used only to construct supervision targets. It is not used as a test-time feature. At test time, the router receives only clean query-time features and predicts whether the fusion expert should be activated.

Relation-level priors are also computed from development-side expert outcomes only. For a relation $r$, \texttt{relation\_gain\_prior} denotes the mean value of $\Delta(q)$ over development queries with relation $r$. \texttt{relation\_fusion\_win\_rate} denotes the proportion of such development queries for which $RR_f(q)>RR_s(q)$. \texttt{relation\_support} records the number of development queries used to estimate these relation-level statistics. \texttt{relation\_is\_visual\_prior} is a development-derived binary prior indicating whether the relation belongs to a visually favorable relation group or shows positive development-side fusion tendency under the clean feature construction. These priors are attached to test queries only by relation identity and never use test labels or test-side expert outcomes.

Observed-side modality features are computed only from the known side of the query. For tail prediction $(h,r,?)$, the observed side is the head entity $h$; for head prediction $(?,r,t)$, the observed side is the tail entity $t$. Thus, \texttt{observed\_has\_img}, \texttt{observed\_text\_img\_cosine}, and \texttt{observed\_img\_missing\_replaced} describe only the observed entity. The clean router never uses the hidden target entity, target-side image availability, target-aware subgroup labels, or answer-aware expert scores.

\paragraph{Router model families.}
We evaluate four router families. First, the legal clean rule baseline activates the fusion expert only under conservative query-observable conditions. Second, binary clean classifiers, including logistic regression and XGBoost classifiers, predict whether $\Delta(q)$ exceeds a gain margin $\delta$ and then apply a threshold to the predicted fusion probability. Third, structured threshold policies reuse the same clean prediction scores but replace a single global threshold with direction-specific or bucket-specific thresholds. Fourth, target-aligned routers replace the binary label with richer supervision, including regression-based gain prediction and ordinal gain buckets.

For binary routing, the label is
\[
y(q)=\mathbf{1}[\Delta(q)>\delta].
\]
For regression-based routing, the target is the scalar gain $\Delta(q)$ itself. The implemented regression router is trained with a squared-error objective, so the model learns $\widehat{\Delta}(q)$ by minimizing prediction error on development-side gain values. At inference time, the regression score is converted into a hard routing decision through a development-selected threshold $\theta$:
\[
\alpha_{reg}(q)=\mathbf{1}[\widehat{\Delta}(q)>\theta].
\]
For ordinal routing, $\Delta(q)$ is partitioned into ordered gain buckets, and the final selection uses whether the predicted bucket indicates positive expected fusion gain.

\paragraph{Threshold selection and test protocol.}
All routing hyperparameters are selected on the development split only. This includes the gain margin $\delta$, the single global threshold $\tau$, direction-specific thresholds $\tau_{head}$ and $\tau_{tail}$, bucket-specific thresholds, ordinal decision rules, and the regression decision threshold $\theta$. Once these choices are fixed, the selected router is applied once to the test split for final reporting.

The test split is never used to fit the router, construct relation priors, tune thresholds, choose the gain margin, or select the router family. Test labels and test-side expert outcomes are used only after routing decisions have been made, in order to compute filtered ranking metrics and paired significance results.

\paragraph{Bootstrap evaluation.}
The main significance evidence is computed on the clean routing line using paired query-level bootstrap. For each pair of compared policies, we compute reciprocal-rank differences on the same test query instances, resample these paired query outcomes with replacement, and recompute the MRR difference on each bootstrap sample. The reported confidence intervals are taken from the empirical bootstrap distribution of paired $\Delta$MRR.

This paired design is necessary because all clean routing policies are evaluated on the same query set. The confidence intervals therefore quantify whether one policy improves over another under matched query conditions, rather than comparing unrelated aggregate scores.
```

---

## 3. Optional minor adjustment to the earlier feature table explanation

If the current feature table still looks too engineering-oriented, add the following sentence immediately after the clean feature table:

```tex
All relation-level features are development-derived priors rather than test-time labels: they summarize how the relation behaved on development queries and are then transferred to test queries only through relation identity.
```

---

## 4. Patch status

This patch directly addresses:

- router model family specification;
- dev-only threshold selection;
- regression target and squared-error training objective;
- relation-prior construction;
- observed-side feature legality;
- paired bootstrap evaluation;
- Introduction compression without removing the controlled claim.
