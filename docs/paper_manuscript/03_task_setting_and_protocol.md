# Task Setting and Method

## 1. Task Setting and Protocol

We study multimodal knowledge graph completion (MMKGC) on OpenBG-IMG under a standard link prediction setting. Each fact is represented as a triple `(h, r, t)`, where `h` and `t` denote head and tail entities and `r` denotes the relation. Given a partially observed query, the task is to rank candidate entities for the missing position. Tail prediction ranks candidate tails for `(h, r, ?)`, whereas head prediction ranks candidate heads for `(?, r, t)`.

This setting differs from structure-only KGC because entities may additionally have associated text and image information. In principle, these auxiliary modalities can enrich entity representations and improve ranking quality. In practice, however, modality availability is not uniform across entities or query conditions. As a result, the present paper is not only about multimodal representation learning in the abstract. It is also about how the evaluation protocol shapes where multimodal evidence can plausibly matter.

All experiments in this paper use the current OpenBG-IMG `paper_split`, which is the unified split employed throughout the project for official model comparison, subgroup analysis, relation-group analysis, and routing evaluation. We intentionally do not redefine the benchmark split for the final paper-stage study, because the purpose of this work is precisely to analyze and operationalize the empirical consequences of the current protocol rather than to replace it with a new one.

The most important property of the current `paper_split` is the asymmetry between target position and image availability. On the head side, target entities can still be image-available for a substantial portion of test queries. On the tail side, however, target entities are effectively always `no_img` under the current test distribution. This means that target position is not merely a formal direction choice. Under the present protocol, it is entangled with modality availability.

This asymmetry is central to the paper. It explains why multimodal gain can appear clearly in some local regimes, especially when the prediction target is image-supported, while overall performance can still remain dominated by structure-favorable conditions. It also explains why clean routing is difficult: the deployable decision boundary is not formed in a neutral symmetric space, but inside a protocol where head-side and tail-side queries operate under systematically different gain regimes.

All base models are trained under a unified train/dev/test workflow. The development split is used for early stopping and checkpoint selection, and all final paper-facing metrics are reported on the test split. Router training follows the same separation principle at the feature-label level: query-level gain labels and relation priors are constructed from development-side expert outcomes only, and the trained router is then applied to test-side query features without using test labels during training. This development-only selection discipline is important because the paper moves from empirical diagnosis to operational method; final claims should therefore remain grounded in development-side supervision and test-side reporting.

All formal comparisons use filtered ranking metrics on the test set, including MRR, Hits@1, Hits@3, and Hits@10. All main paper-stage comparisons use `direction=both`, so both head prediction and tail prediction are included in evaluation. Because the current `paper_split` is asymmetric, subgroup analysis must also be defined in a protocol-aware way. Rather than assuming that both directions contain both image-supported and image-missing targets, the present paper analyzes the only three meaningful target-side regimes under the current test distribution:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define `tail_has_img` for the final analysis because it does not meaningfully exist under the current test distribution. This detail is important not only for the bounded-gain diagnosis, but also for the method contribution: the router is not attempting to solve an abstract symmetric multimodal KGC problem. It is attempting to make query-level decisions under this specific three-regime protocol.

The present paper also relies on complementary evaluation lines. The first is the **official model-comparison line**, which aggregates completed paper-stage runs through their formal `test_metrics.json` outputs and is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`. The second is the **clean routing line**, which is based on query-level exported expert outcomes and unified recomputation under legal query-time constraints. This line is used for `Gate-only`, `Residual-only`, Oracle routing, the clean rule baseline, naive global-threshold clean routing, structured clean threshold policies, and target-aligned clean supervision. A third line, used only for analysis, is the **post-hoc selector line**, which may use target-aware or confidence-rich signals to study upper-bound-style separability, but is not part of the deployable clean claims. These lines answer different questions and must not be mixed in the same table.

The main message of this protocol section is therefore simple: the conclusions of this paper are inseparable from the current OpenBG-IMG setting. The `paper_split` creates a meaningful but asymmetric missing-visual evaluation space, and this protocol not only reveals that multimodal gain is bounded, but also creates the structured clean decision problem that the method is designed to solve.

## 2. Clean Routing Formulations for Selective Activation

The previous section establishes a protocol-aware empirical tension. Under the current OpenBG-IMG setting, multimodal gain is visible but not globally reliable. `Full Model` improves over weaker multimodal baselines, yet stronger structural alternatives—especially `Residual-only`—remain superior at the global level. This means that the key problem is no longer simply how to build a richer multimodal encoder. Instead, the problem becomes whether multimodal fusion should be activated for the current query at all, and if so, how that decision should be made under clean query-time legality.

We therefore move from always-on multimodal fusion to **clean query-level selective activation**. The central idea is simple: if multimodal gain is conditional rather than uniformly available, then the system should predict when fusion is likely to help and use a stronger structural fallback otherwise. To operationalize this idea, we study query-level expert selection between a fusion expert and a structural expert.

We define two fixed experts:

- **Fusion expert:** `Gate-only`
- **Structural expert:** `Residual-only`

This choice is deliberate. `Gate-only` represents relation-aware multimodal fusion without an explicit structural compensation branch. `Residual-only` represents structure-heavy compensation within the same internal model family. Together, they are the clearest diagnostic endpoints in the current framework and provide a clean basis for selective routing. We do not use `Full Model` itself as the routed fusion expert in the first version of the method, because `Full Model` already contains a residual path internally and would blur the interpretation of the selection problem.

For a query `q` and a candidate entity `e`, let `s_f(q, e)` denote the score from the fusion expert and `s_s(q, e)` denote the score from the structural expert. The operational problem can then be stated as follows:

> Given a query `q`, decide whether multimodal fusion is expected to provide enough gain over the structural expert to justify activation under a clean query-time constraint.

This formulation differs from standard multimodal modeling in one important respect. We are not asking the model to use multimodal evidence everywhere. We are asking it to decide **when multimodal evidence is worth using** under a protocol where gain is known to be bounded.

### 2.1 Clean legality constraint

The key methodological boundary of this paper is that the main deployable routing claims must remain **clean**. A clean router may use only information that is available at query time. In particular, it may use:

- the query direction (`head` or `tail` prediction),
- relation identity and development-derived relation priors,
- query-observable modality indicators derived from the observed side of the query,
- and router outputs computed from these legal inputs.

By contrast, the deployable clean router may **not** use:

- hidden target-side information,
- target-side image-availability labels,
- target-aware regimes,
- correct-target scores,
- or any other signals that require the true missing entity or post-hoc expert outcomes at inference time.

This distinction is essential. Stronger target-aware or confidence-aware selectors may still be analyzed later as post-hoc tools, but they are not part of the main clean method claims.

### 2.2 Naive global-threshold clean routing

The simplest clean routing formulation uses a learned probability `p(q)` and a single global threshold `\tau`. Let `x_q` denote a clean feature vector available at query time. The router predicts

\[
p(q) = P(y=1 \mid x_q),
\]

where `y=1` means that the fusion expert should be selected. At inference time, we define

\[
\alpha(q) = \mathbf{1}[p(q) > \tau],
\]

and the final score becomes

\[
s_{\text{final}}(q,e) = \alpha(q) s_f(q,e) + (1-\alpha(q)) s_s(q,e).
\]

This global-threshold formulation is intentionally simple. It serves as a minimal deployable clean baseline rather than the final structured policy proposed by the paper. As shown later in the experiments, this baseline is too coarse to capture the strongly asymmetric gain structure induced by the current protocol.

### 2.3 Structured clean threshold policies

The new results of this paper show that the weakness of the naive clean router is not solely due to signal scarcity. It is also caused by policy granularity. Under the present protocol, head-side and tail-side queries operate under different gain regimes, so a single global threshold is too rigid.

We therefore study a stronger family of **structured clean threshold policies**. The most important member is **direction-specific thresholding**, which replaces the single global threshold with two query-direction-dependent thresholds:

\[
\alpha(q)=
\begin{cases}
\mathbf{1}[p(q) > \tau_{head}], & q \text{ is a head-prediction query}, \\
\mathbf{1}[p(q) > \tau_{tail}], & q \text{ is a tail-prediction query}.
\end{cases}
\]

This formulation keeps the router clean, because query direction is always observable at inference time, but it allows the policy to adapt to the asymmetric decision structure induced by the protocol.

We also study optional bucketized thresholding variants, such as relation-prior buckets or query-observable groupings, as supporting structured-policy extensions. These are not all equally strong in practice, but they serve a conceptual purpose: they show that the main limitation of naive clean routing lies in enforcing one globally shared decision boundary over a space that is known to be heterogeneous.

### 2.4 Target-aligned supervision for clean routing

The second major refinement concerns the training target. In the original gain-threshold formulation, supervision is derived from a coarse binary gain label. For each query `q`, let `rank_f(q)` and `rank_s(q)` be the filtered ranks of the correct target entity under the fusion and structural experts, respectively, and define reciprocal ranks as

\[
RR_f(q) = \frac{1}{rank_f(q)}, \qquad RR_s(q) = \frac{1}{rank_s(q)}.
\]

The gain difference is then

\[
\Delta(q) = RR_f(q) - RR_s(q).
\]

The original binary label is defined as

\[
y(q) = \mathbf{1}[\Delta(q) > \delta],
\]

where `\delta` is a gain margin. This label is useful as a first operationalization of bounded gain, but it is also coarse: it collapses different gain magnitudes into the same binary decision class.

To better align supervision with the actual expert difference, we therefore study stronger clean targets:

1. **Binary gain label** as the naive baseline;
2. **Regression target**, which predicts `\Delta(q)` directly;
3. **Ordinal gain buckets**, which partition `\Delta(q)` into multiple ordered intervals.

These target-aligned formulations preserve more information about gain magnitude than the original binary label and therefore provide a stronger basis for clean routing under shallow and uneven decision boundaries.

### 2.5 Clean router features

The clean routing line uses only legal query-time features. These include:

- `direction`,
- `relation_id`,
- development-derived relation priors such as `relation_gain_prior`, `relation_fusion_win_rate`, `relation_support`, and `relation_is_visual_prior`,
- and observed-side modality indicators such as `observed_has_img`, `observed_text_img_cosine`, and `observed_img_missing_replaced`.

These features are intentionally limited. They do not attempt to reconstruct the hidden target, and they do not use target-aware or answer-aware confidence signals. Their purpose is to test how much of bounded multimodal gain can be recovered under a truly deployable query-time constraint.

## 3. Router Training Details

### 3.1 Query-level feature and label construction

Router training is performed on query-level feature tables constructed from the fixed fusion and structural experts. For every development query `q`, we record the reciprocal ranks produced by the fusion expert and the structural expert, and compute the expert difference:

\[
\Delta(q)=RR_f(q)-RR_s(q).
\]

This development-side expert difference is used only to create router supervision. It is not used as a test-time feature. At test time, the router receives only clean query-time features and predicts whether the fusion expert should be activated.

Relation-level priors are also computed from the development split only. For a relation `r`, `relation_gain_prior` denotes the average development-side reciprocal-rank gain of the fusion expert over the structural expert for that relation. `relation_fusion_win_rate` denotes the proportion of development queries for which the fusion expert outperforms the structural expert. `relation_support` records the number of development queries available for that relation, and `relation_is_visual_prior` is a development-derived indicator of whether the relation tends to behave as visually favorable under the clean feature construction. These relation priors are then attached to test queries by relation identity without using test labels or test expert outcomes.

Observed-side modality features are computed only from the known side of the query. For tail prediction `(h,r,?)`, the observed side is the head entity `h`; for head prediction `(?,r,t)`, the observed side is the tail entity `t`. The clean router therefore never uses the hidden target entity, target-side image availability, or target-aware subgroup labels.

### 3.2 Router model families

The clean routing experiments evaluate three families of routing models. The first is the legal clean rule baseline, which activates the fusion expert only under conservative query-observable conditions. The second is naive global-threshold learned routing, where classifiers such as logistic regression and XGBoost output a fusion-selection probability under a single threshold. The third family contains the stronger clean policies studied in the current paper: direction-specific thresholding, optional bucketized thresholding, regression-based gain prediction, and ordinal gain modeling.

For binary routing, the classifier is trained to predict whether `\Delta(q)` exceeds a development-defined gain margin `\delta`. For regression-based routing, the model predicts the scalar target `\Delta(q)` directly, using a standard squared-error regression objective in the implemented regressor family. For ordinal routing, the target is formed by partitioning `\Delta(q)` into ordered gain buckets, and the final routing decision is derived from whether the predicted bucket indicates positive expected fusion gain.

### 3.3 Threshold selection and test protocol

All routing hyperparameters are selected on the development split only. This includes the gain margin `\delta`, the single global threshold `\tau`, direction-specific thresholds `\tau_head` and `\tau_tail`, bucketized thresholds, and the regression decision threshold `\theta`. After these settings are selected, the trained router is applied once to the test split for final reporting.

This protocol is important because the routing task is vulnerable to leakage. The test split is never used to fit the router, construct relation priors, tune thresholds, choose the gain margin, or select the router family. Test labels and test-side expert outcomes are used only after routing decisions have been made, in order to compute filtered ranking metrics and paired significance results.

### 3.4 Bootstrap evaluation

The main significance evidence is computed on the clean routing line using paired query-level bootstrap. For each pair of compared methods, we compute reciprocal-rank differences on the same set of test query instances, resample these paired query outcomes with replacement, and recompute the MRR difference on each bootstrap sample. The reported confidence intervals are taken from the empirical bootstrap distribution of paired `\Delta`MRR.

This paired design matters because all clean routing methods are evaluated on the same test query set. The bootstrap intervals therefore quantify whether one clean policy improves over another under matched query conditions, rather than comparing unrelated aggregate scores.

## 4. Analysis-Only Post-hoc Selector and Evaluation Basis

In addition to the clean routing line, we retain a stronger post-hoc selector for analysis only. This selector may use target-aware or confidence-rich information that is unavailable to a deployable clean router. Its purpose is not to support the main method claim, but to study offline separability and remaining headroom relative to Oracle-like selection.

The evaluation basis follows directly from this distinction. The routing framework is evaluated under a unified query-level recomputed clean line because the router, Oracle selection, and clean rule selection all operate on query-level exported expert outcomes. To keep this comparison fair, fixed experts inside that table are also reported on the same recomputed aggregation rather than mixed with the official main-result summary line. This leads to the complementary evaluation lines described above:

- the **official model-comparison line** supports claims about the ranking of the original models under the paper protocol,
- the **clean routing line** supports claims about deployable selective activation under legal query-time constraints,
- and the **post-hoc selector line** supports analysis of stronger offline separability.

The section-level takeaway is therefore as follows:

> Because multimodal gain in the current OpenBG-IMG protocol is conditional rather than globally reliable, we reformulate MMKGC from always-on multimodal fusion to clean query-level selective activation. The paper studies not only a naive global-threshold clean baseline, but also stronger structured clean policies and more target-aligned clean supervision, while retaining post-hoc selectors only as analysis tools rather than deployable methods.
