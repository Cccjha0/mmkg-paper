# Task Setting and Method

## 1. Task Setting and Protocol

We study multimodal knowledge graph completion (MMKGC) on OpenBG-IMG under a standard link prediction setting. Each fact is represented as a triple `(h, r, t)`, where `h` and `t` denote head and tail entities and `r` denotes the relation. Given a partially observed query, the task is to rank candidate entities for the missing position. Tail prediction ranks candidate tails for `(h, r, ?)`, whereas head prediction ranks candidate heads for `(?, r, t)`.

This setting differs from structure-only KGC because entities may additionally have associated text and image information. In principle, these auxiliary modalities can enrich entity representations and improve ranking quality. In practice, however, modality availability is not uniform across entities or query conditions. As a result, the present paper is not only about multimodal representation learning in the abstract. It is also about how the evaluation protocol shapes where multimodal evidence can plausibly matter.

All experiments in this paper use the current OpenBG-IMG `paper_split`, which is the unified split employed throughout the project for official model comparison, subgroup analysis, relation-group analysis, and routing evaluation. We intentionally do not redefine the benchmark split for the final paper-stage study, because the purpose of this work is precisely to analyze and operationalize the empirical consequences of the current protocol rather than to replace it with a new one.

The most important property of the current `paper_split` is the asymmetry between target position and image availability. On the head side, target entities can still be image-available for a substantial portion of test queries. On the tail side, however, target entities are effectively always `no_img` under the current test distribution. This means that target position is not merely a formal direction choice. Under the present protocol, it is entangled with modality availability.

This asymmetry is central to the paper. It explains why multimodal gain can appear clearly in some local regimes, especially when the prediction target is image-supported, while overall performance can still remain dominated by structure-favorable conditions. In the earlier analysis-oriented version of the paper, this asymmetry motivated a bounded-gain interpretation. In the present gain-threshold version, it also motivates the method itself: if multimodal gain depends on protocol-shaped query conditions, then multimodal activation should be treated as a selective decision rather than a globally uniform default.

All base models are trained under a unified train/dev/test workflow. The development split is used for early stopping and checkpoint selection, and all final paper-facing metrics are reported on the test split. Router training follows the same separation principle at the feature-label level: query-level gain labels and relation priors are constructed from development-side expert outcomes only, and the trained router is then applied to test-side query features without using test labels during training. This development-only selection discipline is important because the paper moves from empirical diagnosis to operational method; final claims should therefore remain grounded in development-side supervision and test-side reporting.

All formal comparisons use filtered ranking metrics on the test set, including MRR, Hits@1, Hits@3, and Hits@10. All main paper-stage comparisons use `direction=both`, so both head prediction and tail prediction are included in evaluation. Because the current `paper_split` is asymmetric, subgroup analysis must also be defined in a protocol-aware way. Rather than assuming that both directions contain both image-supported and image-missing targets, the present paper analyzes the only three meaningful target-side regimes under the current test distribution:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define `tail_has_img` for the final analysis because it does not meaningfully exist under the current test distribution. This detail is important not only for the bounded-gain diagnosis, but also for the method contribution: the router is not attempting to solve an abstract symmetric multimodal KGC problem. It is attempting to make query-level decisions under this specific three-regime protocol.

The present paper also relies on two complementary evaluation dialects. The first is the **official model-comparison line**, which aggregates completed paper-stage runs through their formal `test_metrics.json` outputs and is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`. The second is the **routing-compatible line**, which is based on query-level exported expert outcomes and recomputed aggregation. This second line is necessary because Oracle routing, rule-based routing, and learned routing all operate on query-level expert predictions. To keep the routing comparison fair, fixed experts inside that comparison are also reported on the same query-level recomputed basis rather than mixed with the official main-result line. These two lines answer different questions and must not be mixed in the same table.

The main message of this protocol section is therefore simple: the conclusions of this paper are inseparable from the current OpenBG-IMG setting. The `paper_split` creates a meaningful but asymmetric missing-visual evaluation space, and this protocol not only reveals that multimodal gain is bounded, but also creates the decision problem that the method is designed to solve.

## 2. Gain-Threshold Routing Framework

The previous sections establish a protocol-aware empirical tension. Under the current OpenBG-IMG setting, multimodal gain is visible but not globally reliable. `Full Model` improves over weaker multimodal baselines, yet stronger structural alternatives—especially `Residual-only`, and in the official model-comparison line also `ComplEx`—remain superior at the global level. This means that the key problem is no longer simply how to build a richer multimodal encoder. Instead, the problem becomes whether multimodal fusion should be activated for the current query at all.

We therefore move from always-on multimodal fusion to **gain-aware selective activation**. The central idea is simple: if multimodal gain is conditional rather than uniformly available, then the model should predict when fusion is likely to help and use a stronger structural fallback otherwise. To operationalize this idea, we propose a lightweight **gain-threshold routing** framework that performs query-level expert selection between a fusion expert and a structural expert.

We study link prediction on OpenBG-IMG under the current paper protocol. A query may be written either as `(h, r, ?)` for tail prediction or as `(?, r, t)` for head prediction. Under the present split, target position is entangled with image availability, so multimodal usefulness is not symmetric across the evaluation space. The test space is naturally decomposed into three target-side regimes: `head_has_img`, `head_no_img`, and `tail_no_img`. These regimes are not equally favorable to multimodal fusion. In particular, `head_has_img` is the clearest multimodal-favorable regime, whereas `tail_no_img` remains strongly structure-favorable. The problem can therefore be reframed as follows:

> Given a query `q`, decide whether multimodal fusion is expected to produce sufficient gain over a stronger structural alternative, and activate the corresponding expert accordingly.

This formulation differs from standard multimodal modeling in one important respect. We are not asking the model to use multimodal evidence everywhere. We are asking it to decide **when multimodal evidence is worth using** under a protocol where gain is known to be bounded.

We define two fixed experts:

- **Fusion expert:** `Gate-only`
- **Structural expert:** `Residual-only`

This choice is deliberate. `Gate-only` represents relation-aware multimodal fusion without an explicit structural compensation branch. `Residual-only` represents structure-heavy compensation within the same internal model family. Together, they are the clearest diagnostic endpoints in the current framework and provide a clean basis for selective routing. We do not use `Full Model` itself as the routed fusion expert in the first version of the method. Doing so would make the interpretation less clear, because `Full Model` already contains a residual path internally. By contrast, routing between `Gate-only` and `Residual-only` keeps the competition interpretable as **fusion versus structural fallback**.

For a query `q` and a candidate entity `e`, let `s_f(q, e)` denote the score from the fusion expert and `s_s(q, e)` denote the score from the structural expert. A query-level router receives a feature vector `x_q` and predicts the probability that the fusion expert should be selected:

\[
p(q) = P(y=1 \mid x_q).
\]

At inference time, we apply a hard threshold `\tau` and define the routing decision as:

\[
\alpha(q) = \mathbf{1}[p(q) > \tau].
\]

The final candidate score is then

\[
s_{\text{final}}(q,e) = \alpha(q) s_f(q,e) + (1-\alpha(q)) s_s(q,e).
\]

This design has two advantages. First, it keeps the experts fixed and changes only the selection mechanism, so any improvement can be attributed to selective activation rather than to added model capacity or end-to-end co-adaptation. Second, it matches the current problem structure well: the goal is not to blend experts softly everywhere, but to activate the multimodal path only when it is sufficiently likely to help.

To train the router, we construct query-level supervision from development-set expert performance. For each query `q`, let `rank_f(q)` be the filtered rank of the correct target entity under the fusion expert and `rank_s(q)` the corresponding rank under the structural expert. We define reciprocal ranks as

\[
RR_f(q) = \frac{1}{rank_f(q)}, \qquad RR_s(q) = \frac{1}{rank_s(q)}.
\]

The gain difference is then

\[
\Delta(q) = RR_f(q) - RR_s(q).
\]

Using this quantity, we define the binary gain label:

\[
y(q) = \mathbf{1}[\Delta(q) > \delta],
\]

where `\delta` is a gain margin that controls how much positive advantage the fusion expert must demonstrate before the query is labeled as gain-positive. This label should be interpreted carefully. It does **not** mean that the query belongs to a universally multimodal-favorable semantic class. Instead, it means that under the current protocol and current expert pair, the fusion expert yields sufficiently larger reciprocal-rank improvement than the structural expert for this specific query.

The method is intentionally lightweight and controlled. We do not train a large end-to-end mixture-of-experts system, and we do not introduce token-level or representation-level routing. Instead, we show that under the current protocol, bounded multimodal gain can already be operationalized as a query-level selective-activation problem between two fixed and interpretable experts.

## 3. Router Features, Training, and Evaluation Basis

A naive router could easily degenerate into a shallow heuristic such as “use fusion whenever images are available.” To prevent this, we design the router feature space to cover multiple complementary views of the query.

The first feature group consists of **protocol-aware condition features**, which directly encode the conditions under which multimodal gain is known to vary under the current benchmark. These features include `direction`, `target_has_img`, `target_regime`, `relation_id`, `relation_gain_prior`, `relation_fusion_win_rate`, and `relation_support`. Their role is to let the router represent the empirical reality that multimodal usefulness depends on target position, modality availability, and relation context under the current role-conditioned missing-modality protocol.

The second feature group consists of **modality-consistency features**, including `text_img_cosine` and `img_is_missing_replaced`. These features describe whether the query appears to have coherent multimodal evidence and whether the image branch is using real visual support or a missing-image replacement vector.

The third feature group consists of **expert-confidence features**, including `fusion_margin`, `struct_margin`, `fusion_correct_score`, `struct_correct_score`, and `delta_margin`. These signals go beyond the question “is fusion plausible?” and address the stronger question “is the fusion expert currently trustworthy enough to prefer over the structural expert?” In the later ablation and interpretability analyses, this confidence-aware information becomes one of the main reasons why learned routing outperforms shallow heuristics.

For ablation, the full feature space is also organized into incremental feature sets:

- `F1`: minimal image-availability signal
- `F2`: protocol-aware condition features
- `F3`: protocol-aware + modality-consistency features
- `F4`: protocol-aware + modality-consistency + expert-confidence features

This staged organization allows us to test whether routing success comes only from modality availability or from a richer protocol-aware decision boundary.

We evaluate three classes of routers. The first baseline is a **rule-based router** that selects fusion only when the target has image support and the relation-level gain prior is positive. This baseline represents the strongest simple heuristic interpretation of the bounded-gain diagnosis. The second router is **logistic regression**, which provides a lightweight and interpretable learned decision boundary. The third router is an **XGBoost-based classifier**, which can model nonlinear interactions among protocol-aware, relation-level, modality, and confidence features.

Router training is performed on development-side query-level feature tables constructed from the fixed experts. The supervised target is the gain label defined above. Training uses only development-set data, and the resulting router is then applied to test-side query-level features without seeing test outcomes during training. For the learned routers, we evaluate multiple gain margins `\delta` and multiple routing thresholds `\tau`. This separation is important: `\delta` controls how gain-positive labels are defined during training, whereas `\tau` controls how conservatively the router activates fusion at inference time.

This training protocol is tied directly to the evaluation basis. The routing framework is evaluated under a unified query-level recomputed line because the router, Oracle selection, and rule-based selection all operate on query-level exported expert outcomes. To keep the comparison fair, fixed experts inside the routing table are also reported under the same recomputed aggregation rather than mixed with the official main-result summary line. This leads to the two complementary evaluation dialects described above: the official model-comparison line supports claims about the ranking of the original models under the paper protocol, whereas the routing-compatible line supports claims about selective activation under a shared query-level basis.

The section-level takeaway is therefore as follows:

> Because multimodal gain in the current OpenBG-IMG protocol is conditional rather than globally reliable, we reformulate MMKGC from always-on multimodal fusion to query-level selective activation. The proposed gain-threshold router predicts whether fusion is worth activating and then routes each query between a fusion expert and a structural expert through a hard-threshold decision rule.
