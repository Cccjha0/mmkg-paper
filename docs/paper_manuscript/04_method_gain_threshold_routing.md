# Method: Gain-Threshold Routing for Selective Multimodal Activation

## 1. Overview

The previous sections establish a protocol-aware empirical tension. Under the current OpenBG-IMG setting, multimodal gain is visible but not globally reliable. `Full Model` improves over weaker multimodal baselines, yet stronger structural alternatives—especially `Residual-only`, and in the official main-result line also `ComplEx`—remain superior at the global level. This means that the key problem is no longer simply how to build a richer multimodal encoder. Instead, the problem becomes whether multimodal fusion should be activated for the current query at all.

We therefore move from always-on multimodal fusion to **gain-aware selective activation**. The central idea is simple: if multimodal gain is conditional rather than uniformly available, then the model should predict when fusion is likely to help and use a stronger structural fallback otherwise. To operationalize this idea, we propose a lightweight **gain-threshold routing** framework that performs query-level expert selection between a fusion expert and a structural expert.

Importantly, this design does not attempt to train a new large end-to-end multimodal architecture. Instead, it isolates the contribution of selective activation itself. The experts are fixed, the router is lightweight, and the method is evaluated under the same protocol-aware conditions that motivated the bounded-gain analysis in the first place.

## 2. Problem Formulation

We study link prediction on OpenBG-IMG under the current paper protocol. A query may be written either as `(h, r, ?)` for tail prediction or as `(?, r, t)` for head prediction. Under the present split, target position is entangled with image availability: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. As a result, multimodal usefulness is not symmetric across the evaluation space.

Earlier analysis showed that the test space is naturally decomposed into three target-side regimes:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

These regimes are not equally favorable to multimodal fusion. In particular, `head_has_img` is the clearest multimodal-favorable regime, whereas `tail_no_img` remains strongly structure-favorable. The problem can therefore be reframed as follows:

> Given a query `q`, decide whether multimodal fusion is expected to produce sufficient gain over a stronger structural alternative, and activate the corresponding expert accordingly.

This formulation differs from standard multimodal modeling in one important respect. We are not asking the model to use multimodal evidence everywhere. We are asking it to decide **when multimodal evidence is worth using** under a protocol where gain is known to be bounded.

## 3. Dual-Expert Gain-Threshold Routing

### 3.1 Fixed experts

We define two fixed experts.

- **Fusion expert:** `Gate-only`
- **Structural expert:** `Residual-only`

This choice is deliberate. `Gate-only` represents relation-aware multimodal fusion without an explicit structural compensation branch. `Residual-only` represents structure-heavy compensation within the same internal model family. Together, they are the clearest diagnostic endpoints in the current framework and provide a clean basis for selective routing.

We do not use `Full Model` itself as the routed fusion expert in the first version of the method. Doing so would make the interpretation less clear, because `Full Model` already contains a residual path internally. By contrast, routing between `Gate-only` and `Residual-only` keeps the competition interpretable as **fusion versus structural fallback**.

### 3.2 Query-level routing formulation

For a query `q` and a candidate entity `e`, let:

- `s_f(q, e)` denote the score from the fusion expert
- `s_s(q, e)` denote the score from the structural expert

A query-level router receives a feature vector `x_q` and predicts the probability that the fusion expert should be selected:

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

## 4. Gain Label Construction

To train the router, we construct query-level supervision from development-set expert performance. For each query `q`, let:

- `rank_f(q)` be the filtered rank of the correct target entity under the fusion expert
- `rank_s(q)` be the filtered rank of the correct target entity under the structural expert

We define reciprocal ranks as:

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

where `\delta` is a gain margin that controls how much positive advantage the fusion expert must demonstrate before the query is labeled as gain-positive.

This label should be interpreted carefully. It does **not** mean that the query belongs to a universally multimodal-favorable semantic class. Instead, it means that under the current protocol and current expert pair, the fusion expert yields sufficiently larger reciprocal-rank improvement than the structural expert for this specific query.

In the first-round routing experiments, we evaluate `\delta \in {0.00, 0.01, 0.02}`. All labels are constructed from development-set query outcomes only. Test labels are never used to train the router, and relation priors are also estimated strictly from train/dev-side information to avoid leakage.

## 5. Protocol-Aware Router Features

A naive router could easily degenerate into a shallow heuristic such as “use fusion whenever images are available.” To prevent this, we design the router feature space to cover multiple complementary views of the query.

### 5.1 Protocol-aware condition features

The first feature group directly encodes the protocol conditions under which multimodal gain is known to vary. These features include:

- `direction`
- `target_has_img`
- `target_regime`
- `relation_id`
- `relation_gain_prior`
- `relation_fusion_win_rate`
- `relation_support`

These features let the router represent the empirical reality that multimodal usefulness depends on target position, modality availability, and relation context under the current role-conditioned missing-modality protocol.

### 5.2 Modality-consistency features

The second feature group describes whether the query appears to have coherent multimodal evidence. These features include:

- `text_img_cosine`
- `img_is_missing_replaced`

The cosine feature measures the compatibility between text and image representations, while the missing-image indicator distinguishes real visual support from the learnable replacement used when an entity lacks an image. In the present results, modality-consistency features appear to provide weaker gains than protocol or confidence features, but they remain useful as a complementary signal.

### 5.3 Expert-confidence features

The third feature group captures whether one expert appears more trustworthy for the current query. These features include:

- `fusion_margin`
- `struct_margin`
- `fusion_correct_score`
- `struct_correct_score`
- `delta_margin`

These signals go beyond the question “is fusion plausible?” and address the stronger question “is the fusion expert currently confident enough to trust more than the structural expert?” In the later feature-ablation study, this confidence-aware information is one of the main reasons why the learned router outperforms shallow heuristics.

### 5.4 Feature-set view for ablation

For ablation, the full feature space is also organized into incremental feature sets:

- `F1`: minimal image-availability signal
- `F2`: protocol-aware condition features
- `F3`: protocol-aware + modality-consistency features
- `F4`: protocol-aware + modality-consistency + expert-confidence features

This staged organization allows us to test whether routing success comes only from modality availability or from a richer protocol-aware decision boundary.

## 6. Router Baselines and Training

We evaluate three classes of routers.

### 6.1 Rule-based router

The first baseline is a rule-based router that selects fusion only when the target has image support and the relation-level gain prior is positive. This baseline represents the strongest simple heuristic interpretation of the earlier gain-boundary analysis. It is deliberately shallow and serves as a lower bound for learned routing.

### 6.2 Logistic regression router

The second router is logistic regression. Its purpose is to provide a lightweight and interpretable learned decision boundary. If logistic regression already improves over the rule-based router, this would indicate that gain-positive queries are learnable from the proposed feature space without requiring a highly flexible nonlinear model.

### 6.3 Gradient-boosted router

The third router is an XGBoost-based classifier. This router can model nonlinear interactions among protocol-aware, relation-level, modality, and confidence features. In the first-round experiments, it is the strongest learned router and provides the best trade-off between routing precision and retained multimodal benefit.

### 6.4 Training protocol

Router training is performed on development-side query-level feature tables constructed from the fixed experts. The supervised target is the gain label defined above. Training uses only development-set data, and the resulting router is then applied to test-side query-level features without seeing test outcomes during training.

For the learned routers, we evaluate multiple gain margins `\delta` and multiple routing thresholds `\tau`. This separation is important:

- `\delta` controls how gain-positive labels are defined during training
- `\tau` controls how conservatively the router activates fusion at inference time

By scanning both, we can separate the question of **what counts as useful gain** from the question of **how cautious the final routing decision should be**.

## 7. Hard-Threshold Inference and Evaluation Dialects

The routing framework is evaluated under a unified query-level recomputed line. This detail matters because the router, the Oracle selector, and the rule-based selector all operate on query-level exported expert outcomes. To keep the comparison fair, fixed experts in the routing table are also reported under the same query-level recomputed aggregation rather than mixed with the official main-result summary line.

This leads to two distinct but complementary evaluation dialects in the paper:

1. **Official main-result dialect**
   - based on `test_metrics.json`
   - used for the seven-model paper-facing main table
   - supports claims about the original model family under the formal paper protocol

2. **Routing dialect**
   - based on query-level exported expert predictions and recomputed aggregation
   - used for Oracle, rule-based routing, logistic routing, XGBoost routing, and the fixed experts inside the router comparison
   - supports claims about selective activation under a shared routing-compatible line

These two dialects must not be mixed in the same table. The first addresses the performance ordering of the original models. The second addresses whether query-level selective activation improves over fixed experts once all rows are computed under the same routing-compatible protocol.

## 8. Design Rationale and Scope

The proposed method is intentionally lightweight and controlled. Several design decisions reflect this goal.

First, the experts are fixed rather than jointly optimized with the router. This isolates the value of selective activation itself.

Second, routing is performed at the query level rather than at the representation or token level. This matches the protocol-aware problem framing and keeps the method simple enough to interpret.

Third, we use hard-threshold routing in the first version of the method rather than soft mixing. Hard routing aligns more naturally with the bounded-gain hypothesis: the central claim is that multimodal activation should be selective, not merely softly down-weighted everywhere.

Finally, the method is protocol-specific by design. It is not intended to prove that all MMKGC settings should use the same expert pair or the same router features. Instead, it shows that under the current OpenBG-IMG protocol, bounded multimodal gain can be turned into a useful decision problem and exploited through selective activation.

## 9. Section-Level Takeaway

The main message of this section is:

> Because multimodal gain in the current OpenBG-IMG protocol is conditional rather than globally reliable, we reformulate MMKGC from always-on multimodal fusion to query-level selective activation. The proposed gain-threshold router predicts whether fusion is worth activating and then routes each query between a fusion expert and a structural expert through a hard-threshold decision rule.
