# Analysis and Discussion

## 1. Overview

The experiments establish two facts that are individually clear but conceptually incomplete unless interpreted together.

First, under the official paper protocol, the strongest fixed model remains `Residual-only`, while `Full Model` is only the strongest multimodal variant inside the internal family. This preserves the earlier bounded-gain diagnosis: multimodal information is useful, but not globally dominant.

Second, under the unified routing line, learned selective activation outperforms both fixed experts and the original always-on multimodal design. The best learned router (`xgb + delta=0.01 + tau=0.7`) reaches `0.3160` MRR, exceeding fixed `Residual-only`, rule-based routing, and the original `Full Model`, while still remaining below Oracle routing.

Taken together, these results suggest a more precise interpretation of the problem. The main obstacle in the current protocol is not the absence of multimodal value. It is the absence of a mechanism that activates multimodal evidence only under conditions where it is likely to help and suppresses it where stronger structural fallback is more reliable. This section explains that interpretation and clarifies what the current results do and do not imply.

## 2. Why Selective Activation Works Under the Current Protocol

### 2.1 Bounded gain is a routing problem in disguise

The earlier analysis showed that multimodal usefulness is unevenly distributed across the test space. The clearest favorable regime appears in `head_has_img`, while the globally dominant `tail_no_img` regime remains strongly structure-favorable. Relation-group evidence further shows that multimodal benefit is relation-dependent but does not scale into grouped-level structural dominance. Behavior analysis adds a mechanism-level view: the fusion path is real and adaptive, but final branch preference remains residual-dominant, especially under weak visual support.

Once these pieces are viewed together, the natural conclusion is that the original always-on multimodal setup is mismatched to the protocol. The protocol does not reward multimodal activation uniformly. It rewards it only under specific query conditions. In that sense, bounded multimodal gain is already a routing problem in disguise. The real challenge is not to force one model path to dominate everywhere, but to decide when multimodal evidence is likely to be worth trusting.

The gain-threshold router is effective precisely because it makes this hidden decision problem explicit. It converts the earlier diagnosis from a descriptive statement—multimodal gain is conditional—into an operational rule: activate the fusion expert only when the current query is likely to benefit from it.

### 2.2 Why always-on `Full Model` underperforms the routed system

The contrast between `Full Model` and the learned router is especially instructive. `Full Model` already contains both multimodal fusion and structural compensation, yet it remains far below the best routing result. This gap should not be read as evidence that `Full Model` learned nothing useful. Instead, it reflects a limitation of always-on branch interaction under the current protocol.

The behavior analysis already showed that `Full Model` learns a meaningful fusion branch, but that the final representation remains residual-dominant. This is a reasonable internal response to the current data regime: because a large part of the evaluation space is structure-favorable, the model learns to trust the structural path more heavily. The difficulty is that such a shared internal compromise cannot cleanly separate the local multimodal-favorable region from the global structure-favorable region.

By contrast, the router does not need to learn a single globally balanced representation. It can preserve the fusion expert for one subset of queries and rely on the structural expert for another subset. In other words, the routed system wins not because it has a stronger fusion module than `Full Model`, but because it is better matched to the protocol's heterogeneous gain structure.

## 3. Why Learned Routing Beats Rule-Based Heuristics

### 3.1 The protocol is informative but not simple

At first glance, the current protocol might suggest an easy routing rule: use fusion when images are available and avoid it when they are not. The experimental results show that this intuition is incomplete. The rule-based router improves only negligibly over the fixed structural expert, while learned routing yields a clear gain.

This difference indicates that the gain boundary is not equivalent to a single protocol flag. Target regime matters, but it is only one part of the decision. Relation context matters as well, because multimodal gain is not aligned perfectly with a coarse visual/nonvisual split. Expert-confidence signals matter too, because even within the same subgroup the fixed experts are not equally reliable on every query.

### 3.2 Evidence from feature ablation

The feature ablation results make this point especially clear. Minimal image-availability features alone (`F1`) perform poorly. Large improvements appear only after protocol-aware condition features are added, and another clear jump appears when expert-confidence features are introduced. This shows that routing success depends on a richer decision boundary that combines subgroup, relation-prior, and confidence information.

The learned router therefore does not merely discover that some queries have images. It discovers that multimodal activation is worthwhile only when several conditions align: the protocol regime is not strongly hostile, the relation-level prior is not unfavorable, and the fusion expert is not obviously less trustworthy than the structural expert for the current query.

### 3.3 Why XGBoost becomes the strongest learned router

The stronger performance of XGBoost over logistic regression fits this picture. The protocol-aware gain structure is not purely linear. There are interactions among direction, target regime, relation priors, and confidence margins. A nonlinear router is better able to model these interactions and therefore better able to make conservative but useful activation decisions. The improvement over logistic regression is not dramatic enough to suggest a radically different problem, but it is stable enough to indicate that query-level routing is not governed by a single simple threshold in the raw feature space.

## 4. How the Current Results Reposition `Gate-only`, `Residual-only`, and `Full Model`

### 4.1 `Gate-only` is locally useful, not globally reliable

The current paper should not position `Gate-only` as a failed model. It remains the clearest fusion endpoint in the internal family and performs well in the multimodal-favorable subgroup structure, especially on the head side when image support is present. Its value in the new paper is conceptual as much as empirical: it represents what pure relation-aware multimodal activation looks like without explicit structural fallback.

The routing results strengthen this interpretation. `Gate-only` is not globally reliable enough to serve as a standalone best model, but it is useful enough to function as the fusion expert inside a selective system. This is exactly the kind of role one would expect for a locally strong but globally unstable multimodal component.

### 4.2 `Residual-only` is more than an ablation baseline

The current results also make it even clearer that `Residual-only` should not be described as a mere ablation. Under both the official model-comparison line and the routing line, it is the strongest fixed structural reference inside the internal family. Behavior analysis further shows that the complete model becomes increasingly residual-biased when image support weakens, which means that `Residual-only` captures a genuine task-level preference rather than an accidental simplification.

For this reason, `Residual-only` is best understood as the structurally reliable fallback expert around which the routing problem is organized. Its strength is exactly why selective activation is needed in the first place.

### 4.3 `Full Model` remains analytically important even if it is not the final winner

The original analysis paper treated `Full Model` as the central diagnostic model, and that remains true in the new paper. Its role has changed, however. It is no longer the method to be defended as the likely global winner. Instead, it becomes the clearest demonstration that always-on multimodal enhancement is not sufficient under the current protocol.

This is still an important contribution. Without `Full Model`, the paper could only compare pure fusion with pure structure. With `Full Model`, the paper can show that even when both paths are integrated inside one richer model, protocol-shaped gain heterogeneity still prevents always-on multimodal modeling from becoming globally optimal. This is the immediate bridge from the old analysis narrative to the new routing method.

## 5. What Relation-Group Evidence Still Contributes

The relation-group results should not be used to claim that the router succeeds because the relation ontology has solved the problem in advance. The grouped evidence does not support such a simple reading. Even coarse visual relations do not become globally favorable to multimodal models as a group.

What relation-group evidence does contribute is something more subtle and more useful: it shows that multimodal gain is relation-dependent, but not in a way that can be captured by one naive semantic grouping. At the relation level, `Full Model` often improves over `Gate-only`, yet still loses to `Residual-only` on most retained relations. This is exactly the kind of pattern that motivates query-level routing rather than global semantic partitioning.

In other words, relation-group analysis supports the selective-activation story indirectly. It tells us that multimodal benefit is not random, but it also tells us that the decision boundary is too fine-grained and too interaction-dependent to be handled by a simple grouped rule.

## 6. What the Current Results Do Not Prove

The strongest claim supported by the present paper is that **selective activation is effective under the current OpenBG-IMG protocol**. Several stronger claims are not supported and should be avoided.

First, the results do not show that multimodal fusion is globally superior to structural modeling in general. Even in the new paper, structural fallback remains indispensable, and the strongest fixed model is still structure-heavy.

Second, the results do not show that the current gain-threshold router is the final or universally best way to exploit bounded multimodal gain. The method is intentionally lightweight and controlled. It is designed to demonstrate that the bounded-gain hypothesis can be operationalized, not to exhaust the entire routing design space.

Third, the results do not prove that the same expert pair, thresholding strategy, or feature set would transfer unchanged to other MMKGC benchmarks or to a different split of OpenBG-IMG. The entire contribution is protocol-aware by construction.

These limits are not weaknesses to hide. They are part of what makes the paper credible. The present work is strongest when it is framed as a careful, protocol-grounded demonstration that query-level selective activation can exploit bounded multimodal gain more effectively than always-on fusion under the current setting.

## 7. Broader Implications

The current results suggest three broader implications for MMKGC research.

First, protocol-aware evaluation matters more than aggregate leaderboard ranking alone. The current study shows that target position, modality availability, and relation structure can jointly shape whether multimodal information appears useful at all.

Second, strong structural competitors remain essential anchors for interpretation. If the paper had compared multimodal variants only to weak baselines, the bounded-gain story would have been far less informative. The central tension appears precisely because strong structural alternatives remain competitive.

Third, selective activation deserves attention as a first-class design principle in multimodal graph learning. The present paper does not claim to solve that agenda completely, but it does show that under heterogeneous modality conditions, deciding **when to trust multimodal evidence** can be as important as deciding **how to fuse it**.

## 8. Section-Level Takeaway

The main message of this section is:

> The success of gain-threshold routing does not overturn the earlier bounded-gain analysis. It operationalizes it. Under the current OpenBG-IMG protocol, multimodal gain is real but query-dependent, and learned selective activation works because it preserves multimodal benefit where it is locally useful while falling back to stronger structural modeling where fusion is less reliable.
