# Experiments

## 1. Experimental Setup

All experiments are conducted on the current OpenBG-IMG `paper_split` under the unified protocol described in the previous section. Model selection is based on development performance, and all final paper-facing metrics are computed on the test set using filtered ranking with `direction=both`. Unless otherwise stated, results are aggregated over three random seeds and reported as `mean ± std` when the source line supports that form.

The overall experimental design contains two complementary evaluation lines.

The first is the **official model-comparison line**, which is based on the aggregated `test_metrics.json` files from the completed paper-stage runs. This line is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`. It answers the question: how do the original models rank under the formal paper protocol?

The second is the **routing line**, which is based on query-level exported expert predictions and unified recomputation. This line is used for `Gate-only`, `Residual-only`, Oracle selection, rule-based routing, logistic routing, and XGBoost routing. It answers a different question: once all rows are placed on the same query-level routing-compatible line, does selective activation improve over fixed experts?

These two lines serve different purposes and must not be mixed in the same table. The official line supports claims about the original model family under the formal paper protocol. The routing line supports claims about gain-aware selective activation.

## 2. Compared Methods and Evaluation Dialects

### 2.1 Official model-comparison line

The official line compares the seven completed paper-stage models:

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`
- `ComplEx`
- `TuckER`

This line establishes the empirical status of the original model family and the structural baselines before any routing is introduced.

### 2.2 Routing line

The routing line compares:

- `Gate-only` (fixed fusion expert, query-level recomputed)
- `Residual-only` (fixed structural expert, query-level recomputed)
- `Full Model` (paper baseline, reported for context)
- Oracle routing
- Rule-based routing
- learned routing with logistic regression
- learned routing with XGBoost

The router is trained on development-side query-level features and evaluated on test-side query-level exported results. Gain margins `delta` and routing thresholds `tau` are varied in the router experiments, while the final routing comparison focuses on the strongest learned settings.

## 3. Official Main Results

### 3.1 Seven-model comparison under the official line

Table X reports the official seven-model comparison under the formal paper protocol. The overall ordering is:

1. `Residual-only`
2. `ComplEx`
3. `Full Model`
4. `Gate-only`
5. `Early Fusion`
6. `Text-only`
7. `TuckER`

Using the official aggregated line, `Residual-only` reaches `0.2930 ± 0.0008` MRR, `ComplEx` reaches `0.2588 ± 0.0018`, and `Full Model` reaches `0.2100 ± 0.0097`. Within the internal multimodal family, `Full Model` is therefore the strongest multimodal variant, outperforming `Gate-only` (`0.1739 ± 0.0044`), `Early Fusion` (`0.1666 ± 0.0013`), and `Text-only` (`0.1261 ± 0.0043`).

### 3.2 Interpretation

These official results establish the core tension inherited from the earlier analysis. The most complete multimodal architecture is not the strongest overall model under the current protocol. Instead, the strongest global result comes from the structure-heavy `Residual-only` model, while the classical structural baseline `ComplEx` also remains stronger than `Full Model`. At the same time, `Full Model` consistently improves over the weaker multimodal baselines, which means that multimodal modeling is not useless even though it does not dominate globally.

The official main table therefore supports two conclusions at once. First, multimodal fusion provides real value inside the internal family. Second, stronger structural compensation remains globally preferred under the present protocol. Taken together, these observations motivate the routing experiments: if multimodal gain is real but bounded, perhaps it should be activated selectively rather than kept on everywhere.

## 4. Routing Results Under the Unified Query-Level Line

### 4.1 Main routing comparison

Table Y reports the routing comparison under the unified query-level recomputed line. On this line, the fixed fusion expert `Gate-only` reaches `0.1688` MRR, while the fixed structural expert `Residual-only` reaches `0.2930`. The rule-based router yields `0.2939`, which is only marginally above the fixed structural expert. By contrast, learned routers improve more substantially:

- logistic (`delta=0.01`, `tau=0.7`): `0.3111`
- XGBoost (`delta=0.01`, `tau=0.7`): `0.3160`

The best learned router is therefore `XGBoost + delta=0.01 + tau=0.7`, with overall MRR `0.3159569`. This result is higher than the fixed `Residual-only` expert (`0.2930494`), higher than the rule-based router (`0.2939469`), and also higher than the original `Full Model` (`0.2100`, reported from the official line for context). Oracle routing reaches `0.3337373`, showing that the routing problem still contains additional exploitable headroom.

### 4.2 Interpretation

The routing table supports the central method claim of the paper. Once the earlier bounded-gain finding is converted into a query-level expert-selection problem, a lightweight router can improve over both fixed experts and over the original always-on multimodal design. Importantly, the rule-based router provides only negligible improvement, whereas learned routing provides a clear gain. This means that the problem cannot be solved well by a simple heuristic such as “use fusion whenever images are available.” Query-level selective activation is useful precisely because the gain boundary is more complex than a single protocol flag.

## 5. Subgroup Evaluation

### 5.1 Why subgroup evaluation remains necessary

Although the routing table shows overall improvement, overall metrics alone do not reveal whether the learned router behaves consistently with the protocol-aware gain structure that motivated the method. We therefore evaluate all routing models over the same three meaningful target-side regimes:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

### 5.2 Main subgroup pattern

The subgroup results show a clear and coherent pattern.

For the fixed experts, the contrast remains sharp. `Gate-only` performs best among the fixed endpoints in `head_has_img` (`0.01235`), whereas `Residual-only` dominates the globally stronger `tail_no_img` regime (`0.58325`). `Full Model` lies between them, reflecting the earlier bounded-gain picture.

The learned routers follow the expected selective pattern. As the threshold `tau` increases, fusion coverage declines across all regimes, but the decline is especially consequential in `tail_no_img`, where stronger structural fallback is more reliable. For the strongest learned setting (`xgb`, `delta=0.01`, `tau=0.7`), the subgroup metrics are:

- `head_has_img`: `0.01161`
- `head_no_img`: `0.01253`
- `tail_no_img`: `0.62003`

At the same time, subgroup fusion coverage for this setting is:

- `head_has_img`: `0.1501`
- `head_no_img`: `0.1059`
- `tail_no_img`: `0.2326`

### 5.3 Interpretation

These results show that the learned router is not behaving randomly or globally. Instead, it becomes more conservative in structure-favorable settings while still preserving a pathway for multimodal activation in the conditions where local gain is more plausible. This is exactly the expected behavior if the router has learned the protocol-aware gain structure rather than a shallow shortcut.

## 6. Threshold Scan

### 6.1 Main threshold trade-off

To understand how routing conservativeness affects performance, we scan inference thresholds `tau`. The XGBoost router shows a clear trade-off between fusion coverage, gain precision, and final MRR:

- `tau=0.1`: MRR `0.2514`, coverage `0.483`, gain precision `0.327`
- `tau=0.3`: MRR `0.2921`, coverage `0.333`, gain precision `0.452`
- `tau=0.5`: MRR `0.3084`, coverage `0.252`, gain precision `0.555`
- `tau=0.7`: MRR `0.3160`, coverage `0.185`, gain precision `0.670`
- `tau=0.9`: MRR `0.3142`, coverage `0.096`, gain precision `0.846`

The logistic router follows the same qualitative direction, but its entire curve remains below the XGBoost curve.

### 6.2 Interpretation

The threshold scan provides one of the clearest validations of the selective-activation hypothesis. Performance does not improve by activating fusion more aggressively. Instead, MRR rises as the router becomes more selective and filters fusion more cautiously, then plateaus once fusion coverage becomes very small. This means that under the current protocol, multimodal activation is beneficial only when precision is high enough. The best setting does not correspond to maximum fusion usage, but to a balanced threshold at which harmful fusion has been significantly suppressed while some local multimodal benefit is still retained.

## 7. Feature Ablation

### 7.1 Ablation design

To test whether routing success comes only from shallow image-availability cues, we evaluate incremental feature sets `F1` to `F4`.

- `F1`: minimal image-availability signal
- `F2`: protocol-aware condition features
- `F3`: protocol-aware + modality-consistency features
- `F4`: protocol-aware + modality-consistency + expert-confidence features

### 7.2 Main ablation results

The ablation results show a clear hierarchy.

For logistic regression:

- `F1`: MRR `0.1651`
- `F2`: MRR `0.2333`
- `F3`: MRR `0.2333`
- `F4`: MRR `0.2501`

For XGBoost:

- `F1`: MRR `0.1651`
- `F2`: MRR `0.2306`
- `F3`: MRR `0.2303`
- `F4`: MRR `0.2556`

### 7.3 Interpretation

The ablation evidence strongly argues against a trivial explanation of the routing gains. `F1`, which essentially captures only whether the target is image-supported, performs poorly for both models. A large jump appears when protocol-aware condition features are added (`F2`), showing that direction, subgroup structure, and relation-level priors are central. Modality-consistency signals (`F3`) add only limited extra benefit in the present setting. By contrast, the final jump from `F3` to `F4` shows that expert-confidence signals such as `fusion_margin`, `struct_margin`, and `delta_margin` are major contributors to successful routing. The learned router therefore depends on a richer query-level decision boundary than a simple “has image” rule.

## 8. Relation-Group Evaluation

### 8.1 Why relation-group evidence is still included

Relation-group evaluation remains part of the paper because the earlier bounded-gain analysis showed that multimodal usefulness is not determined by target regime alone. Relation characteristics also matter. We therefore retain relation-group evidence in the experiments section as supporting analysis for the selective-activation narrative.

### 8.2 Main findings

Using the focused three-model comparison (`Gate-only`, `Full Model`, `Residual-only`), the grouped ordering remains stable across all three coarse relation groups:

1. `Residual-only`
2. `Full Model`
3. `Gate-only`

The grouped MRR values are:

- `visual_relations`: `Gate-only 0.1350`, `Full Model 0.1826`, `Residual-only 0.2691`
- `weak_visual_relations`: `Gate-only 0.2194`, `Full Model 0.2252`, `Residual-only 0.3197`
- `ambiguous_material_relations`: `Gate-only 0.1337`, `Full Model 0.1948`, `Residual-only 0.2842`

With the `MIN20` support filter at the relation level, `Full Model` outperforms `Gate-only` on most retained relations, but surpasses `Residual-only` on only a minority of them.

### 8.3 Interpretation

These results do not support a naive claim that “visual relations as a whole favor multimodal models.” Instead, they support the more careful statement that multimodal gain is relation-dependent but still bounded. This makes relation-group evidence a useful supporting bridge between the earlier gain-boundary analysis and the present routing results. It helps justify why query-level selective activation is necessary: relation conditions matter, but not in a sufficiently simple way to be handled by a single coarse rule.

## 9. Overall Experimental Summary

The experiments lead to a coherent overall conclusion.

First, under the official paper protocol, `Residual-only` remains the strongest fixed model overall, while `Full Model` remains the strongest multimodal variant inside the internal family. This preserves the core tension established in the earlier analysis.

Second, under the unified routing line, selective activation is effective. The best learned router (`xgb + delta=0.01 + tau=0.7`) reaches `0.3160` MRR, outperforming the fixed structural expert, the rule-based router, and the original always-on `Full Model`, while still leaving room to Oracle routing.

Third, subgroup, threshold-scan, feature-ablation, and relation-group evidence all support the same interpretation. Multimodal gain is not globally reliable. It is local, protocol-shaped, and query-dependent. Once that bounded-gain structure is made explicit, however, a lightweight router can exploit it through selective activation.

The main message of this section is therefore:

> The experiments do not show that multimodal fusion is universally stronger than structural modeling. They show that under the current OpenBG-IMG protocol, multimodal gain can be used more effectively when activation is made selective rather than always-on.
