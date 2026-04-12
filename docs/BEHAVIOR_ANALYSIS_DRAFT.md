# Behavior Analysis Draft

## 1. Motivation

The gain-boundary analysis shows where multimodal gain appears, but it does not yet explain why that gain remains bounded. In particular, the completed subgroup and relation-type results establish a clear empirical pattern: the `Full Model` consistently improves over `Gate-only`, yet still fails to surpass `Residual-only` globally. This means that the `Full Model` is not simply ineffective, but that its multimodal contribution is being shaped by an internal competition between fusion-based evidence and structural compensation.

To explain this mechanism, we analyze the behavior of the learned model from three complementary perspectives. First, we examine gate statistics to determine whether multimodal fusion actually changes with relation context. Second, we analyze residual contribution and its variation across subgroups to test whether structural compensation becomes stronger when image support is weak. Third, we combine branch-preference evidence to determine whether fusion and residual operate as complementary or competing paths. Together, these analyses explain why multimodal gain is real but does not become globally dominant under the current protocol.

## 2. Gate Behavior

### 2.1 Why gate analysis matters

In the current model family, the gate branch is the main mechanism for relation-aware multimodal fusion. If multimodal usefulness is truly context-dependent, then the gate should not behave like a fixed global scalar. Instead, it should vary across relation settings and potentially across different target conditions. Conversely, if gate values are nearly constant or unrelated to relation context, then the fusion path would not provide strong evidence of adaptive multimodal reasoning.

For this reason, we first analyze training-time gate summaries and then relation-aware gate behavior on actual test queries. The goal is not to prove that gate alone explains final performance, but to determine whether it learns a nontrivial contextual pattern.

### 2.2 Global gate statistics

The global behavior summaries show that both `Gate-only` and `Full Model` learn stable gate statistics, including overall mean and variance as well as separate means for image-supported and image-missing entities. At the best epoch, the overall gate mean of `Full Model` is lower than that of `Gate-only`, indicating that the complete model does not simply rely on stronger gating everywhere. In addition, both models consistently show `g_mean_img < g_mean_noimg`, which means that gate behavior already differs across image-available and image-missing conditions.

This is an important first observation. It shows that gate behavior is not random and that the learned model does respond differently under distinct modality conditions. However, global gate statistics alone are not sufficient for interpretation, because they still average over different relation types and query contexts.

### 2.3 Relation-aware gate variation

To address this limitation, we further analyze gate values on real test `(entity, relation)` pairs. The relation-aware summary shows that gate means differ across coarse relation groups, and that these differences are stable enough to be visible at both the group level and the per-relation level. In other words, gate does respond to relation context rather than behaving like a fixed fusion weight.

At the same time, the magnitude of this variation remains limited in explanatory power. Although gate values do differ across relation groups, the observed variation does not line up strongly enough with the final performance ranking to explain the entire main-result pattern by itself. For example, even when relation-aware gate changes are clearly present, `Residual-only` still remains stronger than `Full Model` at the grouped level.

### 2.4 Gate-level interpretation

The gate analysis therefore supports a deliberately moderate conclusion: gate is genuinely adaptive, but it is not the dominant force behind the final ranking. The most accurate summary is:

- `gate is relation-aware, but not relation-dominant`

This conclusion matters because it rules out two overly simple interpretations. The first false interpretation would be that the fusion path is essentially dead or constant. The second false interpretation would be that relation-aware gating alone should have been sufficient to make the `Full Model` globally strongest. The evidence supports neither. Gate behavior is meaningful and structured, but it is only one part of a larger branch interaction.

## 3. Residual Behavior

### 3.1 Why residual analysis matters

If gate alone cannot explain the final ranking, the next question is whether the residual path becomes especially important in the weak-visual regimes where multimodal usefulness declines. This question is directly tied to the earlier residual-dominance diagnosis, which already showed that the final mixed representation is persistently biased toward the residual branch. What remains to be explained is whether this residual preference becomes even stronger under image-limited conditions.

To answer this, we analyze both global residual statistics and relation-aware residual contribution on real test queries.

### 3.2 Global residual preference

The first result is that the `Full Model` learns a clear residual preference over training. By the best epoch, the average branch weights are already strongly skewed toward the residual side, with `mix_w_residual` substantially larger than `mix_w_fusion`. This shows that the final decision path is not balanced between the two branches. Even when fusion remains active, the overall branch preference is clearly residual-biased.

This observation is already important because it explains why simple global fusion success should not be expected. If the model systematically learns to trust residual compensation more than fused multimodal evidence, then the multimodal path may still help locally without becoming dominant in the final ranking.

### 3.3 Subgroup-aware residual strengthening

The second and more informative result is that residual contribution becomes stronger precisely when image support is weaker. In the relation-aware residual summaries, the effective residual norm on the head side is consistently higher for `head_noimg` than for `head_has_img` across all three coarse relation groups. The same pattern appears in the residual-to-fused ratio, which also increases under `head_noimg`.

This means that residual preference is not merely a global average effect. The model actively leans more heavily on structural compensation when the target-side visual condition becomes weaker. In other words, the residual path is not only globally preferred; it is also conditionally reinforced in exactly the regimes where multimodal gain is hardest to sustain.

### 3.4 Residual-level interpretation

This behavior gives the mechanism-level explanation for an important part of the gain boundary. When image support is strong enough, the fusion path can contribute useful ranking signal. But when image support weakens, the model increasingly relies on structural compensation instead. Therefore, the residual branch should be understood not only as a globally strong path, but also as the preferred fallback under weak-visual conditions.

This is why the main paper should not describe `Residual-only` as merely an ablation baseline. Under the current protocol, it captures a real and powerful structural preference of the task.

## 4. Fusion vs Residual

### 4.1 Why branch interaction is the key mechanism

The `Full Model` combines two meaningful paths rather than one active path and one dead path. The fusion branch is adaptive and relation-aware, while the residual branch is strong and increasingly preferred under weaker image support. The central mechanism question is therefore not whether one branch exists and the other does not, but how their interaction determines the final performance boundary.

### 4.2 Evidence that fusion is useful

Several completed analyses show that fusion is not dead. Gradients on the fusion and projection components remain nonzero during training. Relation-aware gate values vary across relation settings. Most importantly, the completed relation-type analysis shows that the `Full Model` outperforms `Gate-only` on most medium-support and high-support relations under the `MIN20` filter. These observations collectively show that fusion learns meaningful signal and contributes real local improvements.

This point is crucial because it prevents an overly negative interpretation of the `Full Model`. The fact that `Residual-only` wins globally does not mean that multimodal fusion failed to learn anything. On the contrary, the evidence shows that fusion is active, structured, and locally beneficial.

### 4.3 Evidence that residual dominates final preference

At the same time, the final branch preference remains clearly residual-dominant. Mix-weight trajectories show that residual preference strengthens over training, and relation-aware residual analysis shows that this preference becomes even stronger when image support is weak. Thus, the branch interaction is not symmetric. Fusion adds useful information, but the model does not trust it equally across conditions.

This asymmetry explains the core empirical pattern of the paper:

- `Full Model` improves over `Gate-only` because fusion contributes useful local signal
- `Full Model` still fails to surpass `Residual-only` because the final representation remains dominated by the structurally stronger residual path

### 4.4 Final behavior-level judgment

The most accurate description of the branch interaction is therefore:

- `residual-dominant asymmetric complementarity`

There is complementarity because fusion produces stable gains over simpler multimodal baselines. There is asymmetry because the final mixed representation and weak-visual fallback behavior remain biased toward residual compensation. This formulation captures both sides of the evidence and is more faithful than describing the two branches as either purely cooperative or purely redundant.

## 5. Why This Section Matters for the Paper

This section provides the mechanism layer of the paper. The gain-boundary analysis already showed that multimodal gain is local and bounded. The behavior analysis now explains why that boundary appears. The answer is not that fusion is ineffective, nor that the model is broken. Rather, the answer is that under the current protocol:

- fusion is real and adaptive
- residual is stronger and increasingly preferred
- weak image support further amplifies residual reliance

This is the internal explanation for why the `Full Model` can be locally helpful without becoming globally dominant.

## 6. Section-Level Takeaway

The most important message of this section should be:

> Under the current OpenBG-IMG protocol, the `Full Model` learns a meaningful multimodal fusion path, but that path operates inside a representation system whose final preference remains residual-dominant, especially when visual support is weak. This explains why multimodal gain is visible but bounded.
