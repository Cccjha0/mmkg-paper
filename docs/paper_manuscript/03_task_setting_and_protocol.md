# Task Setting and Method

## 1. Task Setting and Protocol

We study multimodal knowledge graph completion (MMKGC) on OpenBG-IMG under a standard link prediction setting. Each fact is represented as a triple `(h, r, t)`, where `h` and `t` denote head and tail entities and `r` denotes the relation. Given a partially observed query, the task is to rank candidate entities for the missing position. Tail prediction ranks candidate tails for `(h, r, ?)`, whereas head prediction ranks candidate heads for `(?, r, t)`.

This setting differs from structure-only KGC because entities may additionally have associated text and image information. In principle, these auxiliary modalities can enrich entity representations and improve ranking quality. In practice, however, modality availability is not uniform across entities or query conditions. As a result, the present paper is not only about multimodal representation learning in the abstract. It is also about how the evaluation protocol shapes where multimodal evidence can plausibly matter and how selectively such evidence should be activated.

All experiments in this paper use the current OpenBG-IMG `paper_split`, which is the unified split employed throughout the project for official model comparison, subgroup analysis, relation-group analysis, query-level routing evaluation, and candidate-level routing evaluation. We intentionally do not redefine the benchmark split for the final paper-stage study, because the purpose of this work is precisely to analyze and operationalize the empirical consequences of the current protocol rather than to replace it with a new one.

The most important property of the current `paper_split` is the asymmetry between target position and image availability. On the head side, target entities can still be image-available for a substantial portion of test queries. On the tail side, however, target entities are effectively always `no_img` under the current test distribution. This means that target position is not merely a formal direction choice. Under the present protocol, it is entangled with modality availability.

This asymmetry is central to the paper. It explains why multimodal gain can appear clearly in some local regimes, especially when the prediction target is image-supported, while overall performance can still remain dominated by structure-favorable conditions. It also explains why query-level clean routing is difficult: the deployable decision boundary is not formed in a neutral symmetric space, but inside a protocol where head-side and tail-side queries operate under systematically different gain regimes.

All base models are trained under a unified train/dev/test workflow. The development split is used for early stopping and checkpoint selection, and all final paper-facing metrics are reported on the test split. Query-level router training follows the same separation principle at the feature-label level: gain labels and relation priors are constructed from development-side expert outcomes only, and the trained router is then applied to test-side query features without using test labels during training. Candidate-level router training is also separated from final reporting: candidate-level training uses development-side candidate sets and labels, while final reported candidate-router results are computed on the test split under filtered ranking.

All formal comparisons use filtered ranking metrics on the test set, including MRR, Hits@1, Hits@3, and Hits@10. All main paper-stage comparisons use `direction=both`, so both head prediction and tail prediction are included in evaluation. Because the current `paper_split` is asymmetric, subgroup analysis must also be defined in a protocol-aware way. Rather than assuming that both directions contain both image-supported and image-missing targets, the present paper analyzes the only three meaningful target-side regimes under the current test distribution:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define `tail_has_img` for the final analysis because it does not meaningfully exist under the current test distribution. This detail is important not only for the bounded-gain diagnosis, but also for the method contribution: the router is not attempting to solve an abstract symmetric multimodal KGC problem. It is attempting to make selective activation decisions under this specific three-regime protocol.

The present paper relies on four complementary evaluation lines:

1. **Official model-comparison line.** This line aggregates completed paper-stage runs through formal `test_metrics.json` outputs and is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`.
2. **Strict query-level clean line.** This line uses query-level exported expert outcomes with recomputed aggregation under metadata-only legal query-time constraints. It is used for the clean rule baseline, naive global-threshold clean routing, direction-specific thresholding, and regression-based clean gain prediction.
3. **Score-aware candidate-level line.** This line evaluates candidate-level routers that may use non-answer-aware expert score, confidence, and disagreement features after the fixed experts have scored candidates. These routers are deployable after expert scoring, but they are not strict metadata-only clean routers.
4. **Oracle / post-hoc line.** This line is used only for upper-bound analysis. It may use answer-aware expert outcomes and is not deployable.

Rows from these lines answer different questions and must not be mixed as if they shared the same information conditions. In particular, the strongest strict query-level clean result and the strongest score-aware candidate-level result should both be reported, but they should be described as belonging to different deployability settings.

## 2. Fixed Expert Pair

Both routing stages use the same pair of fixed experts:

- **Fusion expert:** `Gate-only`
- **Structural expert:** `Residual-only`

For a query `q` and candidate entity `e`, let `s_f(q,e)` denote the score from the fusion expert and `s_s(q,e)` denote the score from the structural expert.

This expert pair is deliberately interpretable. `Gate-only` represents relation-aware multimodal fusion without an explicit residual compensation branch. `Residual-only` represents a structure-heavy fallback within the same internal model family. The purpose of routing is therefore not to combine arbitrary black-box systems, but to decide when the multimodal fusion path should affect ranking relative to a strong structural fallback.

The paper studies this expert-selection problem in two stages. Stage 1 asks how far strict metadata-only query-level routing can go. Stage 2 asks whether more fine-grained candidate-level score-aware routing can recover additional deployable gain without using answer-aware information.

## 3. Stage 1: Strict Query-level Clean Selective Activation

Stage 1 formulates selective activation at the query level. The router makes one decision for the whole query: use the fusion expert for all candidates, or fall back to the structural expert for all candidates.

### 3.1 Clean legality constraint

A strict query-level clean router may use only information that is available before the missing entity is known. In particular, it may use:

- the query direction (`head` or `tail` prediction),
- relation identity and development-derived relation priors,
- query-observable modality indicators derived from the observed side of the query,
- and router outputs computed from these legal inputs.

It may **not** use:

- hidden target-side information,
- target-side image-availability labels,
- target-aware regimes,
- correct-target scores,
- reciprocal ranks computed on the test target,
- or any other answer-aware signal that requires knowing the true missing entity.

This distinction is central to the paper. The query-level clean line measures what can be recovered from metadata-only deployable signals. It is intentionally stricter than the score-aware candidate-level line introduced later.

### 3.2 Naive global-threshold clean routing

The simplest query-level clean formulation uses a learned probability `p(q)` and a single global threshold `\tau`. Let `x_q` denote a strict clean feature vector available at query time. The router predicts

\[
p(q)=P(y=1\mid x_q),
\]

where `y=1` means that the fusion expert should be selected. At inference time,

\[
\alpha(q)=\mathbf{1}[p(q)>\tau].
\]

The final query-level score for every candidate is then

\[
s_{\text{final}}(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e).
\]

This formulation is a minimal deployable baseline. It intentionally tests whether one global clean decision boundary is sufficient under the asymmetric OpenBG-IMG protocol.

### 3.3 Direction-specific structured thresholding

Because the current protocol makes head-side and tail-side queries operate under different gain regimes, one global threshold can be too coarse. We therefore evaluate direction-specific thresholding, which keeps the router clean but gives head-prediction and tail-prediction queries different operating points:

\[
\alpha(q)=
\begin{cases}
\mathbf{1}[p(q)>\tau_{head}], & q \text{ is a head-prediction query}, \\
\mathbf{1}[p(q)>\tau_{tail}], & q \text{ is a tail-prediction query}.
\end{cases}
\]

This policy remains strict clean because query direction is known at inference time. It tests whether the original clean router fails partly because its policy boundary is too coarse rather than because legal query-time signals are entirely absent.

### 3.4 Target-aligned query-level supervision

The original query-level gain-threshold formulation uses a coarse binary label. For each development query `q`, let `rank_f(q)` and `rank_s(q)` be the filtered ranks of the correct target entity under the fusion and structural experts, respectively. Their reciprocal ranks are

\[
RR_f(q)=\frac{1}{rank_f(q)}, \qquad RR_s(q)=\frac{1}{rank_s(q)}.
\]

The expert gain is

\[
\Delta(q)=RR_f(q)-RR_s(q).
\]

The binary gain label is

\[
y(q)=\mathbf{1}[\Delta(q)>\delta],
\]

where `\delta` is a development-selected gain margin. This label is useful, but it discards the magnitude of the expert difference.

To preserve more target-aligned information during training, we also evaluate regression-based gain prediction:

\[
\widehat{\Delta}(q)\approx \Delta(q).
\]

At test time, the predicted gain is converted into a query-level selection decision through a development-selected threshold `\theta`:

\[
\alpha_{reg}(q)=\mathbf{1}[\widehat{\Delta}(q)>\theta].
\]

The regression router uses the same strict clean feature vector as the binary router. It differs by training on a scalar gain target rather than a hard binary label. In the revised paper, this regression router is treated as the strongest strict query-level clean baseline, not as the final method endpoint.

### 3.5 Strict clean feature families

The strict query-level clean line uses only legal query-time features, including:

- `direction`,
- `relation_id`,
- development-derived relation priors such as `relation_gain_prior`, `relation_fusion_win_rate`, `relation_support`, and `relation_is_visual_prior`,
- observed-side modality indicators such as `observed_has_img`, `observed_text_img_cosine`, and `observed_img_missing_replaced`.

All relation-level priors are estimated on the development split and then attached to test queries by relation identity. Observed-side modality features are computed only from the known entity in the query: for `(h,r,?)`, the observed side is `h`; for `(?,r,t)`, the observed side is `t`. The strict clean router never uses hidden target identity or target-side modality labels.

## 4. Stage 2: Score-aware Candidate-level Selective Activation

Stage 1 is intentionally strict, but it also exposes a limitation: metadata-only query-level signals recover only modest gain. Stage 2 therefore moves selective activation from the query level to the candidate level. Instead of selecting one expert for the whole query, the router predicts a soft fusion weight for each query-candidate pair.

The candidate-level router is defined as

\[
\alpha(q,e)=\sigma(g_\theta(x_q,x_e,z_{q,e})),
\]

where:

- `x_q` denotes query-level legal features,
- `x_e` denotes candidate-level metadata features,
- `z_{q,e}` denotes non-answer-aware score-aware features derived from the fixed experts,
- and `\sigma` maps the router output to a soft weight in `[0,1]`.

The mixed candidate score is

\[
s_{\text{mix}}(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e).
\]

This formulation is more fine-grained than query-level routing. It allows the model to keep the structural expert dominant for most candidates while applying fusion-based correction only where the expert score patterns suggest that fusion is useful.

### 4.1 Candidate-router variants

We evaluate three candidate-level router variants:

| Variant | Feature type | Purpose |
|---|---|---|
| `CA-S1` | clean candidate metadata | Tests whether static candidate modality metadata is sufficient. |
| `CA-S2` | score-aware expert features | Tests whether expert score, confidence, and disagreement signals provide deployable routing evidence. |
| `CA-S3` | clean candidate metadata + score-aware features | Tests whether static clean metadata adds reliable gain on top of score-aware signals. |

The feature distinction is important. `CA-S1` uses static candidate metadata and therefore asks whether candidate-level modality information alone is enough. `CA-S2` uses score-aware expert evidence and is the main score-aware candidate-level router. `CA-S3` combines both feature families.

### 4.2 Legality boundary of score-aware routing

`CA-S2` and `CA-S3` are **not** strict metadata-only clean routers. They are better described as **score-aware deployable candidate-level routers**. They may use expert scores, score margins, confidence-like statistics, and disagreement patterns computed after the fixed experts have scored candidates. These signals are available at inference time if both experts are run.

However, they remain non-answer-aware. The candidate-level router may not use:

- the hidden target label as an input feature,
- reciprocal rank of the correct target,
- target-side image-availability labels as answer-aware labels,
- Oracle expert choice,
- or any post-hoc information requiring knowledge of which candidate is correct.

This boundary should be stated explicitly in the paper. The key claim is not that `CA-S2` is a strict clean router. The correct claim is that `CA-S2` is a deployable score-aware router that uses non-answer-aware expert score patterns to recover more of the Oracle headroom than strict query-level clean routing.

### 4.3 Candidate-level pairwise training objective

The candidate-level routers are trained with a pairwise ranking objective. For a query `q`, a positive candidate `e^+`, and a negative candidate `e^-`, the loss is

\[
\mathcal{L}_{pair}=-\log \sigma\left(s_{\text{mix}}(q,e^+)-s_{\text{mix}}(q,e^-)\right).
\]

This objective directly trains the router to assign mixed scores that rank the true candidate above negative candidates. The fixed expert scores remain the ingredients of the mixture; the candidate router learns the candidate-level mixing weight `\alpha(q,e)`.

For efficiency, candidate-router training uses top-K candidate sets rather than full-entity scoring at every training step. Negative candidates are sampled from a mixture of:

- hard negatives from the `Gate-only` top-K list,
- hard negatives from the `Residual-only` top-K list,
- and random negatives drawn from the union candidate pool.

Final reporting is stricter than the training approximation: the reported `CA-S1`, `CA-S2`, and `CA-S3` test metrics are computed under full filtered ranking over all candidate entities, not only over the top-K training candidate sets.

## 5. Evaluation and Bootstrap Testing

The revised method requires careful separation of evaluation lines.

| Evaluation line | Used for | Deployability |
|---|---|---|
| Official model-comparison line | Original base models | Standard model evaluation |
| Strict query-level clean line | Clean rule, global threshold, direction-specific thresholding, regression clean router | Metadata-only deployable |
| Score-aware candidate-level line | `CA-S1`, `CA-S2`, `CA-S3` | Deployable after expert scoring; non-answer-aware but not strict metadata-only clean |
| Oracle / post-hoc line | Oracle routing and answer-aware upper bounds | Not deployable |

Bootstrap significance is computed with paired query-level resampling. For each compared pair of policies, reciprocal-rank differences are computed on the same test query instances, then resampled with replacement to obtain an empirical distribution of delta-MRR. This paired design is necessary because routing methods are evaluated on the same query set, and the confidence interval should measure matched improvement rather than independent aggregate variation.

The method-level takeaway is therefore:

> Under protocol-shaped incomplete visual support, strict query-level clean routing is a useful but limited first stage. The stronger method contribution is score-aware candidate-level selective activation, which remains non-answer-aware but uses expert score patterns to make sparse candidate-level corrections beyond what metadata-only query-level routing can recover.
