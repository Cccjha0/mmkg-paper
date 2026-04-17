# Task Setting and Protocol

## 1. Task Setting

We study multimodal knowledge graph completion (MMKGC) on OpenBG-IMG under a standard link prediction setting. Each fact is represented as a triple `(h, r, t)`, where `h` and `t` denote head and tail entities, and `r` denotes the relation. Given a partially observed query, the task is to rank candidate entities for the missing position. Tail prediction ranks candidate tails for `(h, r, ?)`, while head prediction ranks candidate heads for `(?, r, t)`.

This setting differs from structure-only KGC because entities may additionally have associated text and image information. In principle, these auxiliary modalities can enrich entity representations and improve ranking quality. In practice, however, modality availability is not uniform across entities or query conditions. As a result, the present paper is not only about multimodal representation learning in the abstract. It is also about how the evaluation protocol shapes where multimodal evidence can plausibly matter.

## 2. Dataset and Paper Split

All experiments in this paper use the current OpenBG-IMG `paper_split`, which is the unified split employed throughout the project for official model comparison, subgroup analysis, relation-group analysis, behavior analysis, and routing evaluation. We intentionally do not redefine the benchmark split for the final paper-stage study, because the purpose of this work is precisely to analyze and operationalize the empirical consequences of the current protocol rather than to replace it with a new one.

The most important property of the current `paper_split` is the asymmetry between target position and image availability. On the head side, target entities can still be image-available for a substantial portion of test queries. On the tail side, however, target entities are effectively always `no_img` under the current test distribution. This means that target position is not merely a formal direction choice. Under the present protocol, it is entangled with modality availability.

This asymmetry is central to the paper. It explains why multimodal gain can appear clearly in some local regimes, especially when the prediction target is image-supported, while overall performance can still remain dominated by structure-favorable conditions. In the earlier analysis-oriented version of the paper, this asymmetry motivated a bounded-gain interpretation. In the present gain-threshold version, it also motivates the method itself: if multimodal gain depends on protocol-shaped query conditions, then multimodal activation should be treated as a selective decision rather than a globally uniform default.

## 3. Unified Training and Selection Protocol

All models are trained under a unified train/dev/test workflow. The development split is used only for model selection and early stopping, while all final paper-facing metrics are reported on the test split. This separation is important because the current paper moves from analysis to method. The router and all paper conclusions must therefore remain grounded in development-side supervision and test-side reporting, without mixing exploratory observations into final claims.

For each completed base model, the reported checkpoint is the `best.ckpt` selected according to development performance. We do not compare arbitrary terminal checkpoints. This choice is especially important in the current setting because branch preference, gate behavior, and residual contribution evolve during training; a consistent checkpoint-selection rule prevents later analysis from depending on inconsistent model states.

The same selected runs are reused across downstream analyses. Overall comparison, subgroup analysis, relation-group evaluation, behavior summaries, and case interpretation all trace back to the same paper-stage runs and the same checkpoint-selection principle. This consistency is essential because it allows the paper to interpret gain boundary and selective activation as properties of one coherent protocol rather than as artifacts of incompatible evaluation settings.

## 4. Evaluation Protocol

All formal comparisons use filtered ranking metrics on the test set. The reported metrics include:

- MRR
- Hits@1
- Hits@3
- Hits@10

All main paper-stage comparisons use `direction=both`, meaning that both head prediction and tail prediction are included in evaluation. This protocol is stricter than earlier tail-focused exploratory observations and is one reason why the final global ordering differs from some earlier expectations. In particular, the `both` protocol exposes head-side difficulty much more directly and makes the asymmetry of the current split far more consequential for interpretation.

In addition to overall metrics, later sections report protocol-consistent subgroup and relation-level evaluations. These are not separate ad hoc experiments. They reuse the same test split, the same filtered-ranking setup, and the same selected checkpoints, but restrict aggregation to specific subsets of queries or relations. This is important because it means the later analyses and the routing experiments all decompose the same underlying evaluation problem rather than inventing parallel ones.

## 5. Protocol-Aware Subgroup Structure

Because the current `paper_split` is asymmetric, subgroup analysis must also be defined in a protocol-aware way. Rather than assuming both directions contain both image-supported and image-missing targets, we analyze the only three meaningful target-side regimes under the present test distribution:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define `tail_has_img` for the final analysis because it does not meaningfully exist under the current test distribution. This detail is important not only for the old gain-boundary analysis, but also for the present routing method. The router is not attempting to solve an abstract symmetric multimodal KGC problem. It is attempting to make decisions under this specific three-regime protocol.

This subgroup structure also clarifies how later claims should be interpreted. When the paper states that multimodal gain is strongest in `head_has_img`, it does not mean that head prediction is universally easier or that tail prediction is inherently structure-only across all datasets. It means that under the current OpenBG-IMG protocol, the most favorable multimodal regime coincides with head-side queries whose target entities actually have image support.

## 6. Protocol-Aware Relation Analysis

The relation-type analysis is conducted under the same protocol rather than under a separately constructed benchmark. We use a coarse grouping file that organizes relations into:

- `visual_relations`
- `weak_visual_relations`
- `ambiguous_material_relations`

All grouped and relation-level metrics are computed from the same selected paper-stage runs and the same filtered-ranking procedure. For relation-level interpretation, we additionally apply a minimum-support filter so that extremely small-support relations do not dominate the narrative.

The purpose of this grouping is analytical rather than taxonomic. It is not intended to prove that visual relations as a whole are globally favorable to multimodal models. Instead, it is intended to test whether multimodal gain is relation-dependent and whether that dependence supports a global or only local advantage. Under the current protocol, the evidence supports the latter interpretation. In the present paper, this matters because relation dependence becomes one of the reasons why a learned query-level router can outperform a shallow global rule.

## 7. Official Results vs Routing Results

The present paper contains two complementary evaluation dialects.

The first is the **official model-comparison line**, which aggregates the completed paper-stage runs through their formal `test_metrics.json` outputs. This line is used for the seven-model comparison among `Text-only`, `Early Fusion`, `Gate-only`, `Residual-only`, `Full Model`, `ComplEx`, and `TuckER`.

The second is the **routing line**, which is based on query-level exported expert outcomes and recomputed aggregation. This line is necessary because Oracle routing, rule-based routing, and learned routing all operate on query-level expert predictions. To keep the routing comparison fair, fixed experts inside that comparison are also reported on the same query-level recomputed line.

These two lines answer different questions and must not be mixed in the same table. The official line establishes the ranking of the original models under the paper protocol. The routing line establishes whether selective activation improves over fixed experts once all rows are placed on a shared routing-compatible basis.

## 8. Why Protocol Details Matter for the New Paper

The protocol described above is not a neutral implementation detail. It is the foundation of both the paper's analysis and its method contribution. Under a more symmetric split or a differently balanced modality distribution, the relative strength of multimodal fusion, structural compensation, and routing could differ. In the current setting, however, the combination of target-position asymmetry, image-availability asymmetry, relation dependence, and branch competition produces a specific empirical pattern:

- multimodal gain is visible but local
- stronger structural compensation remains globally dominant
- multimodal usefulness is therefore query-dependent rather than globally reliable

This is exactly why the paper moves from gain-boundary analysis to gain-aware selective activation. The protocol does not merely explain the old tension; it creates the decision problem that the router is designed to solve.

## 9. Section-Level Takeaway

The main message of this section is:

> The conclusions of this paper are inseparable from the current protocol. The OpenBG-IMG `paper_split` creates a meaningful but asymmetric missing-visual setting, and this protocol not only reveals that multimodal gain is bounded, but also motivates the central method contribution of the paper: query-level selective activation through gain-threshold routing.
