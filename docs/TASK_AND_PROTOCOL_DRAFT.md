# Task Setting and Experimental Protocol Draft

## 1. Task Setting

We study multimodal knowledge graph completion (MMKGC) on OpenBG-IMG under a standard link prediction setting. Each fact is represented as a triple \((h, r, t)\), where \(h\) and \(t\) denote head and tail entities, and \(r\) denotes the relation. The task is to rank candidate entities for a missing head or missing tail given the remaining elements of the triple. Following common KGC practice, we evaluate both head prediction and tail prediction rather than restricting evaluation to a single direction.

The key characteristic of this setting is that entities may have associated textual and visual information in addition to graph structure. This makes OpenBG-IMG a suitable testbed for studying when multimodal information can improve completion quality. At the same time, it also introduces a realistic challenge: multimodal availability is not uniform across entities or prediction conditions, so the usefulness of images cannot be assumed to be global or symmetric.

## 2. Dataset and Split

All experiments are conducted on the current OpenBG-IMG `paper_split`, which is the unified split used throughout the project for model comparison, subgroup analysis, behavior analysis, and case study. We do not resplit the dataset for the final paper-stage experiments, because one goal of this work is precisely to analyze the empirical consequences of the current protocol rather than define a new benchmark setting.

An important property of the current `paper_split` is its asymmetry between target position and image availability. On the head side, target entities can still be image-available for a substantial portion of test queries. On the tail side, however, target entities are effectively always `no_img` under the current test distribution. This means that target position is not only a query-direction choice; it is also entangled with modality availability. As a result, head-side and tail-side evaluation should not be interpreted as fully symmetric evidence about multimodal usefulness.

This asymmetry is central to the paper. It explains why multimodal gain can appear clearly in some local regimes, especially when the prediction target is image-supported, while overall results can still remain dominated by structure-heavy conditions. We therefore treat the `paper_split` not as a neutral background detail, but as an essential part of the protocol that shapes the gain-boundary conclusions reported later in the paper.

## 3. Compared Models

Under this unified protocol, we compare seven models:

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`
- `ComplEx`
- `TuckER`

The first five form the internal model family used to analyze multimodal fusion, residual compensation, and their interaction. `ComplEx` and `TuckER` are included as structural reference baselines under the same training and evaluation pipeline. In the final analysis, `ComplEx` emerges as the competitive structural baseline under the current protocol, while `TuckER` serves as a classical structural reference model but does not emerge as competitive in the resulting ranking.

## 4. Training Protocol

All models are trained under a unified train/dev/test workflow. The development set is used only for model selection and early stopping, while all final metrics reported in the paper come from the test set. This separation is important because one of the goals of the project is to avoid mixing exploratory dev-set observations with final paper claims.

For each model family, we report results over at least three random seeds. The final paper-facing numbers are aggregated as `mean ± std` across seeds. This choice is intended to reduce sensitivity to isolated runs and make all comparisons consistent across structural and multimodal variants.

The multimodal models use the current raw-cache-based input pipeline, in which text and image raw feature caches are loaded and then projected through trainable layers such as `text_proj` and `img_proj`. Older cache formats are retained only for compatibility and are not used as the main evidence base for final paper claims.

## 5. Model Selection and Checkpoints

For each run, the final reported checkpoint is the `best.ckpt` selected according to development performance. We do not report the final training epoch by default, since the goal is to compare models under a consistent selection rule rather than by arbitrary terminal checkpoints. This is especially important for the current study because branch preference, gate behavior, and residual contribution all evolve during training; using a unified `best.ckpt` selection rule prevents the analysis from depending on inconsistent checkpoint choices.

The same selection logic is also reused in the downstream analyses. Subgroup evaluation, relation-type evaluation, behavior summaries, and case extraction all trace back to the same paper-stage runs and the same checkpoint-selection principle. This ensures that later analytical sections are grounded in the same model instances as the main result table.

## 6. Evaluation Protocol

We evaluate all models with filtered ranking metrics on the test set. The reported metrics include:

- MRR
- Hits@1
- Hits@3
- Hits@10

All final paper-stage comparisons use `direction=both`, meaning that both head prediction and tail prediction are included in evaluation. This choice is stricter than earlier tail-focused observations and is one reason why the final global ranking differs from some early exploratory expectations. In particular, the `both` protocol exposes the difficulty of head-side prediction more directly and makes the asymmetry of the current split more consequential for interpretation.

In addition to overall metrics, later sections of the paper also report protocol-consistent subgroup and relation-level evaluations. These are not separate ad hoc experiments: they reuse the same test split, the same filtered ranking setup, and the same selected checkpoints, but restrict aggregation to specific subsets of queries or relations. This consistency is important because it allows the paper to interpret gain boundary as a property of the current evaluation protocol rather than as an artifact of incompatible analysis settings.

## 7. Protocol-Aware Subgroup Structure

Because the current `paper_split` is asymmetric, our subgroup analysis is defined in a protocol-aware way. Rather than assuming both directions contain `has_img` and `no_img` targets, we analyze:

- `head_has_img`
- `head_no_img`
- `tail_no_img`

We do not define a `tail_has_img` subgroup for the final analysis because it does not meaningfully exist under the current test distribution. This detail must be made explicit in the paper, since otherwise the reader may incorrectly assume that the multimodal gain analysis is direction-symmetric.

This protocol-aware subgrouping is also the reason why some later conclusions must be interpreted carefully. For example, when we say that multimodal gain is strongest in `head_has_img`, we do not mean that head prediction is always easier or that tail prediction is inherently structure-only in all datasets. We mean that, under the current OpenBG-IMG protocol, the most favorable multimodal regime coincides with head-side queries whose prediction target actually has image support.

## 8. Protocol-Aware Relation Analysis

The relation-type analysis is also conducted under the same protocol rather than under a separately constructed benchmark. We use a coarse grouping file that organizes relations into:

- `visual_relations`
- `weak_visual_relations`
- `ambiguous_material_relations`

All grouped and relation-level metrics are computed from the same selected paper-stage runs and the same filtered-ranking evaluation procedure. For relation-level interpretation, we additionally use a minimum-support filter so that relations with extremely small test support do not dominate the narrative. This makes the relation analysis better aligned with paper writing, where local wins should be interpreted only when relation support is large enough to be meaningful.

Importantly, the relation-type analysis is not intended to prove that visual relations as a whole are globally favorable to multimodal models. Instead, it is designed to test whether multimodal gain is relation-dependent and whether that dependence supports a global or only local advantage. Under the current protocol, the evidence supports the latter interpretation.

## 9. Why Protocol Details Matter for the Paper

The protocol described above is not a neutral implementation detail. It directly shapes the paper's main claim. Under a more symmetric or differently balanced split, the relative strength of multimodal fusion and structural compensation might differ. In the current setting, however, the combination of target-position asymmetry, image-availability asymmetry, and branch competition produces a specific empirical pattern:

- multimodal gain is visible but local
- structure-heavy compensation remains globally dominant

This is why the paper should be framed as a protocol-aware gain-boundary analysis rather than as a universal statement about MMKGC in general. The contribution is not that all multimodal models are bounded in the same way everywhere, but that under the current OpenBG-IMG protocol we can clearly identify the conditions under which multimodal information helps and the conditions under which stronger structural modeling remains dominant.

## 10. Section-Level Takeaway

The most important message of this section should be:

> The final conclusions of the paper are inseparable from the current protocol. The OpenBG-IMG `paper_split` creates a meaningful but asymmetric missing-visual setting, and the observed gain boundary must therefore be interpreted as a protocol-aware empirical finding rather than a universal law of MMKGC.
