# Main Results

## 1. Overall Comparison

Table X reports the final test results of all compared models under the unified OpenBG-IMG protocol. All numbers are aggregated over three seeds and reported as `mean +- std`. Following the protocol described in the previous section, model selection is based on `best.ckpt` from the development set, while all final metrics are computed on the test set using filtered ranking with `direction=both`.

The overall ranking is:

1. `Residual-only`
2. `ComplEx`
3. `Full Model`
4. `Gate-only`
5. `Early Fusion`
6. `Text-only`
7. `TuckER`

This ranking immediately establishes the core empirical tension of the paper. The most complete multimodal model is not the strongest overall model under the current protocol. Instead, the best global result is achieved by the structure-heavy `Residual-only` model, while the classical structural baseline `ComplEx` also remains stronger than the `Full Model`. At the same time, the `Full Model` is clearly stronger than the simpler multimodal baselines, which indicates that multimodal modeling is not useless, even though it does not dominate globally.

## 2. Main Quantitative Findings

The first important observation is that `Full Model` consistently improves over the weaker multimodal references. Compared with `Gate-only`, it obtains a clear MRR gain, and it also remains stronger than `Early Fusion` and `Text-only`. This shows that adding the residual compensation path to relation-aware multimodal fusion produces a real improvement over simpler multimodal formulations.

The second observation is that this improvement is still insufficient to overcome stronger structure-oriented competitors. `Residual-only` remains substantially ahead of `Full Model`, and `ComplEx` also outperforms it under the same protocol. Therefore, the empirical role of the `Full Model` is not that of a globally dominant architecture, but that of the strongest multimodal variant within the internal fusion-oriented family.

The third observation concerns `TuckER`. Although `TuckER` is a classical structural reference model, it does not emerge as a competitive structural baseline in the current ranking. This is important for interpretation. The main global competitors of the `Full Model` are `Residual-only` and `ComplEx`, not `TuckER`. Accordingly, later discussion should not treat all structural baselines as equally strong under the present protocol.

## 3. What the Main Results Already Tell Us

Even before any subgroup or mechanism analysis, the main table already supports two high-level conclusions.

First, multimodal modeling can produce meaningful gains relative to simpler fusion baselines. If multimodal information were entirely unhelpful under the current setting, `Full Model` would not consistently outperform `Gate-only` and `Early Fusion`. The fact that it does so means that fusion-based multimodal signal contributes something real.

Second, stronger structural compensation remains globally preferred under the current protocol. The superior performance of `Residual-only`, together with the competitiveness of `ComplEx`, shows that the current task is not naturally solved by adding modalities alone. Instead, structural regularity remains the dominant factor in the final global ranking.

Taken together, these two observations imply that the central research problem is not whether multimodal information can help at all, but under what conditions that help is strong enough to matter.

## 4. Head-Tail Asymmetry in the Main Table

The overall table also reveals a strong asymmetry between head-side and tail-side performance. Across nearly all models, `head_mrr` is much lower than `tail_mrr`. This means that the final `direction=both` evaluation is substantially stricter than earlier tail-focused impressions and that head prediction is the more difficult side of the task under the current protocol.

This pattern matters for two reasons. First, it confirms that the final global ranking cannot be interpreted only through tail-side behavior. Second, it suggests that overall performance is the result of two very different regimes being mixed together inside the same metric. This motivates the later subgroup analysis, where target position and image availability are studied explicitly rather than being left implicit inside the overall scores.

## 5. What the Main Results Do Not Yet Explain

Although the main result table is necessary, it is not sufficient to support the paper's final claim on its own. In particular, the table does not answer:

- where the `Full Model` gains over simpler multimodal models actually come from
- why these gains fail to overturn stronger structure-heavy competitors
- whether multimodal usefulness depends on image availability, relation type, or branch behavior

For this reason, the main results section should be written as a tension-establishing section rather than a conclusion section. Its job is to make the core empirical pattern explicit:

- `Full Model` is meaningfully better than weaker multimodal baselines
- but not strong enough to become globally dominant

The rest of the paper then explains why this pattern occurs.

## 6. Transition to the Analysis Sections

Based on the main table alone, a natural but incomplete interpretation would be that structural models simply dominate and multimodal modeling is not worth pursuing. Our later analysis shows that this interpretation is too coarse. The completed subgroup, relation-type, behavior, and case analyses reveal that multimodal gain is real, but it appears under specific boundary conditions rather than uniformly across the task.

Therefore, the correct role of the main results section is to introduce the central tension of the paper, not to resolve it. The next sections address this tension directly by asking:

- when does multimodal information help?
- why is that gain local rather than global?
- why does stronger structural compensation remain dominant under the current protocol?

## 7. Section-Level Takeaway

The most important message of this section should be:

> Under the unified OpenBG-IMG protocol, the `Full Model` is the strongest multimodal variant in the internal family, but it is not the strongest overall model. This establishes the key empirical tension of the paper: multimodal fusion helps, yet stronger structural compensation remains globally dominant.
