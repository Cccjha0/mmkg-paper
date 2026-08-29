# M-Hyper OpenBG-IMG Adaptation

## Source anchor

The adapter follows the released `M_Hyper_B` path in
`external/M-Hyper/src/models.py` at source commit
`cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76`. The official README commands for
MKG-W, MKG-Y, and DB15K use the same formal anchor:

- rank: 128
- optimizer: Adagrad
- learning rate: 0.1
- batch size: 1000
- wN3 weight: 0.005
- epochs: 200
- validation interval: 5 epochs

The released loop does not early-stop; it completes 200 epochs and retains the
best validation checkpoint. The formal adapter therefore sets
`early_stop_patience: null` while continuing to select the checkpoint by Dev
MRR only.

The formal OpenBG configuration preserves these values. In particular,
`batch_size=1000` is used directly on the A100; the earlier conservative
resource suggestion of 128 is not used by the formal config or any reported
result.

## Preserved model path

The port preserves the released path actually used by `M_Hyper_B`:

- structural, visual, and textual independent embeddings plus the fused base;
- FERF projections and reconstruction terms;
- relation-aware modality fusion with relation-specific temperatures;
- sparse Gaussian modality noise with preserve ratio 0.2;
- the released complex product over the concatenated four-part representation;
- reciprocal-relation one-vs-all cross entropy;
- released wN3 factor weighting;
- auxiliary loss equal to consistency plus reconstruction loss.

The released forward computes three independence terms but comments them out
of the returned loss. The adapter does the same. It also deliberately preserves
the released reconstruction target's final `proj_no_stru` item. This is source
fidelity, not a correction invented for OpenBG.

## Protocol adaptations

The following adaptations are required by the locked OpenBG baseline protocol:

1. Fixed `text_feat_raw.pt` and `img_feat_raw.pt` are used instead of the
   source datasets' embeddings. Missing image rows remain the cache's zero raw
   vectors. `has_img.pt` is used only for evaluator subgroup reporting.
2. The source model fits PCA over the complete entity-feature matrix. To avoid
   Dev/Test entity information entering data-dependent initialization, this
   adapter fits image and text PCA only on entity IDs visible as a head or tail
   in the training split, then transforms all entity rows. The fitted entity
   IDs are recorded as `model.pca_fit_entity_ids` for auditing.
   Under `mmkg_general_v1`, each modality fit is further restricted to
   train-visible entities whose explicit modality mask is true; the two exact
   subsets are exposed as `pca_fit_image_entity_ids` and
   `pca_fit_text_entity_ids`. Missing projected and independent modality paths
   are masked, while the frozen OpenBG path remains unchanged.
3. Training triples are augmented with `(t, r + R, h)` only inside the
   one-vs-all training engine. The original 136 relations remain the evaluator
   interface; head candidates are scored through reciprocal relations.
4. The repository's unified evaluator is retained: strict higher-is-better
   ranking, identical filtered facts, filtered head and tail prediction, and
   their 0.5/0.5 aggregation. The source evaluator is not copied because its
   tie rule and released validation loader do not satisfy this protocol.
5. Clean entity representations are cached only during evaluation. Training
   invalidates the cache, and the evaluator rebuilds it after checkpoint loads.
   This changes computation reuse, not scores or candidate sets.

Formal model selection remains DEV ONLY. `evaluation.run_test` can be set to
`false` for configuration search; the formal frozen configuration sets it to
`true` for the final three-seed runs.
