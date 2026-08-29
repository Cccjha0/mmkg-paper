# Recent Baselines: Formal Experimental Protocol

## Status and scope

This document is the binding protocol for adding recent baseline models to the
OpenBG-IMG comparison.  It applies to every newly added baseline, including
MoMoK, APKGC, NativE, M-Hyper, and any later model.  A result that does not satisfy this
protocol must be labelled exploratory and must not be used as a formal
comparison result.

The purpose is to compare methods under the same data, inputs, and evaluation
contract while allowing each method a reasonable, documented configuration.

## Fixed data and input contract

All formal runs must use the following immutable inputs:

| Item | Required value |
| --- | --- |
| Dataset | OpenBG-IMG `paper_split` |
| Train / dev / test triples | 220,087 / 5,000 / 10,000 |
| Train split | `data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_train.tsv` |
| Dev split | `data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_dev.tsv` |
| Test split | `data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_test.tsv` |
| Text features | `data/cache/openbg_img/text_feat_raw.pt` |
| Image features | `data/cache/openbg_img/img_feat_raw.pt` |
| Image-availability indicator | `data/cache/openbg_img/has_img.pt` |

The raw feature caches are the only permitted text and image feature sources.
Their construction, ordering, values, and entity alignment must not be changed
per model.  Model-specific trainable projections or encoders on top of these
fixed inputs are allowed when they are part of the model implementation.

For every entity without an image, `img_feat_raw.pt` contains the zero raw
embedding and `has_img.pt` is the authoritative missing-image indicator.  No
adapter may rewrite the shared cache, silently drop the entity, or derive
availability from another source.  A released model's own internal missing-
modality mechanism (for example APKGC's Gaussian replacement) is permitted
only when it is driven by `has_img.pt` and does not mutate the shared raw cache;
otherwise the model receives the same zero embedding.  The adapter's exact
handling of these two inputs must be recorded for each baseline.

No additional image collection, text corpus, pretrained feature cache, or
transductive feature fitting on dev/test entities is permitted unless it is
applied identically to every compared model and this protocol is formally
amended before running the experiment.

## Fixed evaluation contract

All formal dev and test measurements use the repository's current filtered,
bidirectional ranking evaluator without modification:

- filtered tail prediction: `(h, r, ?)`;
- filtered head prediction: `(?, r, t)`;
- both directions are evaluated for every triple;
- the reported overall value for each metric is
  `0.5 * head + 0.5 * tail`.

Filtering uses the established set of labelled true facts from train, dev, and
test, while retaining the target fact as a candidate.  Candidates are all
entities in the shared entity vocabulary.  Tie handling remains the existing
strict-greater rule: `rank = 1 + number of candidates with score greater than
the target score`.

The only primary metrics are MRR, Hits@1, Hits@3, and Hits@10.  Directional or
image-availability subgroup metrics may be reported as diagnostics, but they
do not replace the four overall metrics and may not be used to select a model
unless the selection rule is amended in advance.

## Model selection and tuning

All model-selection decisions are **DEV ONLY**.  This includes hyperparameter
choice, training duration or early stopping, checkpoint selection, loss
variants, fusion choices, and any model-specific tuning.  The test split may
be evaluated only after these choices are frozen; it must never guide a retry,
configuration change, checkpoint choice, or selection among variants.

The primary selection objective is dev overall MRR under the fixed evaluation
contract.  Every candidate configuration is run with formal seeds `1`, `2`,
and `3`; configurations are ranked by the arithmetic mean of their three
selected-checkpoint dev MRR values.  If that mean is exactly tied, select the
configuration with higher mean dev Hits@10; if it remains tied, select the
configuration with the smaller declared training budget.  This rule prevents a
configuration from being selected because of one lucky seed and avoids
discretionary selection.

Each baseline begins from its original paper's reported or released default
configuration where it is compatible with this dataset and input contract.
It may then use a finite, predeclared dev search over relevant hyperparameters,
for example:

- learning rate;
- batch size;
- hidden or embedding dimension;
- negative-sampling ratio;
- adversarial weight or temperature;
- other optimizer/loss settings required by the original method.

The shared configuration (`lr=0.001`, `batch_size=4096`, `neg_ratio=10`,
`epochs=200`) is a repository default and a reference point, not a mandatory
configuration for every recent baseline.  Forcing it on models such as MoMoK,
APKGC, or NativE would not be a fair comparison.  Fairness means fixed data,
fixed cache inputs, fixed evaluation, and dev-only selection—not identical
optimization hyperparameters.

Before a search starts, record the candidate configurations, their source
(paper default, released code default, or stated rationale), search budget,
and the dev selection objective.  The selected configuration must be chosen
from that record.  Any expansion of the search after seeing results must be
recorded as a new dev-only search stage; test results remain unavailable until
the revised configuration is frozen.

## Seeds, reporting, and audit record

For the selected model/configuration, use the already-trained three formal
seeds `1`, `2`, and `3`.  Each seed selects its checkpoint on dev only, then is
evaluated once on test.  Report test MRR, Hits@1, Hits@3, and Hits@10 as the
arithmetic mean and sample standard deviation across the three seed-level test
values: `mean ± std` (with `ddof=1`).

For every reported baseline, retain a machine-readable or tabular audit record
containing:

- model name, paper/code version or commit, and any integration changes;
- exact data split and cache paths (plus checksums when available);
- explicit missing-image / `has_img.pt` handling;
- declared dev search space, trial results, and selected configuration;
- seed, random-state controls, selected dev checkpoint, and its dev metrics;
- per-seed test metrics and the final `mean ± std` summary;
- hardware/software environment and the command or configuration used.

Do not average checkpoints, ensemble configurations, choose a lucky seed, or
reuse test outcomes for a new decision.  Any exception requires an explicit
amendment to this document before the affected formal experiment is run.

## Compliance check

Before publishing a result, confirm all of the following:

1. The run used the exact `paper_split` files and the three fixed caches.
2. Missing images remained zero in the shared raw cache; any official internal
   replacement used the shared indicator and was documented.
3. Both filtered head and filtered tail rankings were computed by the current
   evaluator, with equal-weight aggregation.
4. All selection and tuning decisions used dev results only.
5. The frozen configuration has three independent seed runs and reports
   `mean ± sample std` on test.
