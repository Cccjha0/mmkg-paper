# OpenBG-IMG DynaSemble reproduction protocol

## Scope

This is a faithful external-baseline adaptation of DynaSemble to the already
frozen OpenBG-IMG M-Hyper + NativE expert pair. It does not alter either expert
or any method in this repository. DynaSemble is evaluated as published: a
query-dependent score-distribution ensemble trained only on validation data.

Source provenance:

- Paper: Nandi et al., *DynaSemble: Dynamic Ensembling of Textual and
  Structure-Based Models for Knowledge Graph Completion*, ACL 2024.
- Repository: `dair-iitd/KGC-Ensemble`.
- Source commit: `48d66b915f64899798f736129fa8c4d0a40fdb78`.
- Audited files: `NBFNet/script/selector.py`,
  `NBFNet/script/train_selector.py`, and the two-model YAML configurations.

## Frozen method configuration

For every query, each expert is scored over the complete filtered entity
candidate set and min-max normalized separately. The released code uses the
two features `1 - mean(normalized_scores)` and unbiased score variance for
each expert. The four concatenated features are passed through the released
selector topology `Linear(4,16) -> Linear(16,16) -> ReLU -> Linear(16,1) ->
ReLU`.

As in the released two-model experiments, one expert has fixed weight 1 and
the other receives the learned non-negative weight. NativE is the fixed-weight
expert and M-Hyper receives the learned weight. This parameterization remains
expressive in the relative weight ratio and is fixed before evaluation.

The following values come from the paper and released two-model code and are
not searched on OpenBG-IMG:

- hidden dimension: 16;
- Adam learning rate: `5e-5`;
- margin: 2.0;
- strict negatives: 9,999 plus the gold entity;
- training batch size: 16;
- epochs: 1;
- second hidden linear weight initialization: uniform `[0,2]`; all other
  parameters retain the released code's PyTorch-default initialization;
- selector random seed: the paired expert seed (1, 2, or 3).

The released training direction behavior is preserved: DEV triples are
deterministically shuffled, the first half of each batch is trained as tail
queries, and the second half as head queries. Thus each DEV triple contributes
one directional selector-training observation in the single epoch.

## Information and locking boundary

Each of the three paired expert seeds receives its own DynaSemble selector.
Only the 5,000 DEV triples and strict negative candidates are used for selector
optimization. There is no hyperparameter grid, early stopping, or TEST-driven
choice. After DEV, `dev_lock.json` records the complete method configuration,
expert run manifest, baseline-selection hash, selector hashes, and training
summaries.

TEST refuses to run without this lock and refuses selectors, expert runs, or a
baseline selection whose hashes or manifests differ. Rerunning DEV is also
refused once a TEST summary exists. Existing TEST results must not be used to
change any DynaSemble option.

The evaluator preserves the exact fixed-expert endpoint: whenever the released
selector's final ReLU produces a zero M-Hyper weight, the ensemble rank is set
to the already-audited NativE rank. This prevents single-precision min-max
rounding from changing a mathematically identical endpoint by one rank.

The true-fact filtering universe and both-direction full-entity scoring are
identical to the existing heterogeneous-complementarity evaluator. Training
uses strict negatives from that frozen filtered candidate universe. Evaluation
uses exact filtered full ranking, never top-k approximation.

## Matched comparison

The report includes exactly the following main rows:

1. M-Hyper;
2. fixed Query-zscore 0.5;
3. DEV-locked Global alpha;
4. DynaSemble;
5. answer-aware Oracle.

For DEV and TEST it reports pooled MRR/Hits, each of the three expert seeds,
head/tail directions, and 95% normal intervals over original-triple cluster
means. Each cluster contains both directions and all three expert seeds. The
script also reproduces the pre-existing matched query rows and aborts on any
rank mismatch or reciprocal-rank error larger than `5e-7`.

DEV DynaSemble metrics are in-sample selector-training diagnostics. Only the
immutable TEST execution is external generalization evidence.
