# MKG-W and DB15K experiment guide

This guide prepares both external datasets for the shared-input `mmkg_general_v1` protocol. It does not redefine any OpenBG-IMG experiment.

## Data contract

The raw HDF5 files stay unchanged. Canonical output is written below `data/datasets/<dataset>/processed/`:

```text
train.tsv                 # canonical head, relation, tail
valid.tsv
test.tsv
entity2id.json
relation2id.json
text_feat.pt              # [N, Dt], mean-pooled per HDF5 key
img_feat.pt               # [N, Dv], mean-pooled per HDF5 key
has_text.pt               # [N] bool, derived from key presence
has_img.pt                # [N] bool, derived from key presence
manifest.json
audit_report.json
```

An all-zero feature row never determines missingness. The masks come from explicit entity-to-HDF5-key alignment. The preprocessor rejects duplicate/non-contiguous mappings, invalid triple IDs, dirty canonical splits, missing crosswalks, inconsistent HDF5 dimensions, and any NaN/Inf value. Every raw input is checked against `docs/EXTERNAL_SOURCES_LOCK.json` before preprocessing.

## 1. Prepare MKG-W first

The official OpenKE split and mapping files already exist under `external/NATIVE/benchmarks/MKG-W`. MKG-W entities are Wikidata resources, while the supplied HDF5 keys are English Wikipedia titles, so a checked crosswalk is mandatory. Run these commands in the server environment (with `torch` and `h5py` installed):

```bash
mkdir -p data/datasets/mkg_w/raw data/datasets/mkg_w/processed

python ml/training/scripts/build_mkg_w_feature_key_map.py \
  --entity2id external/NATIVE/benchmarks/MKG-W/entity2id.txt \
  --feature-h5 data/datasets/MKG_W_img_BEIT_16-224.h5 \
  --output data/datasets/mkg_w/raw/feature_key_map.tsv

python ml/training/scripts/preprocess_external_mmkg.py \
  --dataset mkg_w \
  --benchmark-dir external/NATIVE/benchmarks/MKG-W \
  --text-h5 data/datasets/MKG_W_description_sentences.h5 \
  --image-h5 data/datasets/MKG_W_img_BEIT_16-224.h5 \
  --feature-key-map data/datasets/mkg_w/raw/feature_key_map.tsv \
  --cross-modal-space independent \
  --output-dir data/datasets/mkg_w/processed
```

Use `--audit-only` on the preprocessing command to inspect counts, dimensions, coverage, finite ranges and NaN/Inf counts without writing `.pt` tensors. Review both `feature_key_map.tsv.report.json` and `audit_report.json` before training. Cross-modal cosine is disabled by default; `shared` is legal only when the two feature files are explicitly documented as one aligned embedding space.

## 2. Prepare DB15K after MKG-W validation

DB15K text keys use DBpedia resource basenames. Image keys use Freebase MIDs, so the official MMKB SameAs file is required; ID order is never used as a fallback.

The pinned NativE mirror's DB15K `valid2id.txt` is not a legal validation split: its 9,904 rows contain 2,423 duplicates, 6,584 unique triples overlap TRAIN, and 824 unique triples overlap TEST. It remains source-locked for provenance and is audited, but it is never used for checkpoint selection. The versioned `db15k_train_holdout_v1` policy instead deterministically holds out 10% of the pinned TRAIN (seed 2025), stratifies by relation, preserves TRAIN coverage for every DEV entity/relation, and leaves the pinned TEST sequence unchanged.

```bash
mkdir -p data/datasets/db15k/raw data/datasets/db15k/processed
curl -L https://raw.githubusercontent.com/mniepert/mmkb/master/DB15K/DB15K_SameAsLink.txt \
  -o data/datasets/db15k/raw/DB15K_SameAsLink.txt

python ml/training/scripts/preprocess_external_mmkg.py \
  --dataset db15k \
  --benchmark-dir external/NATIVE/benchmarks/DB15K \
  --text-h5 data/datasets/MMKB_description_sentences.h5 \
  --image-h5 data/datasets/MMKB_img_BEIT_16-224.h5 \
  --db15k-same-as data/datasets/db15k/raw/DB15K_SameAsLink.txt \
  --split-policy db15k_train_holdout_v1 \
  --db15k-valid-fraction 0.10 \
  --split-seed 2025 \
  --cross-modal-space independent \
  --output-dir data/datasets/db15k/processed
```

The repaired canonical counts are TRAIN 71,300 / DEV 7,922 / TEST 9,902. The manifest records source and canonical counts, the source leakage audit, split construction policy/seed, source lock, mapping/split hashes, HDF5 hashes, dimensions, numeric health, pooling and modality coverage. The loader revalidates these artifacts and rejects a manifest whose split policy differs from the YAML config. All old DB15K checkpoints and `processed/` split artifacts must therefore be discarded and regenerated; MKG-W and OpenBG-IMG are unchanged.

## 3. Server training commands

Do not use the smoke configs as evidence. Run each seed with the corresponding common seed config. Examples for MKG-W:

MKG-W uses its complete clean official validation split. DB15K uses `db15k_train_holdout_v1` because the pinned mirrored validation file is contaminated. `termination_policy: dev_early_stop` and `termination_policy: fixed_budget` control only when training ends; best-checkpoint selection remains DEV ONLY in both cases.

```bash
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_gate_only.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_complex.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_residual_only.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_gate_residual.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_mhyper.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_adamf_mat.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_native.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_apkgc.yaml
```

Replace `mkg_w` with `db15k` after the MKG-W load/train/valid/filtered-test chain passes. The dataset-neutral structural alias is `mmkg_complex`.

Within each dataset, all configs use the same canonical splits, features, masks and evaluator. Hyperparameters remain architecture-specific and must be selected on validation only. The checked-in starting-anchor configs therefore set `evaluation.run_test: false`. After the configuration is frozen from validation results, change only that field to `true` for the final three-seed runs; final test evaluation is locked to the selected validation checkpoint.

Released-script anchors are dataset-specific and must not be copied across datasets:

| Model | MKG-W anchor | DB15K anchor |
| --- | --- | --- |
| NativE | `dim=250`, `margin=4`, batch 1024, 128 negatives, 1000 epochs | `dim=250`, `margin=12`, batch 1024, 128 negatives, 1000 epochs |
| AdaMF-MAT | `dim=200`, `margin=12`, batch 1024, 128 negatives, 1000 epochs | `dim=250`, `margin=12`, batch 1024, 128 negatives, 1000 epochs |
| APKGC | `num_proj=1`, `Mformer_hd_mean` | `num_proj=2`, `Mformer_hd_graph`; both use `dim=128`, batch 1024, 32 negatives and 8000 epochs |
| M-Hyper | `rank=128`, batch 1000, Adagrad 0.1, `wN3=0.005`, 200 epochs | Same released command anchor |

The canonical tensors remain unchanged across methods, while missing-modality handling stays architecture-specific and mask-driven:

| Model family | Missing text / image inside the model |
| --- | --- |
| Gate-only, Gate+Residual | learned missing vectors selected by `has_text` / `has_img` |
| Residual-only | learned entity residual path only; modality masks are evaluation diagnostics |
| NativE, AdaMF-MAT | absent projected modality is zero-masked before fusion |
| APKGC | Gaussian replacement computed from observed rows, following the released model |
| M-Hyper | absent projected and independent modality paths are zero-masked; modality PCA fits observed train-visible rows only |
| ComplEx | structural-only; masks are used only for evaluation diagnostics |
| General-v2 Fusion (`mmkg_gate_only_v2`) | explicit masked text/image weights; single available modality gets weight 1; T0V0 uses an explicit learned fallback; independent text/image dropout changes availability only during training |
| General-v2 Structural (`mmkg_structural_v2`) | pure learned entity structure plus ComplEx relation scoring; no text/image tensors or availability masks are constructor inputs or state-dict entries |

The unified evaluator ranks over every mapped entity, filters all other known positives from train+valid+test, evaluates head and tail prediction, uses the existing strict-`>` tie rule, and reports the equal-direction average. This is the same evaluator used by every method; external paper numbers are not mixed into the main comparison.

## 4. Query export, router and score-aware analysis

Export Gate-only, Residual-only and Ours query results from their server run directories with `scripts/export_query_eval.py`. General exports report the real relation name, both target masks, and one of `head_T1V1` … `tail_T0V0`. Target masks are for reporting only.

```bash
python scripts/export_query_eval.py --expert gate_only --run-dir <mkg_w_gate_run> --split dev \
  --out outputs/mkg_w/router/dev/gate_only_query_eval_seed1.csv \
  --summary-json outputs/mkg_w/router/dev/gate_only_query_eval_seed1_summary.json
python scripts/export_query_eval.py --expert residual_only --run-dir <mkg_w_residual_run> --split dev \
  --out outputs/mkg_w/router/dev/residual_only_query_eval_seed1.csv
# Repeat for test only after model/hyperparameter selection is locked on valid.
```

Build relation priors separately for each dataset from its validation exports. Never reuse OpenBG priors and never use test outcomes when estimating priors. Then build the query-time legal general router features:

```bash
python scripts/build_relation_priors.py \
  --gate-dir outputs/mkg_w/router/dev \
  --residual-dir outputs/mkg_w/router/dev \
  --out-dir outputs/mkg_w/router/priors \
  --split dev

for delta in 0.00 0.01 0.02; do
  python scripts/build_gain_labels.py \
    --delta "$delta" \
    --split dev \
    --gate-dir outputs/mkg_w/router/dev \
    --residual-dir outputs/mkg_w/router/dev \
    --out-dir outputs/mkg_w/router/dev
done

python scripts/build_router_features.py \
  --router-mode clean \
  --protocol-version mmkg_general_v1 \
  --processed-dir data/datasets/mkg_w/processed \
  --prior-csv outputs/mkg_w/router/priors/relation_gain_stats_gamma_0.000.csv \
  --gate-dev-dir outputs/mkg_w/router/dev \
  --residual-dev-dir outputs/mkg_w/router/dev \
  --label-dir outputs/mkg_w/router/dev \
  --gate-test-dir outputs/mkg_w/router/test \
  --residual-test-dir outputs/mkg_w/router/test \
  --out-dir outputs/mkg_w/router/features
```

The clean profile contains only direction, dataset-local validation priors and observed-entity modality features. Target availability is excluded to prevent answer-aware leakage. General score-aware features also exclude raw numeric relation IDs; alpha selection remains dataset-local on validation.

Train general clean routers with `--feature-set G1`, `G2`, or `G3`; frozen OpenBG experiments continue using `C1`–`C4`.

`G4` is an optional general-v2 context scaffold. It adds relation×direction DEV-prior fields plus TRAIN-only observed-degree/relation-frequency fields. Missing optional inputs are neutral zeros. Raw relation IDs, target masks, correct-target scores and outcome ranks remain forbidden. Do not enable G4 in a reported experiment until the dataset-local TRAIN/DEV statistic builders and their provenance have been locked.

For score-aware combination, first export dev and test candidate scores into a dataset-specific directory, then select alpha on dev and apply the locked selection to test:

```bash
python scripts/export_candidate_scores.py --gate-run-dir <mkg_w_gate_run> \
  --residual-run-dir <mkg_w_residual_run> --split dev --direction both \
  --top-k 100 \
  --out-csv outputs/mkg_w/candidate_router/scores/dev_seed1_top100.csv \
  --summary-json outputs/mkg_w/candidate_router/scores/dev_seed1_top100_summary.json

python scripts/eval_score_ensemble_baselines.py \
  --score-dir outputs/mkg_w/candidate_router/scores \
  --output-dir outputs/mkg_w/score_ensemble/eval \
  --selection-only \
  --score-normalization query_zscore \
  --relation-shrinkage-lambda 20
```

For `mmkg_general_v1`, this command validates that every dev/test summary belongs to one dataset and one `top_k`, reports deltas against that dataset's Residual-only scores, and keeps plots under the dataset-specific output directory. Summary discovery no longer assumes `top100`; the JSON `top_k` field is authoritative. It never reads or overwrites the frozen OpenBG E5/CA-S2 paper artifacts.

## 5. Generalized-v2 DEV-first funnel

The v2 aliases exist only under `mmkg_general_v1`. They do not replace `mmkg_gate_only`, `mmkg_residual_only`, or any `openbg_img_*` alias. All checked-in v2 configs keep `evaluation.run_test: false`.

One-seed fixed-expert commands (run on the training server, not a low-compute workstation):

```bash
# DB15K
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/db15k_complex.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/db15k_gate_only.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/db15k_gate_v2.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/db15k_structural_v2.yaml

# MKG-W
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_complex.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_gate_only.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_gate_v2.yaml
python -m ml.training.scripts.run_train --common ml/configs/common_seed1.yaml --config ml/configs/mkg_w_structural_v2.yaml
```

Export DEV query outcomes for each v2 pair and compute fixed-strategy complementarity:

```bash
python scripts/export_query_eval.py --expert fusion_v2 --run-dir <dataset_gate_v2_run> --split dev \
  --out outputs/<dataset>/v2/router/dev/fusion_v2_query_eval_seed1.csv
python scripts/export_query_eval.py --expert structural_v2 --run-dir <dataset_structural_v2_run> --split dev \
  --out outputs/<dataset>/v2/router/dev/structural_v2_query_eval_seed1.csv
python scripts/analyze_general_expert_complementarity.py \
  --fusion outputs/<dataset>/v2/router/dev/fusion_v2_query_eval_seed1.csv \
  --structural outputs/<dataset>/v2/router/dev/structural_v2_query_eval_seed1.csv \
  --split dev --out-dir outputs/<dataset>/v2/complementarity/dev
```

The analysis reports fusion/structural wins, ties, mean delta RR, both fixed MRRs, hard-selection Oracle MRR, Oracle headroom, direction, the eight target-side modality regimes, relation, and relation-support buckets. Target masks are post-hoc reporting fields only.

Export candidate scores into the same dataset-local directory, then compare normalization and alpha policies without retraining either expert:

```bash
python scripts/export_candidate_scores.py --gate-run-dir <dataset_gate_v2_run> \
  --residual-run-dir <dataset_structural_v2_run> --split dev --direction both --top-k 100 \
  --out-csv outputs/<dataset>/v2/scores/dev_seed1_top100.csv \
  --summary-json outputs/<dataset>/v2/scores/dev_seed1_top100_summary.json

python scripts/eval_score_ensemble_baselines.py \
  --score-dir outputs/<dataset>/v2/scores \
  --output-dir outputs/<dataset>/v2/score_ensemble/query_zscore \
  --selection-only --score-normalization query_zscore \
  --relation-shrinkage-lambda 20
```

Repeat the last command with `none` and `rank_based`. Use `--relation-shrinkage-lambda 0` for unshrunk relation alpha. Nonzero lambda is rejected for `openbg_legacy_v1`. `--selection-only` never loads test summaries. Once every choice is frozen on validation, export final test scores and rerun without `--selection-only`; test is only for locked reporting.

## 6. Verification commands (server)

The following suite includes model/evaluator tests and should run on the server, not the local low-compute machine:

```bash
python -m pytest ml/training/tests/test_dataset_bundle_contract.py \
  ml/training/tests/test_external_preprocessing_contract.py \
  ml/training/tests/test_external_anchor_configs.py \
  ml/training/tests/test_protocol_isolation.py \
  ml/training/tests/test_general_protocol_toy.py \
  ml/training/tests/test_apkgc_epoch_noise.py \
  ml/training/tests/test_openbg_legacy_checkpoint_contract.py
```

The generalized-v2 additions are covered by:

```bash
python -m pytest ml/training/tests/test_general_v2_models.py \
  ml/training/tests/test_score_combination_v2.py \
  ml/training/tests/test_general_router_contract.py \
  ml/training/tests/test_general_protocol_toy.py \
  ml/training/tests/test_protocol_isolation.py
```

The OpenBG lock has two levels. This read-only check validates the already-exported seed-1 metrics and query artifacts without loading models:

```bash
python ml/training/scripts/verify_openbg_legacy_regression.py
```

The release checkpoint gate also verifies immutable checkpoint sizes/SHA256 values and strict state-dict loading. Missing external checkpoints are a hard failure in that job:

```bash
REQUIRE_OPENBG_LEGACY_CHECKPOINTS=1 python -m pytest \
  ml/training/tests/test_openbg_legacy_checkpoint_contract.py
```

For a full checkpoint regression, re-export Gate-only, Residual-only and Full/Ours with the old configs and compare them to `ml/training/tests/fixtures/openbg_legacy_regression_v1.json`. The legacy profile keeps the original split, cache files, `has_img`, target regimes, router columns, raw-relation score-aware feature order and checkpoint keys.

Place re-exported CSVs under one directory with the fixture basenames, then run `python ml/training/scripts/verify_openbg_legacy_regression.py --query-dir <reexport_dir>` to compare the semantic `(query_id, direction, rank)` vectors independent of CSV line endings.
