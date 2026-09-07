# MKG-Y Y0/Y1 Existing-Checkpoint Acceptance Audit

Date: 2026-09-07

## Decision

- Y0 data/engineering adaptation: `Y0_ACCEPTED`
- Y1 standalone checkpoint reproduction: `Y1_ACCEPTED_WITH_PROVENANCE_NOTES`
- Progression: `Y2_DEV_EXPORT_ALLOWED`

No TEST row file was opened and no checkpoint was trained, selected, or modified.

## Y0 canonical data acceptance

The canonical MKG-Y package contains 15,000 entities, 28 relations, and TRAIN/DEV/TEST counts of 21,310/2,665/2,663. The existing preprocessing audit declares zero duplicates and zero cross-split overlaps.

| Modality | Coverage | Aligned | Missing | Dimension | Numeric health |
| --- | ---: | ---: | ---: | ---: | --- |
| text | 82.1% | 12,316 | 2,684 | 384 | finite |
| image | 96.4% | 14,466 | 534 | 383 | finite |

## Y1 three-seed DEV checkpoint acceptance

| Model | Seeds | DEV MRR mean ± sd | Range | H@1 | H@3 | H@10 | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mmkg_mhyper | 3 | 0.352631 ± 0.001696 | 0.350979–0.354367 | 0.318824 | 0.371044 | 0.407755 | True |
| mmkg_native | 3 | 0.324409 ± 0.002119 | 0.322780–0.326805 | 0.248655 | 0.379675 | 0.428455 | True |
| mmkg_adamf_mat | 3 | 0.320742 ± 0.000429 | 0.320414–0.321228 | 0.238086 | 0.380613 | 0.434897 | True |

All nine checkpoints are non-empty valid ZIP-format torch archives, use the official canonical MKG-Y package, run full filtered bidirectional DEV evaluation, retain `run_test: false`, and have distinct hashes across seeds within each model.

## Provenance notes

- 6/9 runs (all seed-2/3 runs) retain the legacy filename `metrics_seed1.csv`; their directory suffix, `common.yaml`, and merged `system.seed` agree on the correct seed.
- 9/9 embedded run manifests contain the earlier whole-registry source-lock hash; after excluding only that registry digest, each embedded MKG-Y manifest equals the current canonical manifest.

The whole-registry source-lock hash changed after training, but the embedded and current MKG-Y manifests differ only in that registry hash; every MKG-Y record-level source, mapping, split, and canonical feature hash remains identical. Seed identity is taken from the frozen run directory plus `system.seed`, never inferred from the legacy metric filename.

## Machine-readable outputs

- [data acceptance](../../outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance/data_acceptance.json)
- [checkpoint inventory](../../outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance/checkpoint_inventory.csv)
- [standalone DEV summary](../../outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance/standalone_dev_summary.csv)
- [acceptance decision](../../outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance/acceptance_decision.json)
- [audit manifest](../../outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance/audit_manifest.json)

## Operational audit

- TEST row access = 0
- TEST evaluation = 0
- checkpoint training/reselection/modification = 0
- hyperparameter change = 0

This acceptance permits only the next frozen DEV export/replication stage. It does not unlock TEST.
