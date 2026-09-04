# AACPI Phase 3A Candidate Asset Audit

**Audit date:** 2026-09-04
**Scope:** MKG-W and DB15K DEV only

## Finding

The Phase 1 `dev_query_rows.csv` checkpoint files and AACPI utility tables do not contain full candidate scores, normalized candidate vectors, candidate identities, or per-action candidate rankings. They contain target ranks/RR, the 21-point alpha RR curve, and the frozen 13 query-geometry summaries. Candidate structure cannot be recovered from these values and will not be inferred from RR.

The repository has a historical `candidate_router_dev_top100.parquet`, but it is a Gate/Residual router export with its own expert definitions and target-aware export schema. It does not cover the frozen M-Hyper, NativE, and AdaMF-MAT pairs required by Phase 3A and is not an eligible Phase 3A source.

All 18 selected expert checkpoints needed for the six dataset/pair/seed combinations are present under `ml/artifacts/outputs/`, along with their `config_merged.json` files. Existing DEV full-ranking summaries record the exact run directory for expert A and B at each seed. Therefore candidate landscapes can be reconstructed without retraining or checkpoint reselection.

The current local checkout does not contain the canonical processed `manifest.json`, `train.tsv`, `valid.tsv`, mappings, or modality tensors referenced by those configs; its processed dataset directories contain only audit reports. Full reconstruction is therefore intentionally blocked locally. The server run must use the same frozen processed assets that produced the checkpoints. The builder checks every required file plus the manifest-declared TRAIN/DEV counts and hashes, and aborts rather than substituting or guessing when an asset is absent.

## Safe reconstruction plan

`scripts/build_aacpi_action_response_features.py` reconstructs full candidate scores from the frozen checkpoints and only canonical TRAIN/DEV assets. It does not call the general dataset loader because that loader opens all three split files. The Phase 3A loader reads `train.tsv`, `valid.tsv`, entity/relation mappings, and canonical modality tensors, and constructs a model-compatible bundle with an empty TEST list.

Candidate response features use unfiltered full-candidate rankings. This is required for answer-agnostic inference: the exact filtered evaluator keeps the current correct target while removing other true answers, which requires target identity. Filtered rank/RR remains solely in the already frozen Phase 1 utility supervision.

The builder reuses the repository model factory, expert direction scorer, and canonical `query_zscore` implementation. It writes compact query/action response features and a source manifest rather than persisting dense score matrices. The manifest hashes each checkpoint, configuration, canonical TRAIN/DEV input, mappings, modality tensors, utility table, and feature contract.

## Boundary

- expert retraining: no;
- checkpoint reselection: no;
- TEST rows opened: no;
- TEST evaluation commands: no;
- RR-derived candidate reconstruction: no;
- Phase 1 utility/action/alpha changes: no.

The reconstruction is GPU-intensive and must be run on the server after the implementation commit is available.
