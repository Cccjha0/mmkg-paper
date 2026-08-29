# Multi-dataset implementation map

OpenBG-IMG is frozen under `openbg_legacy_v1`; MKG-W and DB15K use the explicit `mmkg_general_v1` path. Old YAML files intentionally remain unchanged and infer the legacy loader.

| File / area | Current responsibility | OpenBG-specific assumption found | Required change | Legacy risk | Verification |
|---|---|---|---|---|---|
| `ml/training/src/data/tsv_reader.py` | TSV parsing | `ent_*` / `rel_*` tokens | Add integer canonical reader; retain old parser | High | Legacy artifact fixture; parser unit tests |
| `ml/training/scripts/run_train.py` | Train entry point | Reads paths itself | Consume `DatasetBundle` | High | Old configs still infer legacy |
| `ml/training/src/models/build_model.py` | Model construction | OpenBG cache names and paths | Inject bundle; map dataset-neutral model aliases | High | Legacy state-dict key test |
| `ml/training/src/models/openbg_img_gated_lp.py` | Gate/Residual/Ours core | Text assumed present | General-only `has_text` and `t_missing`; keep legacy object layout | High | Four-state toy test |
| `ml/training/src/eval/filtered_ranking.py` | Filtered bidirectional ranking | Only image subgroup reporting | Add independent T/V subgroup metrics | Medium | Existing evaluator tests plus toy four-state test |
| `scripts/export_query_eval.py` | Query-level ranks/scores | Synthetic relation names and OpenBG target regimes | Bundle mapping and protocol-specific regimes | High | Frozen legacy CSV hashes |
| `router/feature_utils.py` | Clean/post-hoc features | `text_emb.pt`, `img_emb.pt`, `has_img.pt`; no text mask | Separate general clean profile with observed T/V masks | High | Legacy profile remains default |
| `scripts/build_relation_priors.py` | Validation relation priors | Paths imply one dataset | Reuse definition with dataset-local validation exports | Medium | Dataset/protocol stored beside outputs |
| `scripts/eval_score_ensemble_baselines.py` | Score-aware combination | Raw relation ID used as float | Freeze for legacy; exclude it in general profile | High | Default feature order unchanged |
| Recent baseline adapters | M-Hyper, AdaMF-MAT, NativE, APKGC | OpenBG-named constructors/cache loading | Feed shared canonical tensors through `build_model` | Medium | Dataset-neutral config aliases |
| `ml/training/scripts/preprocess_external_mmkg.py` | New canonical preprocessing | N/A | Official splits, explicit keyed alignment, pooling, masks, hashes | Low | Audit and preprocessing tests |

The migration boundary is deliberately narrow: format and alignment logic live in the data layer, while training and evaluation receive integer triples and aligned tensors only.
