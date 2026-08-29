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
| `ml/training/src/models/general_mmkg/structural_expert.py` | Pure general-v2 structural expert | N/A | No modality constructor inputs/buffers; ComplEx-compatible scoring | Low | Modality-independence and filtered-ranking toy tests |
| `ml/training/src/models/general_mmkg/availability_fusion.py` | General-v2 multimodal expert | N/A | Explicit four-state masks, T0V0 fallback, independent modality dropout | Low | Four-state weights/no-NaN test |
| `router/score_combination.py` | General score normalization/interpolation primitive | N/A | `none`, query z-score, rank-based; no target input | Low | Leakage and filtered-candidate tests |
| `scripts/analyze_general_expert_complementarity.py` | Dataset-local post-hoc analysis | N/A | Fixed wins/ties, Oracle headroom and subgroup exports | Low | Synthetic CSV contract/manual DEV export |
| Recent baseline adapters | M-Hyper, AdaMF-MAT, NativE, APKGC | OpenBG-named constructors/cache loading | Feed shared canonical tensors through `build_model` | Medium | Dataset-neutral config aliases |
| `ml/training/scripts/preprocess_external_mmkg.py` | New canonical preprocessing | N/A | Official splits, explicit keyed alignment, pooling, masks, hashes | Low | Audit and preprocessing tests |

The migration boundary is deliberately narrow: format and alignment logic live in the data layer, while training and evaluation receive integer triples and aligned tensors only.

## Frozen boundary and generalized-v2 boundary

Frozen legacy behavior comprises `openbg_legacy_v1`, every existing `openbg_img_*` config and alias, the old gated/residual classes and their state-dict layout, C1–C4 feature profiles/order, OpenBG target-regime definitions, raw-score interpolation default, evaluator tie/filter semantics, and all existing result artifacts. None of those are migrated in place.

Generalized-v2 behavior is opt-in through `mmkg_gate_only_v2` and `mmkg_structural_v2` under `mmkg_general_v1`. Score normalization defaults to `none`; relation shrinkage defaults to lambda zero. The optional G4 fields are neutral until dataset-local TRAIN graph statistics and DEV-only relation×direction priors are supplied. Formal test evaluation remains disabled in all v2 anchor configs until validation choices are frozen.
