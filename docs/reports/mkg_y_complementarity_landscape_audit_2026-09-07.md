# MKG-Y Y-E1 Available Complementarity Landscape Audit

Date: 2026-09-07

## Outcome

Descriptive replication outcome: **Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT**.

Y-E1 has no method-development or progression gate. It describes the complete frozen DEV action geometry and does not train a selector or start Y-E2.

## Main findings

- All 3/3 pairs have positive action-grid Oracle headroom with an original-triple clustered 95% CI lower bound above zero.
- Available headroom ranges from 0.036846 to 0.044070 MRR.
- Positive-opportunity prevalence ranges from 63.88% to 65.94% across pairs.
- The frozen Global alphas are mkg_y_mhyper_native=0.35, mkg_y_mhyper_adamf=0.35, mkg_y_native_adamf=0.55.
- These results establish available complementarity only. They do not establish cross-seed stability, inference-time observability, or deployability.

## Frozen protocol and boundary

- Dataset: MKG-Y; pairs: M-Hyper + NativE, M-Hyper + AdaMF-MAT, and NativE + AdaMF-MAT.
- Evidence: Y2 exact per-query filtered RR for alpha `0.00:0.05:1.00`, three seeds, and head/tail directions.
- Score normalization: `query_zscore`; Global alpha is taken unchanged from each Y2 DEV selection.
- MKG-Y DEV filtering uses TRAIN+DEV known facts only so no TEST row is opened or used as a filter fact.
- Bootstrap resamples original triples while retaining all three seeds and both directions.
- Supported relation-direction groups require at least 60 seed-direction observations.
- TEST access = 0; checkpoint evaluation = 0; retraining = 0; reselection = 0; policy training = 0.

## Pair-level results

| Pair | alpha0 | Global MRR | Oracle MRR | Headroom | Clustered 95% CI | Positive opportunities | G median | W median | D median* | Plateau mean | Fragmented positive | Direction consistency** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | 0.35 | 0.339323 | 0.376169 | 0.036846 | [0.034396, 0.039433] | 63.88% | 0.000113 | 0.286 | 0.05 | 0.440 | 11.57% | 0.641 |
| MKG-Y / M-Hyper + AdaMF-MAT | 0.35 | 0.337307 | 0.381377 | 0.044070 | [0.041320, 0.046907] | 65.22% | 0.000166 | 0.286 | 0.05 | 0.438 | 12.97% | 0.642 |
| MKG-Y / NativE + AdaMF-MAT | 0.55 | 0.328287 | 0.366131 | 0.037844 | [0.035450, 0.040339] | 65.94% | 0.000099 | 0.429 | 0.05 | 0.483 | 17.21% | 0.585 |

\* `D` is summarized over positive-opportunity queries only. ** Macro average over supported relation × direction groups.

## Figures

1. [Global-to-Oracle dumbbell](../../outputs/complementarity_identifiability/mkg_y_y_e1_landscape/figure1_global_to_oracle.svg)
2. [Highlighted action landscapes](../../outputs/complementarity_identifiability/mkg_y_y_e1_landscape/figure2_action_landscape_heatmaps.svg)
3. [G W D distributions](../../outputs/complementarity_identifiability/mkg_y_y_e1_landscape/figure3_gwd_distributions.svg)
4. [Relation by direction consistency](../../outputs/complementarity_identifiability/mkg_y_y_e1_landscape/figure4_relation_direction_consistency.svg)

## Reproducibility outputs

- Per-query geometry: `outputs/complementarity_identifiability/mkg_y_y_e1_landscape/per_query_action_geometry.csv.gz`
- Pair statistics: `outputs/complementarity_identifiability/mkg_y_y_e1_landscape/pair_statistics.csv`
- Machine-readable assessment: `outputs/complementarity_identifiability/mkg_y_y_e1_landscape/replication_assessment.json`
- Audit manifest: `outputs/complementarity_identifiability/mkg_y_y_e1_landscape/audit_manifest.json`

## Source hashes

| Pair | Role | Path | SHA256 |
| --- | --- | --- | --- |
| all | y2_export_verification | `outputs/complementarity_identifiability/mkg_y_y2_full_ranking/export_verification.json` | `b9d3e6ef09ef7933923e512d209f5147042034747c7f435e1ffa746e544dbb42` |
| all | y2_audit_manifest | `outputs/complementarity_identifiability/mkg_y_y2_full_ranking/audit_manifest.json` | `caa1e701dfa6b3bd99097a2580cd070cd730dd905720d3bfc6e143f2d54d9b41` |
| all | y_e1_contract | `docs/protocols/MKG_Y_Y_E1_LANDSCAPE_CONTRACT.json` | `e60dfaed5c0e56a76befb9b3a932f761956674308c4caf643e5f4c651f119785` |
| all | y_e1_implementation | `scripts/audit_mkg_y_e1_landscape.py` | `496bbd8ac4ec3963779de3f5673cf06580a8b131e3b70d79164f87e1aca991b2` |
| all | shared_landscape_implementation | `scripts/audit_complementarity_landscape.py` | `161a4979b764f7509ffddaa4090b2c17942dfb8ea7c50eab93b3661f839e5df2` |
| mkg_y_mhyper_native | y2_source_manifest | `outputs/complementarity_identifiability/mkg_y_y2_full_ranking/mkg_y_mhyper_native_dev_source_manifest.json` | `007353bb7d1f4269ef9a9d07c43d209fa226dbf2f4148a568fc89623b822dff2` |
| mkg_y_mhyper_native | source_query_rows | `outputs/mkg_y/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_query_rows.csv` | `b37c8f69052a6b8abe6e5a878daaafdc6203d3ca31f98a3dbb9ffa853352569b` |
| mkg_y_mhyper_native | source_selection | `outputs/mkg_y/anchored_dynamic/mhyper_native_seed123/full_ranking/selection.json` | `c41b24bf8225752e95111e5169c670d76eec64e99710349373168457e6338772` |
| mkg_y_mhyper_native | source_summary | `outputs/mkg_y/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_summary.json` | `c2a16dc7ff6e2993e8dc0c598956f0eedb33ed84afcfaf2f22358356fde8cc32` |
| mkg_y_mhyper_adamf | y2_source_manifest | `outputs/complementarity_identifiability/mkg_y_y2_full_ranking/mkg_y_mhyper_adamf_dev_source_manifest.json` | `ca19a0cfcf63c18c1c0a4a2046efd409d61226a654e5e04625652d88e2a35826` |
| mkg_y_mhyper_adamf | source_query_rows | `outputs/mkg_y/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_query_rows.csv` | `bcb61387bfb88970a893fd1649d1626933a70bddd0d938939fa7e7b61e2b5d5d` |
| mkg_y_mhyper_adamf | source_selection | `outputs/mkg_y/anchored_dynamic/mhyper_adamf_seed123/full_ranking/selection.json` | `b25a19769b733bd9350866b17c3b6207c890460c3cc25cfdcbe358b1b139bba9` |
| mkg_y_mhyper_adamf | source_summary | `outputs/mkg_y/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_summary.json` | `83dc12669cee784285b9300bc70143a912be275cfb10a922fa7b07faa4cba69e` |
| mkg_y_native_adamf | y2_source_manifest | `outputs/complementarity_identifiability/mkg_y_y2_full_ranking/mkg_y_native_adamf_dev_source_manifest.json` | `215772f489a1818cb830afb19c97aae42f1db18aca1c45d3f68f78bdbfe036dc` |
| mkg_y_native_adamf | source_query_rows | `outputs/mkg_y/anchored_dynamic/native_adamf_seed123/full_ranking/dev_query_rows.csv` | `d00e97dcd7b94949eab2fd9e8ef30c153d0797656aaee94ce94c831700e091b3` |
| mkg_y_native_adamf | source_selection | `outputs/mkg_y/anchored_dynamic/native_adamf_seed123/full_ranking/selection.json` | `683eb814d3a9bcf0eaa954498c745138a091b0395e953c94546829cea1e683d3` |
| mkg_y_native_adamf | source_summary | `outputs/mkg_y/anchored_dynamic/native_adamf_seed123/full_ranking/dev_summary.json` | `ac09c4795247bdd76936bfce87ae6b650d5a070af9f9cd1fd4af9eb4e0d72a0c` |

All source hashes were verified before analysis. The analytical input contains DEV rows only.
