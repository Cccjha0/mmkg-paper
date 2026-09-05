# Complementarity Landscape Audit — Experiment 1

Date: 2026-09-05

## Outcome

Frozen Available-complementarity gate: **GO**. Headroom criterion passed by 6/6 pairs; positive-opportunity criterion passed by 6/6 pairs.

This is a descriptive DEV-only audit. It trains no selector, runs no policy, and does not start Experiment 2.

## Main findings

- Available complementarity is unambiguous in all six dataset-pairs: action-grid Oracle headroom ranges from 0.040841 to 0.057545 MRR, and every original-triple clustered 95% CI excludes zero on the positive side.
- Positive opportunities are common (44.997% to 63.564%), so the GO result is not driven by a very small tail of queries. The gain distribution is nevertheless highly zero-heavy: both DB15K M-Hyper pairs have median `G=0`.
- The landscapes are strongly stepped: mean plateau ratios range from 0.501 to 0.560. Among positive queries, the nearest beneficial action is usually one grid step from Global (median `D=0.05` for every pair), while mean `D` ranges from 0.086 to 0.136 because a smaller subset requires larger moves.
- Beneficial regions are usually connected, but not universally: 6.79% to 14.95% of positive-opportunity queries have multiple beneficial components, with as many as 5 to 8 components depending on the pair.
- The proposed contrast between M-Hyper-centered and NativE + AdaMF-MAT geometry is not uniformly supported by Available-complementarity alone. MKG-W M-Hyper + NativE and NativE + AdaMF-MAT have similar mean widths (0.292 versus 0.284) and fragmentation among positive queries (14.94% versus 14.45%), even though their direction alignment differs modestly.
- Four pairs have `alpha0=1.0`. Their apparent direction consistency of 1.0 is mechanically one-sided because no action exists above Global; it must not be interpreted as evidence that relation or context identifies a preferred correction direction. Only non-boundary pairs provide a genuinely two-sided direction-consistency diagnostic in this audit.

## Frozen protocol and operational boundary

- Datasets: MKG-W and DB15K; pairs: M-Hyper + NativE, M-Hyper + AdaMF-MAT, and NativE + AdaMF-MAT.
- Evidence: exact filtered per-query RR from the manifest-linked `full_ranking/dev_query_rows.csv` files.
- Action grid: `0.00:0.05:1.00`; normalization: `query_zscore`; seeds: 1, 2, 3; directions: head and tail.
- Global alpha comes unchanged from each manifest-linked DEV `selection.json`.
- Confidence intervals resample original triples and keep their three seeds and two prediction directions clustered.
- TEST access = 0; checkpoint retraining = 0; checkpoint reselection = 0; historical result modification = 0.

## Metric definitions

For every seed-direction query, `U_q(alpha) = RR_q(alpha) - RR_q(alpha0)` and `G_q = max RR - RR(alpha0)`.

- Beneficial basin width `W`: fraction of the 21 frozen grid actions with strictly positive `U`.
- Minimum beneficial distance `D`: minimum absolute alpha displacement from Global among positive-utility actions; undefined when no opportunity exists.
- Deterministic best alpha: maximize exact RR, then minimize distance to Global, then choose the smaller alpha.
- Best-action direction: toward expert A for alpha above Global, toward expert B below Global, otherwise stay.
- Plateau ratio: `(21 - number of distinct RR values) / 20`; 0 is fully varying and 1 is completely flat.
- Fragmentation: number of contiguous positive-utility regions on the ordered grid; fragmented means more than one region.
- A relation/direction context group is marked supported at `60` or more seed-direction query observations.
- Direction consistency is the larger of the toward-A and toward-B proportions among positive-opportunity queries; signed preference retains which side dominates.

## Pair-level results

| Dataset / pair | alpha0 | Global MRR | Oracle MRR | Headroom | Clustered 95% CI | Positive opportunities | G median | W median | D median* | Plateau mean | Fragmented positive | Direction consistency** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper + NativE | 0.60 | 0.356356 | 0.402981 | 0.046625 | [0.044355, 0.048932] | 63.56% | 0.000303 | 0.333 | 0.05 | 0.523 | 14.94% | 0.734 |
| MKG-W / M-Hyper + AdaMF-MAT | 1.00 | 0.352822 | 0.393663 | 0.040841 | [0.038510, 0.043266] | 52.66% | 0.000028 | 0.095 | 0.05 | 0.501 | 6.79% | 1.000 |
| MKG-W / NativE + AdaMF-MAT | 0.95 | 0.334855 | 0.382500 | 0.047645 | [0.045520, 0.049777] | 61.21% | 0.000067 | 0.048 | 0.05 | 0.508 | 14.45% | 0.717 |
| DB15K / M-Hyper + NativE | 1.00 | 0.382587 | 0.432166 | 0.049579 | [0.047550, 0.051588] | 45.00% | 0.000000 | 0.000 | 0.05 | 0.560 | 7.18% | 1.000 |
| DB15K / M-Hyper + AdaMF-MAT | 1.00 | 0.382587 | 0.428040 | 0.045453 | [0.043692, 0.047285] | 45.59% | 0.000000 | 0.000 | 0.05 | 0.552 | 7.25% | 1.000 |
| DB15K / NativE + AdaMF-MAT | 1.00 | 0.321022 | 0.378567 | 0.057545 | [0.055881, 0.059262] | 52.04% | 0.000021 | 0.095 | 0.05 | 0.547 | 9.35% | 1.000 |

\* `D` is summarized only over positive-opportunity queries. ** Macro average over supported relation × direction groups.

## Frozen GO gate

- Headroom: 6/6 pairs have positive Available headroom with clustered-bootstrap lower 95% bound above zero (threshold: at least 5/6).
- Opportunity prevalence: 6/6 pairs have positive-opportunity rate at least 25% (threshold: at least 5/6).
- Decision: **GO** — Experiment 2 is eligible, but is not started by this audit

The gate result only governs whether Experiment 2 is scientifically eligible. Experiment 2 was not started in this task.

## Figures

1. [Global-to-Oracle dumbbell](../../outputs/complementarity_identifiability/exp1_landscape/figure1_global_to_oracle.svg)
2. [MKG-W action-landscape heatmaps](../../outputs/complementarity_identifiability/exp1_landscape/figure2_action_landscape_heatmaps.svg)
3. [G/W/D distributions](../../outputs/complementarity_identifiability/exp1_landscape/figure3_gwd_distributions.svg)
4. [Relation × direction consistency](../../outputs/complementarity_identifiability/exp1_landscape/figure4_relation_direction_consistency.svg)

## Reproducibility outputs

- Per-query geometry: `outputs/complementarity_identifiability/exp1_landscape/per_query_action_geometry.csv.gz`
- Pair statistics: `outputs/complementarity_identifiability/exp1_landscape/pair_statistics.csv` and JSON equivalent.
- Distribution summaries: `outputs/complementarity_identifiability/exp1_landscape/gwd_plateau_distribution_summary.csv`.
- Fragmentation summaries: `outputs/complementarity_identifiability/exp1_landscape/fragmentation_statistics.csv`.
- Direction groups: `outputs/complementarity_identifiability/exp1_landscape/direction_consistency.csv`.
- Action summaries: `outputs/complementarity_identifiability/exp1_landscape/action_level_summary.csv`.
- Machine-readable audit/gate/source record: `outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json`.

## Source hashes

| Pair | Role | Path | SHA256 |
| --- | --- | --- | --- |
| db15k_mhyper_adamf | source_query_rows | `outputs/db15k/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_query_rows.csv` | `431d98b55b288fff46c7454be1e0e4409e47c75226b8441ec2ec02fe9ba1a6d2` |
| db15k_mhyper_adamf | source_selection | `outputs/db15k/anchored_dynamic/mhyper_adamf_seed123/full_ranking/selection.json` | `4bbac94200d8d11821670e4a183f92975baba5c2531bf13f50aedf0fba64f282` |
| db15k_mhyper_adamf | source_full_ranking_summary | `outputs/db15k/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_summary.json` | `f9d277474077d25cbde9112db4e5d74ae9d3de8403b72ac1421531736c7f9720` |
| db15k_mhyper_adamf | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/db15k_mhyper_adamf_dev_utility_table.csv.gz` | `bd4b7623e1eb60cf84718c42c93b2d47abbf3aaa411e5c04592d48a0aa410eca` |
| db15k_mhyper_adamf | aacpi_source_manifest | `outputs/aacpi/utility_tables/db15k_mhyper_adamf_dev_source_manifest.json` | `0f95a6073e85b4d226f57d68e45f5013bc11091f604d9d129a1eb877156e920e` |
| db15k_mhyper_native | source_query_rows | `outputs/db15k/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_query_rows.csv` | `f85d657f7183dc960bba762418ea3695ab0af78b59d45e158e1cd0a34079a081` |
| db15k_mhyper_native | source_selection | `outputs/db15k/anchored_dynamic/mhyper_native_seed123/full_ranking/selection.json` | `b0378a6bc3bc7f8ee50dc1f0aeee2d0073df801372d4f657dd35113ed07d2c04` |
| db15k_mhyper_native | source_full_ranking_summary | `outputs/db15k/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_summary.json` | `499286e155a8c5f03202531cfe7f4a46015c3b8475c5ada4862c096fec0a2420` |
| db15k_mhyper_native | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/db15k_mhyper_native_dev_utility_table.csv.gz` | `a3726b3e6194512edc6f87738ab85812f030fd51bb12fc68ed2fee4efe83f748` |
| db15k_mhyper_native | aacpi_source_manifest | `outputs/aacpi/utility_tables/db15k_mhyper_native_dev_source_manifest.json` | `3d5bd8a8069717933434d2a9dab6053decbe69cbe9b4dc06d74ddc84cab34b6d` |
| db15k_native_adamf | source_query_rows | `outputs/db15k/anchored_dynamic/native_adamf_seed123/full_ranking/dev_query_rows.csv` | `0b132d2a13d5df045d71adda1cf6282230f5db26ef18a5d2052caeb58da22bfd` |
| db15k_native_adamf | source_selection | `outputs/db15k/anchored_dynamic/native_adamf_seed123/full_ranking/selection.json` | `cce837ceb70ef765e6cc0ac6230b6a5241a7167e10621cb205339dbaa767c499` |
| db15k_native_adamf | source_full_ranking_summary | `outputs/db15k/anchored_dynamic/native_adamf_seed123/full_ranking/dev_summary.json` | `260c0e2590c340fd8e99a6c039048cb4e38db9f9673564689864c0ef302eec56` |
| db15k_native_adamf | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/db15k_native_adamf_dev_utility_table.csv.gz` | `b5769ce5d98a8b5198567bd1f824a9404ae889ca43030ee3b65d1caf1677ed3b` |
| db15k_native_adamf | aacpi_source_manifest | `outputs/aacpi/utility_tables/db15k_native_adamf_dev_source_manifest.json` | `f9001315914573bc95444c27c09def97e1eddf0fea4bac2d1ee81b6f1d96e20a` |
| mkgw_mhyper_adamf | source_query_rows | `outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_query_rows.csv` | `e389449b44379ecb18fe248127c6b062b2f5cc7894fb052536a442528ab4287b` |
| mkgw_mhyper_adamf | source_selection | `outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123/full_ranking/selection.json` | `f1c688963e62b52470c30ee722713bb372c51df61421503396cfbc3e5e8ad198` |
| mkgw_mhyper_adamf | source_full_ranking_summary | `outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123/full_ranking/dev_summary.json` | `2a1293bb2f5a5e8a681e67fe7091d5f3c99c3357761b4569c7b3efc6610eb5a5` |
| mkgw_mhyper_adamf | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/mkgw_mhyper_adamf_dev_utility_table.csv.gz` | `33765d12c6832c7fa53256e76ed1b7085bf87e1f68ee4ecdd86951283010cb2f` |
| mkgw_mhyper_adamf | aacpi_source_manifest | `outputs/aacpi/utility_tables/mkgw_mhyper_adamf_dev_source_manifest.json` | `8d9c1171dd9630742cfe8f80ea6a00adba580a76dc973c200fc77d60d78f8b8c` |
| mkgw_mhyper_native | source_query_rows | `outputs/mkg_w/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_query_rows.csv` | `8b7dfb913f3a0d00eef156200e4a656f4062f2b2306b85cf8e68130749254b98` |
| mkgw_mhyper_native | source_selection | `outputs/mkg_w/anchored_dynamic/mhyper_native_seed123/full_ranking/selection.json` | `28f40889a625e53cf14c69111f2225f1d998019987bf6109a46c06126a43db6c` |
| mkgw_mhyper_native | source_full_ranking_summary | `outputs/mkg_w/anchored_dynamic/mhyper_native_seed123/full_ranking/dev_summary.json` | `ed8bcb7375bb87f5f32e9bdc05496e59adc7ab6c596fcb83c5577786df9902d1` |
| mkgw_mhyper_native | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/mkgw_mhyper_native_dev_utility_table.csv.gz` | `d830f6978fd2380adcc8726fd53d8a9786990fa53d7fee9ef6d8a80e4d9ccc73` |
| mkgw_mhyper_native | aacpi_source_manifest | `outputs/aacpi/utility_tables/mkgw_mhyper_native_dev_source_manifest.json` | `57a24f419069845d35f4338ed8caa641dd50c290b6947cd14e8e83f65fdac0ac` |
| mkgw_native_adamf | source_query_rows | `outputs/mkg_w/anchored_dynamic/native_adamf_seed123/full_ranking/dev_query_rows.csv` | `980197c29b8d0c447e298b862d43064e0b86c088a134347136fa87a64ad877ed` |
| mkgw_native_adamf | source_selection | `outputs/mkg_w/anchored_dynamic/native_adamf_seed123/full_ranking/selection.json` | `15bd47d17171a3e603c5a5fb2f27bc33875ef6fd7f78a95be3f12f67d3fdfeaf` |
| mkgw_native_adamf | source_full_ranking_summary | `outputs/mkg_w/anchored_dynamic/native_adamf_seed123/full_ranking/dev_summary.json` | `73a0308e5afbedc714d357cbea2f73fc5c090e931eae3ae01a2f03d02acf1b4e` |
| mkgw_native_adamf | aacpi_utility_table_not_consumed | `outputs/aacpi/utility_tables/mkgw_native_adamf_dev_utility_table.csv.gz` | `20da7555f8a10a04a36dab57396f8726f1628cbffea59793fab586b4fcc16b61` |
| mkgw_native_adamf | aacpi_source_manifest | `outputs/aacpi/utility_tables/mkgw_native_adamf_dev_source_manifest.json` | `d4e01934588ec2919542b1ed4998d2a0992f300889cf313529f398eda6ff86b0` |

All manifest-declared hashes were recomputed before analysis. The utility tables are recorded for provenance but were not read as analytical input.
