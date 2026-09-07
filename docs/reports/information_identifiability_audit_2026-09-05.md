# Information–Identifiability Audit — Experiment 2

Date: 2026-09-05

## Outcome

Frozen preliminary query-level gate: **QUERY-LEVEL PRELIMINARY NO-GO**.

These are finite-model strict-OOF probes and are reported as Empirical Identifiable Headroom, not theoretical `C_identifiable`.

## Primary nested OOF probes

| Pair | X | Fold-specific Global MRR | OOF MRR | OOF gain | 95% clustered CI | Recovery | Positive gain | Negative transfer | Changed | Train gain | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper+NativE | X1 | 0.355802 | 0.363651 | +0.007849 | [+0.005991, +0.009754] | 16.8% | 37.1% | 23.3% | 93.1% | +0.017745 | +0.009896 |
| MKG-W / M-Hyper+NativE | X2 | 0.355802 | 0.363841 | +0.008039 | [+0.006174, +0.009943] | 17.2% | 36.1% | 23.0% | 90.5% | +0.019008 | +0.010969 |
| MKG-W / M-Hyper+NativE | X3 | 0.355802 | 0.364002 | +0.008200 | [+0.006368, +0.010029] | 17.6% | 36.8% | 22.1% | 90.2% | +0.019561 | +0.011361 |
| MKG-W / M-Hyper+NativE | X4 | 0.355802 | 0.363141 | +0.007339 | [+0.005445, +0.009243] | 15.7% | 36.3% | 22.4% | 89.9% | +0.019684 | +0.012345 |
| MKG-W / M-Hyper+NativE | X5 | 0.355802 | 0.362850 | +0.007047 | [+0.005222, +0.008864] | 15.1% | 35.1% | 21.5% | 86.1% | +0.018611 | +0.011563 |
| MKG-W / M-Hyper+NativE | X6 | 0.355802 | 0.359484 | +0.003682 | [+0.001583, +0.005760] | 7.9% | 36.6% | 22.3% | 87.1% | +0.010968 | +0.007285 |
| MKG-W / M-Hyper+AdaMF | X1 | 0.352822 | 0.356511 | +0.003689 | [+0.002721, +0.004681] | 9.0% | 21.8% | 7.0% | 34.9% | +0.007495 | +0.003806 |
| MKG-W / M-Hyper+AdaMF | X2 | 0.352822 | 0.357645 | +0.004823 | [+0.003715, +0.005958] | 11.8% | 22.4% | 6.4% | 34.6% | +0.010042 | +0.005219 |
| MKG-W / M-Hyper+AdaMF | X3 | 0.352822 | 0.358243 | +0.005421 | [+0.004177, +0.006675] | 13.3% | 23.6% | 7.1% | 38.2% | +0.012309 | +0.006887 |
| MKG-W / M-Hyper+AdaMF | X4 | 0.352822 | 0.358731 | +0.005909 | [+0.004669, +0.007196] | 14.5% | 21.4% | 7.2% | 34.2% | +0.012424 | +0.006515 |
| MKG-W / M-Hyper+AdaMF | X5 | 0.352822 | 0.358139 | +0.005317 | [+0.004163, +0.006512] | 13.0% | 22.0% | 7.6% | 37.0% | +0.012460 | +0.007143 |
| MKG-W / M-Hyper+AdaMF | X6 | 0.352822 | 0.351294 | -0.001528 | [-0.003467, +0.000508] | -3.7% | 30.9% | 16.4% | 67.6% | +0.009226 | +0.010754 |
| MKG-W / NativE+AdaMF | X1 | 0.334855 | 0.335505 | +0.000649 | [-0.000213, +0.001510] | 1.4% | 8.1% | 7.0% | 25.1% | +0.005835 | +0.005185 |
| MKG-W / NativE+AdaMF | X2 | 0.334855 | 0.335752 | +0.000897 | [-0.000029, +0.001838] | 1.9% | 9.4% | 8.1% | 29.2% | +0.006515 | +0.005618 |
| MKG-W / NativE+AdaMF | X3 | 0.334855 | 0.335633 | +0.000778 | [-0.000177, +0.001720] | 1.6% | 9.5% | 7.9% | 29.2% | +0.006786 | +0.006008 |
| MKG-W / NativE+AdaMF | X4 | 0.334855 | 0.335894 | +0.001039 | [+0.000093, +0.002000] | 2.2% | 10.5% | 8.7% | 31.7% | +0.008234 | +0.007196 |
| MKG-W / NativE+AdaMF | X5 | 0.334855 | 0.335803 | +0.000947 | [+0.000044, +0.001827] | 2.0% | 9.5% | 8.1% | 27.9% | +0.007917 | +0.006970 |
| MKG-W / NativE+AdaMF | X6 | 0.334855 | 0.332203 | -0.002652 | [-0.004042, -0.001278] | -5.6% | 15.7% | 17.8% | 52.2% | +0.008688 | +0.011340 |
| DB15K / M-Hyper+NativE | X1 | 0.382587 | 0.384003 | +0.001416 | [+0.000560, +0.002317] | 2.9% | 18.9% | 13.5% | 41.9% | +0.005932 | +0.004516 |
| DB15K / M-Hyper+NativE | X2 | 0.382587 | 0.384368 | +0.001781 | [+0.001008, +0.002561] | 3.6% | 18.3% | 12.9% | 40.3% | +0.005956 | +0.004176 |
| DB15K / M-Hyper+NativE | X3 | 0.382587 | 0.385077 | +0.002490 | [+0.001595, +0.003384] | 5.0% | 18.2% | 14.8% | 42.8% | +0.008457 | +0.005967 |
| DB15K / M-Hyper+NativE | X4 | 0.382587 | 0.384712 | +0.002125 | [+0.001280, +0.002993] | 4.3% | 18.4% | 13.6% | 42.2% | +0.009172 | +0.007046 |
| DB15K / M-Hyper+NativE | X5 | 0.382587 | 0.384706 | +0.002119 | [+0.001451, +0.002814] | 4.3% | 17.6% | 11.9% | 40.0% | +0.007703 | +0.005584 |
| DB15K / M-Hyper+NativE | X6 | 0.382587 | 0.380004 | -0.002583 | [-0.003818, -0.001319] | -5.2% | 20.9% | 15.9% | 57.3% | +0.008825 | +0.011408 |
| DB15K / M-Hyper+AdaMF | X1 | 0.382587 | 0.382749 | +0.000162 | [-0.000324, +0.000670] | 0.4% | 16.4% | 9.4% | 33.2% | +0.002812 | +0.002649 |
| DB15K / M-Hyper+AdaMF | X2 | 0.382587 | 0.383239 | +0.000652 | [+0.000140, +0.001172] | 1.4% | 14.9% | 9.6% | 31.3% | +0.004558 | +0.003906 |
| DB15K / M-Hyper+AdaMF | X3 | 0.382587 | 0.383419 | +0.000832 | [+0.000305, +0.001347] | 1.8% | 16.5% | 9.3% | 34.1% | +0.004444 | +0.003612 |
| DB15K / M-Hyper+AdaMF | X4 | 0.382587 | 0.383511 | +0.000924 | [+0.000349, +0.001509] | 2.0% | 16.0% | 9.4% | 32.9% | +0.005040 | +0.004116 |
| DB15K / M-Hyper+AdaMF | X5 | 0.382587 | 0.383694 | +0.001107 | [+0.000544, +0.001669] | 2.4% | 15.8% | 9.1% | 32.6% | +0.005733 | +0.004626 |
| DB15K / M-Hyper+AdaMF | X6 | 0.382587 | 0.376402 | -0.006185 | [-0.007286, -0.005100] | -13.6% | 19.6% | 13.5% | 53.5% | +0.004719 | +0.010904 |
| DB15K / NativE+AdaMF | X1 | 0.321022 | 0.321558 | +0.000536 | [-0.000004, +0.001082] | 0.9% | 16.6% | 7.7% | 32.4% | +0.003813 | +0.003277 |
| DB15K / NativE+AdaMF | X2 | 0.321022 | 0.322299 | +0.001277 | [+0.000785, +0.001791] | 2.2% | 14.5% | 6.6% | 27.5% | +0.004382 | +0.003105 |
| DB15K / NativE+AdaMF | X3 | 0.321022 | 0.322260 | +0.001238 | [+0.000688, +0.001792] | 2.2% | 16.1% | 7.4% | 31.3% | +0.005730 | +0.004492 |
| DB15K / NativE+AdaMF | X4 | 0.321022 | 0.322586 | +0.001564 | [+0.001039, +0.002114] | 2.7% | 16.3% | 7.2% | 31.5% | +0.005386 | +0.003823 |
| DB15K / NativE+AdaMF | X5 | 0.321022 | 0.322089 | +0.001067 | [+0.000606, +0.001547] | 1.9% | 15.4% | 6.8% | 29.9% | +0.004257 | +0.003189 |
| DB15K / NativE+AdaMF | X6 | 0.321022 | 0.318177 | -0.002845 | [-0.004117, -0.001546] | -4.9% | 24.9% | 16.3% | 58.5% | +0.011058 | +0.013903 |

## Utility-identifiability diagnostics

| Pair | X | Spearman(pred U, U) | Positive AP lift | Harmful AP lift | Positive-vs-harmful AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper+NativE | X1 | +0.1944 | +0.0862 | +0.0200 | 0.6582 |
| MKG-W / M-Hyper+NativE | X2 | +0.1786 | +0.0861 | +0.0167 | 0.6454 |
| MKG-W / M-Hyper+NativE | X3 | +0.1804 | +0.0828 | +0.0193 | 0.6468 |
| MKG-W / M-Hyper+NativE | X4 | +0.1726 | +0.0781 | +0.0176 | 0.6429 |
| MKG-W / M-Hyper+NativE | X5 | +0.1640 | +0.0813 | +0.0103 | 0.6352 |
| MKG-W / M-Hyper+NativE | X6 | +0.1955 | +0.1060 | +0.0633 | 0.6417 |
| MKG-W / M-Hyper+AdaMF | X1 | +0.2946 | +0.2200 | +0.0366 | 0.6583 |
| MKG-W / M-Hyper+AdaMF | X2 | +0.3052 | +0.2237 | +0.0437 | 0.6631 |
| MKG-W / M-Hyper+AdaMF | X3 | +0.3179 | +0.2191 | +0.0521 | 0.6683 |
| MKG-W / M-Hyper+AdaMF | X4 | +0.3238 | +0.2248 | +0.0534 | 0.6715 |
| MKG-W / M-Hyper+AdaMF | X5 | +0.3172 | +0.2120 | +0.0534 | 0.6700 |
| MKG-W / M-Hyper+AdaMF | X6 | +0.2029 | +0.1087 | +0.0458 | 0.6027 |
| MKG-W / NativE+AdaMF | X1 | +0.1133 | +0.0316 | +0.0045 | 0.5670 |
| MKG-W / NativE+AdaMF | X2 | +0.1166 | +0.0303 | +0.0178 | 0.5691 |
| MKG-W / NativE+AdaMF | X3 | +0.1244 | +0.0316 | +0.0208 | 0.5732 |
| MKG-W / NativE+AdaMF | X4 | +0.1188 | +0.0270 | +0.0184 | 0.5705 |
| MKG-W / NativE+AdaMF | X5 | +0.1196 | +0.0340 | +0.0183 | 0.5706 |
| MKG-W / NativE+AdaMF | X6 | +0.0854 | +0.0258 | +0.0264 | 0.5479 |
| DB15K / M-Hyper+NativE | X1 | +0.1600 | +0.1486 | -0.0069 | 0.5947 |
| DB15K / M-Hyper+NativE | X2 | +0.1617 | +0.1526 | +0.0063 | 0.5939 |
| DB15K / M-Hyper+NativE | X3 | +0.1625 | +0.1447 | +0.0091 | 0.5924 |
| DB15K / M-Hyper+NativE | X4 | +0.1673 | +0.1485 | +0.0100 | 0.5957 |
| DB15K / M-Hyper+NativE | X5 | +0.1649 | +0.1487 | +0.0077 | 0.5949 |
| DB15K / M-Hyper+NativE | X6 | +0.1327 | +0.0779 | +0.0380 | 0.5845 |
| DB15K / M-Hyper+AdaMF | X1 | +0.2014 | +0.1503 | +0.0255 | 0.6120 |
| DB15K / M-Hyper+AdaMF | X2 | +0.2044 | +0.1494 | +0.0391 | 0.6105 |
| DB15K / M-Hyper+AdaMF | X3 | +0.2042 | +0.1437 | +0.0400 | 0.6087 |
| DB15K / M-Hyper+AdaMF | X4 | +0.2063 | +0.1459 | +0.0405 | 0.6094 |
| DB15K / M-Hyper+AdaMF | X5 | +0.2108 | +0.1521 | +0.0430 | 0.6148 |
| DB15K / M-Hyper+AdaMF | X6 | +0.1579 | +0.0790 | +0.0474 | 0.5943 |
| DB15K / NativE+AdaMF | X1 | +0.1262 | +0.1009 | +0.0093 | 0.5744 |
| DB15K / NativE+AdaMF | X2 | +0.1327 | +0.1139 | +0.0128 | 0.5777 |
| DB15K / NativE+AdaMF | X3 | +0.1473 | +0.1165 | +0.0182 | 0.5841 |
| DB15K / NativE+AdaMF | X4 | +0.1509 | +0.1195 | +0.0176 | 0.5861 |
| DB15K / NativE+AdaMF | X5 | +0.1458 | +0.1153 | +0.0158 | 0.5842 |
| DB15K / NativE+AdaMF | X6 | +0.0967 | +0.0494 | +0.0175 | 0.5530 |

## Preliminary gate by information representation

| X | Robust-positive pairs | Both NativE+AdaMF positive | >=10% recovery pairs | MKG-W NativE+AdaMF CI lower > 0 | Pass |
| --- | ---: | --- | ---: | --- | --- |
| X1 | 3/6 | True | 1/6 | False | False |
| X2 | 5/6 | True | 2/6 | False | False |
| X3 | 5/6 | True | 2/6 | False | False |
| X4 | 6/6 | True | 2/6 | True | False |
| X5 | 6/6 | True | 2/6 | True | False |
| X6 | 1/6 | False | 0/6 | False | False |

Experiment 3 remains the required next comparison regardless of this preliminary result. No query selector or final policy was developed.

## Machine-readable results

- `metrics_by_x_learner.csv`: every dataset/pair/X/learner result
- `primary_nested_probe_metrics.csv`: fold-wise inner-selected primary probes
- `nested_learner_selections.csv`: fold-specific learner choices
- `runs/<pair>/<x>/<learner>/seed_direction_metrics.csv`: seed, direction, and seed × direction results

## Figures

1. [figure1_information_identifiability_curve](../../outputs/complementarity_identifiability/exp2_information/figure1_information_identifiability_curve.svg)
2. [figure2_available_vs_empirical](../../outputs/complementarity_identifiability/exp2_information/figure2_available_vs_empirical.svg)
3. [figure3_x_learner_heatmap](../../outputs/complementarity_identifiability/exp2_information/figure3_x_learner_heatmap.svg)
4. [figure4_train_oof_gap](../../outputs/complementarity_identifiability/exp2_information/figure4_train_oof_gap.svg)

## Operational audit

- TEST access = 0
- full-DEV Global used for held-out folds = 0
- checkpoint training/reselection = 0
- AACPI resurrection = 0
- final policy development = 0
- candidate embeddings = 0

All input and output hashes are recorded in the machine-readable `audit_manifest.json`.
