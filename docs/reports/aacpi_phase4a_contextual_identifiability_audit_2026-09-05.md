# AACPI Phase 4A Contextual Identifiability Audit

**Frozen decision:** NO-GO

This audit uses DEV-only outer-held-out predictions and runs no policy or TEST evaluation.

## Pair results

| Dataset | Pair | Rep | H1 | Spearman | Positive AP lift | Harmful AP lift | Sign AUROC | Top-10% U | Top-10% P+ | Top-10% P- |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| db15k | db15k_mhyper_adamf | C0 | FAIL | 0.093226 | +0.076024 | +0.005713 | 0.547582 | -0.002274 | 45.995% | 23.116% |
| db15k | db15k_mhyper_adamf | C1 | FAIL | 0.096338 | +0.087834 | +0.001712 | 0.545837 | -0.001642 | 49.098% | 24.415% |
| db15k | db15k_mhyper_adamf | C2 | FAIL | 0.089499 | +0.074447 | +0.006432 | 0.544488 | -0.002418 | 46.863% | 25.320% |
| db15k | db15k_mhyper_adamf | C3 | FAIL | 0.074495 | +0.042070 | +0.009236 | 0.545924 | -0.004829 | 39.741% | 17.462% |
| db15k | db15k_mhyper_adamf | C4 | FAIL | 0.078688 | +0.047024 | +0.009167 | 0.544670 | -0.004905 | 40.336% | 18.498% |
| db15k | db15k_mhyper_native | C0 | FAIL | 0.071288 | +0.062933 | -0.003518 | 0.537141 | +0.002421 | 43.938% | 19.539% |
| db15k | db15k_mhyper_native | C1 | FAIL | 0.056608 | +0.054165 | -0.006271 | 0.520909 | +0.002040 | 42.245% | 23.652% |
| db15k | db15k_mhyper_native | C2 | PASS | 0.056425 | +0.045956 | +0.001817 | 0.532968 | +0.001191 | 40.877% | 18.177% |
| db15k | db15k_mhyper_native | C3 | FAIL | 0.037185 | +0.039517 | -0.011735 | 0.507306 | +0.000795 | 39.494% | 19.834% |
| db15k | db15k_mhyper_native | C4 | FAIL | 0.056379 | +0.048267 | -0.008918 | 0.525766 | +0.001825 | 41.040% | 17.435% |
| db15k | db15k_native_adamf | C0 | PASS | 0.086563 | +0.068695 | +0.006953 | 0.544957 | +0.006698 | 52.291% | 18.508% |
| db15k | db15k_native_adamf | C1 | PASS | 0.114580 | +0.081918 | +0.008057 | 0.559464 | +0.009916 | 54.473% | 18.813% |
| db15k | db15k_native_adamf | C2 | PASS | 0.096982 | +0.074603 | +0.008801 | 0.556212 | +0.004438 | 51.828% | 19.303% |
| db15k | db15k_native_adamf | C3 | PASS | 0.073764 | +0.049808 | +0.007315 | 0.543662 | +0.003277 | 47.389% | 18.124% |
| db15k | db15k_native_adamf | C4 | PASS | 0.076684 | +0.048209 | +0.008728 | 0.542834 | +0.004841 | 47.247% | 18.456% |
| mkg_w | mkgw_mhyper_adamf | C0 | PASS | 0.157590 | +0.121008 | +0.010269 | 0.588671 | +0.005313 | 61.859% | 14.958% |
| mkg_w | mkgw_mhyper_adamf | C1 | PASS | 0.159271 | +0.119044 | +0.013137 | 0.588399 | +0.008504 | 61.557% | 15.504% |
| mkg_w | mkgw_mhyper_adamf | C2 | PASS | 0.136561 | +0.096706 | +0.010615 | 0.571875 | +0.005289 | 58.244% | 16.624% |
| mkg_w | mkgw_mhyper_adamf | C3 | PASS | 0.127927 | +0.087251 | +0.006977 | 0.571470 | +0.008271 | 55.360% | 14.364% |
| mkg_w | mkgw_mhyper_adamf | C4 | PASS | 0.092723 | +0.047171 | +0.004171 | 0.549432 | +0.000801 | 46.219% | 15.319% |
| mkg_w | mkgw_mhyper_native | C0 | PASS | 0.220586 | +0.107881 | +0.108481 | 0.658203 | +0.006548 | 46.879% | 22.782% |
| mkg_w | mkgw_mhyper_native | C1 | PASS | 0.196304 | +0.085255 | +0.099756 | 0.640574 | +0.005657 | 42.027% | 22.568% |
| mkg_w | mkgw_mhyper_native | C2 | PASS | 0.240036 | +0.132964 | +0.116534 | 0.670538 | +0.008711 | 50.402% | 23.357% |
| mkg_w | mkgw_mhyper_native | C3 | PASS | 0.174433 | +0.082732 | +0.069144 | 0.627352 | +0.004804 | 43.035% | 21.432% |
| mkg_w | mkgw_mhyper_native | C4 | PASS | 0.212089 | +0.098565 | +0.090937 | 0.655166 | +0.005805 | 43.562% | 21.501% |
| mkg_w | mkgw_native_adamf | C0 | FAIL | 0.040870 | +0.015106 | +0.018078 | 0.527146 | -0.002800 | 30.948% | 29.646% |
| mkg_w | mkgw_native_adamf | C1 | FAIL | 0.054105 | +0.023916 | +0.012594 | 0.536735 | -0.000365 | 32.484% | 27.261% |
| mkg_w | mkgw_native_adamf | C2 | FAIL | 0.038583 | +0.018872 | +0.011764 | 0.524868 | -0.002715 | 31.930% | 29.568% |
| mkg_w | mkgw_native_adamf | C3 | FAIL | 0.031537 | +0.023191 | +0.002367 | 0.521059 | -0.003140 | 32.593% | 31.634% |
| mkg_w | mkgw_native_adamf | C4 | FAIL | 0.043807 | +0.020290 | +0.004045 | 0.529247 | -0.001808 | 31.579% | 27.580% |

## Frozen gates

| Rep | H1 pairs | Native pairs | Critical dirs | Critical seeds | Primary | Strong | Contribution | Stability | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |
| C1 | 3/6 | False | 1/2 | 2/3 | False | False | True | True | NO-GO |
| C2 | 4/6 | False | 0/2 | 0/3 | False | False | False | True | NO-GO |
| C3 | 3/6 | False | 0/2 | 1/3 | False | False | False | True | NO-GO |
| C4 | 3/6 | False | 1/2 | 1/3 | False | False | True | True | NO-GO |

## MKG-W / NativE + AdaMF-MAT

- C0: Spearman 0.040870; AP lifts +0.015106/+0.018078; sign AUROC 0.527146; top-10% U -0.002800; P+/P- 30.948%/29.646%; passing actions 0/5; directions 0/2; seeds 0/3; supported-relation coverage 22.95%.
- C1: Spearman 0.054105; AP lifts +0.023916/+0.012594; sign AUROC 0.536735; top-10% U -0.000365; P+/P- 32.484%/27.261%; passing actions 2/5; directions 1/2; seeds 2/3; supported-relation coverage 22.02%.
- C2: Spearman 0.038583; AP lifts +0.018872/+0.011764; sign AUROC 0.524868; top-10% U -0.002715; P+/P- 31.930%/29.568%; passing actions 1/5; directions 0/2; seeds 0/3; supported-relation coverage 20.21%.
- C3: Spearman 0.031537; AP lifts +0.023191/+0.002367; sign AUROC 0.521059; top-10% U -0.003140; P+/P- 32.593%/31.634%; passing actions 0/5; directions 0/2; seeds 1/3; supported-relation coverage 13.33%.
- C4: Spearman 0.043807; AP lifts +0.020290/+0.004045; sign AUROC 0.529247; top-10% U -0.001808; P+/P- 31.579%/27.580%; passing actions 2/5; directions 1/2; seeds 1/3; supported-relation coverage 11.65%.

## Core audit answers

1. Structural context: C1 mean sign-AUROC increment is -0.001963; interpret together with its gate row above.
2. Modality context: C2 mean sign-AUROC increment is -0.000458; activity lift is reported separately in `context_increments.csv`.
3. Frozen latent context: C3 mean sign-AUROC increment is -0.014488; the strongest mean sign increment is C2.
4. C3 critical-pair top-confidence utility is -0.003140 (nonpositive).
5. C4 critical-pair top-confidence utility is -0.001808; the strongest mean top-decile increment is C1.
6. Beneficial-vs-harmful and nonzero-activity increments are separate columns; the frozen gate uses sign separation and actual top-decile utility.
7. Prior-PASS stability: C1=True, C2=True, C3=True, C4=True.
8. Phase 4A GO representations: none.
9. Evidence for Context-Conditioned Conservative Policy: insufficient; policy remains prohibited.
10. Frozen conclusion: the frozen contextual families did not clear the information-sufficiency gates; stop feature expansion and treat an answer-agnostic identifiability ceiling as the active explanation.

## Integrity audit

- TEST rows accessed: **0**; TEST commands: **0**; policy evaluations: **0**.
- Expert retraining: **0**; checkpoint reselection: **0**.
- AACPI V2, Phase 3A results, alpha0, action grid, utility target, and historical feature contracts modified: **no**.
- Structural/modality statistics are TRAIN-only; latent states are frozen and target-independent.
- Fold-fitted PCA and scaling use current training groups only; outer original-triple leakage is zero.
- C0 reproduces Phase 3A R3 within the frozen `1e-18` CSV round-trip tolerance; all source and generated artifacts are hash-auditable.
