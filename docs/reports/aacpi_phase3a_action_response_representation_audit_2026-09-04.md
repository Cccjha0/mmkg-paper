# AACPI Phase 3A Action-Response Representation Audit

**Frozen decision:** NO-GO

This report uses DEV-only outer-fold OOF predictions. It evaluates representation recovery and runs no policy.

## Pair results

| Dataset | Pair | Rep | H1 | Spearman | Positive AP lift | Harmful AP lift | Pos-vs-harm AUROC | Top-10% U | Top-10% P+ | Top-10% P- |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| db15k | db15k_mhyper_adamf | R0 | FAIL | 0.083701 | +0.068424 | -0.002902 | 0.536759 | -0.004243 | 44.391% | 26.398% |
| db15k | db15k_mhyper_adamf | R1 | FAIL | 0.106361 | +0.086073 | +0.002946 | 0.551619 | -0.001979 | 45.679% | 27.507% |
| db15k | db15k_mhyper_adamf | R2 | FAIL | 0.084028 | +0.079272 | +0.003231 | 0.535142 | -0.003305 | 47.778% | 27.513% |
| db15k | db15k_mhyper_adamf | R3 | FAIL | 0.093226 | +0.076024 | +0.005713 | 0.547582 | -0.002274 | 45.995% | 23.116% |
| db15k | db15k_mhyper_native | R0 | FAIL | 0.039591 | +0.019897 | +0.001052 | 0.527223 | -0.000859 | 34.471% | 16.315% |
| db15k | db15k_mhyper_native | R1 | FAIL | 0.093034 | +0.066904 | -0.002894 | 0.554540 | +0.002472 | 43.654% | 18.435% |
| db15k | db15k_mhyper_native | R2 | FAIL | 0.044034 | +0.064850 | -0.016326 | 0.502605 | +0.002659 | 45.616% | 26.540% |
| db15k | db15k_mhyper_native | R3 | FAIL | 0.071288 | +0.062933 | -0.003518 | 0.537141 | +0.002421 | 43.938% | 19.539% |
| db15k | db15k_native_adamf | R0 | PASS | 0.106047 | +0.067160 | +0.009870 | 0.562017 | +0.002837 | 51.491% | 18.408% |
| db15k | db15k_native_adamf | R1 | FAIL | 0.088571 | +0.088767 | -0.001534 | 0.544446 | +0.004220 | 54.237% | 21.732% |
| db15k | db15k_native_adamf | R2 | PASS | 0.090036 | +0.074889 | +0.008745 | 0.550058 | +0.004042 | 53.889% | 19.818% |
| db15k | db15k_native_adamf | R3 | PASS | 0.086563 | +0.068695 | +0.006953 | 0.544957 | +0.006698 | 52.291% | 18.508% |
| mkg_w | mkgw_mhyper_adamf | R0 | PASS | 0.142492 | +0.091109 | +0.002929 | 0.572980 | +0.000888 | 55.467% | 15.630% |
| mkg_w | mkgw_mhyper_adamf | R1 | PASS | 0.162961 | +0.109229 | +0.006630 | 0.584001 | +0.004807 | 57.182% | 15.338% |
| mkg_w | mkgw_mhyper_adamf | R2 | PASS | 0.154500 | +0.121319 | +0.013008 | 0.589130 | +0.004245 | 62.775% | 15.095% |
| mkg_w | mkgw_mhyper_adamf | R3 | PASS | 0.157590 | +0.121008 | +0.010269 | 0.588671 | +0.005313 | 61.859% | 14.958% |
| mkg_w | mkgw_mhyper_native | R0 | PASS | 0.213460 | +0.090125 | +0.085011 | 0.655916 | +0.005183 | 41.554% | 19.615% |
| mkg_w | mkgw_mhyper_native | R1 | PASS | 0.224710 | +0.100854 | +0.108433 | 0.660680 | +0.004085 | 44.560% | 21.851% |
| mkg_w | mkgw_mhyper_native | R2 | PASS | 0.229574 | +0.131890 | +0.099255 | 0.663717 | +0.007935 | 50.387% | 23.795% |
| mkg_w | mkgw_mhyper_native | R3 | PASS | 0.220586 | +0.107881 | +0.108481 | 0.658203 | +0.006548 | 46.879% | 22.782% |
| mkg_w | mkgw_native_adamf | R0 | FAIL | 0.023478 | +0.005774 | -0.004458 | 0.515313 | -0.003081 | 28.469% | 26.941% |
| mkg_w | mkgw_native_adamf | R1 | FAIL | 0.030902 | +0.013602 | +0.001711 | 0.520417 | -0.003784 | 30.168% | 29.568% |
| mkg_w | mkgw_native_adamf | R2 | FAIL | 0.035345 | +0.016175 | +0.007336 | 0.523810 | -0.003953 | 31.977% | 29.872% |
| mkg_w | mkgw_native_adamf | R3 | FAIL | 0.040870 | +0.015106 | +0.018078 | 0.527146 | -0.002800 | 30.948% | 29.646% |

## Frozen gates

| Rep | H1 pairs | Prior FAIL improved | MKG-W Native improved | Primary | Recovery | Stability | Decision |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| R1 | 2/6 | 0/3 | False | False | False | True | NO-GO |
| R2 | 3/6 | 0/3 | False | False | False | True | NO-GO |
| R3 | 3/6 | 0/3 | False | False | False | True | NO-GO |

## MKG-W / NativE + AdaMF-MAT falsification pair

- R0: Spearman 0.023478; positive/harmful AP lift +0.005774/-0.004458; positive-vs-harmful AUROC 0.515313; top-decile U -0.003081.
- R1: Spearman 0.030902; positive/harmful AP lift +0.013602/+0.001711; positive-vs-harmful AUROC 0.520417; top-decile U -0.003784.
- R2: Spearman 0.035345; positive/harmful AP lift +0.016175/+0.007336; positive-vs-harmful AUROC 0.523810; top-decile U -0.003953.
- R3: Spearman 0.040870; positive/harmful AP lift +0.015106/+0.018078; positive-vs-harmful AUROC 0.527146; top-decile U -0.002800.

## Required audit answers

1. R2 clear-improvement pairs: 0/6; its frozen gate result is False.
2. MKG-W / NativE + AdaMF-MAT recovery under R1/R2/R3: R1=no, R2=no, R3=no.
3. By mean positive-vs-harmful AUROC gain, the larger contribution is R1; R1=+0.0076, R2=-0.0010.
4. R3 minus R2 mean sign-AUROC change is +0.0065; this is the independent context increment.
5. Sign separation and activity lift are reported separately; the decision uses sign AUROC/AP and top-decile utility, never activity alone.
6. Prior PASS stability: R1=True, R2=True, R3=True.
7. Phase 3A GO representations: none.
8. Basis for a later conservative-policy phase: insufficient; do not enter policy development.

MKG-W / NativE + AdaMF-MAT supported-relation H1 coverage by representation: R0=16.02%, R1=16.80%, R2=14.01%, R3=22.95%.

## Integrity audit

- TEST rows accessed: **0**.
- TEST evaluation commands: **0**.
- Expert checkpoints modified: **no**.
- Phase 1 utility tables, AACPI V2, action grids, alpha0, and Base13 modified: **no**.
- Every new representation field follows the frozen answer-agnostic contract.
- Every prediction is outer-fold held out; each original triple occurs in one outer fold.
- No Advantage-Greedy or other policy was run.
- Source and output hashes are recorded in the Phase 3A manifests.
