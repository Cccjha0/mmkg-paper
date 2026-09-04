# AACPI V2 Phase 2B OOF Audit

All results are DEV-only. No TEST evaluation or policy evaluation was performed.

## Frozen H1 result

Decision: **NO-GO**.

Pairs passing all four signals: 3/6 (required: 4/6).

Both NativE + AdaMF-MAT pairs pass: false (required: true).

| Dataset | Pair | Spearman | Positive AP lift | Harmful AP lift | Highest 10% actual mean U | Pair pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| mkg_w | mkgw_mhyper_native | 0.213460 | +0.090125 | +0.085011 | +0.005183 | PASS |
| mkg_w | mkgw_mhyper_adamf | 0.142492 | +0.091109 | +0.002929 | +0.000888 | PASS |
| mkg_w | mkgw_native_adamf | 0.023478 | +0.005774 | -0.004458 | -0.003081 | FAIL |
| db15k | db15k_mhyper_native | 0.039591 | +0.019897 | +0.001052 | -0.000859 | FAIL |
| db15k | db15k_mhyper_adamf | 0.083701 | +0.068424 | -0.002902 | -0.004243 | FAIL |
| db15k | db15k_native_adamf | 0.106047 | +0.067160 | +0.009870 | +0.002837 | PASS |

## Integrity checks

- Utility and frozen search-space SHA-256 values match every run audit.
- Every prediction row matches its frozen utility row and actual advantage.
- Every original triple and query belongs to exactly one outer fold.
- Every query-action row has one finite outer-fold OOF prediction.
- Each selected outer-fold configuration reproduces the frozen inner-CV selection rule.
- Reference-action advantages and RR values satisfy the anchor identities.

## Interpretation

DB15K NativE + AdaMF-MAT passes all four signals, but MKG-W NativE + AdaMF-MAT is near random for sign discrimination: harmful AP is below prevalence and its highest predicted-advantage bucket has negative actual mean advantage. The frozen H1 gate therefore fails.

Under the preregistered response, Phase 2C and conservative policy work do not proceed from this run. The next research question is identifiability with the frozen 13 score-geometry features; MLP capacity and the action grid must not be expanded in response to this result.
