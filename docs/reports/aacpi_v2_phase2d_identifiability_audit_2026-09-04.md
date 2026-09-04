# AACPI V2 Phase 2D Advantage Identifiability Audit

Phase 2B remains a frozen NO-GO. This report uses DEV-only outer-fold OOF predictions and performs no policy evaluation.

## PASS / FAIL comparison

| Dataset | Pair | H1 | Spearman | Positive AP lift | Harmful AP lift | Top-10% actual mean U | Top-10% P(U>0) | Top-10% P(U<0) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mkg_w | mkgw_mhyper_native | PASS | 0.213460 | +0.090125 | +0.085011 | +0.005183 | 41.554% | 19.615% |
| mkg_w | mkgw_mhyper_adamf | PASS | 0.142492 | +0.091109 | +0.002929 | +0.000888 | 55.467% | 15.630% |
| mkg_w | mkgw_native_adamf | FAIL | 0.023478 | +0.005774 | -0.004458 | -0.003081 | 28.469% | 26.941% |
| db15k | db15k_mhyper_native | FAIL | 0.039591 | +0.019897 | +0.001052 | -0.000859 | 34.471% | 16.315% |
| db15k | db15k_mhyper_adamf | FAIL | 0.083701 | +0.068424 | -0.002902 | -0.004243 | 44.391% | 26.398% |
| db15k | db15k_native_adamf | PASS | 0.106047 | +0.067160 | +0.009870 | +0.002837 | 51.491% | 18.408% |

## Same-pair contrast: NativE + AdaMF-MAT

MKG-W is nearly unidentifiable in the pooled OOF result (Spearman 0.0235, positive AP lift +0.0058, harmful AP lift -0.0045). Its top prediction decile remains at mean U -0.0031. DB15K retains useful rank and sign structure (Spearman 0.1060, positive/harmful AP lifts +0.0672/+0.0099) and reaches top-decile mean U +0.0028.

## Calibration and advantage concentration

| Pair | Bottom-decile actual U | Top-decile actual U | Top-bottom change | Monotone improving transitions (of 5) |
| --- | ---: | ---: | ---: | ---: |
| mkgw_mhyper_native | -0.012125 | +0.005183 | +0.017308 | 5 |
| mkgw_mhyper_adamf | -0.026386 | +0.000888 | +0.027274 | 5 |
| mkgw_native_adamf | -0.017969 | -0.003081 | +0.014888 | 3 |
| db15k_mhyper_native | -0.015897 | -0.000859 | +0.015038 | 4 |
| db15k_mhyper_adamf | -0.044143 | -0.004243 | +0.039901 | 5 |
| db15k_native_adamf | -0.027556 | +0.002837 | +0.030393 | 5 |

MKG-W NativE + AdaMF-MAT is not completely unordered: its bottom-to-top change is positive, but improvement stops at a negative utility ceiling and only three of five adjacent bucket transitions improve. Its top decile is therefore weakly ranked but unsafe for intervention. DB15K for the same pair reaches positive utility and a materially higher positive rate in the top decile.

## Action and regime conditioning

| Pair | Available actions | Passing actions | Action Spearman range | Passing directions (of 2) | Passing seeds (of 3) | Passing-relation query coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| mkgw_mhyper_native | 8 | 0 | 0.0261 to 0.1842 | 2 | 3 | 68.73% |
| mkgw_mhyper_adamf | 4 | 0 | 0.0812 to 0.1997 | 1 | 1 | 18.65% |
| mkgw_native_adamf | 5 | 0 | 0.0047 to 0.0322 | 0 | 0 | 16.02% |
| db15k_mhyper_native | 4 | 0 | 0.0010 to 0.0761 | 0 | 1 | 11.49% |
| db15k_mhyper_adamf | 4 | 0 | 0.0113 to 0.1290 | 0 | 0 | 12.00% |
| db15k_native_adamf | 4 | 1 | 0.0658 to 0.1296 | 1 | 3 | 47.16% |

For MKG-W NativE + AdaMF-MAT, all five actions, both directions, and all three seeds fail. Its 12 passing supported relations cover only 16.02% of supported query instances, and its largest relation fails. Relation pockets show heterogeneity but do not rescue the pooled formulation. DB15K NativE + AdaMF-MAT has one passing action, one passing direction, all three seeds passing, and 47.16% coverage by passing supported relations.

## Frozen-feature separability

| Pair | Max positive-vs-rest AUROC lift | Max harmful-vs-rest AUROC lift | Max positive-vs-harmful AUROC lift | Max abs(feature-U Spearman) |
| --- | ---: | ---: | ---: | ---: |
| mkgw_mhyper_native | 0.1886 | 0.2044 | 0.0092 | 0.0278 |
| mkgw_mhyper_adamf | 0.2926 | 0.0819 | 0.1193 | 0.3253 |
| mkgw_native_adamf | 0.1547 | 0.1440 | 0.0352 | 0.0483 |
| db15k_mhyper_native | 0.2367 | 0.1960 | 0.0727 | 0.1496 |
| db15k_mhyper_adamf | 0.2451 | 0.1406 | 0.0818 | 0.1784 |
| db15k_native_adamf | 0.1201 | 0.1185 | 0.0857 | 0.1557 |

MKG-W NativE + AdaMF-MAT has apparently strong positive-versus-rest and harmful-versus-rest single-feature separation, but the same features often move positive and harmful rows together away from the large zero plateau. Once zero rows are removed, its best positive-versus-harmful AUROC lift is only 0.0352; DB15K reaches 0.0857. The maximum absolute feature/continuous-U Spearman likewise differs by more than threefold (0.0483 versus 0.1557). The MKG-W representation mainly identifies utility activity, not the sign and magnitude required for safe correction.

## Winner/action mismatch

| Dataset | Pair | Anchor-optimal | Endpoint predicts deviation but anchor is best | Opposite direction, excluding ties and anchor | Direction agreement, excluding ties and anchor |
| --- | --- | ---: | ---: | ---: | ---: |
| mkg_w | mkgw_mhyper_native | 39.203% | 21.460% | 10.750% | 89.250% |
| mkg_w | mkgw_mhyper_adamf | 52.007% | 40.868% | 29.457% | 70.543% |
| mkg_w | mkgw_native_adamf | 41.854% | 28.901% | 17.600% | 82.400% |
| db15k | db15k_mhyper_native | 60.395% | 50.323% | 35.609% | 64.391% |
| db15k | db15k_mhyper_adamf | 59.190% | 51.335% | 33.939% | 66.061% |
| db15k | db15k_native_adamf | 53.564% | 44.388% | 22.186% | 77.814% |

For MKG-W NativE + AdaMF-MAT, the opposite-direction rate is 17.60%; for DB15K it is 22.19%. Endpoint preference also predicts a deviation on many queries whose true local optimum is the anchor. This directly supports the claim that expert endpoint preference is not equivalent to beneficial mixture correction.

## Available versus identifiable complementarity

| Dataset | Pair | Local headroom | Opportunity rate | Positive AP lift | Harmful AP lift | Top-10% U | Historical Anchored DEV delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mkg_w | mkgw_mhyper_native | 0.027552 | 60.797% | +0.090125 | +0.085011 | +0.005183 | +0.006617 |
| mkg_w | mkgw_mhyper_adamf | 0.010751 | 47.993% | +0.091109 | +0.002929 | +0.000888 | +0.002382 |
| mkg_w | mkgw_native_adamf | 0.029620 | 58.146% | +0.005774 | -0.004458 | -0.003081 | -0.000441 |
| db15k | db15k_mhyper_native | 0.014411 | 39.605% | +0.019897 | +0.001052 | -0.000859 | +0.001300 |
| db15k | db15k_mhyper_adamf | 0.014677 | 40.810% | +0.068424 | -0.002902 | -0.004243 | +0.000450 |
| db15k | db15k_native_adamf | 0.025114 | 46.436% | +0.067160 | +0.009870 | +0.002837 | +0.000293 |

Historical Anchored values are retrospective DEV cross-fit/P3 references. No new policy was run.

## Failure taxonomy

| Pair | H1 | Modes | Feature max positive/harmful AUROC lift | Max abs(feature-U Spearman) | Action Spearman range | Relation-pass coverage | Outcome |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| mkgw_mhyper_native | PASS | — | 0.1886/0.2044 | 0.0278 | 0.1581 | 68.73% | H1 PASS; failure outcome not applicable |
| mkgw_mhyper_adamf | PASS | — | 0.2926/0.0819 | 0.3253 | 0.1185 | 18.65% | H1 PASS; failure outcome not applicable |
| mkgw_native_adamf | FAIL | F3 | 0.1547/0.1440 | 0.0483 | 0.0275 | 16.02% | Outcome B — structured heterogeneity bottleneck |
| db15k_mhyper_native | FAIL | F3;F6 | 0.2367/0.1960 | 0.1496 | 0.0751 | 11.49% | Outcome B — structured heterogeneity bottleneck |
| db15k_mhyper_adamf | FAIL | F2;F3;F4;F5;F6 | 0.2451/0.1406 | 0.1784 | 0.1177 | 12.00% | Outcome B — structured heterogeneity bottleneck |
| db15k_native_adamf | PASS | — | 0.1201/0.1185 | 0.1557 | 0.0639 | 47.16% | H1 PASS; failure outcome not applicable |

The preregistered F1 flag is not triggered for MKG-W NativE + AdaMF-MAT because positive-versus-rest and harmful-versus-rest AUROC both exceed its threshold. The positive-versus-harmful diagnostic shows that much of this is plateau/activity separation rather than correction-sign separation. This report retains the frozen F1 classification and records the sign bottleneck as interpretation; it does not retroactively alter the taxonomy rule.

## Research-direction decision

Primary Phase 2D conclusion: **Outcome B — structured heterogeneity bottleneck**.

### 1. Why the same NativE + AdaMF-MAT pair differs across datasets

Both datasets contain substantial available complementarity, so opportunity supply is not the explanation. The difference is identifiability: MKG-W has no passing action, direction, or seed regime, weak direct positive-versus-harmful feature separation, and a negative top-decile utility. DB15K has stable signal across all three seeds, one passing direction and action, much stronger feature/continuous-utility structure, and almost half of supported query instances in passing relation regimes.

### 2. Dominant failure mechanisms

The aggregate failure set is heterogeneous, which supports Outcome B. MKG-W NativE + AdaMF-MAT is the strongest representation/sign-identifiability bottleneck: relation pockets are too narrow to change that assessment. DB15K M-Hyper + AdaMF-MAT is principally F4/F5: it ranks some positive opportunity but its high-score region stays harmful and harmful-action detection is below prevalence. DB15K M-Hyper + NativE shows seed/relation dependence and weak aggregate calibration.

### 3. Endpoint winner versus beneficial mixture correction

Across pairs, 10.75%-35.61% of endpoint-nontie beneficial corrections point opposite the endpoint winner, and 21.46%-51.33% of endpoint-nontie queries are actually anchor-optimal. This is large enough to establish a structural weakness in winner supervision. It is not sufficient by itself to explain every Anchored result: MKG-W M-Hyper + AdaMF-MAT has high mismatch yet historical Anchored DEV gain remains positive. Anchored failure is best explained by winner/action mismatch interacting with weak advantage identifiability.

### 4. Signal remaining in the frozen geometry

Yes, but it is not uniformly the required signal. Several features identify plateau versus nonzero movement, and DB15K contains sign and magnitude structure that the OOF predictor uses. In MKG-W NativE + AdaMF-MAT, the same summaries mainly indicate that an action may change rank, without reliably indicating whether the change helps or harms. No frozen action, direction, or seed subgroup exposes a stable missed solution.

### 5. Next research hypothesis

The most defensible next step is a separately preregistered richer score representation with query/context augmentation, designed specifically to distinguish beneficial from harmful rank movement. Structured regime information is a secondary hypothesis and should be evaluated as representation, not used to fit ad hoc relation-specific policies. The DB15K success means score-based dynamic combination is not disproved in all settings, but continuing with the same 13 summaries or a larger MLP is not justified.

This task does not implement the next hypothesis. AACPI V2 remains failed and must not be retroactively changed.

## Integrity audit

- TEST rows accessed: **0**.
- TEST evaluation commands executed: **0**.
- Phase 1 utility tables modified: **no**.
- Phase 2B predictor retrained: **no**.
- Frozen 13 features, action grids, and alpha0 values modified: **no**.
- Every prediction diagnostic uses recorded outer-fold OOF predictions.
- Every original triple remains in exactly one outer fold.
- Phase 1 and Phase 2B source hashes and frozen Phase 2B metrics were reproduced exactly.
- Every Phase 2D table and figure is rebuilt by `scripts/analyze_aacpi_advantage_identifiability.py`.
