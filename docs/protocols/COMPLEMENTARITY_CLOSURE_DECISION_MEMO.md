# Three-Dataset Complementarity Closure Decision Memo

Date: 2026-09-08
Status: `FROZEN_THREE_DATASET_DEV_CLOSURE_BEFORE_TEST`

This memo freezes the complete MKG-W, DB15K, and MKG-Y DEV evidence chain before the one-time confirmatory TEST evaluation. The commit containing this memo, the TEST contract, and every TEST script is the only unlock boundary. TEST cannot change the frozen route, claims, gates, title, or narrative.

## 1. Frozen classifications and route

- Core Experiment 3 route: `ROUTE_C_LIMITS`
- Core Experiment 4: `E4_INTERMEDIATE`
- Core Experiment 5: `E5_INTERMEDIATE`
- Core Experiment 6 stability: `E6_STABLE_HIGH`
- Core Experiment 6 gain concentration: `E6_GAIN_HIGHLY_CONCENTRATED`
- Core joint interpretation: `FINAL_SELECTIVE_RARE_OPPORTUNITY`
- Part II high-value selectivity audit: secondary closure diagnostic completed; positive realized gain at frozen 10% coverage in 6/6 core pairs, with clustered CI lower above zero in 5/6; negative transfer remains material and no selector route is reopened.
- MKG-Y Y0/Y1 and Y-E1–Y-E6: completed as descriptive external replication, not as a new gate.
- Three-dataset synthesis: `CROSS_DATASET_CLOSURE_SYNTHESIS_COMPLETE`

The frozen paper route remains a Limits/Boundary paper. No new selector, representation, granularity, or policy tuning is authorized.

## 2. Frozen three-dataset DEV conclusion

Across nine dataset/pairs, Raw Available, 2-of-3 Stable, 3-of-3 Stable, and LOSO headroom all have clustered CI lower above zero in 9/9 pairs. Frozen X4 has positive clustered CI lower in 8/9, but median recovery of available headroom is only 4.3%. The frozen local-identifiability criterion holds in 2/9 pairs. Top 10% of original triples contribute at least half the Oracle gain in 7/9 pairs.

The conclusion is therefore fixed:

> Heterogeneous MMKGC expert complementarity is real and often stable across seeds, but it is only weakly recoverable from the frozen inference-time observable space and is frequently concentrated in a small set of high-value opportunities.

Raw Oracle headroom is not evidence that an unrestricted router can recover it.

## 3. Final research questions

1. How much heterogeneous-expert Oracle headroom is available across three established MMKGC benchmarks?
2. How much survives seed-consensus, direct cross-seed transfer, and LOSO diagnostics?
3. How much is locally represented in the frozen X4 inference-time observable space and recovered by the frozen X4 probe?
4. Is effective Oracle gain diffuse or concentrated in rare high-value original triples?
5. Do the frozen DEV conclusions extrapolate to a single untouched TEST evaluation?

## 4. Frozen primary claims

- The evidence hierarchy is `Raw Available → Seed-Stable / Transferable → Locally Observable → Empirically Deployable`; these quantities are not interchangeable.
- Consensus and 2-/3-of-3 values are all-seed ex-post upper diagnostics. Direct transfer and LOSO are held-out-seed diagnostics. Frozen X4 is the inference-time observable probe.
- Stable opportunities exist across MKG-W, DB15K, and MKG-Y, while observable/deployable recovery remains small and pair-dependent.
- Oracle gain is commonly concentrated in a small fraction of original triples, supporting selective-opportunity and limits framing rather than universal adaptive routing.
- TEST is confirmatory only. A failure to reproduce any DEV pattern will be reported as dataset/pair dependence without post-hoc method or narrative changes.

## 5. Frozen main figures

1. Cross-dataset evidence-chain decomposition.
2. Stability-to-observability comparison.
3. Cumulative Oracle-gain concentration.
4. Frozen high-value selectivity coverage/utilization summary.
5. One-time TEST confirmation table/figure using the same diagnostic roles.

## 6. Frozen TEST policy and baselines

The machine-readable contract is `docs/protocols/COMPLEMENTARITY_CLOSURE_TEST_CONTRACT.json`.

- Datasets: MKG-W, DB15K, MKG-Y.
- Expert pairs: M-Hyper+NativE, M-Hyper+AdaMF-MAT, NativE+AdaMF-MAT on each dataset.
- Checkpoints: the exact three accepted seed-1/2/3 checkpoints listed and hashed in the contract.
- Directions: head and tail.
- Filtering: exact filtered ranking with TRAIN+DEV+TEST truth facts, retaining the current target.
- Score normalization: `query_zscore`.
- Alpha grid: `{0.00,0.05,...,1.00}`.
- TEST Global baseline: one full-DEV-selected alpha0 per pair: `0.60,1.00,0.95,1.00,1.00,1.00,0.35,0.35,0.55` in contract pair order.
- Bootstrap: 10,000 original-triple clustered replicates; each sampled triple retains both directions, all seeds, and all actions.

Experiment 2 DEV OOF gain used a fold-specific outer-train alpha0. TEST uses the single full-DEV alpha0. These values must be annotated as different baseline estimands and must not be presented as a numerically homogeneous policy funnel.

## 7. Frozen final X4 fit/application

For every pair, use the modal `(learner, hyperparameter configuration)` among the five already-frozen nested-OOF outer-fold selections. A vote tie is broken by higher mean inner probe gain among voting folds, then frozen learner order, then frozen hyperparameter order. The selected configuration is explicitly stored in the TEST contract.

Refit that configuration once on all DEV query-action rows with `U(q,alpha)=RR(q,alpha)-RR(q,alpha0_full_DEV)`. Every X4 action includes `alpha`, `delta_alpha`, and `abs_delta_alpha`. Apply it once to answer-agnostic TEST X4 features. The Global action is present and its predicted advantage is fixed to zero. No TEST label participates in fitting, normalization, learner/configuration selection, or action selection.

This is the frozen empirical X4 probe, not a newly developed policy.

## 8. Frozen TEST metrics

For every pair report:

- Expert A/B and frozen Global MRR, Hits@1, Hits@3, Hits@10.
- Raw Oracle, all-seed Consensus, 2-of-3 Stable, 3-of-3 Stable headroom and recovery.
- Direct cross-seed transfer and LOSO headroom/recovery.
- Frozen X4 TEST MRR, gain versus the full-DEV-alpha0 Global baseline, clustered CI, recovery, changed rate, positive transfer rate, and negative transfer rate.
- Stable-opportunity rates.
- Top1/5/10/20 Oracle-gain shares, Q50, Gini, and Effective Support.

These are descriptive TEST confirmations. No DEV classification gate is recomputed or replaced.

## 9. One-time TEST commands

Only after the freezing commit is pushed, run exactly:

```powershell
Set-Location G:\mmkg-project-research

.\scripts\run_complementarity_closure_test.ps1 -Mode Preflight -Python python

.\scripts\run_complementarity_closure_test.ps1 -Mode Evaluate -Python python -Device cuda

.\scripts\run_complementarity_closure_test.ps1 -Mode Analyze -Python python
```

`Preflight` verifies committed protocol/code, every frozen DEV source, all 27 checkpoint hashes, the nine alpha0 values, and the nine final X4 configurations without opening TEST. `Evaluate` is the only high-compute stage and is intended for the server GPU.

## 10. Forbidden post-TEST changes

- new feature or representation
- new selector or policy
- action-grid or normalization change
- gate or metric redefinition
- TEST-driven threshold, coverage, learner, or hyperparameter selection
- title, claims, route, or narrative rewritten in response to TEST
- checkpoint retraining or reselection
- returning to DEV for tuning
- rerunning a completed TEST pair after inspecting its result

An interrupted/incomplete operational run may only resume the already frozen computation. Any repair must preserve estimands, inputs, checkpoints, features, action grid, and configuration and must be documented.

Frozen route: `ROUTE_C_LIMITS`
Frozen interpretation: `FINAL_SELECTIVE_RARE_OPPORTUNITY`

TEST_STATUS_READY_FOR_ONE_TIME_CONFIRMATORY_RUN
