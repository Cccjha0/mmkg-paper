# Cross-Dataset Complementarity Closure Synthesis

Date: 2026-09-08
Split: DEV only
Datasets: MKG-W, DB15K, MKG-Y

## Frozen status

This report synthesizes the already frozen Experiment 1-6 and MKG-Y Y-E1-Y-E6 evidence. It does not define a new gate, rescale the six-pair gates, reselect a route, or develop a policy.

- Core route remains `ROUTE_C_LIMITS`.
- Core closure interpretation remains `FINAL_SELECTIVE_RARE_OPPORTUNITY`.
- MKG-Y remains a descriptive external replication.

## Main evidence chain

| Dataset / pair | Raw available | 2-of-3 stable | LOSO stable | Best group OOF | Frozen X4 OOF | Local signal | Top10 share |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| MKG-W / M-Hyper+NativE | +0.04663 | +0.02868 (61.5%) | +0.01454 (31.2%) | +0.00473 (L3) | +0.00734 (15.7%) | yes | 50.1% |
| MKG-W / M-Hyper+AdaMF | +0.04084 | +0.02818 (69.0%) | +0.01227 (30.0%) | +0.00065 (L4) | +0.00591 (14.5%) | no | 58.1% |
| MKG-W / NativE+AdaMF | +0.04764 | +0.02274 (47.7%) | +0.00702 (14.7%) | +0.00027 (L2) | +0.00104 (2.2%) | no | 45.9% |
| DB15K / M-Hyper+NativE | +0.04958 | +0.03373 (68.0%) | +0.01241 (25.0%) | +0.00106 (L3) | +0.00213 (4.3%) | no | 55.6% |
| DB15K / M-Hyper+AdaMF | +0.04545 | +0.02988 (65.7%) | +0.00932 (20.5%) | +0.00000 (L1) | +0.00092 (2.0%) | no | 55.2% |
| DB15K / NativE+AdaMF | +0.05755 | +0.03057 (53.1%) | +0.00455 (7.9%) | +0.00006 (L2) | +0.00156 (2.7%) | no | 40.7% |
| MKG-Y / M-Hyper+NativE | +0.03685 | +0.01961 (53.2%) | +0.00938 (25.5%) | +0.00082 (L2) | +0.00370 (10.1%) | no | 56.2% |
| MKG-Y / M-Hyper+AdaMF | +0.04407 | +0.02406 (54.6%) | +0.01185 (26.9%) | +0.00492 (L2) | +0.00677 (15.4%) | yes | 51.8% |
| MKG-Y / NativE+AdaMF | +0.03784 | +0.01455 (38.4%) | +0.00389 (10.3%) | +0.00079 (L2) | -0.00044 (-1.2%) | no | 52.9% |

Raw Oracle, all-seed consensus and two-/three-of-three quantities are upper diagnostics. LOSO is a held-out-seed diagnostic. Best-group and X4 values are strict grouped OOF results. These quantities are not interchangeable.

## Cross-dataset summary

| Dataset | Median raw | Median Stable-2 recovery | Median Stable-3 recovery | Median LOSO recovery | Median X4 recovery | X4 CI>0 | Local signal | Top10>=50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | 0.04663 | 61.5% | 36.5% | 30.0% | 14.5% | 3/3 | 1/3 | 2/3 |
| DB15K | 0.04958 | 65.7% | 40.4% | 20.5% | 2.7% | 3/3 | 0/3 | 2/3 |
| MKG-Y | 0.03784 | 53.2% | 26.8% | 25.5% | 10.1% | 2/3 | 1/3 | 3/3 |

Across all nine pairs:

- Raw available headroom has CI lower > 0 for 9/9 pairs.
- 2-of-3 and 3-of-3 stable headroom have CI lower > 0 for 9/9 and 9/9 pairs.
- LOSO headroom has CI lower > 0 for 9/9 pairs.
- Frozen X4 OOF gain has CI lower > 0 for 8/9 pairs; median available-headroom recovery is 4.3%.
- The frozen local-identifiability criterion holds for only 2/9 pairs.
- Top 10% of triples contribute at least half the Oracle gain for 7/9 pairs; Q50 is at most 10% for 7/9.

## Interpretation

The stable layer does not collapse: every pair retains significant two-of-three, three-of-three and LOSO headroom. The sharpest loss occurs between stable/transferable opportunity and observable action structure. X4 produces significant OOF gains in most pairs, but its median recovery is small and local neighborhoods rarely satisfy the frozen action-consistency criterion.

MKG-Y strengthens the external-validity side of this conclusion. Its M-Hyper pairs reproduce positive X4 gains, while MKG-Y NativE+AdaMF has negative X4 gain despite positive stable and LOSO headroom. Across all three datasets, NativE+AdaMF never passes the frozen local-signal criterion. This is evidence for a pair-dependent identifiability boundary, not for a universally recoverable routing target.

The concentration results further limit the claim: effective Oracle gain often comes from a small fraction of original triples. This supports the frozen Limits / selective-rare-opportunity framing. It does not justify reopening selector development or treating the Oracle bound as deployable headroom.

## Paper claim boundary

- Large Oracle headroom is not evidence that a better router can recover all complementarity.
- Seed-stable opportunities exist across all three datasets, but frozen inference-time X4 evidence recovers only a minority of available headroom.
- Local X4 neighborhoods rarely satisfy the frozen action-signal criterion, so action consistency cannot be claimed as generally identifiable.
- Oracle gains are often concentrated in a small original-triple subset, supporting a limits and selective-opportunity framing rather than universal adaptation.

## Figures

1. `figure1_cross_dataset_evidence_chain.svg`
2. `figure2_recovery_heatmap.svg`
3. `figure3_stability_observability.svg`
4. `figure4_concentration_deployability.svg`

## Integrity audit

- TEST access = 0
- checkpoint execution/retraining/reselection = 0
- new selector / representation / policy tuning = 0
- gate or route change = 0
- historical result modification = 0
- all direct source/output hashes recorded = yes
- TEST remains locked
- subsequent stage started = 0

CROSS_DATASET_CLOSURE_SYNTHESIS_COMPLETE
