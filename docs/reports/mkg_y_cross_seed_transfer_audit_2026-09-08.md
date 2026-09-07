# MKG-Y Y-E4 Cross-Seed Transfer and LOSO Stability Audit

Date: 2026-09-08

## Outcome

**Y_E4_CROSS_SEED_REPLICATION_REPORTED** (descriptive external replication; no progression gate or route selection).

Direct transfer applies each source seed's deterministic Oracle alpha to an independently trained target seed without target reselection or gain clipping. LOSO chooses from two training seeds and evaluates once on the held-out seed. Both are stability diagnostics, not deployable inference-time policies.

## Pair-level results

| Pair | Raw Oracle | Cross-seed transfer (95% CI) | Transfer recovery | LOSO (95% CI) | LOSO recovery | Exact alpha agreement | Direction agreement | B / Z / H | Frozen X4 OOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | +0.036846 | +0.006802 [+0.005731, +0.007896] | 18.5% | +0.009381 [+0.007907, +0.010906] | 25.5% | 35.5% | 57.0% | 58.3% / 10.2% / 31.5% | +0.003705 |
| MKG-Y / M-Hyper + AdaMF-MAT | +0.044070 | +0.008830 [+0.007510, +0.010189] | 20.0% | +0.011849 [+0.010073, +0.013676] | 26.9% | 33.5% | 56.5% | 59.0% / 11.3% / 29.7% | +0.006773 |
| MKG-Y / NativE + AdaMF-MAT | +0.037844 | +0.002640 [+0.001851, +0.003484] | 7.0% | +0.003892 [+0.002626, +0.005188] | 10.3% | 34.8% | 47.8% | 52.3% / 12.8% / 34.9% | -0.000441 |

## Interpretation

Cross-seed transfer is significantly positive in 3/3 pairs; LOSO stable headroom is significantly positive in 3/3 pairs. Median transfer recovery is 18.5%, and median LOSO recovery is 25.5%.

The two M-Hyper pairs retain 18.5–20.0% of Raw Oracle under direct seed transfer and 25.5–26.9% under LOSO. NativE + AdaMF-MAT is much more fragile: direct transfer retains 7.0%, LOSO retains 10.3%, and its frozen X4 OOF gain remains negative. Thus cross-seed stability is detectable but strongly pair-dependent and does not by itself imply inference-time observability.

The gap from Raw Oracle to direct cross-seed transfer quantifies seed-specific action fragility. LOSO is a less restrictive all-but-one-seed stability diagnostic and must not be interpreted as an observable query policy. Frozen X4 remains the strict-OOF inference-time observable probe and uses its fold-specific Global baseline.

## Ordered seed transfers

| Pair | 1→2 | 1→3 | 2→1 | 2→3 | 3→1 | 3→2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | +0.005548 | +0.006424 | +0.008006 | +0.007755 | +0.005984 | +0.007095 |
| MKG-Y / M-Hyper + AdaMF-MAT | +0.007723 | +0.007862 | +0.008767 | +0.009409 | +0.010262 | +0.008955 |
| MKG-Y / NativE + AdaMF-MAT | +0.000893 | +0.004282 | +0.003082 | +0.002564 | +0.003006 | +0.002014 |

## Figures

1. `figure1_headroom_funnel.svg` — Raw Oracle → direct cross-seed transfer → LOSO → frozen X4.
2. `figure2_cross_seed_transfer_matrix.svg` — pair-specific 3×3 seed transfer matrices.
3. `figure3_transfer_recovery_forest.svg` — direct-transfer recovery with clustered intervals.
4. `figure4_stability_vs_transfer.svg` — direction agreement versus transfer recovery.

## Operational audit

- TEST access = 0
- checkpoint inference/retraining/reselection = 0
- target-seed action reselection = 0
- negative-gain clipping = 0
- new selector/feature/representation = 0
- original-triple clustered bootstrap = 10,000 replicates
- all direct source and output hashes recorded = yes

MKG-Y TEST remains locked. Y-E5 was not started by this audit.
