# Experiment 4 — Cross-Seed Transferable Complementarity Audit

Date: 2026-09-07
Split: DEV only
Frozen prior route: `ROUTE_C_LIMITS`

## Outcome

Frozen classification: **E4_INTERMEDIATE**.

The audit transfers each source seed's deterministic Oracle alpha directly to the other independent seeds; target-seed reselection and negative-utility clipping are absent. LOSO is reported only as a cross-seed stability diagnostic, not an inference-time policy.

## Interpretation

A statistically detectable cross-seed component exists in all six pairs, but it is partial: Transfer Recovery ranges from 5.9% to 23.9%, with a median of 17.1%, and no pair reaches the frozen 25% threshold. LOSO recovery is larger (median 22.8%) but also remains well below the Raw Oracle ceiling.

The two NativE+AdaMF pairs show the weakest transfer recovery, especially DB15K. Thus the evidence rejects both extremes: complementarity is not wholly seed-specific because every transfer CI excludes zero, but the transferable component is not large enough for `E4_SUBSTANTIALLY_TRANSFERABLE`. Most Raw Oracle headroom disappears once the source-seed action must survive an independent seed.

## Pair-level results

| Pair | Raw Oracle (95% CI) | Cross-seed transfer (95% CI) | Recovery (95% CI) | LOSO stable (95% CI) | LOSO recovery | Exact alpha | Direction agreement | B / Z / H | Frozen X4 OOF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MKG-W / M-Hyper+NativE | +0.046625 [+0.044381, +0.049000] | +0.010809 [+0.009589, +0.012084] | 23.2% [21.2%, 25.2%] | +0.014542 [+0.012927, +0.016206] | 31.2% | 35.8% | 63.0% | 62.0% / 11.9% / 26.1% | +0.007339 |
| MKG-W / M-Hyper+AdaMF | +0.040841 [+0.038547, +0.043245] | +0.009750 [+0.008425, +0.011125] | 23.9% [21.3%, 26.3%] | +0.012267 [+0.010617, +0.014009] | 30.0% | 38.9% | 71.3% | 67.4% / 7.6% / 25.0% | +0.005909 |
| MKG-W / NativE+AdaMF | +0.047645 [+0.045526, +0.049912] | +0.005643 [+0.004612, +0.006741] | 11.8% [9.9%, 13.9%] | +0.007020 [+0.005593, +0.008532] | 14.7% | 34.3% | 49.6% | 53.6% / 15.6% / 30.8% | +0.001039 |
| DB15K / M-Hyper+NativE | +0.049579 [+0.047591, +0.051630] | +0.009275 [+0.008112, +0.010470] | 18.7% [16.8%, 20.5%] | +0.012410 [+0.010935, +0.013919] | 25.0% | 38.0% | 62.4% | 55.6% / 12.6% / 31.9% | +0.002125 |
| DB15K / M-Hyper+AdaMF | +0.045453 [+0.043644, +0.047300] | +0.007034 [+0.006029, +0.008067] | 15.5% [13.6%, 17.3%] | +0.009324 [+0.008034, +0.010657] | 20.5% | 38.2% | 63.4% | 56.9% / 11.2% / 31.9% | +0.000924 |
| DB15K / NativE+AdaMF | +0.057545 [+0.055882, +0.059238] | +0.003387 [+0.002491, +0.004271] | 5.9% [4.4%, 7.3%] | +0.004554 [+0.003341, +0.005763] | 7.9% | 27.4% | 50.5% | 55.5% / 15.5% / 29.0% | +0.001564 |

Frozen X4 OOF is copied from Experiment 2 and therefore remains relative to its fold-specific outer-train Global baseline. The first three funnel stages use the unchanged full-DEV Experiment 1 alpha0. This baseline distinction is retained rather than silently redefining the frozen X4 probe.

## Ordered seed transfers

| Pair | 1→2 | 1→3 | 2→1 | 2→3 | 3→1 | 3→2 |
|---|---:|---:|---:|---:|---:|---:|
| MKG-W / M-Hyper+NativE | +0.010802 | +0.011385 | +0.011301 | +0.010402 | +0.011329 | +0.009632 |
| MKG-W / M-Hyper+AdaMF | +0.009377 | +0.010245 | +0.009658 | +0.010197 | +0.009103 | +0.009919 |
| MKG-W / NativE+AdaMF | +0.004903 | +0.007025 | +0.005990 | +0.005433 | +0.006374 | +0.004132 |
| DB15K / M-Hyper+NativE | +0.009412 | +0.009910 | +0.008749 | +0.009208 | +0.009206 | +0.009167 |
| DB15K / M-Hyper+AdaMF | +0.007990 | +0.007105 | +0.006322 | +0.006261 | +0.006330 | +0.008194 |
| DB15K / NativE+AdaMF | +0.004063 | +0.003030 | +0.004708 | +0.002847 | +0.003534 | +0.002141 |

## Agreement and denominator checks

Exact alpha agreement requires all three seed actions to match. Direction agreement maps actions to LEFT/ANCHOR/RIGHT relative to unchanged alpha0. Non-anchor direction agreement and pairwise exact-alpha agreement are retained in `pair_summary.csv`. B/Z/H is conditioned on a strictly positive, non-anchor source Oracle action; the three rates sum to one for every pair.

## Frozen classification gate

- Transfer CI lower > 0: 6/6 pairs; datasets represented: db15k, mkg_w.
- Transfer Recovery >=25%: 0/6 pairs.
- LOSO CI lower > 0: 6/6 pairs.
- Median Transfer Recovery: 17.1%.
- Median LOSO Recovery: 22.8%.
- Transfer not significantly positive: 0/6 pairs.

The classification is evaluated exactly from the preregistered gates. No Experiment 1–3 result or `ROUTE_C_LIMITS` was changed.

## Figures

1. `figure1_headroom_funnel.svg` — Raw Oracle → Cross-seed Transfer → LOSO Stable → frozen X4 OOF gain.
2. `figure2_cross_seed_transfer_matrix.svg` — six shared-scale 3×3 seed matrices.
3. `figure3_transfer_recovery_forest.svg` — transfer recovery and paired clustered-bootstrap CI.
4. `figure4_stability_vs_transfer.svg` — direction agreement versus transfer recovery.

## Integrity audit

- TEST access = 0
- TEST commands = 0
- checkpoint retraining = 0
- checkpoint reselection = 0
- new selector = 0
- new representation = 0
- alpha grid modified = no
- alpha0 modified = no
- Experiment 1 result modified = no
- original-triple bootstrap intact = yes (10000 percentile replicates, seed 20260907)
- all direct source and output hashes recorded = yes (`audit_manifest.json`; manifest self-hash excluded to avoid recursion)
- Experiment 5 started = 0

E4_INTERMEDIATE
