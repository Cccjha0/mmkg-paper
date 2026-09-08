# Complementarity Closure — One-Time Frozen TEST Audit

This report applies only the metrics and commands frozen in the committed three-dataset DEV closure decision memo. It does not alter any DEV classification, gate, title, or narrative.

The TEST baseline is the single full-DEV-selected alpha0 for each pair. Experiment 2 DEV OOF gains used fold-specific outer-train alpha0 values, so the DEV and TEST gain baselines are related but not numerically identical estimands.

| Pair | Global MRR | Raw | Consensus | 2-of-3 | 3-of-3 | Direct transfer | LOSO | Frozen X4 | Top10 | Q50 | Effective support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MKG-W / M-Hyper + NativE | 0.364305 | 0.050880 | 0.045565 | 0.031852 | 0.019328 | 0.012860 | 0.016688 | +0.009189 | 47.3% | 10.9% | 29.2% |
| MKG-W / M-Hyper + AdaMF-MAT | 0.359721 | 0.045090 | 0.037721 | 0.031459 | 0.020810 | 0.011111 | 0.013776 | +0.007054 | 54.1% | 8.9% | 24.3% |
| MKG-W / NativE + AdaMF-MAT | 0.344365 | 0.050125 | 0.041572 | 0.023320 | 0.011402 | 0.004798 | 0.006112 | +0.000659 | 44.1% | 12.0% | 32.4% |
| DB15K / M-Hyper + NativE | 0.374567 | 0.048813 | 0.040966 | 0.032967 | 0.019140 | 0.009148 | 0.012725 | +0.002935 | 54.0% | 8.9% | 23.3% |
| DB15K / M-Hyper + AdaMF-MAT | 0.374567 | 0.043848 | 0.035541 | 0.028845 | 0.017023 | 0.006687 | 0.009305 | +0.001990 | 55.3% | 8.5% | 22.9% |
| DB15K / NativE + AdaMF-MAT | 0.312761 | 0.057590 | 0.044438 | 0.030084 | 0.014212 | 0.002672 | 0.003744 | +0.002275 | 40.8% | 13.2% | 35.9% |
| MKG-Y / M-Hyper + NativE | 0.375584 | 0.035474 | 0.032133 | 0.017026 | 0.008783 | 0.005274 | 0.007016 | +0.005781 | 56.1% | 8.3% | 23.0% |
| MKG-Y / M-Hyper + AdaMF-MAT | 0.372394 | 0.042098 | 0.038494 | 0.020450 | 0.010573 | 0.006144 | 0.007867 | +0.009842 | 51.4% | 9.7% | 25.9% |
| MKG-Y / NativE + AdaMF-MAT | 0.360187 | 0.043831 | 0.039155 | 0.017224 | 0.004926 | 0.002826 | 0.003938 | -0.001111 | 48.7% | 10.4% | 28.2% |

| Pair | Expert A MRR/H@1/H@3/H@10 | Expert B MRR/H@1/H@3/H@10 | Global MRR/H@1/H@3/H@10 | X4 MRR/H@1/H@3/H@10 |
|---|---:|---:|---:|---:|
| MKG-W / M-Hyper + NativE | 0.359721/0.302020/0.382546/0.469272 | 0.338037/0.242201/0.395609/0.496178 | 0.364305/0.297302/0.393542/0.486313 | 0.373494/0.307791/0.400328/0.496217 |
| MKG-W / M-Hyper + AdaMF-MAT | 0.359721/0.302020/0.382546/0.469272 | 0.296319/0.200944/0.356458/0.448838 | 0.359721/0.302020/0.382546/0.469272 | 0.366775/0.307635/0.389448/0.478903 |
| MKG-W / NativE + AdaMF-MAT | 0.338037/0.242201/0.395609/0.496178 | 0.296319/0.200944/0.356458/0.448838 | 0.344365/0.253510/0.397052/0.497075 | 0.345025/0.254017/0.397871/0.497621 |
| DB15K / M-Hyper + NativE | 0.374567/0.299569/0.409278/0.517337 | 0.312761/0.206810/0.378089/0.499512 | 0.374567/0.299569/0.409278/0.517337 | 0.377502/0.301690/0.413654/0.522083 |
| DB15K / M-Hyper + AdaMF-MAT | 0.374567/0.299569/0.409278/0.517337 | 0.282588/0.158907/0.365263/0.496415 | 0.374567/0.299569/0.409278/0.517337 | 0.376557/0.300882/0.412122/0.520703 |
| DB15K / NativE + AdaMF-MAT | 0.312761/0.206810/0.378089/0.499512 | 0.282588/0.158907/0.365263/0.496415 | 0.312761/0.206810/0.378089/0.499512 | 0.315037/0.208308/0.380748/0.503232 |
| MKG-Y / M-Hyper + NativE | 0.378813/0.345851/0.396545/0.435223 | 0.340651/0.256290/0.404118/0.458756 | 0.375584/0.323132/0.406309/0.458818 | 0.381365/0.335774/0.406309/0.456878 |
| MKG-Y / M-Hyper + AdaMF-MAT | 0.378813/0.345851/0.396545/0.435223 | 0.339117/0.251721/0.404118/0.460571 | 0.372394/0.314432/0.407060/0.464263 | 0.382236/0.336025/0.406058/0.461071 |
| MKG-Y / NativE + AdaMF-MAT | 0.340651/0.256290/0.404118/0.458756 | 0.339117/0.251721/0.404118/0.460571 | 0.360187/0.290650/0.406559/0.463825 | 0.359076/0.288397/0.406309/0.463888 |

Raw Oracle and all-seed consensus/stable quantities remain ex-post upper diagnostics. Direct transfer and LOSO are seed-stability diagnostics. Frozen X4 is the only inference-time observable probe in this table.

- TEST access: one-time, after committed decision memo
- checkpoint reselection/retraining: 0
- feature/selector/grid/gate changes: 0
- DEV classifications, route, claims, title, and narrative modified: 0

Frozen DEV interpretation retained: `FINAL_SELECTIVE_RARE_OPPORTUNITY`

FROZEN_TEST_VALIDATION_COMPLETE
