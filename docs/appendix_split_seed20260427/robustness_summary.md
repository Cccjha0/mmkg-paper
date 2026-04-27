# Appendix Robustness Summary: seed20260427

Purpose: appendix robustness check only. The split was selected by split statistics, not model results.

| Method | Level | Feature type | MRR | Delta vs Residual | Delta vs E5 |
|---|---|---|---:|---:|---:|
| Residual-only | fixed | structural | 0.2944 +/- 0.0008 | -- | -- |
| Gate-only | fixed | fusion | 0.1718 +/- 0.0032 | -0.1226 | -- |
| E5 regression clean router | query | strict clean | 0.2992 | 0.0048 | -- |
| CA-S2 score-aware candidate router | candidate | score-aware | 0.3134 +/- 0.0015 | 0.0190 | 0.0142 |

## Trend Check

- Role-modality asymmetry holds: `True`.
- Residual-only remains above Gate-only overall: `True`.
- E5 gives a modest gain over Residual-only: `True`.
- CA-S2 remains above E5: `True`.

## Interpretation

The appendix split preserves the same qualitative ordering as the main result: CA-S2 score-aware candidate routing remains strongest, E5 improves modestly over the structural baseline, and Gate-only remains globally unstable under the role-modality asymmetric protocol.
