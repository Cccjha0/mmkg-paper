# Protocol Audit Sanity Summary

Entity degree is computed from training triples only. Test triples are used only to group target entities by prediction-side regime.

| Regime | #Queries | Mean degree | Median degree | Q25 | Q75 | Mean log-degree |
|---|---:|---:|---:|---:|---:|---:|
| `head_has_img` | 7048 | 10.86 | 11.00 | 8.00 | 13.00 | 2.42 |
| `head_no_img` | 2952 | 11.54 | 11.00 | 8.00 | 14.00 | 2.42 |
| `tail_no_img` | 10000 | 3169.69 | 1517.00 | 274.00 | 3368.00 | 6.82 |

This sanity check does not eliminate structural confounding. It records the most direct degree alternative so the role--modality interpretation remains explicitly protocol-aware.
