# Anchored Dynamic: four-pair locked TEST summary

| Dataset | Expert pair | Tier | Global MRR | Query-soft MRR | Δ Query-soft | Anchored MRR | Δ Anchored | 95% CI | Oracle MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | confirmatory | 0.364305 | 0.370506 | +0.006201 | 0.370176 | +0.005872 | [+0.004708, +0.007035] | 0.401902 |
| MKG-W | M-Hyper + AdaMF-MAT | confirmatory | 0.359721 | 0.352470 | -0.007251 | 0.361705 | +0.001984 | [+0.001370, +0.002598] | 0.390732 |
| DB15K | M-Hyper + NativE | secondary replication | 0.374567 | 0.365621 | -0.008946 | 0.375838 | +0.001271 | [+0.000768, +0.001775] | 0.409286 |
| DB15K | M-Hyper + AdaMF-MAT | secondary replication | 0.374567 | 0.347971 | -0.026595 | 0.375281 | +0.000714 | [+0.000339, +0.001089] | 0.405001 |

The confidence interval is a normal 95% interval over paired original-triple cluster means. Each cluster retains all seeds and both prediction directions.
