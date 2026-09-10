# Paper A Efficiency and Complexity Benchmark

## Status

The benchmark protocol and executable workers are prepared, but no timing numbers are reported from this local environment. PyTorch/CUDA is unavailable here, and the requested benchmark is intended for the A100 host. Empty result tables are deliberate and must not be interpreted as zero cost.

Run on the A100 environment:

```powershell
python scripts/benchmark_paper_a_efficiency.py --device cuda --warmup 3 --repetitions 5 --queries-per-direction 256
```

Add `--include-adamf` for the two M-Hyper + AdaMF-MAT pairs. Set `--queries-per-direction 0` only when full-TEST timing is desired; candidate ranking remains exact and full-entity for either setting.

## Complexity accounting

Let Q be the number of evaluated directional queries, E the entity count, and C_p/C_s the per-candidate costs of the primary/secondary experts. Primary scoring is O(Q E C_p); every dual-expert method is O(Q E (C_p + C_s)). After frozen scores exist, Global normalization and exact mixing/ranking are O(Q E). DynaSemble adds O(Q E) score statistics plus a constant-size 4-16-16-1 selector. Anchored Dynamic adds O(Q E) query geometry and z-normalization plus a constant-size locked logistic combiner. Thus neither adaptive combiner changes the exact full-ranking asymptotic dependence on E; the benchmark measures their different constants separately.

## Measurement boundary

Base scoring includes the frozen evaluator's candidate construction, model forward pass, standard filtered masking, CPU transfer, and endpoint rank. Feature time begins only after both base score matrices exist. DynaSemble uses its released min-max statistics and selector. Anchored Dynamic uses the locked query-geometry schema, classifier, beta, threshold, fallback, and exact alpha grid. The ranking candidate set and filtering protocol are never reduced; only the number of TEST queries timed may be sampled, and that count is recorded.

Peak GPU is the process allocator peak after the method's required models are loaded, so Primary is not charged for a secondary model. Peak CPU is sampled RSS at stage boundaries and should be treated as an approximate process peak.

No timing result from TEST is used for model, pair, beta, threshold, orientation, or operating-point selection.
