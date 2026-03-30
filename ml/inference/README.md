# Inference Module

This module turns training artifacts into stable Python inference entry points.
It currently supports checkpoint loading, tail prediction, attribute completion,
multimodal entity inspection, similar-entity retrieval, demo scripts, and basic latency benchmarking.

## Current Status

Implemented:

- runtime loader: [runtime.py](/E:/learn/R&D/mmkg-project/ml/inference/runtime.py)
- predictor interface: [predictor.py](/E:/learn/R&D/mmkg-project/ml/inference/predictor.py)
- shared parsing/formatting helpers: [utils.py](/E:/learn/R&D/mmkg-project/ml/inference/utils.py)
- CLI inference entry: [run_predict.py](/E:/learn/R&D/mmkg-project/ml/inference/scripts/run_predict.py)
- demo script: [demo_openbg_img.py](/E:/learn/R&D/mmkg-project/ml/inference/scripts/demo_openbg_img.py)
- latency benchmark script: [benchmark_inference.py](/E:/learn/R&D/mmkg-project/ml/inference/scripts/benchmark_inference.py)

Supported tasks:

1. Given `(head, relation)`, predict top-k tails.
2. Given an entity, complete attributes over configured attribute relations.
3. Given an entity, inspect multimodal availability and embedding summaries.
4. Given an entity, retrieve similar entities in a chosen representation space.

## Runtime Contract

### Inputs

- Internal inputs use integer ids.
- External inputs accept `ent_XXXXXX` and `rel_XXXX`.
- Legacy run directories are supported. Old config paths such as `cache/...` and `datasets/...` are rewritten to `data/cache/...` and `data/datasets/...` at runtime.

### Outputs

All public predictor methods now return the same top-level structure:

```json
{
  "task": "tail",
  "model": "openbg_img_gate_residual",
  "device": "cpu",
  "inputs": {},
  "results": [],
  "latency_ms": 123.4
}
```

Stable top-level fields:

- `task`
- `model`
- `device`
- `inputs`
- `results`
- `latency_ms` when timing is available

Task names currently used:

- `tail`
- `tail_batch`
- `attr`
- `entity`
- `similar`

## Text and Metadata Resolution

The inference layer automatically loads:

- `*_entity2text.tsv`
- `*_relation2text.tsv`
- `*_entity2text_en.tsv`
- `*_relation2text_en.tsv`

based on the configured training dataset path.

Entity formatting uses zero-padded tokens:

- entities: `ent_007314`
- relations: `rel_0096`

This is required for consistent lookup into dataset text maps.

### Bilingual Output Fields

The predictor now supports bilingual display fields for presentation and UI use.

Current behavior:

- `*_text` remains the compatibility field and currently points to Chinese text
- `*_text_zh` is the Chinese source text
- `*_text_en` is the English display text when an English map exists

This applies to:

- `head_text`, `head_text_zh`, `head_text_en`
- `relation_text`, `relation_text_zh`, `relation_text_en`
- `entity_text`, `entity_text_zh`, `entity_text_en`

English resolution priority:

1. `*_entity2text_en.tsv` / `*_relation2text_en.tsv`
2. safe fallback: if the original text is already ASCII, reuse it for English display

Recommended workflow:

- curate relation English text manually
- generate entity English text offline into `*_entity2text_en.tsv`
- keep model training and internal ids unchanged

## Attribute Completion

`attr` is implemented as repeated tail prediction over a relation set.

Priority order for attribute relation selection:

1. `inference.attribute_relations` in config
2. automatic fallback: top frequent relations from training data

Current OpenBG-IMG experiment configs already include explicit `attribute_relations`.

## Multimodal Entity View

`get_entity_multimodal()` currently returns:

- entity id and token
- entity text in bilingual fields
- text/image embedding availability
- whether an image exists
- inferred `image_path` when available
- available representation spaces
- per-space embedding summary
- fused summary when available
- gate summary when available

## Similar Entity Query

Supported spaces:

- `text`
- `image`
- `fused`
- `entity_repr`

Current implementation uses exact retrieval, chunked over all entities.
Approximate indexing is not implemented yet.

## Scripts

### Run Inference

```powershell
python ml/inference/scripts/run_predict.py `
  --run_dir ml/artifacts/outputs/openbg_img_gated_vec_res_rel/<timestamp>_seed1 `
  --task tail `
  --head ent_007314 `
  --relation rel_0096 `
  --topk 5 `
  --device cpu
```

### Run Demo

```powershell
python ml/inference/scripts/demo_openbg_img.py `
  --run_dir ml/artifacts/outputs/openbg_img_gated_vec_res_rel/<timestamp>_seed1 `
  --entity ent_007314 `
  --relation rel_0096 `
  --topk 5 `
  --device cpu
```

### Run Benchmark

```powershell
$env:CONDA_NO_PLUGINS='true'
$env:PYTHONIOENCODING='utf-8'
conda run -n pytorch_env python ml/inference/scripts/benchmark_inference.py `
  --run_dir ml/artifacts/outputs/openbg_img_gated_vec_res_rel/<timestamp>_seed1 `
  --entity ent_007314 `
  --relation rel_0096 `
  --topk 5 `
  --device cpu `
  --warmup 1 `
  --repeats 2 `
  --batch_size 2
```

## CPU Baseline

Measured on March 12, 2026 with:

- model: `openbg_img_gate_residual`
- run dir: `ml/artifacts/outputs/openbg_img_gated_vec_res_rel/<timestamp>_seed1`
- device: `cpu`
- entity: `ent_007314`
- relation: `rel_0096`
- topk: `5`
- repeats: `2`

Observed mean latency:

- `tail`: about `272 ms`
- `tail_batch` with 2 queries: about `552 ms`
- `attr`: about `1375 ms`
- `entity`: about `5.5 ms`
- `similar`: about `130 ms`

Practical interpretation:

- `entity` and `similar` are already cheap enough for normal synchronous API usage.
- `tail` is acceptable for synchronous usage on CPU.
- `attr` is noticeably slower because it runs multiple tail queries serially.
- `tail_batch` is currently just repeated single-query execution, not a fused batch scorer.

## Current Limitations

- benchmark results are currently documented only for CPU
- text output still shows terminal-encoding issues in the current shell on some Chinese strings
- entity English text depends on `*_entity2text_en.tsv`; if that file is empty, `entity_text_en` will usually be empty
- `tail_batch` is structurally standardized but not computationally optimized
- attribute completion still depends on a manually curated relation list for best quality
- no backend API layer has been added yet
- no ANN index or retrieval cache has been added for large-scale similarity search

## Next Tasks

Recommended next steps:

1. optimize `tail_batch` into a true batched scorer instead of repeated single-query calls
2. measure GPU latency baselines for the same tasks
3. add a thin backend wrapper around the predictor contract
4. fix terminal and API text-encoding presentation end-to-end
5. optionally add approximate retrieval for `similar`

