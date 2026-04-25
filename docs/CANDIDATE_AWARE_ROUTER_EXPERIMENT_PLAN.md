# Candidate-aware / Score-aware Router Experiment Plan

## 0. Document Purpose

This document defines the next-stage experimental plan for improving the current routing component in the MMKG paper project.

The current paper has already established a protocol-aware selective activation framework on OpenBG-IMG. The strongest existing clean router is the **Regression-based clean router**, which achieves approximately **0.2982 MRR**, compared with **Residual-only at 0.2930 MRR** and **Oracle routing at 0.3337 MRR**.

The key question for the next stage is therefore:

> Can a stronger router recover more deployable multimodal gain by moving beyond coarse query-level routing?

The proposed direction is not to simply replace the current clean router with a more complicated classifier. Instead, this plan introduces a more fine-grained routing framework:

> **Candidate-aware soft selective activation with ranking-aligned training.**

The main idea is to move from query-level hard expert selection to candidate-level soft expert weighting, optionally enhanced with score-aware deployable features.

---

## 1. Current Repository Context

The current repository already contains a mature query-level routing pipeline. The existing codebase includes:

- `scripts/build_router_feature_table.py`
- `scripts/run_threshold_scan.py`
- `scripts/run_clean_dual_threshold_scan.py`
- `scripts/run_clean_prior_bucket_policy.py`
- `scripts/run_clean_hybrid_prior_first.py`
- `scripts/run_clean_probability_calibration.py`
- `scripts/train_clean_regression_router.py`
- `scripts/eval_clean_regression_router.py`
- `scripts/train_clean_ordinal_router.py`
- `scripts/eval_clean_ordinal_router.py`
- `router/experiment_utils.py`
- `router/feature_utils.py`
- `router/router_models.py`
- `router/routing_utils.py`

The existing clean router feature sets are query-level feature sets, mainly including:

- direction
- relation id
- relation-level development priors
- observed-side image availability
- observed-side text-image cosine
- observed-side missing-image replacement flag

This means the current clean router makes one decision per query:

```text
use Gate-only
or
use Residual-only
```

This design is clean and deployable, but it is also coarse. The next-stage experiments should therefore preserve the existing query-level clean routing results as baselines, while adding a new candidate-aware routing line.

---

## 2. Core Motivation

### 2.1 Limitation of the Current Query-level Router

The current router predicts a query-level selection variable:

```math
\alpha(q) \in \{0,1\}
```

This means that for each query, the system chooses either:

- the fusion expert, Gate-only; or
- the structural expert, Residual-only.

However, under OpenBG-IMG, the usefulness of multimodal information may vary not only by query, but also by candidate entity.

For the same query, different candidate entities may have different properties:

- some candidates have images;
- some candidates have missing visual modality;
- some candidates have strong text-image consistency;
- some candidates are favored by Gate-only;
- some candidates are favored by Residual-only;
- some relations may be visually sensitive, while others may be structure-dominant.

Therefore, a query-level router may be too coarse to recover the full available gain.

### 2.2 Proposed Direction

The proposed router predicts a candidate-level soft routing weight:

```math
\alpha(q,e) \in [0,1]
```

The final score for candidate entity `e` under query `q` becomes:

```math
s(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e)
```

where:

- `s_f(q,e)` is the Gate-only score;
- `s_s(q,e)` is the Residual-only score;
- `alpha(q,e)` is the candidate-aware routing weight.

This formulation allows the model to use fusion more strongly for candidates where multimodal evidence appears useful, while falling back to the structural expert when fusion appears unreliable.

---

## 3. Experimental Lines

The full next-stage experiment plan contains five lines.

| Line | Name | Purpose | Priority |
|---|---|---|---|
| Line 0 | Reproduction and baseline locking | Reproduce and freeze current routing results | Must do |
| Line 1 | Candidate-aware soft router | Move from query-level selection to candidate-level soft weighting | Highest |
| Line 2 | Score-aware query router | Use deployable expert score-distribution signals | High |
| Line 3 | Ranking-aligned router | Train router with pairwise/listwise ranking objective | High |
| Line 4 | Combined router | Combine candidate-aware, score-aware, and ranking-aligned design | Final stage |

Recommended execution order:

```text
Line 0 -> Line 1 -> Line 2 -> Line 3 -> Line 4
```

Do not start with the most complex combined router. The experiments should identify which factor actually contributes to improvement:

1. candidate-level granularity;
2. score-aware deployable signals;
3. ranking-aligned training objective;
4. their combination.

---

## 4. Line 0: Reproduction and Baseline Locking

### 4.1 Goal

Before adding new experiments, lock the current results used by the paper.

Target baseline values:

| Method | MRR | Role |
|---|---:|---|
| Residual-only | 0.2930 | fixed structural expert |
| Clean rule | 0.2943 | legal rule baseline |
| Naive global clean router | 0.2939 | global-threshold learned router |
| Direction-specific threshold | 0.2974 | structured clean policy |
| Regression-based clean router | 0.2982 | strongest current clean router |
| Oracle routing | 0.3337 | post-hoc upper bound |

### 4.2 Required Output Files

```text
outputs/router/eval/clean/baseline_locked_summary.csv
outputs/router/eval/clean/baseline_locked_query_rows.csv
outputs/router/eval/clean/baseline_locked_manifest.json
```

Suggested `baseline_locked_manifest.json`:

```json
{
  "dataset": "OpenBG-IMG paper_split",
  "evaluation": "filtered ranking",
  "direction": "both",
  "experts": ["Gate-only", "Residual-only"],
  "residual_mrr": 0.2930,
  "clean_rule_mrr": 0.2943,
  "direction_specific_mrr": 0.2974,
  "regression_clean_router_mrr": 0.2982,
  "oracle_mrr": 0.3337,
  "seeds": [1, 2, 3]
}
```

### 4.3 Stop Rule

If reproduced values differ from the paper values by more than `±0.0005 MRR`, stop and check:

- whether the same split is used;
- whether the same query-level expert outcomes are used;
- whether the official model-comparison line is being mixed with the clean routing line;
- whether seed aggregation is consistent.

---

## 5. Line 1: Candidate-aware Soft Router

## 5.1 Research Question

> Can candidate-aware soft routing recover more deployable multimodal gain than query-level clean routing?

This line is the most important part of the next-stage plan.

---

## 5.2 Candidate-level Scoring Requirement

For each query `q`, export the scores of both experts over candidate entities:

```text
Gate-only:      s_f(q,e)
Residual-only:  s_s(q,e)
```

Because OpenBG-IMG has around 27,910 entities, full candidate score export may be expensive. Therefore, use a two-stage design.

### Stage A: Top-K Candidate Reranking

For each query, export:

```text
topK candidates from Gate-only
topK candidates from Residual-only
correct target entity
union(topK_gate, topK_residual, correct target)
```

Recommended value:

```text
K = 100
```

This produces a manageable candidate set per query and is suitable for fast method validation.

### Stage B: Full-ranking Evaluation

Only if Stage A shows promising improvement, run full-ranking evaluation over all candidate entities.

Full-ranking results are required before making a strong main-paper claim.

---

## 5.3 New Script: `scripts/export_candidate_scores.py`

### Purpose

Export candidate-level scores from Gate-only and Residual-only experts.

### Example Command

```bash
python scripts/export_candidate_scores.py ^
  --gate-run-dir ml/artifacts/outputs/openbg_img_gate_only/seed1 ^
  --residual-run-dir ml/artifacts/outputs/openbg_img_residual_only/seed1 ^
  --split test ^
  --direction both ^
  --top-k 100 ^
  --include-target true ^
  --out-parquet outputs/candidate_router/scores/test_seed1_top100.parquet
```

Run separately for:

```text
seed1
seed2
seed3
```

and for:

```text
dev
test
```

### Output Files

```text
outputs/candidate_router/scores/dev_seed1_top100.parquet
outputs/candidate_router/scores/dev_seed2_top100.parquet
outputs/candidate_router/scores/dev_seed3_top100.parquet
outputs/candidate_router/scores/test_seed1_top100.parquet
outputs/candidate_router/scores/test_seed2_top100.parquet
outputs/candidate_router/scores/test_seed3_top100.parquet
```

### Required Columns

```text
query_id
seed
split
direction
relation_id
head_id
tail_id
observed_entity_id
target_entity_id
candidate_entity_id
is_target
candidate_rank_gate
candidate_rank_residual
score_gate
score_residual
score_diff
score_mean
score_max
in_gate_topk
in_residual_topk
```

### Important Leakage Rule

`is_target` and `target_entity_id` must not be used as router features. They are allowed only for training labels and evaluation.

---

## 5.4 New Script: `scripts/build_candidate_router_table.py`

### Purpose

Build candidate-aware router feature tables from exported candidate scores.

### Inputs

```text
outputs/candidate_router/scores/dev_seed*_top100.parquet
outputs/candidate_router/scores/test_seed*_top100.parquet
cache/text_emb.pt
cache/img_emb.pt
cache/has_img.pt
relation priors
```

The existing `router.feature_utils.load_cache_bundle()` can be reused to load:

```text
text_emb.pt
img_emb.pt
has_img.pt
```

### Outputs

```text
outputs/candidate_router/features/candidate_router_dev_top100.parquet
outputs/candidate_router/features/candidate_router_test_top100.parquet
outputs/candidate_router/features/feature_contract.json
```

---

## 5.5 Feature Design

### A. Query-level Clean Features

Reuse the current clean C4 features:

```text
direction
relation_id
relation_gain_prior
relation_fusion_win_rate
relation_support
relation_is_visual_prior
observed_has_img
observed_text_img_cosine
observed_img_missing_replaced
```

These maintain comparability with the existing query-level clean router.

### B. Candidate-level Clean Features

Add candidate-side modality features:

```text
candidate_has_img
candidate_text_img_cosine
candidate_img_missing_replaced
candidate_text_norm
candidate_img_norm
candidate_is_observed_entity
```

These are considered clean because all candidate entities are visible during ranking.

### C. Score-aware Deployable Features

Add expert score features:

```text
score_gate
score_residual
score_diff
score_mean
score_abs_diff
gate_rank_in_union
residual_rank_in_union
in_gate_topk
in_residual_topk
```

These should be treated as a separate **score-aware deployable line**, not as strict metadata-only clean features.

### D. Forbidden Features

The following fields must not be used as deployable router features:

```text
is_target
target_entity_id
target_has_img
target_regime
rank_gate
rank_residual
rr_gate
rr_residual
rr_gain
fusion_correct_score
struct_correct_score
correct_score
```

They are allowed only for:

```text
training labels
evaluation
post-hoc analysis
oracle upper bound
```

---

## 5.6 Feature Contract

Create `outputs/candidate_router/features/feature_contract.json`:

```json
{
  "strict_clean_features": [
    "direction",
    "relation_id",
    "relation_gain_prior",
    "relation_fusion_win_rate",
    "relation_support",
    "relation_is_visual_prior",
    "observed_has_img",
    "observed_text_img_cosine",
    "observed_img_missing_replaced",
    "candidate_has_img",
    "candidate_text_img_cosine",
    "candidate_img_missing_replaced"
  ],
  "score_aware_features": [
    "score_gate",
    "score_residual",
    "score_diff",
    "score_mean",
    "score_abs_diff",
    "gate_rank_in_union",
    "residual_rank_in_union",
    "in_gate_topk",
    "in_residual_topk"
  ],
  "forbidden_features": [
    "is_target",
    "target_entity_id",
    "target_has_img",
    "target_regime",
    "rank_gate",
    "rank_residual",
    "rr_gate",
    "rr_residual",
    "rr_gain",
    "fusion_correct_score",
    "struct_correct_score",
    "correct_score"
  ]
}
```

---

## 5.7 New Script: `scripts/train_candidate_soft_router.py`

### Purpose

Train a router that outputs candidate-level soft weights:

```math
\alpha(q,e) \in [0,1]
```

### Model Variants

| Variant | Name | Features | Purpose |
|---|---|---|---|
| CA-S1 | candidate-clean-soft | query clean + candidate modality | Test whether candidate modality helps |
| CA-S2 | candidate-score-soft | query clean + score-aware features | Test whether expert score signals help |
| CA-S3 | candidate-full-soft | query clean + candidate modality + score-aware features | Strongest candidate-level version |

### Suggested MLP Architecture

```text
Input features
  -> Linear(hidden=128)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(64)
  -> ReLU
  -> Linear(1)
  -> Sigmoid
  -> alpha(q,e)
```

### Output Files

```text
outputs/candidate_router/models/ca_s1_clean_soft/model.pt
outputs/candidate_router/models/ca_s2_score_soft/model.pt
outputs/candidate_router/models/ca_s3_full_soft/model.pt
outputs/candidate_router/models/*/config.json
outputs/candidate_router/models/*/train_log.csv
```

---

## 5.8 Loss Functions

### 5.8.1 Pointwise BCE Loss

This is the simplest sanity-check objective.

Positive sample:

```text
correct target entity
```

Negative samples:

```text
non-target candidates in union topK
```

Loss:

```math
\mathcal{L}_{BCE}=-y\log p-(1-y)\log(1-p)
```

This is easy to implement but should not be the final main method, because it is not directly aligned with MRR.

---

### 5.8.2 Pairwise Ranking Loss

This is the recommended main objective.

For each query, define:

```text
e+ = correct target entity
e- = negative candidate entity
```

Mixed score:

```math
s(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e)
```

Pairwise loss:

```math
\mathcal{L}_{pairwise}=-\log\sigma(s(q,e^+)-s(q,e^-))
```

Recommended negative sampling ratio:

```text
gate_hard : residual_hard : random = 2 : 2 : 1
```

where:

- `gate_hard` means hard negatives from Gate-only topK;
- `residual_hard` means hard negatives from Residual-only topK;
- `random` means random negatives from the union candidate set.

---

### 5.8.3 Listwise Loss

This is optional and should only be attempted after pairwise routing shows improvement.

For each query, compute softmax over the union topK candidate set:

```math
P(e|q)=\frac{\exp(s(q,e))}{\sum_{e'}\exp(s(q,e'))}
```

Loss:

```math
\mathcal{L}_{listwise}=-\log P(e^+|q)
```

This objective is more ranking-aligned but more computationally expensive.

---

## 5.9 New Script: `scripts/eval_candidate_soft_router.py`

### Purpose

Evaluate candidate-aware soft router by recomputing mixed scores and ranks.

### Evaluation Mode A: Top-K Reranking

Rerank only the union topK candidate set.

Output metrics:

```text
candidate_topk_rerank_mrr
candidate_topk_hits1
candidate_topk_hits3
candidate_topk_hits10
```

This is useful for fast validation, but it is not enough for a final main-paper claim.

### Evaluation Mode B: Full-ranking Evaluation

Rerank all candidate entities.

This is required for the final paper result.

### Output Files

```text
outputs/candidate_router/eval/ca_s1_clean_soft_top100_eval.csv
outputs/candidate_router/eval/ca_s2_score_soft_top100_eval.csv
outputs/candidate_router/eval/ca_s3_full_soft_top100_eval.csv
outputs/candidate_router/eval/ca_s1_clean_soft_full_eval.csv
outputs/candidate_router/eval/ca_s2_score_soft_full_eval.csv
outputs/candidate_router/eval/ca_s3_full_soft_full_eval.csv
```

---

## 5.10 Line 1 Main Result Table

| Method | Routing level | Feature type | Objective | MRR | Delta vs Residual | Delta vs E5 | Oracle gap recovered |
|---|---|---|---|---:|---:|---:|---:|
| Residual-only | fixed | structure | none | 0.2930 | — | — | 0% |
| E5 Regression clean | query | strict clean | delta-RR regression | 0.2982 | +0.0051 | — | about 12.6% |
| CA-S1 | candidate | clean candidate | pairwise | TBD | TBD | TBD | TBD |
| CA-S2 | candidate | score-aware | pairwise | TBD | TBD | TBD | TBD |
| CA-S3 | candidate | clean + score-aware | pairwise | TBD | TBD | TBD | TBD |
| Oracle | query oracle | answer-aware | oracle | 0.3337 | +0.0407 | +0.0355 | 100% |

Oracle gap recovery:

```math
\text{recovered}=\frac{MRR_{method}-MRR_{residual}}{MRR_{oracle}-MRR_{residual}}
```

---

## 6. Line 2: Score-aware Query Router

## 6.1 Research Question

> Are expert score distributions deployable signals for stronger routing?

This line tests whether expert confidence signals can improve routing while still avoiding answer-aware leakage.

---

## 6.2 Motivation

Strict clean metadata-only routing uses only query metadata and observed-side modality indicators. It does not know whether Gate-only or Residual-only is confident for the current query.

At deployment time, however, the system can run both experts and inspect their score distributions without knowing the correct answer.

Possible deployable score-aware features include:

```text
Gate-only top1 score
Gate-only top2 score
Gate-only top1-top2 margin
Gate-only top10 mean
Gate-only top10 standard deviation
Gate-only top100 entropy
Residual-only top1 score
Residual-only top2 score
Residual-only top1-top2 margin
Residual-only top10 mean
Residual-only top10 standard deviation
Residual-only top100 entropy
score_top1_diff
margin_diff
entropy_diff
top10_overlap
top100_overlap
```

---

## 6.3 New Script: `scripts/build_score_aware_query_table.py`

### Inputs

```text
outputs/candidate_router/scores/dev_seed*_top100.parquet
outputs/candidate_router/scores/test_seed*_top100.parquet
```

### Outputs

```text
outputs/candidate_router/features/score_aware_query_dev.parquet
outputs/candidate_router/features/score_aware_query_test.parquet
```

### Required Columns

```text
query_id
seed
direction
relation_id
gate_top1_score
gate_top2_score
gate_margin_1_2
gate_top10_mean
gate_top10_std
gate_entropy_top100
residual_top1_score
residual_top2_score
residual_margin_1_2
residual_top10_mean
residual_top10_std
residual_entropy_top100
score_top1_diff
margin_diff
entropy_diff
top10_overlap
top100_overlap
rr_gate
rr_residual
rr_gain
gain_label_d001
```

The fields `rr_gate`, `rr_residual`, and `rr_gain` are labels/evaluation fields, not deployable features.

---

## 6.4 New Scripts

```text
scripts/train_score_aware_query_router.py
scripts/eval_score_aware_query_router.py
```

### Feature Sets

| Feature Set | Content |
|---|---|
| SA-Q1 | clean C4 only |
| SA-Q2 | score distribution only |
| SA-Q3 | clean C4 + score distribution |
| SA-Q4 | clean C4 + score distribution + direction-specific threshold |

### Models

Start with:

```text
Logistic Regression
XGBoost
MLP
Regression target
```

Do not start with a complex neural router unless these simpler models show clear improvement.

---

## 6.5 Main Table

| Method | Feature type | Routing level | MRR | Delta vs E5 | Interpretation |
|---|---|---|---:|---:|---|
| E5 Regression clean | strict clean | query | 0.2982 | — | current best clean method |
| SA-Q1 | clean C4 | query | TBD | TBD | sanity check |
| SA-Q2 | score-aware only | query | TBD | TBD | score signal utility |
| SA-Q3 | clean + score-aware | query | TBD | TBD | combined deployable signal |
| SA-Q4 | clean + score + structured threshold | query | TBD | TBD | strongest query-level score-aware policy |

---

## 7. Line 3: Ranking-aligned Router

## 7.1 Research Question

> Does directly optimizing a ranking objective improve routing compared with predicting query-level RR gain?

The current regression router predicts:

```math
\Delta(q)=RR_f(q)-RR_s(q)
```

This is better than binary labels, but it is still not directly optimizing the final ranking.

---

## 7.2 Query-level Ranking-aligned Router

Keep query-level soft routing:

```math
\alpha(q) \in [0,1]
```

Final score:

```math
s(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e)
```

Train with pairwise loss:

```math
\mathcal{L}=-\log\sigma(s(q,e^+)-s(q,e^-))
```

This tests whether ranking-aligned training helps even without candidate-level alpha.

---

## 7.3 Candidate-level Ranking-aligned Router

Use:

```math
\alpha(q,e) \in [0,1]
```

and the same pairwise/listwise ranking objective.

This is the main recommended direction.

---

## 7.4 New Scripts

```text
scripts/train_ranking_aligned_router.py
scripts/eval_ranking_aligned_router.py
```

Example command:

```bash
python scripts/train_ranking_aligned_router.py ^
  --train-table outputs/candidate_router/features/candidate_router_dev_top100.parquet ^
  --router-level candidate ^
  --feature-set CA-S3 ^
  --loss pairwise ^
  --negatives-per-query 20 ^
  --epochs 30 ^
  --batch-size 128 ^
  --lr 1e-3 ^
  --out-dir outputs/candidate_router/models/rank_ca_s3_pairwise
```

---

## 7.5 Objective Ablation Table

| Method | Level | Objective | MRR | Delta vs E5 |
|---|---|---|---:|---:|
| E5 Regression | query | delta-RR regression | 0.2982 | — |
| Q-Rank | query | pairwise ranking | TBD | TBD |
| CA-Pointwise | candidate | BCE | TBD | TBD |
| CA-Pairwise | candidate | pairwise ranking | TBD | TBD |
| CA-Listwise | candidate | listwise CE | TBD | TBD |

This table is important because it separates the effect of routing granularity from the effect of training objective.

---

## 8. Line 4: Combined Router

## 8.1 Method

The final combined method uses:

```text
candidate-level alpha
+ score-aware features
+ pairwise/listwise ranking loss
+ optional direction-specific calibration
```

Formal definition:

```math
\alpha(q,e)=\sigma(f_\theta(x_q,x_e,x_{score}))
```

```math
s(q,e)=\alpha(q,e)s_f(q,e)+(1-\alpha(q,e))s_s(q,e)
```

---

## 8.2 Variants

| Variant | Name | Description |
|---|---|---|
| COMB-1 | candidate + score + pairwise | main combined version |
| COMB-2 | candidate + score + listwise | high-cost version |
| COMB-3 | candidate + score + direction calibration | final enhancement |

---

## 8.3 Result Interpretation

| Result Range | Interpretation |
|---|---|
| below 0.2982 | new router does not beat current E5 |
| 0.2982–0.3020 | small improvement; do not overclaim |
| 0.3020–0.3050 | useful improvement; worth reporting as additional result |
| 0.3050–0.3100 | strong enough to update the paper method section |
| above 0.3100 | very valuable; check leakage carefully |
| close to 0.3337 | highly suspicious unless rigorously verified |

---

## 9. Strict Leakage Control

## 9.1 Forbidden Deployable Features

The following must not be used as input features for deployable routers:

```text
is_target
target_entity_id
target_has_img
target_regime
rank_gate
rank_residual
rr_gate
rr_residual
rr_gain
fusion_correct_score
struct_correct_score
correct_score
```

These features can only appear in:

```text
training labels
evaluation files
oracle analysis
post-hoc analysis
```

---

## 9.2 Allowed Candidate-aware Clean Features

```text
candidate_entity_id
candidate_has_img
candidate_text_img_cosine
candidate_img_missing_replaced
candidate_text_norm
candidate_img_norm
direction
relation_id
relation priors from dev
observed_has_img
observed_text_img_cosine
```

Rationale:

Candidate entity metadata is visible during ranking because the model scores all candidate entities.

---

## 9.3 Allowed Score-aware Deployable Features

```text
score_gate(q,e)
score_residual(q,e)
score_diff(q,e)
top-k score margin
score entropy
top-k overlap
```

Rationale:

These features are obtained from model inference outputs and do not require knowing the correct answer.

However, they must be reported as **score-aware deployable routing**, not as strict metadata-only clean routing.

---

## 10. Execution Checklist

## Phase 0: Lock Baselines

```text
[ ] Reproduce Residual-only clean-line MRR
[ ] Reproduce E5 regression router MRR = 0.2982
[ ] Reproduce Oracle routing MRR = 0.3337
[ ] Write baseline_locked_summary.csv
[ ] Write baseline_locked_manifest.json
```

---

## Phase 1: Export Candidate Scores

```text
[ ] Implement scripts/export_candidate_scores.py
[ ] Export dev split Gate-only / Residual-only top100 candidates
[ ] Export test split Gate-only / Residual-only top100 candidates
[ ] Run for seeds 1, 2, and 3
[ ] Ensure every query includes the correct target
[ ] Check average union topK size
[ ] Check score ranges and missing values
```

---

## Phase 2: Build Candidate Feature Tables

```text
[ ] Implement scripts/build_candidate_router_table.py
[ ] Add query-level clean C4 features
[ ] Add candidate modality features
[ ] Add score-aware features
[ ] Write feature_contract.json
[ ] Verify forbidden features are not included in deployable feature sets
```

---

## Phase 3: Train Candidate-aware Soft Routers

```text
[ ] Train CA-S1 with clean candidate features
[ ] Train CA-S2 with score-aware features
[ ] Train CA-S3 with clean + score-aware features
[ ] First train with pointwise BCE as sanity check
[ ] Then train with pairwise ranking loss as main objective
[ ] Save model.pt, config.json, and train_log.csv
```

---

## Phase 4: Top-K Rerank Evaluation

```text
[ ] Evaluate CA-S1 on top100 reranking
[ ] Evaluate CA-S2 on top100 reranking
[ ] Evaluate CA-S3 on top100 reranking
[ ] Compare with Residual-only, E5, and Oracle
[ ] Check subgroup metrics: head_has_img, head_no_img, tail_no_img
```

Stop rule:

```text
If CA-S1/CA-S2/CA-S3 all fail to exceed 0.2982, do not immediately proceed to full-ranking evaluation.
```

Continue rule:

```text
If CA-S3 exceeds 0.3000, proceed to full-ranking evaluation.
If CA-S3 exceeds 0.3050, strongly consider adding the result to the paper.
```

---

## Phase 5: Full-ranking Evaluation

```text
[ ] Compute Gate-only and Residual-only scores for all candidate entities
[ ] Compute alpha(q,e) for all candidate entities
[ ] Compute mixed score for all candidate entities
[ ] Apply filtered ranking
[ ] Compute MRR, Hits@1, Hits@3, Hits@10
[ ] Save full-ranking query rows
```

Required output:

```text
outputs/candidate_router/eval/ca_s3_full_ranking_eval.csv
outputs/candidate_router/eval/ca_s3_full_ranking_query_rows.csv
```

---

## Phase 6: Significance Testing

Compare:

```text
CA-S3 vs Residual-only
CA-S3 vs E5 Regression clean
CA-S3 vs Clean rule
CA-S3 vs Direction-specific threshold
```

Output:

```text
outputs/candidate_router/eval/significance_ca_s3_vs_residual.json
outputs/candidate_router/eval/significance_ca_s3_vs_e5.json
outputs/candidate_router/eval/significance_ca_s3_vs_clean_rule.json
outputs/candidate_router/eval/significance_ca_s3_vs_direction_specific.json
```

Use paired query-level bootstrap, consistent with the current paper's clean routing significance protocol.

---

## 11. Required Paper Tables and Figures

## 11.1 Main Router Comparison Table

| Method | Routing level | Feature type | Objective | MRR | Delta vs Residual | Delta vs E5 | Oracle gap recovered |
|---|---|---|---|---:|---:|---:|---:|
| Residual-only | fixed | structural | none | 0.2930 | — | — | 0% |
| Clean rule | query | strict clean | rule | 0.2943 | +0.0012 | -0.0039 | TBD |
| Direction-specific threshold | query | strict clean | binary threshold | 0.2974 | +0.0044 | -0.0008 | TBD |
| Regression-based clean router | query | strict clean | delta-RR regression | 0.2982 | +0.0051 | — | about 12.6% |
| CA-S1 | candidate | clean candidate | pairwise | TBD | TBD | TBD | TBD |
| CA-S2 | candidate | score-aware | pairwise | TBD | TBD | TBD | TBD |
| CA-S3 | candidate | clean + score-aware | pairwise | TBD | TBD | TBD | TBD |
| Oracle routing | oracle | answer-aware | oracle | 0.3337 | +0.0407 | +0.0355 | 100% |

---

## 11.2 Feature-line Ablation Table

| Method | Query clean | Candidate modality | Expert score | MRR | Interpretation |
|---|---:|---:|---:|---:|---|
| E5 | yes | no | no | 0.2982 | current strict clean best |
| CA-S1 | yes | yes | no | TBD | candidate modality effect |
| CA-S2 | yes | no | yes | TBD | score-aware effect |
| CA-S3 | yes | yes | yes | TBD | combined effect |

---

## 11.3 Objective Ablation Table

| Method | Level | Objective | MRR | Delta vs E5 |
|---|---|---|---:|---:|
| E5 Regression | query | delta-RR regression | 0.2982 | — |
| Q-Rank | query | pairwise ranking | TBD | TBD |
| CA-Pointwise | candidate | BCE | TBD | TBD |
| CA-Pairwise | candidate | pairwise ranking | TBD | TBD |
| CA-Listwise | candidate | listwise CE | TBD | TBD |

---

## 11.4 Subgroup Result Table

| Method | head_has_img | head_no_img | tail_no_img | Overall |
|---|---:|---:|---:|---:|
| Residual-only | TBD | TBD | TBD | 0.2930 |
| E5 Regression clean | TBD | TBD | TBD | 0.2982 |
| CA-S1 | TBD | TBD | TBD | TBD |
| CA-S2 | TBD | TBD | TBD | TBD |
| CA-S3 | TBD | TBD | TBD | TBD |

This table is critical because the paper's main argument is protocol-aware asymmetry. The new router must show where the improvement comes from.

---

## 11.5 Suggested Figures

### Figure A: Router Granularity Progression

```text
Always-on fusion
      ↓
Query-level hard routing
      ↓
Query-level soft routing
      ↓
Candidate-level soft routing
      ↓
Oracle routing
```

### Figure B: Oracle Gap Recovery

```text
Residual-only -> E5 -> CA-S1 -> CA-S2 -> CA-S3 -> Oracle
```

Annotate:

```text
Recovered Oracle headroom
Remaining Oracle gap
```

### Figure C: Alpha Distribution by Regime

Plot distribution of `alpha(q,e)` for:

```text
head_has_img
head_no_img
tail_no_img
```

This can help explain whether the candidate-aware router behaves consistently with the protocol-aware bounded-gain story.

---

## 12. Interpretation Templates

## 12.1 If Candidate-aware Router Clearly Improves Over E5

Example condition:

```text
CA-S3 MRR >= 0.3050
```

Interpretation:

> The improvement from E5 to candidate-aware routing indicates that the limited gain of query-level clean routing is partly caused by coarse routing granularity. Once expert selection is performed at the query-candidate level and trained with a ranking-aligned objective, deployable routing recovers a larger portion of the Oracle headroom.

Paper claim:

> Query-level selective activation is useful but still too coarse; candidate-aware soft routing provides a stronger deployable mechanism for exploiting bounded multimodal gain.

---

## 12.2 If Score-aware Features Help but Candidate Modality Alone Does Not

Interpretation:

> Candidate modality metadata alone is insufficient to recover substantial routing gain. However, score-aware deployable signals improve routing, suggesting that expert confidence rather than static modality availability is the key missing signal.

Paper claim:

> The main limitation of strict clean routing is not only policy design, but also limited observability. Expert score-distribution signals provide additional deployable evidence for selective activation.

---

## 12.3 If New Routers Do Not Improve Over E5

Interpretation:

> Even after increasing routing granularity and using ranking-aligned objectives, deployable routing remains far below Oracle routing. This suggests that most Oracle separability is driven by answer-aware information that is not recoverable from legal inference-time signals.

Paper claim:

> The remaining Oracle gap is not merely an implementation weakness of the router, but a structural limitation of deployable observability under the current OpenBG-IMG protocol.

This result is still publishable as a stronger limitation analysis.

---

## 13. Minimal Recommended Experiment Set

If only one additional experimental round is feasible, run the following minimal set:

```text
Step 1. export_candidate_scores.py
        Export top100 + correct target.

Step 2. build_candidate_router_table.py
        Build CA-S1 and CA-S3 feature sets.

Step 3. train_candidate_soft_router.py
        Train pairwise candidate-level routers.

Step 4. eval_candidate_soft_router.py
        Evaluate top100 reranking.

Step 5. If CA-S3 > 0.3000, run full-ranking evaluation.
```

Minimal method matrix:

| ID | Method | Status |
|---|---|---|
| M0 | Residual-only | existing |
| M1 | E5 Regression clean | existing |
| M2 | CA-S1 clean candidate pairwise | new |
| M3 | CA-S3 clean + score candidate pairwise | new |
| M4 | Oracle routing | existing |

---

## 14. Expected Outcomes

| Outcome | Likelihood | Meaning |
|---|---:|---|
| CA-S1 gives little or no improvement | high | candidate modality alone may be insufficient |
| CA-S2 / CA-S3 improves over E5 | medium-high | score-aware signals are useful |
| CA-S3 reaches 0.300–0.305 | realistic | useful but moderate improvement |
| CA-S3 reaches 0.305–0.310 | challenging | strong enough to update paper method/results |
| CA-S3 approaches 0.3337 Oracle | unlikely | check for leakage carefully |

Recommended target levels:

```text
Current E5: 0.2982
Target 1:   > 0.3000
Target 2:   > 0.3050
Target 3:   > 0.3100
```

---

## 15. Final Recommendation

The next-stage experimental direction should be:

> **Candidate-aware soft selective activation with ranking-aligned training.**

This direction is better aligned with the paper's current finding that multimodal gain is bounded, local, and protocol-dependent.

The key shift is:

```text
query-level hard selection
        ↓
candidate-level soft weighting
        ↓
ranking-aligned optimization
        ↓
score-aware deployable routing
```

If this line improves over the current E5 result, the paper can move from:

> clean routing recovers only a small part of bounded multimodal gain

into a stronger claim:

> the limited gain of strict query-level clean routing is partly due to coarse routing granularity, and finer candidate-aware selective activation can recover more deployable multimodal gain.

If this line does not improve, the paper still gains a stronger conclusion:

> the remaining Oracle gap is not simply caused by an underpowered query-level router, but reflects a deeper observability limitation under the current OpenBG-IMG protocol.
