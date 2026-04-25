# Remaining Revision Action Plan

This document converts the remaining manuscript weaknesses into an executable revision plan. The current source of truth remains:

- `docs/paper/manuscript_main.tex`

The goal is not to change the paper's central story. The current story is already clear:

> protocol-shaped incomplete visual support -> bounded multimodal gain -> naive clean routing failure -> structured clean routing -> target-aligned clean supervision -> remaining Oracle gap.

The remaining work is to make the manuscript look more like a complete MMKGC/KGC journal paper by strengthening standard experimental components, reproducibility, ablation evidence, and limitations.

---

## Priority 1: Add a standard dataset statistics table

### Problem

The manuscript currently has a strong protocol-regime table showing the test-time target-side asymmetry, but it still lacks the standard dataset statistics table expected in KGC/MMKGC papers.

A reviewer may ask:

> How large is OpenBG-IMG in this paper split? How many entities, relations, triples, and image-supported entities are used?

### Required table

Add a table like this in `Task Setting and Protocol`, before or immediately after the current target-side regime table.

```tex
\begin{table}[t]
\centering
\caption{Dataset statistics of the OpenBG-IMG \texttt{paper\_split}.}
\label{tab:dataset_statistics}
\small
\begin{tabular}{lrrrrrr}
\toprule
Dataset & \#Entities & \#Relations & \#Train & \#Valid & \#Test & \#Image entities \\
\midrule
OpenBG-IMG & TODO & TODO & TODO & TODO & TODO & TODO \\
\bottomrule
\end{tabular}
\end{table}
```

If image coverage is available, add one more column:

```tex
Image coverage
```

### Data needed

Collect from the actual paper split:

- number of unique entities;
- number of relations;
- number of train triples;
- number of valid triples;
- number of test triples;
- number of entities with images;
- image coverage percentage.

Known approximate values from project context may include around 27,910 entities and around 14,690 image-supported entities, but the paper should use script-derived final values rather than memory.

### Recommended implementation

Create a small script, for example:

```text
scripts/build_dataset_statistics.py
```

Expected inputs:

- entity mapping file;
- relation mapping file;
- train/valid/test triple files;
- image availability file, e.g. `has_img.pt` or equivalent metadata.

Expected outputs:

- `docs/dataset_statistics.json`
- `docs/paper_tables/table_dataset_statistics.tex`

### Writing paragraph

Add after the table:

```tex
\Cref{tab:dataset_statistics} summarizes the scale of the OpenBG-IMG \texttt{paper\_split} used in all experiments. The split contains both textual and visual entity information, but image availability is incomplete. This incompleteness is not uniformly distributed across evaluation roles, which motivates the protocol-specific target-side analysis in \Cref{tab:protocol_test_regime_counts}.
```

### Completion criteria

- `tab:dataset_statistics` exists in `manuscript_main.tex`.
- The table uses final script-derived numbers.
- The table is referenced in the text.
- It does not replace the protocol-regime table; both are needed.

---

## Priority 2: Add or strengthen the official model-comparison main table

### Problem

The clean-routing table is now strong, but the standard official model-comparison table needs to be equally visible. KGC/MMKGC reviewers expect a main result table before the routing analysis.

### Available data

The required data already exists in:

- `docs/MAIN_RESULTS_SUMMARY.md`

This file includes mean ± std over 3 seeds for:

- `ComplEx`
- `TuckER`
- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Full Model`
- `Residual-only`

### Required table

Add a table in `Experiments and Discussion`, before the clean-routing table.

```tex
\begin{table}[t]
\centering
\caption{Official model-comparison line on OpenBG-IMG. Results are computed from formal \texttt{test\_metrics.json} outputs and reported as mean $\pm$ standard deviation over three seeds. These rows should not be directly mixed with clean-routing-line rows.}
\label{tab:official_model_comparison}
\small
\begin{tabular}{lccccc}
\toprule
Model & MRR & Hits@1 & Hits@3 & Hits@10 & Notes \\
\midrule
\texttt{ComplEx} & 0.2588 $\pm$ 0.0018 & 0.1920 $\pm$ 0.0026 & 0.2946 $\pm$ 0.0012 & 0.3871 $\pm$ 0.0014 & structural \\
\texttt{TuckER} & 0.0890 $\pm$ 0.0024 & 0.0360 $\pm$ 0.0019 & 0.1054 $\pm$ 0.0042 & 0.1892 $\pm$ 0.0012 & structural \\
\texttt{Text-only} & 0.1261 $\pm$ 0.0043 & 0.0524 $\pm$ 0.0050 & 0.1499 $\pm$ 0.0036 & 0.2761 $\pm$ 0.0046 & text \\
\texttt{Early Fusion} & 0.1666 $\pm$ 0.0013 & 0.0905 $\pm$ 0.0008 & 0.1998 $\pm$ 0.0034 & 0.3155 $\pm$ 0.0022 & multimodal \\
\texttt{Gate-only} & 0.1739 $\pm$ 0.0044 & 0.0946 $\pm$ 0.0108 & 0.2113 $\pm$ 0.0030 & 0.3261 $\pm$ 0.0010 & fusion expert \\
\texttt{Full Model} & 0.2100 $\pm$ 0.0097 & 0.1167 $\pm$ 0.0107 & 0.2658 $\pm$ 0.0131 & 0.3794 $\pm$ 0.0116 & gate + residual \\
\texttt{Residual-only} & \textbf{0.2930 $\pm$ 0.0008} & \textbf{0.2328 $\pm$ 0.0003} & \textbf{0.3306 $\pm$ 0.0021} & \textbf{0.4031 $\pm$ 0.0017} & structural expert \\
\bottomrule
\end{tabular}
\end{table}
```

### Writing paragraph

Add after the table:

```tex
\Cref{tab:official_model_comparison} shows that the strongest official-line result is achieved by \texttt{Residual-only}, while \texttt{Full Model} improves over weaker multimodal variants but does not surpass the structural expert. This motivates the paper's selective-activation formulation: multimodal evidence is useful within local regimes, but always-on fusion is not globally reliable under the current protocol.
```

### Important caution

Do not claim universal superiority of `Residual-only`. Use:

```tex
under the current OpenBG-IMG paper protocol
```

or:

```tex
on the official model-comparison line
```

### Completion criteria

- `tab:official_model_comparison` exists in `manuscript_main.tex`.
- It clearly says official model-comparison line.
- It uses mean ± std over three seeds.
- It is referenced before the clean-routing table.

---

## Priority 3: Add an algorithm box for clean selective activation

### Problem

The Method section is conceptually complete but still text-heavy. A method paper usually benefits from an algorithm box or pipeline summary that turns formulas into an operational procedure.

### Recommended location

Add after `Router Training Details` or before `Experiments and Discussion`.

### Simple LaTeX version without new packages

To avoid package conflicts, use a framed minipage or enumerated algorithm-style block:

```tex
\begin{table}[t]
\centering
\caption{Clean query-level selective activation procedure.}
\label{alg:clean_selective_activation}
\small
\begin{tabular}{p{0.94\textwidth}}
\toprule
\textbf{Input:} query $q$, clean query features $x_q$, fusion expert $s_f$, structural expert $s_s$, trained router $g$, development-selected threshold(s). \\
\midrule
1. Construct clean query-time features $x_q$ using direction, relation-derived development priors, and observed-side modality indicators. \\
2. Predict a fusion probability $p(q)$ or expected gain $\widehat{\Delta}(q)$ using the trained router. \\
3. Apply the selected policy: global threshold, direction-specific threshold, bucket-specific threshold, or regression threshold. \\
4. Set $\alpha(q)=1$ if the fusion expert is selected; otherwise set $\alpha(q)=0$. \\
5. Score each candidate entity $e$ using $s_{final}(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e)$. \\
6. Rank candidates by $s_{final}(q,e)$ and compute filtered ranking metrics. \\
\bottomrule
\end{tabular}
\end{table}
```

### Why this helps

- makes the method easier to understand;
- shows deployable query-time behavior;
- separates training-time gain labels from test-time clean features;
- makes the paper look more like a method paper rather than only an analysis paper.

### Completion criteria

- Algorithm/procedure box appears in Method.
- It explicitly says only clean features are used at query time.
- It references the same scoring equation used in the text.

---

## Priority 4: Add a router ablation table

### Problem

The paper currently shows the final clean-routing progression, but reviewers may ask which component matters most:

- feature set?
- threshold structure?
- target type?
- regression vs binary label?

### Required table

Add a compact ablation table in `Experiments and Discussion`, after the main clean-routing table.

Suggested columns:

```text
Router variant | Feature set | Policy structure | Supervision target | MRR | Key interpretation
```

Example table shell:

```tex
\begin{table}[t]
\centering
\caption{Clean router ablation on feature set, policy structure, and supervision target. All rows are evaluated on the clean routing line.}
\label{tab:router_ablation}
\small
\begin{tabular}{p{0.22\textwidth}p{0.12\textwidth}p{0.20\textwidth}p{0.18\textwidth}cp{0.18\textwidth}}
\toprule
Variant & Features & Policy & Target & MRR & Interpretation \\
\midrule
Clean rule & -- & rule & -- & 0.2943 & legal baseline \\
Naive learned router & C4 & global threshold & binary gain & 0.2939 & global threshold is insufficient \\
Direction-specific threshold & C4 & direction-specific & binary gain & 0.2974 & policy granularity helps \\
Regression clean router & C4 & thresholded gain & scalar $\Delta(q)$ & 0.2982 & target-aligned supervision helps \\
TODO: w/o relation priors & C1/C3 & best policy & scalar/binary & TODO & tests relation prior contribution \\
\bottomrule
\end{tabular}
\end{table}
```

### Data needed

Existing values already known:

- Clean rule: 0.2943
- Naive global clean router: 0.2939
- Direction-specific threshold: 0.2974
- Regression clean router: 0.2982

Potential extra values to collect if available:

- best C1 only;
- best C2 only;
- best C3 only;
- best C4;
- regression without relation priors;
- regression without observed-side modality features.

### Minimal acceptable version

If extra ablation values are not available, use the four known rows and frame it as:

```tex
progressive clean-routing ablation
```

rather than full feature ablation.

### Stronger version

If the router output files already contain feature-set scans, add rows for C1--C4. This would make the ablation much more convincing.

### Completion criteria

- The table separates feature set, policy structure, and target type.
- It does not mix clean routing line with official line.
- It explains why regression and direction-specific thresholding help.

---

## Priority 5: Add a formal Limitations / Threats to Validity subsection

### Problem

The manuscript already mentions controlled claims and Oracle gap, but limitations are spread across the paper. Mature journal papers usually have a concise limitations section.

### Recommended location

Add near the end of `Experiments and Discussion`, before `Conclusion`, or make it a short standalone subsection.

### Suggested text

```tex
\subsection{Limitations and Threats to Validity}

First, the conclusions are protocol-aware rather than universal. The role--modality asymmetry analyzed in this paper is induced by the current OpenBG-IMG \texttt{paper\_split}; different splits or datasets with more balanced target-side image availability may produce different gain boundaries.

Second, the clean router is intentionally restricted to legal query-time features. This makes the routing policy deployable, but it also limits separability. The remaining gap to Oracle routing indicates that some expert-selection information is not visible under the current clean feature set.

Third, the routing framework uses a fixed pair of experts, \texttt{Gate-only} and \texttt{Residual-only}. This design makes the analysis interpretable, but it does not exhaust the possible expert set. Future work could explore additional multimodal and structural experts, soft routing, or end-to-end expert-router training.

Fourth, the main routing conclusions are based primarily on MRR and paired bootstrap delta-MRR. Hit-based routing exports are treated as auxiliary rather than final evidence. This keeps the claim focused, but future work could examine a broader set of ranking and calibration metrics.
```

### Completion criteria

- Limitations are explicit and grouped.
- Oracle gap is discussed as a limitation, not a contribution.
- The section does not weaken the paper; it protects the claim boundary.

---

## Priority 6: Add reproducibility and implementation details

### Problem

The paper now has good router-training details, but standard ML/KG papers often include a short reproducibility paragraph covering code, seeds, hardware, and scripts.

### Recommended location

At the end of `Experimental Setup`.

### Suggested text

```tex
\paragraph{Reproducibility.}
All base-model results are aggregated from formal \texttt{test\_metrics.json} files under a unified filtered-ranking pipeline. We report mean and standard deviation over three random seeds for the official model-comparison line. Router feature tables, threshold scans, clean-routing results, paired bootstrap outputs, and plotting scripts are generated from the same exported query-level expert outcomes. All router thresholds and hyperparameters are selected on the development split only before final test reporting.
```

If hardware details are available, add:

```tex
Experiments were conducted on TODO hardware with TODO GPU memory. Base model hyperparameters are listed in \Cref{tab:hyperparameters}.
```

### Optional table

Add a small hyperparameter table only if data is ready:

```tex
\begin{table}[t]
\centering
\caption{Main training hyperparameters used for official model-comparison runs.}
\label{tab:hyperparameters}
\small
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Embedding dimension & TODO \\
Negative ratio & TODO \\
Optimizer & TODO \\
Learning rate & TODO \\
Batch size & TODO \\
Early stopping metric & dev MRR \\
Seeds & 1, 2, 3 \\
\bottomrule
\end{tabular}
\end{table}
```

### Completion criteria

- Official results are tied to `test_metrics.json`.
- Three-seed reporting is clearly stated.
- Router outputs and bootstrap are reproducible from query-level exports.

---

## Priority 7: Minor title refinement

### Problem

The current title is clear but slightly unnatural:

> Protocol-Aware Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Modality

The phrase `under Incomplete Visual Modality` can sound awkward.

### Recommended replacement

```tex
\title{Protocol-Aware Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Support}
```

### Why

- `visual support` better matches image availability;
- avoids implying that the visual modality itself is technically incomplete;
- still keeps the current title's clarity.

### Completion criteria

- Title updated if the team agrees.
- Running title can stay as `Structured Clean Routing on OpenBG-IMG` or become `Protocol-Aware Selective Activation`.

---

## Priority 8: Replace author placeholders before sending to supervisor

### Problem

The current manuscript still uses placeholder authors, affiliations, and email.

### Required action

Either use real names and affiliations, or anonymize consistently.

Option A: supervisor-facing version:

```tex
\author{Your Name ...}
```

Option B: anonymous review-style version:

```tex
\author{Anonymous Authors}
```

### Completion criteria

- No `Author One`, `Author Two`, `Author Three` remain.
- No placeholder email remains.
- Affiliations are either real or intentionally anonymous.

---

## Recommended execution order

1. Add dataset statistics table.
2. Add official model-comparison table using `docs/MAIN_RESULTS_SUMMARY.md`.
3. Add clean selective activation algorithm box.
4. Add router ablation table.
5. Add limitations subsection.
6. Add reproducibility paragraph and optional hyperparameter table.
7. Decide title wording.
8. Replace author placeholders.

---

## Minimal supervisor-ready checklist

Before sending to the supervisor, the manuscript should have:

- [ ] Dataset statistics table.
- [ ] Official model-comparison table.
- [ ] Clean-routing main table.
- [ ] Problem--response--validation table.
- [ ] Router training details.
- [ ] Algorithm or procedure box.
- [ ] Limitations subsection.
- [ ] Reproducibility paragraph.
- [ ] No placeholder author information.

If time is limited, prioritize the first five items. They most directly affect the paper's perceived completeness and defensibility.
