# Executable Patch Plan: Strengthen Experiments and Discussion

Target file:

- `docs/paper/manuscript_main.tex`

Goal:

Strengthen the `Experiments and Discussion` section without adding new experiments. The current experimental evidence is already sufficient; the remaining improvement is to make the mechanism discussion clearer. This patch adds three small but important clarifications:

1. define `E1` and `E5` before using them repeatedly;
2. explain why the official model-comparison result motivates selective activation;
3. explain why `head_has_img` shows relative multimodal advantage but low absolute MRR.

---

## Patch 1: Define E1 and E5 in `Experimental Setup`

### Location

In `\subsection{Experimental Setup}`, find the paragraph ending with:

```tex
The post-hoc analysis line is retained only for stronger offline separability analysis and upper-bound-style comparisons; it is not part of the deployable clean claim.
```

Insert the following paragraph immediately after it and before `\paragraph{Reproducibility.}`.

### Insert

```tex
For readability, we refer to the best direction-specific clean thresholding policy as \textbf{E1} and the best regression-based clean router as \textbf{E5} in the following discussion. E1 represents the strongest structured-threshold clean policy, whereas E5 represents the strongest target-aligned clean-supervision policy.
```

### Why this helps

Later sections use `E1` and `E5` repeatedly. This sentence prevents readers from feeling that the experiment identifiers appear without definition.

---

## Patch 2: Add mechanism explanation after the official model-comparison table

### Location

In `\subsection{Experimental Setup}`, find the current paragraph after `\Cref{tab:official_model_comparison}` / Table 7, which currently says approximately:

```tex
\Cref{tab:official_model_comparison} shows that the strongest official-line result is achieved by \texttt{Residual-only} under the current OpenBG-IMG paper protocol, while \texttt{Full Model} improves over weaker multimodal variants but does not surpass the structural expert. This motivates the paper's selective-activation formulation: multimodal evidence is useful within local regimes, but always-on fusion is not globally reliable under the current protocol.
```

Replace it with the following strengthened version.

### Replace with

```tex
\Cref{tab:official_model_comparison} shows that the strongest official-line result is achieved by \texttt{Residual-only} under the current OpenBG-IMG paper protocol, while \texttt{Full Model} improves over weaker multimodal variants but does not surpass the structural expert. This behavior is consistent with the protocol-induced dominance of \texttt{tail\_no\_img}: when a large portion of the evaluation space lacks target-side visual support, always-on multimodal fusion may introduce modality noise or dilute the stronger structural regularities captured by the residual branch. This motivates the paper's selective-activation formulation: multimodal evidence can be useful within local regimes, but always-on fusion is not globally reliable under the current protocol.
```

### Why this helps

This paragraph turns Table 7 from a pure result table into a causal motivation for selective activation. It explains why `Residual-only` can be globally stronger even though multimodal information is still useful locally.

---

## Patch 3: Strengthen RQ1 mechanism discussion after subgroup results

### Location

In `\subsection{RQ1: Where does multimodal gain appear?}`, find the paragraph beginning:

```tex
The subgroup results confirm that multimodal gain is not uniformly distributed across this evaluation space.
```

Replace the paragraph with the following stronger version.

### Replace with

```tex
The subgroup results confirm that multimodal gain is not uniformly distributed across this evaluation space. The clearest relative multimodal advantage appears in \texttt{head\_has\_img}, where \texttt{Gate-only} outperforms \texttt{Full Model} and \texttt{Residual-only}, although the absolute MRR in this regime remains low. This pattern suggests that image availability alone is not sufficient to guarantee high absolute ranking performance. In the \texttt{head\_has\_img} regime, visual information makes the fusion path relatively more useful, which explains why \texttt{Gate-only} achieves the best ordering within this subgroup. However, the low absolute MRR indicates that these head-side queries remain intrinsically difficult under the current protocol. By contrast, \texttt{tail\_no\_img} contributes half of all bidirectional queries and is strongly structure-favorable, allowing \texttt{Residual-only} to dominate the aggregate metric. Interpreting these subgroup outcomes together with the regime counts helps explain the broader pattern of the paper: multimodal benefit is visible and meaningful, but it is concentrated in a local subset of queries rather than spread uniformly across the benchmark.
```

### Why this helps

This avoids overstating `head_has_img`. It explains the key nuance:

- `head_has_img` is multimodal-favorable only in a relative sense;
- absolute MRR is still low;
- global performance is dominated by `tail_no_img`, where the structural path is much stronger.

---

## Patch 4: Slightly refine relation-group interpretation

### Location

In the same RQ1 subsection, find the paragraph beginning:

```tex
Relation-group evidence provides a second, but supporting, view of this bounded-gain structure.
```

Replace it with the following version.

### Replace with

```tex
Relation-group evidence provides a second, but supporting, view of this bounded-gain structure. Across the retained coarse relation groups, the grouped ordering remains \texttt{Residual-only} $>$ \texttt{Full Model} $>$ \texttt{Gate-only}, indicating that relation dependence is real but still bounded. The relation-group results therefore do not show a reversal of the global ordering. Instead, they indicate that relation context modulates the degree of multimodal usefulness while remaining bounded by the protocol-level structure-dominant pattern. \Cref{tab:relation_group_support} further keeps this grouped analysis auditable by reporting retained relation and query support after minimum-support filtering, so that the grouped trends are not driven by a handful of low-support cases. Taken together, these observations answer RQ1: under the current OpenBG-IMG protocol, multimodal gain appears in specific protocol-shaped local regions rather than as a globally reliable property of the benchmark.
```

### Why this helps

This prevents overclaiming from Table 12. The relation-group analysis supports bounded gain; it does not prove that multimodal models dominate on visual relations.

---

## Optional Patch 5: Make Table 14 reference use `\Cref`

### Location

In RQ3, the current paragraph begins:

```tex
Table 14 reports the paired bootstrap evidence for the strongest clean-routing comparisons.
```

Replace with:

```tex
\Cref{tab:bootstrap_clean_routing} reports the paired bootstrap evidence for the strongest clean-routing comparisons.
```

Only apply this patch if the label for Table 14 is actually `tab:bootstrap_clean_routing`. If the label is different, use the existing label.

---

## Final expected effect

After applying these patches, the Experiments and Discussion section should satisfy a stronger journal-style logic:

```text
result -> mechanism -> implication -> boundary
```

The paper should not need new experiments. The existing evidence is already sufficient; these edits mainly strengthen interpretation and reviewer readability.
