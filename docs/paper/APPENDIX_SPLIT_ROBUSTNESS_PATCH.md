# Patch: Add Appendix-only Extra Split Robustness Check

This patch describes how to incorporate the `appendix_split_seed20260427` robustness check into the paper without changing the main evaluation line.

## 0. Purpose

The extra split should be used as **appendix-only robustness evidence**.

It should support this controlled claim:

> The qualitative trend remains stable under another valid split: the role--modality asymmetry remains present, strict query-level clean routing still provides only modest gain, and CA-S2 score-aware candidate routing remains clearly stronger than E5.

Do **not** use this split to replace the main `paper_split` results.

---

## 1. Evidence already available in the repository

Existing appendix split files:

```text
docs/appendix_split_seed20260427/dataset_statistics.json
docs/appendix_split_seed20260427/target_side_regime_counts.json
docs/appendix_split_seed20260427/robustness_results.json
docs/appendix_split_seed20260427/robustness_summary.md
docs/appendix_split_seed20260427/table_dataset_statistics.tex
docs/appendix_split_seed20260427/table_target_side_regime_counts.tex
docs/appendix_split_seed20260427/table_robustness_results.tex
```

Key split statistics:

| Item | Value |
|---|---:|
| Entities | 27,910 |
| Relations | 136 |
| Train triples | 220,087 |
| Valid triples | 5,000 |
| Test triples | 10,000 |
| Image entities | 14,690 |
| Image coverage | 52.63% |

Target-side regime counts:

| Regime | Count | Bidirectional ratio |
|---|---:|---:|
| `head_has_img` | 7,018 | 0.3509 |
| `head_no_img` | 2,982 | 0.1491 |
| `tail_has_img` | 0 | 0.0000 |
| `tail_no_img` | 10,000 | 0.5000 |

Robustness results:

| Method | Level | Feature type | MRR | Delta vs Residual | Delta vs E5 |
|---|---|---|---:|---:|---:|
| Residual-only | fixed | structural | 0.2944 +/- 0.0008 | -- | -- |
| Gate-only | fixed | fusion | 0.1718 +/- 0.0032 | -0.1226 | -- |
| E5 regression clean router | query | strict clean | 0.2992 | 0.0048 | -- |
| CA-S2 score-aware candidate router | candidate | score-aware | 0.3134 +/- 0.0015 | 0.0190 | 0.0142 |

Trend checks:

- Role-modality asymmetry holds: `true`
- Residual-only remains above Gate-only overall: `true`
- E5 gives a modest gain over Residual-only: `true`
- CA-S2 remains above E5: `true`

---

## 2. Main-paper editing strategy

Only make small changes in the main text.

Recommended changes:

1. Add one sentence/paragraph in `Experimental Setup`.
2. Add one sentence in `Limitations and Threats to Validity`.
3. Add one sentence in `Conclusion`.
4. Add one new appendix section.

Do **not** add the extra split numbers to:

- Abstract
- Official seven-model comparison table
- Main clean routing table
- Main candidate-router table
- Main significance table

Reason: the extra split is not the official evidence basis. It is a robustness check.

---

## 3. Patch A: Add a short note in Experimental Setup

Target file:

```text
docs/paper/manuscript_main.tex
```

Find the start of the experimental setup section, around:

```latex
All experiments are conducted on the current OpenBG-IMG \texttt{paper\_split} under the unified protocol defined in the previous section.
```

Insert after the first paragraph of Experimental Setup:

```latex
In addition to the main \texttt{paper\_split} results, we report an appendix-only robustness check on an alternative relation-stratified and coverage-protected split, denoted \texttt{appendix\_split\_seed20260427}. This additional split is not used to replace the main evaluation basis; it only tests whether the qualitative role--modality asymmetry and routing trend remain stable under another valid split construction.
```

Expected effect:

- Keeps the main evaluation line unchanged.
- Tells readers the appendix split exists.
- Prevents confusion between main and appendix results.

---

## 4. Patch B: Add one sentence in Limitations

Target file:

```text
docs/paper/manuscript_main.tex
```

Find:

```latex
\subsection{Limitations and Threats to Validity}
```

Find the first limitation paragraph beginning with:

```latex
First, the conclusions are protocol-aware rather than universal.
```

Insert after that paragraph:

```latex
The additional appendix split provides a robustness check for the same qualitative pattern, but it does not make the conclusions dataset-universal: it preserves the role--modality asymmetric structure and confirms the same routing trend, while still remaining within the OpenBG-IMG setting.
```

Expected effect:

- Strengthens the paper against a split-specificity concern.
- Avoids overclaiming universal generalization.

---

## 5. Patch C: Add one sentence in Conclusion

Target file:

```text
docs/paper/manuscript_main.tex
```

Find:

```latex
\section{Conclusion}
```

Find the paragraph discussing CA-S2, likely containing:

```latex
CA-S2 reaches 0.3142 MRR
```

Insert after that paragraph:

```latex
An appendix-only robustness check on an alternative split further supports this qualitative conclusion: the role--modality asymmetry remains present, strict query-level clean routing still provides only modest gain, and CA-S2 remains clearly stronger than E5.
```

Expected effect:

- Reinforces the conclusion without crowding it with appendix numbers.
- Keeps the main conclusion focused.

---

## 6. Patch D: Add a new appendix section

Target file:

```text
docs/paper/manuscript_main.tex
```

Find:

```latex
\appendix
```

Insert the following section after the existing candidate-router diagnostics appendix, or near the end before references/bibliography.

Recommended section:

```latex
\section{Additional Split Robustness Check}
\label{app:additional_split_robustness}

To test whether the main conclusion depends on a single random split, we construct an additional relation-stratified and coverage-protected split, denoted \texttt{appendix\_split\_seed20260427}. This split is used only as an appendix robustness check. It does not replace the main \texttt{paper\_split}, and it is not used for the official model-comparison line.

\input{../appendix_split_seed20260427/table_dataset_statistics.tex}

The additional split keeps the same scale as the main evaluation split: 27{,}910 entities, 136 relations, 220{,}087 training triples, 5{,}000 validation triples, and 10{,}000 test triples. The image coverage also remains unchanged because the entity set and image metadata are shared.

\input{../appendix_split_seed20260427/table_target_side_regime_counts.tex}

The target-side regime counts show that the same role--modality asymmetry remains present. Head-side targets are partially image-available, with 7{,}018 \texttt{head\_has\_img} and 2{,}982 \texttt{head\_no\_img} query instances, whereas tail-side targets remain image-unavailable, with 0 \texttt{tail\_has\_img} and 10{,}000 \texttt{tail\_no\_img} query instances. This supports the interpretation that the role--modality asymmetry is not an artifact of only the main \texttt{paper\_split}.

\input{../appendix_split_seed20260427/table_robustness_results.tex}

The model-level robustness check preserves the same qualitative trend as the main experiments. \texttt{Residual-only} remains much stronger than \texttt{Gate-only} at the global level, confirming that always-on multimodal fusion is not globally reliable under the asymmetric protocol. The E5 regression clean router improves only modestly over \texttt{Residual-only}, while CA-S2 score-aware candidate routing remains clearly stronger than E5. Therefore, the additional split supports the paper's main controlled claim: strict query-level clean routing recovers limited deployable gain, whereas score-aware candidate-level routing exposes substantially more usable routing signal.

These results should not be interpreted as showing exact split-invariant numerical performance. Instead, they provide appendix-level evidence that the central qualitative pattern remains stable under another valid split construction.
```

---

## 7. Important path check for `\input{...}`

The appendix section above uses paths like:

```latex
\input{../appendix_split_seed20260427/table_dataset_statistics.tex}
```

Use this if your compile command is:

```bash
cd docs/paper
pdflatex manuscript_main.tex
```

If you compile from repository root, use:

```latex
\input{docs/appendix_split_seed20260427/table_dataset_statistics.tex}
\input{docs/appendix_split_seed20260427/table_target_side_regime_counts.tex}
\input{docs/appendix_split_seed20260427/table_robustness_results.tex}
```

If LaTeX reports `File not found`, switch between the two path styles.

---

## 8. Patch E: Polish robustness table note

Target file:

```text
docs/appendix_split_seed20260427/table_robustness_results.tex
```

Current issue:

- Residual-only, Gate-only, and CA-S2 are reported as mean +/- std over three seeds.
- E5 is reported as a single aggregate MRR.
- This is acceptable, but it should be explained to avoid reviewer confusion.

Recommended replacement table:

```latex
\begin{table}[t]
\centering
\caption{Appendix robustness results on the OpenBG-IMG \texttt{appendix\_split\_seed20260427}. The split is used only as a robustness check and does not replace the main \texttt{paper\_split} evaluation.}
\label{tab:appendix_split_robustness_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcrrr}
\toprule
Method & Level & Feature type & MRR & $\Delta$ Residual & $\Delta$ E5 \\
\midrule
Residual-only & fixed & structural & 0.2944 $\pm$ 0.0008 & -- & -- \\
Gate-only & fixed & fusion & 0.1718 $\pm$ 0.0032 & -0.1226 & -- \\
E5 regression clean router & query & strict clean & 0.2992 & 0.0048 & -- \\
CA-S2 score-aware candidate router & candidate & score-aware & 0.3134 $\pm$ 0.0015 & 0.0190 & 0.0142 \\
\bottomrule
\end{tabular}

\vspace{0.3em}
\begin{minipage}{0.92\textwidth}
\footnotesize
\emph{Note.} Residual-only, Gate-only, and CA-S2 are reported as mean $\pm$ standard deviation over three seeds. E5 is reported as a single aggregate MRR because only the clean-router aggregate output was retained for this appendix run. The purpose of this table is to verify qualitative trend stability rather than exact numerical invariance across splits.
\end{minipage}
\end{table}
```

---

## 9. Optional RQ4 cross-reference

This is optional. If the main text is already too long, skip it.

Target area:

```latex
\subsection{RQ4: Can candidate-level score-aware routing recover more deployable gain?}
```

At the end of RQ4 or integrated discussion, add one sentence:

```latex
A separate appendix-only robustness check on an alternative split preserves the same qualitative ordering, with CA-S2 remaining clearly stronger than the E5 strict clean router.
```

Recommendation:

- Add this only if you want readers to notice the appendix evidence.
- Skip it if the conclusion already mentions the appendix.

---

## 10. Do not change the abstract

Do not add appendix split results to the abstract.

Main abstract numbers should remain based on the main `paper_split`:

```text
E5 = 0.2982
CA-S2 = 0.3142 +/- 0.0010
CA-S2 over E5 = +0.0160
```

Appendix split numbers are:

```text
E5 = 0.2992
CA-S2 = 0.3134 +/- 0.0015
CA-S2 over E5 = +0.0142
```

Putting both sets of numbers in the abstract would make the evidence hierarchy unclear.

---

## 11. Do not modify main result tables

Do not add appendix split rows into:

- Official model-comparison table
- Clean routing table
- Candidate-router main table
- Paired bootstrap significance table

Reason:

The main tables are based on the main `paper_split`. Mixing appendix split rows into them would create aggregation-basis confusion.

If needed, refer to the appendix only by text:

```latex
Additional split robustness results are provided in \Cref{app:additional_split_robustness}.
```

---

## 12. Recommended final paper structure

After this patch, the paper should conceptually look like:

```text
1 Introduction
2 Related Work
3 Task Setting and Method
4 Experiments and Discussion
   4.1 Experimental Setup
   4.2 RQ1
   4.3 RQ2
   4.4 RQ3
   4.5 RQ4
   4.6 Integrated Discussion
   4.7 Limitations and Threats to Validity
5 Conclusion

Appendix A Clean Routing Progression and Oracle Headroom
Appendix B Additional Delta Sensitivity Results
Appendix C Candidate-level Router Diagnostics
Appendix D Additional Split Robustness Check
References
```

---

## 13. Post-edit checklist

After applying the patch, check:

```text
[ ] manuscript_main.tex compiles successfully.
[ ] No File not found error from appendix table inputs.
[ ] Table labels are unique:
    [ ] tab:dataset_statistics_appendix_split_seed20260427
    [ ] tab:target_side_regime_counts_appendix_split_seed20260427
    [ ] tab:appendix_split_robustness_results
[ ] Main paper_split numbers remain unchanged:
    [ ] Residual-only main MRR = 0.2930
    [ ] E5 main MRR = 0.2982
    [ ] CA-S2 main MRR = 0.3142 +/- 0.0010
[ ] Appendix split numbers are clearly labeled as appendix-only:
    [ ] Residual-only = 0.2944 +/- 0.0008
    [ ] Gate-only = 0.1718 +/- 0.0032
    [ ] E5 = 0.2992
    [ ] CA-S2 = 0.3134 +/- 0.0015
[ ] No sentence implies that the appendix split replaces the main split.
[ ] No sentence implies exact numerical invariance across splits.
[ ] The conclusion says qualitative trend or qualitative conclusion, not identical results.
```

---

## 14. Recommended final claim wording

Use this as the official robustness claim:

```latex
The appendix split robustness check supports qualitative trend stability: the role--modality asymmetry remains present, strict query-level clean routing still recovers only modest gain, and CA-S2 score-aware candidate routing remains clearly stronger than E5.
```

Avoid stronger claims such as:

```text
The result is fully robust across splits.
The method generalizes across all split settings.
The same performance is reproduced.
CA-S2 is universally better.
```

---

## 15. If the manuscript becomes too long

If adding all three appendix tables makes the paper too long, keep only these two tables:

```latex
\input{../appendix_split_seed20260427/table_target_side_regime_counts.tex}
\input{../appendix_split_seed20260427/table_robustness_results.tex}
```

Remove:

```latex
\input{../appendix_split_seed20260427/table_dataset_statistics.tex}
```

Then replace the dataset statistics paragraph with:

```latex
The additional split uses the same scale as the main \texttt{paper\_split}: 220{,}087 training triples, 5{,}000 validation triples, and 10{,}000 test triples over 27{,}910 entities and 136 relations.
```

This shorter version still preserves the important robustness evidence.

---

## 16. Best final version

Recommended final version:

- Main text: add only three short references to the appendix robustness check.
- Appendix: add one new section.
- Appendix section: include target-side regime table and robustness result table.
- Dataset statistics table: optional if page limit allows.
- Robustness result table note: strongly recommended.

This keeps the paper clean: the main results remain official, while the appendix adds useful split-robustness evidence.
