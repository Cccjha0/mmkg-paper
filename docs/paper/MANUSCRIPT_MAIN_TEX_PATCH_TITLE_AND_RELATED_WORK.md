# Patch Instructions: Title and Related Work Revision

Target file:

- `docs/paper/manuscript_main.tex`

Purpose:

This patch addresses two revision priorities:

1. reduce citation-list density in Related Work and make it more problem-driven;
2. revise the title toward a more immediately understandable journal-style framing.

---

## 1. Recommended title update

Current title:

```tex
\title{From Bounded Multimodal Gain to Structured Clean Routing: Selective Activation on OpenBG-IMG}
```

Recommended replacement:

```tex
\title{Protocol-Aware Selective Activation for Multimodal Knowledge Graph Completion under Incomplete Visual Modality}
```

Rationale:

- more immediately understandable to reviewers;
- avoids making `Clean Routing` the first unfamiliar term in the title;
- foregrounds the paper's actual contribution: protocol-aware selective activation under incomplete visual modality.

Alternative if you want to preserve the bounded-gain story more explicitly:

```tex
\title{From Bounded Multimodal Gain to Selective Activation in Multimodal Knowledge Graph Completion}
```

---

## 2. Replace the Related Work section with this version

Recommended replacement range:

Replace the full current `\section{Related Work}` block, ending immediately before `\section{Task Setting and Method}`.

```tex
\section{Related Work}

\subsection{MMKGC and Multimodal Fusion}
Classical knowledge graph completion (KGC) methods, including translational, bilinear, tensor-factorization, convolutional, and graph-based models, provide the structural baselines for link prediction~\cite{nickel2011rescal,bordes2013transe,yang2015distmult,trouillon2016complex,dettmers2018conve,sun2019rotate,balazevic2019tucker,schlichtkrull2018rgcn,vashishth2020compgcn}. These models establish the central reference point for evaluating whether auxiliary modalities add value beyond graph structure. As multimodal knowledge graphs have become more common, MMKGC research has extended this structural paradigm by incorporating textual descriptions, visual features, and other modality-specific signals into entity representations. Early image-aware and multimodal embedding methods, together with benchmark-oriented work such as the \texttt{MMKG} line, share the basic premise that non-structural evidence can complement graph topology when structure alone is insufficient~\cite{xie2017ikrl,mousselly2018multimodal,pezeshkpour2018mkbe,liu2019mmkg}.

Most MMKGC methods therefore focus on \emph{how to fuse} heterogeneous information. Simple early-fusion strategies concatenate or aggregate modality-specific features before scoring, while more advanced methods use learned gates, attention, ensemble mechanisms, relation-aware transformations, or cross-modal interaction to adapt the contribution of each modality~\cite{ngiam2011multimodal,atrey2010multimodalfusion,arevalo2017gmu,zhao2022mose,liang2023hrgat,shang2024lafa,chen2022hybrid,jian2025amit,zhu2025modalities,gao2025mixed}. The common intuition behind these methods is that multimodal usefulness is context-dependent: the same image or text description may be informative for one relation but much less useful for another.

Our work is related to this fusion-oriented literature, and our \texttt{Gate-only} and \texttt{Full Model} variants naturally sit within this tradition. However, the present paper is not primarily a stronger-fusion paper. Prior MMKGC work mainly asks how multimodal evidence should be fused once it is available. By contrast, our starting point is that under the current OpenBG-IMG protocol, multimodal gain is not globally reliable. We therefore shift the emphasis from \emph{how to fuse} toward \emph{when to activate} multimodal evidence under protocol-shaped missing visual support.

\subsection{Missing Modality, Modality Imbalance, and Robustness}
A second strand of related work concerns robustness under incomplete, imbalanced, or unreliable modalities. In many real-world multimodal settings, entities do not have uniformly available text or images, and even when multimodal features exist, their quality may vary substantially. This has motivated broader work on incomplete multimodal learning, modality imbalance, robustness to noisy side information, and the question of whether visual context is always helpful for knowledge-rich prediction problems~\cite{atrey2010multimodalfusion,ramachandram2017deep,baltrusaitis2019multimodal,wang2021rsme}. Within MMKGC specifically, prior studies already suggest that multimodal gains should not be assumed to appear uniformly or to depend only on stronger fusion operators~\cite{wang2021rsme,xu2022relation}.

This perspective is especially important for product-oriented knowledge graphs. In such settings, image availability may be highly uneven across entity subsets, relation types, or prediction directions. Under the current OpenBG-IMG protocol, the issue is even sharper: missingness is not only present, but protocol-shaped, because target position is entangled with modality availability. As a result, missing-modality effects are not merely a nuisance variable; they help determine which modeling path becomes dominant under evaluation.

Our work is aligned with this robustness-oriented perspective, but differs in emphasis. Existing missing-modality work often focuses on making multimodal models more robust when some modalities are absent or noisy. We instead study a protocol-defined missingness pattern and ask where multimodal gain actually appears under that pattern. The resulting problem is not only robustness to missing images, but also clean decision-making under role--modality asymmetry.

\subsection{Selective Activation, Routing, and Structural Baselines}
A third strand of related work comes from selective computation, expert routing, and mixture-of-experts reasoning. Across machine learning settings, routing mechanisms are used when different experts are expected to perform well under different input conditions. Classical mixtures of experts provide the conceptual origin of this perspective, while later conditional-computation systems show how expert routing can become a general design principle for activating only the most suitable path~\cite{jacobs1991adaptive,jordan1994hierarchical,shazeer2017outrageously,lepikhin2021gshard,fedus2022switch}.

This routing view is relevant once multimodal gain is understood as conditional rather than globally reliable. If multimodal evidence helps only in certain protocol-shaped regions of the evaluation space, then always-on fusion becomes only one possible design choice. An alternative is to treat multimodal activation itself as a decision problem: use the multimodal path when the current query is likely to benefit from it, and fall back to a stronger structural path otherwise.

At the same time, our routing problem differs from large-scale mixture-of-experts systems. We do not route among many experts for model scaling. Instead, we study clean query-level selective activation between a fusion expert and a structural expert under strict deployability constraints. The router may use only query-time legal signals, such as direction, relation-derived development priors, and observed-side modality indicators. It may not use the hidden target, target-side image availability, or answer-aware expert scores.

Strong structural baselines are therefore not incidental comparison points in our setting, but part of the explanation itself. Under the current OpenBG-IMG protocol, \texttt{ComplEx} remains a competitive structural anchor and \texttt{Residual-only} is the strongest model within our internal family, whereas \texttt{TuckER} does not emerge as competitive under the present split. This matters because the key empirical tension in our paper is not between multimodal models and weak references, but between local multimodal gains and globally stronger structure-dominant alternatives.

Against this background, the present work should be understood as an analysis-driven method paper. It is not primarily a new-fusion paper, because our main claim is not that a proposed multimodal architecture is globally strongest. It is not primarily a generic missing-modality method paper, because our goal is not to introduce a universal robustness algorithm. Instead, we begin from a protocol-aware diagnosis: under the current OpenBG-IMG setting, multimodal gain is real but bounded, and its usefulness depends on target position, image availability, relation context, and branch competition. We then extend this diagnosis into a method contribution by reformulating bounded multimodal gain as clean query-level selective activation.
```

---

## 3. What this revision fixes

This revision removes the survey-like baseline list and refocuses Related Work around the paper's actual distinction:

- prior MMKGC mainly asks **how to fuse**;
- this paper asks **when to activate**;
- prior missing-modality work mainly studies robustness to absent/noisy modalities;
- this paper studies protocol-shaped missingness and role--modality asymmetry;
- routing/MoE gives methodological context, but this paper studies clean query-level selective activation rather than large-scale expert scaling.
