# Related Work Citation Expansion Framework

## Goal

The current verified bibliography contains **24 real references** in `docs/paper/references.bib`. The target for the next revision is to expand the manuscript toward **50+ references** while keeping the literature review selective, coherent, and fully auditable.

This framework is designed to expand the literature in a controlled way rather than by adding citations randomly. The related-work chapter should grow mainly in three directions:

1. **MMKGC and multimodal fusion**
2. **Missing modality, modality imbalance, and robustness**
3. **Selective activation, routing, and structural baselines**

## Current verified coverage

The current verified bibliography already includes real papers in the following groups:

### A. Structural KGC anchors
- Trouillon et al., 2016 (`ComplEx`)
- Dettmers et al., 2018 (`ConvE`)
- Sun et al., 2019 (`RotatE`)
- Balažević et al., 2019 (`TuckER`)

### B. Early MMKGC / multimodal representation learning
- Xie et al., 2017 (`IKRL`)
- Mousselly-Sergieh et al., 2018 (multimodal translation-based learning)
- Pezeshkpour et al., 2018 (multimodal relational data embedding)
- Liu et al., 2019 (`MMKG`)

### C. Fusion-oriented MMKGC methods
- Zhao et al., 2022 (`MoSE`)
- Liang et al., 2023 (`HRGAT`)
- Shang et al., 2024 (`LAFA`)
- Chen et al., 2022 (Hybrid Transformer)
- Jian et al., 2025 (Adaptive Modality Interaction Transformer)
- Zhu et al., 2025 (`Moodle`)
- Gao et al., 2025 (Mixed-Curvature MMKGC)

### D. Missing modality / robustness / training strategy
- Wang et al., 2021 (Is Visual Context Really Helpful for Knowledge Graph?)
- Xu et al., 2022 (Relation-enhanced Negative Sampling)

### E. Routing / expert selection / multimodal background
- Jacobs et al., 1991 (Adaptive Mixtures of Local Experts)
- Jordan and Jacobs, 1994 (Hierarchical Mixtures of Experts)
- Shazeer et al., 2017 (Sparsely-Gated MoE)
- Ngiam et al., 2011 (Multimodal Deep Learning)
- Baltrušaitis et al., 2019 (Multimodal Machine Learning survey)

### F. Benchmark / survey context
- Deng et al., 2022 (`OpenBG` / OpenBG-IMG source)
- Liang et al., 2024 (ACM CSUR survey on multimodal knowledge graphs)
- Chen et al., 2023 (survey on multimodal knowledge graphs)

## Expansion target to reach 50+

The safest next step is to add citations in blocks, not one by one. The following target blocks are recommended.

### Block 1: Stronger structural KGC background (+6 to +8)
Purpose:
- make the structural-baseline discussion thicker
- show that structure-dominant performance is evaluated against a broader KGC context

Verified real papers suitable for addition:
- Bordes et al., 2013, *Translating Embeddings for Modeling Multi-relational Data* (`TransE`)
- Wang et al., 2014, *Knowledge Graph Embedding by Translating on Hyperplanes* (`TransH`)
- Yang et al., 2015, *Embedding Entities and Relations for Learning and Inference in Knowledge Bases* (`DistMult`)
- Nickel et al., 2016, *A Review of Relational Machine Learning for Knowledge Graphs*
- Schlichtkrull et al., 2018, *Modeling Relational Data with Graph Convolutional Networks* (`R-GCN`)
- Vashishth et al., 2020, *Composition-based Multi-Relational Graph Convolutional Networks* (`CompGCN`)

Recommended placement:
- mainly in Section 2.3, with a small amount in Introduction

### Block 2: More MMKGC method coverage (+8 to +10)
Purpose:
- make the fusion-oriented literature feel less selective and less sparse
- cover 2022--2025 MMKGC methods more densely

Already verified and already in the bibliography:
- Chen et al., 2022
- Zhao et al., 2022
- Liang et al., 2023
- Shang et al., 2024
- Jian et al., 2025
- Zhu et al., 2025
- Gao et al., 2025

Action:
- add these works more explicitly and comparatively in Section 2.1 rather than citing them only once
- later add 2--3 more verified recent MMKGC papers after separate checking

### Block 3: Missing-modality / incomplete multimodal learning (+6 to +8)
Purpose:
- strengthen the claim that uneven modality support is a general modeling issue, not only an OpenBG-IMG artifact

Currently verified anchors:
- Baltrušaitis et al., 2019
- Wang et al., 2021
- Xu et al., 2022
- Deng et al., 2022 (`OpenBG` source)

Action:
- add more verified work on incomplete multimodal learning and robustness only after separate confirmation
- keep this block focused on papers that clearly exist and are clearly relevant, rather than adding generic multimodal papers that are too far from the KG setting

### Block 4: Routing / selective computation / expert selection (+5 to +7)
Purpose:
- support the selective-activation framing more strongly
- make it clear that the routing idea is grounded in a broader methodological tradition

Currently verified anchors:
- Jacobs et al., 1991
- Jordan and Jacobs, 1994
- Shazeer et al., 2017

Verified real additions suitable for later inclusion:
- Lepikhin et al., 2021, *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*
- Fedus et al., 2022, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*

Recommended placement:
- mainly Section 2.3, with one sentence in Introduction if needed

### Block 5: Survey and benchmark context (+4 to +6)
Purpose:
- thicken the review without overloading the technical argument
- show that the paper is positioned inside a broader MMKG/MMKGC literature map

Currently verified anchors:
- Liang et al., 2024
- Chen et al., 2023
- Baltrušaitis et al., 2019
- Deng et al., 2022

Action:
- later add one MMKGC-specific survey or benchmark-oriented source only after separate verification

## Writing rule for the next revision

When expanding toward 50+, citations should be added according to the following rule:

- **Introduction:** add only high-value anchoring references
- **Related Work:** carry most of the citation expansion
- **Method / Experiments / Discussion:** add citations only when they truly support framing or interpretation
- do **not** inflate Sections 3--5 with irrelevant references just to raise the count

## Safe near-term plan

The next verified expansion can be done in two rounds.

### Round 1: move from 24 to about 34 references
Add only highly certain real papers:
- Bordes et al., 2013 (`TransE`)
- Wang et al., 2014 (`TransH`)
- Yang et al., 2015 (`DistMult`)
- Nickel et al., 2016 (KGC review)
- Schlichtkrull et al., 2018 (`R-GCN`)
- Vashishth et al., 2020 (`CompGCN`)
- Lepikhin et al., 2021 (`GShard`)
- Fedus et al., 2022 (`Switch Transformers`)

### Round 2: move from about 34 to 50+
Add more recent MMKGC and missing-modality papers, but only after one-by-one verification before insertion.

## Final note

The point of expanding to 50+ references is not to maximize citation count mechanically. It is to make the new three-part related-work chapter feel complete enough to support the paper's revised positioning:

- not only a stronger-fusion paper,
- not only a missing-modality paper,
- but an analysis-driven selective-activation paper grounded in MMKGC, robustness, and routing literature.
