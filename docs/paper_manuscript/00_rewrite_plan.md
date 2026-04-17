# Paper Rewrite Plan

## Target Positioning

This paper should be positioned as an **analysis-driven method paper** rather than a pure leaderboard paper. The analytical part establishes that multimodal gain in the current OpenBG-IMG protocol is **bounded, conditional, and protocol-aware**. The method part turns this finding into a **gain-threshold selective activation mechanism**.

The central claim should therefore be:

> Under the current OpenBG-IMG protocol, multimodal gain is query-dependent rather than globally reliable. We therefore reformulate the problem from always-on fusion to gain-aware selective activation, and show that a lightweight gain-threshold router can preserve local multimodal benefit while avoiding harmful fusion in structure-favorable regimes.

## New Paper Structure

1. Introduction
2. Related Work
3. Task Setting and Protocol
4. Method: Gain-Threshold Routing for Selective Multimodal Activation
5. Experiments
6. Analysis and Discussion
7. Limitations
8. Conclusion

## Section-by-Section Writing Tasks

| Section | Main Goal | Reusable Material | New Writing Tasks | Output Status |
|---|---|---|---|---|
| 1. Introduction | Move the paper from gain-boundary diagnosis to selective activation | previous introduction draft; new routing contribution paragraph; main result summaries | rewrite the opening problem, add protocol-aware tension, introduce routing as the method response, replace old contribution list | in progress |
| 2. Related Work | Reposition the paper against MMKGC, adaptive fusion, missing modality, and routing | existing related-work draft | add a subsection on routing / expert selection / selective activation; tighten the positioning paragraph | pending |
| 3. Task Setting and Protocol | Preserve protocol-aware foundation | current protocol draft | keep subgroup and relation-group definitions, but make them clearly serve the later method motivation | to refine |
| 4. Method | Define the new proposed method clearly and formally | method draft; router code and result docs | write problem formulation, dual-expert design, gain labels, feature groups, hard-threshold routing, and training details | pending |
| 5. Experiments | Separate official model comparison from router comparison | main results summary; router tables; subgroup table; ablation table | split the section into official main table vs routing table; explicitly state metric dialect differences | pending |
| 6. Analysis and Discussion | Explain why the method works and what it does not prove | relation-type analysis; behavior analysis; router takeaways | explain why selective activation works, why rule-based routing is insufficient, and why protocol dependence matters | pending |
| 7. Limitations | State the boundary of the claims honestly | discussion drafts and protocol notes | write protocol specificity, fixed-expert limitation, and absence of end-to-end co-adaptation | pending |
| 8. Conclusion | Close with a controlled and defensible claim | existing conclusion ideas | rewrite the conclusion around operationalizing bounded gain rather than claiming global multimodal dominance | pending |

## Experiment Writing Rules

### Rule 1: Never mix official main-result metrics with router recomputed metrics in the same table

- Official seven-model comparison should use the `test_metrics.json` aggregation line.
- Router comparison should use the unified query-level recomputed line.
- Fixed experts inside the router section must stay on the router line for fairness with Oracle, rule-based routing, and learned routing.

### Rule 2: Relation-group evidence is supporting evidence, not the central method claim

Relation-group evaluation should remain in the paper because it strengthens the argument that multimodal gain is relation-dependent and bounded. However, it should support the selective-activation narrative rather than compete with the routing results for the main contribution slot.

### Rule 3: The strongest claim is selective activation, not universal multimodal superiority

The paper should not claim that multimodal fusion globally dominates structural models. The results instead support a narrower and more defensible claim: selective activation helps because multimodal gain exists, but only under certain protocol-shaped conditions.

## Recommended Writing Order

1. Rewrite Introduction
2. Rewrite Method
3. Rebuild Experiments
4. Rewrite Analysis and Discussion
5. Revise Related Work
6. Rewrite Limitations and Conclusion
7. Polish Abstract last

## Immediate Next Step

The first active rewrite target is `docs/paper_manuscript/01_introduction.md`. The introduction should establish the old analytical finding and then pivot explicitly to the new method contribution: gain-threshold routing for selective multimodal activation.
