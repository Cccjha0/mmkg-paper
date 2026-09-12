# B15/B16: fixed-policy endpoint ties and harm discrimination

Analysis specification written on 2026-09-12 before producing this review's numerical results. Both datasets have historical TEST exposure; this is a retrospective diagnostic specification, not a preregistration or a new holdout. No selector, action, orientation, threshold or pair is selected here.

## Inputs and replay

Use all six information_boundary_v2 pairs, three base seeds and both directions. DEV uses the original five grouped held-out policies, including each fold's anchor, radius and gate. Recover held-out margins from the saved balanced fold states in winner_signal_review_v1; no new fitting. TEST uses the original locked policy's saved margins and preferences. Reproduce all recorded applied weights, fallback flags, anchor RRs and policy RRs before analysis. Verify hashes against the existing manifests. No scorer or GPU work.

## Endpoint ties (B15)

Tie means exact endpoint RR equality, the same condition excluded from classifier fitting. Check equivalence with endpoint rank equality. Partition every split into ties and non-ties, including unchanged observations. Report counts and fractions, actual weight-change rate, mean absolute weight movement, net RR change relative to the corresponding anchor, harm and benefit counts/rates, unconditional loss, conditional loss and contribution to the whole-split utility. Provide pooled and all pair × seed × direction cells, with all denominators. Reconcile counts and utility with the previous winner/action diagnostics.

Ties are excluded only from the classifier's binary loss: anchor/action selection and evaluation include them. Serving does not know endpoint gold RRs and does not remove ties. Endpoint tie does not imply constant mixture rank. No gold-dependent serving filter is added.

## Harm discrimination (B16)

Use the fixed diagnostic ranking score r=1-c, c=abs(2*p_A-1); larger r hypothesizes greater harm. Never flip this orientation after seeing results. Define a finite ungated proposal with the original radius, clipping, 0.05 projection and tie rule, but tau=0. Raw means ungated, not unbounded or unprojected. Report three fixed populations per pair/split:

1. All valid ungated proposals, including those projected/clipped to the anchor (all).
2. Ungated proposals whose projected weight differs from the anchor (proposed).
3. Actually executed nonzero corrections after the original gate (executed).

Harm means RR(action) - RR(anchor) < -1e-12, using the ungated action for the first two populations and the deployed action for the third. Verify executed is a subset of proposed, with identical raw and deployed actions/RRs there. A weight change need not change RR. For each population report n, harm/non-harm counts, original-triple cluster counts, AUROC, non-interpolated tie-aware average precision (AP), same-population prevalence and AP minus prevalence. The reference AUROC is 0.5; a constant score's AP equals prevalence when harm is present. Global's zero action/harm does not make it a harm classifier. Undefined one-class AUROC and zero-harm AP are not replaced with zeros.

Use 2,000 original-triple cluster bootstrap replicates, RNG seed 20260915 + 2*pair_index + split_index (DEV=0, TEST=1). Resample the full split's original triples with replacement, retaining all six associated seed/direction observations, then apply each fixed population mask. Reuse draws across the three populations. Compute two-sided 95% percentile intervals for prevalence, AUROC, AP and the paired AP-minus-prevalence difference; report each metric's valid replicate count. Empty/one-class replicates are omitted only for metrics that are undefined, with that conditioning disclosed. These are conditional on fixed models, OOF assignments and observed seeds, not training/selection uncertainty, new-checkpoint guarantees, or multiplicity-adjusted tests. Do not infer risk calibration from discrimination or a positive AP lift. Pre-sort scores once and use cluster multiplicities to avoid repeated fitting, sorting or large caches.

Keep the pre-repair confidence_harm report immutable and distinguish its weak, pair-dependent historical result from this corrected analysis. New results may differ; preserve unfavorable findings in either population. Do not choose deployment thresholds from this diagnostic.

## Verification and delivery

Unit-check weighted tied-score AUROC/AP against scikit-learn, multiplicity bootstrap against explicit cluster expansion, undefined populations, and the fact that equal endpoint gold ranks can have a different interior rank. Replay all 474,732 observations and reconcile tie/untied counts, utility sums and active/inactive harm counts. Generate small CSV/JSON summaries, formal PDF tables (including every seed/direction tie cell), a Chinese review and a hash-bound manifest. Do not commit per-query caches or large model files.

Metric references: [scikit-learn AP definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html), [SciPy percentile and paired bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Resampling original triples rather than rows is this study's clustering choice, not an independence guarantee for related graph triples.
