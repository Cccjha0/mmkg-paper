# B13/B14: winner labels, action utility and preference semantics

This retrospective supplement uses the six existing pairs and their audited 21-weight RR caches. It does not score base checkpoints, replace historical results, adopt a new TEST policy or claim confirmation. All counts retain three base seeds and both directions. Existing evaluation fact scopes and raw endpoint dispatch remain unchanged.

## B13: fixed-policy, answer-aware diagnostics

Use the original grouped held-out DEV anchor/radius/action per fold, and the original full-DEV lock on TEST. For each labeled observation, inspect only grid points inside its fixed radius and [0,1]. A downward/upward direction has opportunity when at least one strictly lower/higher permitted grid point improves RR over the anchor by more than 1e-12. Report all four cells (down-only, up-only, both, neither), all endpoint winner classes including ties, and all six pairs/splits. These gold-aware maxima are descriptive opportunity probes, not a new oracle policy, trained target, inference feature, or deployable guarantee.

The winner suggests increasing primary weight for A wins and decreasing it for B wins. Separate absent directions at endpoints from feasible directions without a profitable point. Among non-tied observations with opportunity and a feasible winner-suggested direction, quantify cases with gains only on the opposite side. Also report blocked-direction opportunity counts. For actual nonzero ADC interventions aligned with the endpoint winner, report benefit/harm frequencies, conditional loss, mean loss and utility; a correct direction label does not ensure the chosen magnitude is useful. Record neighboring 0.05-step outcomes separately from best-in-radius outcomes. All tables state their conditioning denominators and retain ties/no-action observations in complete strata and decompositions.

## B14: DEV-only class-weight control

For each of six pairs, reproduce the five original triple-grouped folds. Within each fold fit the same 13-feature median/scaling/liblinear logistic pipeline on the same pooled non-tied training rows. Compare class_weight='balanced' with class_weight=None, keeping C=1, max_iter=2000 and the original random states. This is 30 balanced reconstructions and 30 new small unweighted fits, with one CPU thread; no final full-DEV control model or new TEST application. Preprocessing must match across weighting choices. Stop on convergence failure or failure to reproduce original held-out ADC weights/RR and anchors with the balanced fit.

For both learners, independently select ADC+Global from the same 41 configurations using only the outer training portion's MRR and existing matched-family tie rules. Query-soft uses the same learner without an extra fit or tuning. Score all held-out observations for policy utility. Assess unweighted Brier score, log loss, ROC AUC and 10 equal-width-bin ECE on the naturally weighted held-out **non-tied** endpoint-winner population; also report the training-portion winner prior as a probability-only baseline. Brier/log loss mix discrimination and calibration; ECE is descriptive and bin-dependent. A class-unweighted fit is not automatically calibrated.

Serialize DEV results, all candidate scores/choices, fold fitting scopes, model/preprocessing states and their hashes before the TEST diagnostic phase. The DEV loader refuses TEST paths. TEST diagnostics apply no new learner, calibration or parameter choice. Both weighting alternatives are retained regardless of their outcomes.

## Interpretation

The primary object is the signed logistic margin g; p_A=sigmoid(g) is a class-balanced preference score. Balanced binary weights are n/(2n_c). For an ideal unrestricted weighted-log-loss optimum with fixed class weights, q*=w1*pi/(w1*pi+w0*(1-pi)); this identity does not assert that a regularized, misspecified fitted model attains that optimum. Endpoint ties were excluded, so even a calibrated estimate would concern a specified non-tied winner population. Natural endpoint-win probability and probability of RR harm under a particular action are different events. No harm probability or calibration guarantee is inferred from g, p_A or |2p_A-1|.

Primary references checked for this review: scikit-learn 1.7 LogisticRegression class_weight documentation and its probability-calibration guide; manuscript calibration context also cites Guo et al. (2017). No historical pre-repair calibration snapshot is reused as corrected evidence.
