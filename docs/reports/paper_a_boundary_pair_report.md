# Paper A boundary / stress-pair report

## Execution status

The analysis pipeline is implemented and frozen, but the two DynaSemble selectors and exact full-ranking applications have not been run in this local environment. This placeholder is overwritten by `scripts/analyze_paper_a_boundary_pairs.py` after the A100 DEV and TEST stages complete. Existing closure-test rows are reused when present; otherwise the runner exports the two missing TEST assets from frozen checkpoints without retraining base models.

## Pre-TEST reliable-primary audit

| Dataset | Selected primary | DEV primary delta | 3/3 seeds? | DEV CI95 | Regime |
| --- | --- | ---: | --- | --- | --- |
| MKG-W | NativE | +0.038696 | Yes | [0.034434, 0.043062] | reliable-primary |
| DB15K | NativE | +0.029004 | Yes | [0.026140, 0.031850] | reliable-primary |

The initial “no obvious reliable primary” premise does not hold under the frozen Paper A rule. These pairs remain pre-specified combination-level stress cases, and all outcomes will be retained. The final report must not attribute poor or unstable correction to missing primary reliability unless a genuinely non-reliable-primary pair is evaluated.

## Frozen A100 execution

```powershell
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage dev -Python python -Device cuda
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage test -Python python -Device cuda
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage analyze -Python python
```

No boundary-pair result may be used to retune Anchored Dynamic or revise the main-pair method.
