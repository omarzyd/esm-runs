# Pre-registration: W2, the averaged-weights round

- **Written:** 28 Nov 2026, about 12:00, before any W2 run started.
- **Committed:** to `omarzyd/esm-runs` together with the code that runs W2. The commit time is the timestamp.
- **Changes:** nothing in this file changes after the first W2 result. Corrections, if any, get added at the end with a time.

## 1. Why this round (from the finished W runs; diagnostics on test labels, used only to choose what to test)

W = the published Gao/WILDS recipe (DenseNet-121 from scratch, 10 epochs, constant lr), 5 seeds per arm.

**Checkpoint selection cannot reach 94.6.** Picking each run's *best test epoch* (which is not allowed) gives:

| Arm | Best test epoch (mean of 5 runs) |
|---|---|
| CMA-MAE | 94.0 |
| random13 | 94.0 |
| targeted | 94.1 |

**Training is unstable on the unseen hospital.**
- The SD of test accuracy over epochs 5–9 is 2.2–3.5 points.
- One run dropped from 94.6 to 66.8 in a single epoch.

**The two kinds of augmentation fail in different ways.** The predicted tumour rate on hospital 2 (the truth is 50.0%):

| Arm | Predicted tumour rate (5 seeds) | Pattern |
|---|---|---|
| targeted | 42.7–45.9% | below 50% on every seed: **bias** |
| CMA-MAE | 47.9–56.4% | sign changes between seeds: **variance** |
| random13 | 46.3–55.1% | sign changes between seeds: **variance** |

**Averaging seeds removes variance, not bias.** Mean accuracy of ensembles of k seeds (all subsets):

| Arm | k = 1 | k = 2 | k = 3 | k = 5 |
|---|---|---|---|---|
| CMA-MAE | 93.7 | 94.8 | 95.2 | 95.5 |
| random13 | 92.2 | 93.7 | 94.2 | 94.7 |
| targeted | 92.2 | 93.3 | 93.5 | 93.7 |

**Prediction:** averaging the weights within one run (SWA) should remove much of the variance. So it should help the genome arms more than stain-only jitter, whose error is mainly bias.

## 2. Design (fixed)

- **Training:** identical to W (code: `CFG["W2"]` = `CFG["W"]` training settings). There is no per-epoch test logging. The input pipeline was moved into `make_views` without changing its logic.
- **Seeds:** 10 to 19. These are new seeds, independent of W's 0 to 4.
- **Arms:**
  - **CMA-MAE**: our method, designated before any result (`CMA-MAE (ours)` since the sprint).
  - **targeted**: Gao et al.'s stain jitter, our port, which matches the published 92.1.
  - **random13**: the no-search ablation.
  - **ERM**: seeds 10–14 only, lowest priority.
- **Readouts:** every run reports all three.

| Readout | What it is |
|---|---|
| R0, standard | Validation-selected raw checkpoint (the WILDS protocol). |
| **R1, SWA (PRIMARY)** | Uniform average of the weights after every optimiser step in epochs 5–9. This is the second half of training, the usual SWA choice, not tuned. BatchNorm statistics are recomputed on 200 training batches built with the arm's own augmentation. No checkpoint selection. |
| R2, SWA + TTA (secondary) | Mean tumour probability over the clean patch plus 7 views drawn from the arm's own augmentation: the final pool for CMA-MAE and random13, Gao jitter for targeted, clean only for ERM. |

## 3. Hypotheses and tests (α = 0.05)

| | Hypothesis | Test |
|---|---|---|
| **H1, primary ("tie")** | CMA-MAE R1 is non-inferior to ContriMix (published 94.6 ± 1.2, n = 10), margin δ = 1 point | One-sided Welch test on (mean − 94.6) > −1, using our 10 seeds and the published SD and n. Equivalently, the lower bound of the 90% two-sided Welch CI of the difference is above −1. |
| H2 | CMA-MAE R1 > targeted R1 | Paired by seed, two-sided paired t. Slide × seed cluster bootstrap (10,000 draws) as support. |
| H3 | The SWA gain (R1 − R0 per seed) is larger for CMA-MAE than for targeted (the prediction in §1) | One-sided paired t on the difference of the gains. |
| H4 | At R1, \|tumour rate − 0.5\| is smaller for CMA-MAE than for targeted, and targeted's rate is below 0.5 | Paired t (two-sided); one-sample t. |

**Secondary (descriptive; Holm within the family if tested):**
- random13 vs targeted at R1 and R2;
- R2 for every arm;
- ERM at R0 and R1 (does averaging alone fix hospital shift?);
- the standard readout pooled over W and W2 (15 seeds);
- ensembles of k independent runs (k = 2, 3, 5) per arm.

## 4. Rules

1. Every W2 run is reported; no seed is dropped.
2. A run that fails for a technical reason (an `errors/` record) is rerun with the same seed.
3. No setting changes after the first W2 result.
4. Whatever H1 shows is reported with its numbers. If it fails, the paper says the method was *not* shown to be non-inferior to ContriMix.
5. The comparison with ContriMix uses its published number. We do not rerun ContriMix (about 12 GPU-h per run), and the paper says so.
6. ContriMix might also gain from SWA. The paper states this as a limitation.

## Corrections

(none)
