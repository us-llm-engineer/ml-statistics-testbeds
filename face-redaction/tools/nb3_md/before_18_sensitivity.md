### 3(e) What detector quality would make T1 certifiable? (exploratory)

**Question (derived here, exploratory).** T1 fails at this detector quality. How much better would the detector have to be? The cell shifts the simulator's detector calibration through the parameters of Notebook 2's `simulate_detector` (the face-logit intercept `base`, the false-positive logit scale `fp_gain`, and the clutter rate) and, for levels L2 to L4, also switches the screen/photo policy so that faces on TVs and photographs count as faces to redact (that changes the ground truth, not the detector). It re-evaluates a small core grid (threshold x padding at gap 30, plus strided configurations) on TUNE and CAL and runs the same LTT procedure for T1 at each of the four levels L1 to L4; L0 is the main run.

Why pets matter here: Notebook 2's decomposition shows pets as the largest source of false blur on this synthetic corpus, because the generator gives pets the highest false-positive logit and the longest tracks. A lower `fp_gain` therefore removes that modelling choice's effect. This is **synthetic**; it says nothing about real pets.

**Caveats to keep in view.**
- **Exploratory.** The four levels share the delta budget (delta/4 = 0.025 each, a Bonferroni split), all tested on the already-opened CAL. The levels are not pre-registered targets; this table is a sensitivity analysis, not a certificate.
- "Best config" is chosen by CAL point estimates, so its numbers are optimistic (selection on the same data).
- "Instances needed" is what the binomial test would need if that miss rate persisted on a larger sample; it addresses the miss risk only.

**CHECK 3.4 (a-priori).** The levels are ordered by detector quality: the best CAL point-estimate normalised excess over T1 (max of miss/0.01 and false-blur/0.05; at most 1 means both targets met by point estimate) is non-increasing from L1 to L4.
