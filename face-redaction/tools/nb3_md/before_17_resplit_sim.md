### 3(d) Verification by simulation

**Claim tested.** LTT's guarantee is a statement over repeated samples: if the whole procedure (path from TUNE, test on CAL) were repeated on fresh data, the chosen configuration's *true* risk should exceed its target in at most about a fraction delta of the runs `[P1 Thm 1, facered-q2]`. We check that by re-splitting.

**Design (derived here).** The 2,550 episodes of TUNE + CAL + CAL2 are re-split at random R = 200 times with the main run's sizes (650 TUNE, 950 CAL). For each re-split and each procedure, the cheapest configuration in Lambda-hat is chosen, and its "true" risk is taken from the separate TRUTH population (2,400 episodes, which is itself a finite sample, so "true" carries a small noise). A run is a *violation* if either true risk exceeds its target. Five procedures are simulated: T1; T2 with a single start; T2 with 3 and with 5 starts (each start tested at delta/|J|); and a **T2-tight** boundary case with alpha = 1.4 times the true risk of the main run's choice, so that some configurations are truly unsafe and the guarantee is not vacuous. **Disclosure:** a first attempt with factor 1.0 abstained on every re-split (LTT needs headroom); the factor was loosened to 1.4 *after seeing that*, so it is a demonstration knob, not a claimed threshold.

**CHECKs (a-priori threshold, decided before the run).**
- 3.2 / 3.3 / 3.5: for T1 / T2 / T2-tight, the violation rate over the 200 re-splits is at most delta + 3 binomial standard errors = 0.164.
- 3.6: abstain rate is non-decreasing from 1 to 3 to 5 starts (multi-start pays delta/|J| per path).

Two caveats. First, we test the *chosen* configuration only, which is a weaker event than the theorem's "every configuration in Lambda-hat". Second, CHECK 3.2 for T1 has nothing to test: if a procedure always abstains it never selects, so it cannot violate. Its PASS is vacuous and says only that abstention is safe.
