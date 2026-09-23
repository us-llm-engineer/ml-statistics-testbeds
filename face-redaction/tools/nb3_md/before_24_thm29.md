### 5(d) Stress-test cluster certificates (Thm 29)

**Claim.** For a stress-test generator A (a map from an item to a finite cluster of variants, fixed independently of the audit), the worst-cluster escape probability rho_max (some variant escapes) and the average-variant escape probability rho_avg (expected fraction of escaping variants) satisfy, simultaneously over a finite family G and with probability at least 1 - delta, rho <= U_Q(K, delta / (2|G|)) [P3 Defs 27-28, Thm 29, facered-q9]. The guarantee is relative to the declared generator A only; it is not a general robustness statement over continuous perturbation spaces [P3 Sec. 6.1, facered-q9].

**Generator A (derived here; declared and written to disk before the audit).** Each audited face instance is re-scored under three harder detector calibrations, with the base logit lowered by 0.5, 1.0 and 1.5 (a stand-in for poor light, motion blur, mirror-like surfaces). A cluster is {original, variant 1, 2, 3} of the *same* face. G = {T2 choice, best-effort}, so |G| = 2 and the level is delta / (2|G|) = 0.0125.

**A-priori check 5.7** is an identity, not a coverage test: K_max >= K_rand (a randomly chosen variant escaping implies some variant escapes) and the rho_max bound >= the rho_avg bound. The coverage of Thm 29 itself is not simulated in this notebook.
