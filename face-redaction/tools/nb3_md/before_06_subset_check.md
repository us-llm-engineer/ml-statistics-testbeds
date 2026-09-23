### Check that subsets evaluate like the full corpus

**Claim (derived here).** Evaluating a *subset* corpus with a fresh ground-truth cache gives exactly what restricting an evaluation of the *full* corpus to those episodes gives. If this failed, the split-based accounting in the rest of the notebook (each split evaluated on its own) would be silently wrong.

**CHECK 1.5 (a-priori: exact equality).** For two different configurations, the per-instance miss flags and uncovered-frame counts, the box and false-box counts, and the per-episode reductions are identical.
