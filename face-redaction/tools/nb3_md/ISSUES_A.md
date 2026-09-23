# Issues found while writing the first half of Notebook 3 (stems 01 to 19b)

No code cell was modified. Nothing below invalidates a CHECK verdict; items are for the coordinator to decide.

1. **17_resplit_sim.py, CHECK 3.2 (T1) is vacuous.** T1 abstains on all 200 re-splits, so no configuration is ever selected and "violations 0/200" cannot fail. The markdown says so. Suggested fix: keep it, but label the description "(vacuous if the abstain rate is 1)" or exclude it from PASS tallies.
2. **17_resplit_sim.py, CHECKs 3.2/3.3/3.5 threshold is loose.** The threshold delta + 3 SE = 0.164 is a legitimate a-priori bound, but the observed rates (0, 0, 0; J=3 has 2/200) sit below delta itself, so the tolerance was never used. It tests only the *chosen* configuration, a weaker event than the theorem's "all of Lambda-hat". The markdown discloses both.
3. **18_sensitivity.py, stale comment and brief mismatch.** The code comment says "exploratory levels L1-L3 share the delta budget", and the coordinator's brief says delta/3 per level, but `SENS_DELTA = DELTA / (len(SENS_LEVELS) - 1)` = 0.1/4 = 0.025 and four levels (L1 to L4) are run. The markdown says delta/4 = 0.025 per level (the code wins). Suggested fix: change the comment to "L1-L4".
4. **19_ltt_summary.py, CHECK 3.7 is a tautology.** The chosen config is the cheapest in Lambda-hat by construction, so saving >= 0 always holds. Suggested fix: relabel as informational or replace by "the printed saving equals cost(most conservative) - cost(chosen)". The markdown calls it weak.
5. **19_ltt_summary.py, hard-coded string.** The printed line contains "-- 12x the client's 0.05" as fixed text; it matches today's number (0.607 / 0.05 = 12.1) but would go stale on a re-run with different numbers. Suggested fix: compute it.
6. **08_sizing.py, CHECK 2.4 description vs code.** The description says every kind gets >= 30 instances, the code tests >= 29 and allows the total to differ by up to 7 (`len(FACE_KINDS)`). Observed minimum is 33 and total 629 vs 628, so it passes at the stated threshold too; the tolerance was a rounding allowance. Suggested fix: make the code test >= 30.
7. **04_corpus.py, CHECK 1.3 was redefined after an earlier failure.** The cell comment discloses this (first draft compared against NB2's realised shares within +-0.05 and failed at 0.065). The current comparison is to NB2's design weights within 3 SE, which is a defensible a-priori rule, but the markdown mentions the redefinition.
8. **15_t2_revision.py, wording "(no CAL1 or CAL2 label used)"** in the docstring describes the *inputs* of the T2-v2 rule, which is true, but the rule itself was designed after seeing T2-v1 fail on CAL1. The markdown states this explicitly. Suggested fix: add "(rule designed after the CAL1 failure)" to the `PREREG["T2_v2"]["reason"]` field, which it already partly does.
9. **19b_ltt_viz.py, bottom-right panel, cosmetic.** The red annotation "below this line: T1 met by point estimate (not certified)" overlaps the L3/L4 bar labels ("0.9x", "0.8x"). Suggested fix: move the text above the bars or to the right.
10. **13_targets.py, CHECK 3.1** verifies only that the last event is the targets event and that T2-v1 is looser than T1, not that the rule was applied correctly. The markdown says so. No change needed.

## Statements in the coordinator's brief that differ from the executed outputs

- The brief gave the T2-v2 fixed time as 10:47:07; `data/nb3/prereg_targets.json` from the executed run shows `fixed_at` 2026-09-19 11:14:03 for T2_v2 (11:13:14 for T2_v1), i.e. from the latest run. The markdown does not quote a clock time and points to the file instead.
- delta/3 per sensitivity level: the code and output use delta/4 (issue 3).
- The brief says "some configs meet T1 by point estimate only" for the sensitivity; the outputs show this is true only at L3 (two) and L4 (five), and none at L1 or L2.
