## 5. Sealed audit: pre-register, then open the labels once

Sections 3 and 4 were about *choosing* a configuration. This section asks whether a configuration that is already fixed can be **certified** on data nobody used to design it. The rule of the audit paper is blunt: the configuration, family and thresholds must be "fixed before the audit labels are observed" [P3 Thm 19, facered-q9], and reusing opened labels "violates Assumption 1, and voids the guarantee" [P3 Sec. 2.4, facered-q12].

**How the notebook enforces it.** The AUDIT split is a fresh 700-episode corpus from the same simulator, disjoint at the episode level from TUNE, CAL, CAL2 and TRUTH. The first half of the next cell writes everything that will be tested to `data/nb3/audit_prereg.json` (audit level delta = 0.05, P3's planning default [facered-q10]; the two configurations; the nested threshold family; both epsilons; the stress generator; the 45-second segment length) and appends an event to the log. Only then is the AUDIT split opened. CHECK 5.1 compares the two event-log positions.

**The two audited configurations.**
- **T2 choice**: the cheapest member of the section-3 certified set (thr 0.70, stride 1, gap 10, pad 20, margin 0.3).
- **Best-effort recall-first**: the configuration with the lowest pooled TUNE+CAL+CAL2 miss point estimate (thr 0.40, gap 10, pad 40). LTT did **not** certify it (T1 was empty), so only the audit can say anything about it.

**Five certificates, five different events** (do not merge them):

| | certificate | event it bounds | source |
|---|---|---|---|
| 5(a) | Prop. 26, Clopper-Pearson over audited face **instances** | the conditional miss rate L | [P3 Prop. 26, facered-q9] |
| 5(b) | excluded-pool missed mass, hypergeometric / binomial, on 45-s **segments** | absolute mass r of untouched segments holding a face | [P3 Thm 8, facered-q8; Thm 7, facered-q11]; segments **derived here** |
| 5(c) | fixed-sequence walk over a pre-registered nested family | which prefix passes one epsilon at full delta | [P3 Thm 19, facered-q9] |
| 5(d) | stress-test cluster bounds | escape probability under one *declared* perturbation generator | [P3 Defs 27-28, Thm 29, facered-q9] |
| 5(e) | pooled precision with an episode-cluster bootstrap | the client's precision | **bootstrap, not a paper guarantee** |

### 5(a) Prop. 26: the only basis for a recall claim

Under the audit design of P3 Thm 24, with Q audited positives of which C are missed, "with probability at least 1 - delta, L(g_J) <= U_Q(C_J, delta)" [P3 Prop. 26, facered-q9]. Here the audited positives are face **instances** (tracks), so U is a bound on the client's per-instance miss rate and 1 - U is a recall lower bound. This is the only certificate in the section that speaks to the 99% claim; the missed-mass bound of 5(b) does not, because "the implied bound is L(g) <= 0.5 ... which is not a high-recall guarantee" when faces are rare [P3 Remark 9, facered-q11].

A-priori checks: **5.1** the registration precedes the opening; **5.2** the audit is consistent with the LTT certificate for the T2 choice, i.e. the one-sided 95% Clopper-Pearson *lower* bound on its miss rate does not exceed T2's alpha_miss (a violation would mean the audit contradicts section 3). The client's 99% claim is *not* a pass/fail check: the cell reports whether U <= 0.01, whichever way it falls.

Paper numbers stay separate from these: P3's worked example is N = 100,000 with U_300(0, 0.025) = 0.01222 and M_U = 853 [P3 Sec. 8, facered-q8]; nothing below is meant to reproduce them.
