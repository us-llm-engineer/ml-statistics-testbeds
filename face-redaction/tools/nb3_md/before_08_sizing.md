## 2. Data design and staged-capture sizing

**The question.** How many labelled face instances must the client obtain to *prove* recall >= 99%, and how much footage does that mean?

**What the paper gives (PAPER NUMBERS).**
- The zero-miss planning rule: with no observed misses, certifying a miss rate of at most epsilon needs n0 >= log(1/delta) / log(1/(1 - epsilon)) draws, "rule of three" n0 ~ 3/epsilon at delta = 0.05 `[P3 Sec. 4.6, facered-q7]`. The printed table gives 299 draws for epsilon = 0.01 and 528 for 10 pre-specified prefixes `[P3 Sec. 4.6, facered-q8]`. CHECK 2.1 recomputes all 10 printed numbers.
- **Careful about what those 299 draws certify.** The table is about the *excluded pool*: eta(g) = P(Y = 1 | g(X) = 0). Remark 9 explains why that is not a recall claim: with rare positives, a small missed mass can still be a poor recall, "which is not a high-recall guarantee" `[P3 Remark 9, facered-q11]`. So the 99% claim in this notebook rests on **Proposition 26** over audited face instances: with Q audited positives of which C are missed by the final configuration, L(g) <= U_Q(C, delta) holds with probability at least 1 - delta, where L is the miss rate among positives and U_Q is the exact Clopper-Pearson upper bound `[P3 Prop. 26, facered-q9]`. Recall is 1 - L.

**What is derived here.**
- The same formula, applied to audited *positives* and allowing C misses, gives the minimum Q for each C (CHECK 2.2: C = 0 needs 299, the same arithmetic as the paper's table but a different object; Q increases with C; each Q is an exact threshold).
- Whether the sealed AUDIT split is large enough (CHECK 2.3: at least 299 instances; passing this does *not* mean it can absorb misses, see the printed maximum C).
- Footage minutes and labelling cost. The staged-capture rate (instances per footage minute) and the labelling price are **assumed** parameters you must replace; the natural face density is **synthetic**, measured on this generator's corpus. Note P3 Assumption 2 requires noiseless audit labels, "with no labelling noise" `[P3 Assumption 2, facered-q7]`; in practice that means adjudicated labels, which the price should reflect.
- A staged-capture composition that over-samples the kinds of face the detector misses most (per-kind miss rates measured on TUNE at one reference configuration). CHECK 2.4 tests the design rules: the kind with the highest TUNE miss rate is over-sampled relative to its natural share, and every kind gets at least 30 instances.

**Marginal guarantee.** P1 warns "all the guarantees in this paper are marginal" `[P1 Sec. 5, facered-q3]`, and P3's Proposition 26 is also an average over the audited positives. A 99% certificate across all faces does not say the rate is 99% for children or for small distant faces; that is why the composition matters.
