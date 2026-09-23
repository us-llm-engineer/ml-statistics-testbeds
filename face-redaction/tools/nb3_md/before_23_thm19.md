### 5(c) Fixed-sequence certification along a pre-registered nested family (Thm 19)

**Claim.** If the component detectors, the cut points and the target epsilon are "fixed before the audit labels are observed", test the prefixes "in the fixed order" g_M, g_(M-1), ..., g_1, stop at the first failure, and return the last prefix that passed; then with probability at least 1 - delta either nothing is returned or the returned prefix satisfies r(g) <= epsilon [P3 Thm 19, facered-q9]. Each test runs at the **full** level delta: no delta/M union-bound penalty, at the price of a single pre-registered epsilon and no simultaneous bounds for the other prefixes [P3 Sec. 4.8, facered-q9].

**Family (pre-registered in the JSON before the audit was opened).** The score threshold from 0.40 (g_M, most inclusive) to 0.90 (g_1, least inclusive), 11 members, everything else fixed (gap 30, pad 20, margin 0.3, stride 1). The theorem needs the prefixes to be **nested** (E_M contained in ... contained in E_1). The cell measures whether the pipeline really is nested, on the audit itself.

**Two versions, kept apart.**
- **Segment-mass version, as stated in the paper**: pass_m = the Thm 8 bound on the excluded pool of prefix m is at most epsilon_seg = 0.01, with n_m = 300 fresh draws per prefix.
- **Instance-level twin (derived here)**: the same walk at the same delta, with pass_m = [U_Q(C_m, delta) <= epsilon_inst], where epsilon_inst = 0.05 is T2's alpha_miss, pre-registered. This is Prop. 26 inside a Thm 19 walk. It is **not** stated in the paper. The cell also reports where the walk would stop with a delta/M union bound, to show what "no penalty" is worth.

**A-priori checks.** **5.5** nesting is exact: over the 10 threshold steps no audited instance is missed at a lower threshold yet covered at a higher one, and no segment is excluded at a lower threshold yet boxed at a higher one (zero violations required; otherwise the nesting is only approximate and the theorem's assumption is only approximately met). **5.6** the audited instance miss rate is non-decreasing along the family.
