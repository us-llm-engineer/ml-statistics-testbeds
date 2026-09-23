## 7. Re-verification harness

**Claim.** A certificate belongs to one configuration fixed before its labels were opened [P3 Assumption 1, Sec. 2.4, facered-q12(b)]. When the model, the threshold or the footage changes, the old certificate says nothing about the new system: the protocol is **any design, threshold or model change needs a fresh sealed audit set**. The cell packages what section 5 did into one plain-Python function so the client, or you, can re-run it: `reverify(config, labelled_corpus, detections, delta, targets)`.

**What it recomputes** (from a fresh labelled audit set only): the instance count Q and misses C, the miss rate with the Prop. 26 upper bound and the Clopper-Pearson lower bound [P3 Prop. 26, facered-q9]; LTT p-values for both risks (binomial for the miss indicator, Hoeffding-Bentkus for the episode-averaged false-blur fraction) and their maximum, following LTT's rule for several risks [P1 Prop. 6, facered-q12(a)]; the pooled precision with an episode-cluster bootstrap interval (a bootstrap, not a guarantee).

**Verdict rules (fixed here, before the demonstrations).** LTT targets: **PASS** if both p-values are <= delta; **FAIL** if the data reject the target (the Clopper-Pearson lower bound of the miss rate exceeds alpha_miss, or the Hoeffding lower bound of the false-blur risk exceeds alpha_fb); **INCONCLUSIVE** otherwise, with a reason (for example Q too small to certify even with zero misses). The client's claims get a separate verdict: FAIL if the recall leg or the precision leg is rejected by the data, PASS only if U_Q <= 1% *and* the bootstrap lower limit is >= 0.95, INCONCLUSIVE otherwise.

**What it does not certify.** Any other configuration; a configuration or threshold changed after these labels were seen; footage unlike this generator (distribution shift, which no source covers [facered-q11]); temporal dependence beyond "instances are exchangeable"; noisy labels [P3 Assumption 2, facered-q11]; and the precision interval, which is a bootstrap.

**Demonstrations and a-priori claims** (five fresh audit sets from new seeds, delta = 0.05, targets alpha_miss = 0.05, alpha_fb = 0.45 plus the client's 99% / 95%):

| case | change | a-priori claim |
|---|---|---|
| (i) | none, fresh 500-episode set | LTT verdict PASS (7.2) |
| (ii) | detector base logit lowered by 1.0, from 1.0 to 0.0 (a synthetic model change) | caught: FAIL (7.3) |
| (iii-a) | threshold 0.70 to 0.90 | caught: FAIL (7.4) |
| (iii-b) | threshold 0.70 to 0.60 | shown, no claim made |
| (iv) | none, but a 40-episode set | the client's 99% claim is never PASS at Q < 299 (7.5) |

CHECK 7.1 is a regression test (on the sealed AUDIT the function must reproduce section 5's Q, C and Prop. 26 bound exactly); 7.6 checks that the JSON report round-trips and that two runs give identical results.
