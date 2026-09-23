**What the output says.** Pre-registration precedes opening (5.1 PASS). On Q = 515 audited face instances at delta = 0.05:

| config | missed C | miss rate | U_Q(C, 0.05) | recall lower bound | U <= 1%? |
|---|---|---|---|---|---|
| T2 choice | 10 | 0.0194 | 0.0327 | 0.9673 | **no** |
| best-effort recall-first | 2 | 0.0039 | 0.0122 | 0.9878 | **no** |

- **The client's 99% recall is not certified for either configuration.** The best-effort configuration is close (0.9878), but its bound is above 1%. At its observed rate of 2/515, the "Q needed" column says about 628 audited instances would be enough; that is a larger audit, not a better bound. The T2 choice, at 1.9% observed, can never certify 1% at any Q.
- **T2's own target is corroborated**: its lower confidence bound on the miss rate is 0.0106, below alpha_miss = 0.05 (5.2 PASS), and U = 0.0327 is also below 0.05. The audit does not contradict the section-3 certificate.
- Section 2's sizing table already said what to expect: certifying a 1% miss rate needs Q >= 299, 473 or 628 audited instances for C = 0, 1 or 2 misses; this audit has Q = 515.
- The best-effort configuration buys its recall with false blur (pooled precision in 5(e)).
