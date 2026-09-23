**Reading the outcome** (numbers from the printed table, R = 200).

| Procedure | Abstain rate | Violations |
|---|---|---|
| T1 | 1.000 | 0 / 200 (vacuous: it never selects) |
| T2, single start | 0.240 | 0 / 200 (worst true miss 0.0346 against 0.05) |
| T2, 3 starts | 0.305 | 2 / 200 (0.010) |
| T2, 5 starts | 0.355 | 0 / 200 |
| T2-tight (alpha 0.0362, 0.5089) | 0.230 | 0 / 200 |

- All four CHECKs pass their a-priori thresholds. None needed the delta + 3 SE allowance, since every violation rate is below delta = 0.10 itself, but the guarantee is an upper bound: 0 of 200 is consistent with it and does not show it is tight. Even the tight case sits well below delta.
- **T1 abstains on every re-split**: the failure on the main run is not bad luck of one split.
- **Multi-start costs power** as expected: the abstain rate rises from 0.24 to 0.305 to 0.355.
- When T2 certifies something (152 of 200 runs), the cheapest certified configuration has stride > 1 in 64% of those runs, whereas the main run's certified set contained only stride-1 configurations; the main run is one draw, not the typical case.
