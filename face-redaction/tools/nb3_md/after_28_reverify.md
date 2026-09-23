**What the output says** (delta = 0.05):

| case | Q | C | miss rate | recall lower bound | pooled precision | LTT verdict | client 99%/95% |
|---|---|---|---|---|---|---|---|
| (i) unchanged | 356 | 6 | 0.0169 | 0.9670 | 0.569 | PASS | FAIL |
| (ii) model change | 392 | 77 | 0.1964 | 0.7676 | 0.600 | FAIL | FAIL |
| (iii-a) thr 0.90 | 403 | 104 | 0.2581 | 0.7036 | 0.727 | FAIL | FAIL |
| (iii-b) thr 0.60 | 343 | 6 | 0.0175 | 0.9658 | 0.448 | INCONCLUSIVE | FAIL |
| (iv) 40 episodes | 32 | 4 | 0.1250 | 0.7364 | 0.586 | INCONCLUSIVE | FAIL |

- **The unchanged configuration is not flagged** (7.2 PASS: p_max 0.0010), and the two changes that hurt recall **are caught**: the model shift (miss-rate lower bound 0.1639 > 0.05, 7.3 PASS) and the aggressive threshold (lower bound 0.2224, 7.4 PASS).
- **Lowering the threshold to 0.60 is not clearly good or bad**: misses stay at 6/343, but the false-blur p-value is 0.3053, so the harness says INCONCLUSIVE ("borderline / more data needed"), not PASS.
- **A small audit cannot certify.** With 32 instances the LTT verdict is INCONCLUSIVE (Q = 32 < 59 needed even with zero misses). Its client verdict is **FAIL**, not INCONCLUSIVE: the 4 observed misses already reject the 99% claim and the precision interval sits far below 0.95. The check 7.5 only asserts that the verdict is not PASS at Q < 299; the honest reading is "never PASS at small Q; FAIL when the observed misses already reject the claim, INCONCLUSIVE otherwise".
- **The client's claims FAIL in every case, mostly on the precision leg** (pooled precision 0.45 to 0.73 against 0.95). In case (i) the recall leg alone is not rejected (the miss-rate interval spans the 1% line) but is not established either.
- 7.1 (regression: Q = 515, C = 10, U = 0.0327) and 7.6 (report round trip and determinism) PASS.

**Protocol:** any design, threshold or model change requires a fresh sealed audit set; the certificates are valid only for a configuration fixed before its labels were opened.
