**What the output says (synthetic data, R = 300 calibration sets, alpha = 0.05).**

- **Expectation control holds** (4.1a/b PASS): CRC's mean true miss is 0.0301 (coarse ladder) and 0.0364 (fine ladder), both below 0.05.
- **The "violation" claim was NOT met at the main calibration size** (4.2a, 4.2b and 4.5 are honest FAILs). CRC's per-split violation fraction is 35/300 = 0.117 on both ladders: above delta = 0.10, but inside the Monte Carlo tolerance of it (0.152), and the sweep over alpha peaks at 0.12. The ladder's true miss risks step over 0.05 between 0.043 and 0.051, so many of these violations are small overshoots. At 950 episodes SeqCRC-style calibration is *not* dramatically worse than delta.
- **LTT-1D's violation fraction is 1/300 = 0.003** (4.3a/b PASS), but that is bought with abstention: 0.000 on the coarse ladder, **0.697** on the fine ladder, and 1.00 / 0.99 / 0.70 / 0.15 / 0.003 at alpha = 0.03 / 0.04 / 0.05 / 0.06 / 0.07 (4.6 PASS). LTT answers "no certified configuration" where CRC always answers.
- **The gap opens with small calibration sets** (4.7 PASS): with 100 episodes (about 66 instances) CRC's violation is 0.263 against LTT-1D's 0.053 (abstain 0.91); at 200 / 400 / 950 episodes CRC is 0.293 / 0.230 / 0.140.
- **Face-free episodes dilute an episode-averaged loss** (4.4 PASS): the selected configuration's instance-level true miss averages 0.0862 against alpha = 0.05, and it violates in 300/300 splits.
- **Precision is nobody's win:** mean true pooled precision is 0.47 to 0.59 for every procedure, far from the client's 0.95.

Honest summary: CRC controls what it says it controls (the mean), never abstains, and is cheaper in data; LTT controls the probability, and pays with abstention that grows as the calibration set shrinks.
