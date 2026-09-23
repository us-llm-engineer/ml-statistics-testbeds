**Reading the outcome.**

- **T1 is certified at none of L1 to L4** (`T1_certified` is False in all four rows; the cheapest-certified column is "none (abstain)").
- Some configurations meet T1 **by point estimate only**: none at L1 or L2, two at L3 and five at L4. Point-estimate success is not a certificate; with a finite CAL sample, the test at delta/4 still abstains.
- The normalised excess of the best configuration falls from about 6.2x (L0) to 3.8x, 3.0x, 0.86x and 0.83x at L1 to L4. At L4 the best CAL miss rate is 0.0043 and the best false-blur risk 0.0415 (pooled precision 0.949): the detector is no longer the obstacle, but the calibration sample may be: the miss test alone would need about 1,025 instances at that rate (about 4,750 at L3), against 633 in CAL.
- So even a much better detector needs a larger calibration set before T1 can be *proven*, which links back to the sizing in section 2. CHECK 3.4 passes with excess [3.81, 2.98, 0.86, 0.83].
