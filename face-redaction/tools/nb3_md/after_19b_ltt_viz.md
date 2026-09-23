#### How to read this chart

**Top left.** Each dot is one of 58 configurations at its CAL2 point estimate (x: false-blur risk, y: miss risk). Dashed boxes are the targets: small red T1, larger green T2; stars are certified. No dot lies inside T1. Several grey dots lie inside T2 yet are not certified: a point estimate inside a box is not a certificate.

**Top right.** The x-axis is the path learned on TUNE (0 looked safest); grey is TUNE max-p, green CAL2 max-p, dashed is delta. The walk accepts positions 0 and 1, fails at position 2 (p about 0.7) and stops, although later positions dip below delta. That stopping point is the pattern: path order decides what can be certified.

**Bottom left.** Each dot is one of 200 re-splits; x is the true miss risk of the selected configuration, dashed is alpha_miss. Dots left of the line are honest selections; the two right of it (3-start procedure) are violations. Abstain rates are printed.

**Bottom right.** Bars show how far the best configuration is from T1 at each detector level (1 = both targets met by point estimate). None is certified.

For the job: certification belongs to the procedure, not to a point estimate, and T1 needs both a better detector and more labelled data.
