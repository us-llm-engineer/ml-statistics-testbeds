### 3(b) Evaluate the grid on TUNE, fix the targets, then open CAL

**What this cell does.** It evaluates all 58 configurations on TUNE only and prints the frontier of TUNE point estimates (false-blur risk against miss risk). It also asks, for information only, whether any configuration meets the client's targets **T1** (miss <= 0.01, false-blur <= 0.05) on TUNE.

TUNE point estimates carry **no guarantee**: they are what the path and the targets are learned from. This cell has no `CHECK` because it asserts nothing; it only reports.
