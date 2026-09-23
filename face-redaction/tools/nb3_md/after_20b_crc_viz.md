#### How to read this chart

Four panels, synthetic data; "true" risks come from the 2,400-episode TRUTH population.

- **Top left.** x = miss target alpha; y = fraction of calibration sets (950 episodes) whose selected configuration has true miss above alpha; dashed line = delta = 0.10. Orange (CRC) peaks near 0.12 at alpha = 0.05 and is 0.08 to 0.09 at 0.06 and 0.07. Green (LTT-1D) is flat near 0.003, but this hides abstention: 100% and 99% of runs at alpha = 0.03 and 0.04.
- **Top right.** x = face instances in the calibration set (log); y = fraction of calibration sets. CRC stays above delta at every size (0.26, 0.29, 0.23, 0.14); LTT-1D's violation stays near or below 0.05 and its dotted abstain rate falls from 0.91 to 0.67 as data grows. This is where the two guarantees really differ.
- **Bottom left.** Mean true miss of the selected configuration: CRC 0.036 (instances) and 0.086 (episode-averaged), LTT-1D 0.031, LTT-full 0.024, against alpha = 0.05. Only the episode-averaged bar crosses the line: face-free episodes dilute its loss.
- **Bottom right.** Mean true pooled precision, 0.47 to 0.59, all far below the client's 0.95.

For the job: rest the recall promise on a 1 - delta certificate (LTT, then the sealed audit), not on an expectation bound.
