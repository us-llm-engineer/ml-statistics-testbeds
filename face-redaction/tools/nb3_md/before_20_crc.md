## 4. SeqCRC-style comparison: "in expectation" is not "with probability 1 - delta"

Section 3 used Learn then Test (LTT): a statement about the calibration draw, "with probability at least 1 - delta the chosen configuration's true risk is at most alpha" [P1 Thm 1, facered-q12(a)]. The second paper, SeqCRC, gives a different kind of statement. This section runs both on the **same data, the same miss target (alpha = 0.05, T2's) and the same calibration sets**, so the difference is visible instead of asserted.

**What is implemented**

- **One-parameter CRC** on the confidence threshold: lambda-hat = inf{lambda : (1/(n+1)) sum_i L_i(lambda) + B/(n+1) <= alpha} [P2 Eq. 1, facered-q4]. Here lambda = 1 - score threshold, B = 1, and the loss L_i is the miss indicator of one **face instance** (unit choice: **derived here**). The risk curve is made monotone by a running maximum, in the spirit of P2's monotonisation trick (Sec. IV-E) (**derived here**). The two-step localisation/classification stage of SeqCRC is **not implemented** (budget).
- **What the theorem says.** Theorem 1 gives E[L_test(lambda-hat)] <= alpha, where the expectation is over the calibration data *and* the test point; P2 itself warns: "There is no guarantee that (2) holds for all possible test instances" [P2 Sec. III-A, facered-q4]. So the mean true miss over many calibration draws is controlled; the probability that *one* calibration draw lands on a configuration with true miss above alpha is not. It must never be written as "recall >= 99% with confidence 1 - delta".
- **Exchangeability is the price of admission.** The proof needs the loss sequence to be "i.i.d. (by Assumption 1) and thus exchangeable" [P2 App. A-A, Step 3.c, facered-q4]. Frames within an episode and detections within a track are dependent, and none of the three papers covers video [facered-q11, facered-q12(d)]. Here the loss unit is the face instance and the resampling unit is the episode (**derived here**).
- **Precision is not controlled.** Verbatim: SeqCRC "guarantees a large enough recall but only empirically limits false positives" [P2 Sec. V-D, facered-q5]. Precision can only come from LTT's second risk (section 3).
- **Face-free frames dilute the risk.** P2 gives an image with no ground truth loss zero: "we set the loss to zero" [P2 Sec. IV-E, facered-q6]. On face-sparse footage an *episode-averaged* loss therefore understates the per-instance miss rate; the cell also runs that variant to show the size of the effect (**derived here**).
<!-- cell -->
**Setup and a-priori claims (fixed before the numbers were seen).** Three procedures run on R = 300 random 950-episode calibration sets drawn from the 1,900-episode CAL+CAL2 pool; "true" risks come from the separate 2,400-episode TRUTH population: **CRC (instances)**, **LTT-1D** (miss risk only, fixed sequence from the safest threshold, exact binomial p-values, delta = 0.10) and **CRC (episode-averaged)**. Two threshold ladders were fixed by hand: *coarse* 0.600 to 0.850 in steps of 0.025 and *fine* 0.70 to 0.80 in steps of 0.01; the fine ladder was placed around where the miss curve crosses alpha, i.e. with the knowledge of section 3 (a demonstration knob, not a discovered optimum). With R = 300 and delta = 0.10 the Monte Carlo tolerance is delta + 3 SE = 0.152.

| CHECK | a-priori claim (PASS means the claim was met) |
|---|---|
| 4.1a/b | CRC's *mean* true miss <= alpha + 3 MC SE (expectation control holds) |
| 4.2a/b | CRC's per-split violation fraction **exceeds** 0.152 (written before the run, coarse ladder first) |
| 4.3a/b | LTT-1D's violation fraction <= 0.152 [P1 Thm 1] |
| 4.4 | the episode-averaged CRC's instance-level true miss is >= 1.3 x alpha on average |
| 4.5 / 4.6 | over alpha in {0.03 ... 0.07}: CRC's mean stays <= alpha and its violation exceeds 0.152 at some alpha / LTT-1D's violation stays <= 0.152 at every alpha |
| 4.7 | with a 100-episode calibration set CRC's violation exceeds 0.152 while LTT-1D's does not |

Claims 4.2 and 4.5 were the "SeqCRC violates far more than delta" story. The output decides whether that story survives; it is reported as found below.
