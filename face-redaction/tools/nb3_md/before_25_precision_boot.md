### 5(e) The client's pooled precision, with a bootstrap interval

**Claim (and its limit).** None of the three papers gives an audit certificate for precision: SeqCRC "only empirically limits false positives" [P2 Sec. V-D, facered-q5], and P3 notes that a precision certificate needs a bound on the false-positive mass as well as the burden, which excluded-pool sampling does not provide [facered-q12(a)]. LTT's false-blur risk (section 3) is a different quantity: it is the episode-*averaged* false-blur fraction, whereas the client's precision is *pooled* over all blur box-frames. Both are reported.

The interval below is a **percentile bootstrap over episodes** (B = 2,000; the episode is the resampling cluster because boxes inside an episode are dependent). It is **not** a P1/P2/P3 guarantee and does not carry a 1 - delta statement.

**A-priori check 5.8**: each pooled-precision point estimate lies inside its own 95% bootstrap interval (a sanity check on the code, not on the client's target; whether 0.95 is reached is reported as found).

The cell ends with the protocol for a failed audit, from the audit paper's limitations: the certified configuration is now a *number*, not a design freedom; changing the pipeline in response to opened labels and re-certifying on the same labels voids the guarantee, and peeking to decide whether to sample more without pre-registration is not allowed either [P3 limitations 1-2, facered-q9]. What is allowed: a **fresh, unopened audit split**, or a **pre-registered error-spending plan** across audit rounds.
