## 3. Decide with Learn then Test

**The question.** Is there a configuration in the grid whose *miss risk* and *false-blur risk* are both below the client's targets, with probability at least 1 - delta, and if so which is the cheapest?

### 3(a) The LTT building blocks

The cell implements the pieces compactly; nothing is imported from Notebook 1. What each piece is, and where it comes from:

- **Guarantee.** A random choice lambda-hat is an (alpha, delta)-risk-controlling prediction if P(R(T_lambda-hat) <= alpha) >= 1 - delta `[P1 Def. 1, facered-q2]`. LTT tests, for every grid point, the null H_j: R(lambda_j) > alpha with p-values; any FWER-controlling procedure over those nulls yields the guarantee `[P1 Thm 1, facered-q2]`. If nothing is rejected, "the supremum over an empty set is defined as -inf" `[P1 Thm 1, facered-q2]`, and in practice we "abstain from returning an RCP" `[P1 Sec. 1.1, facered-q3]`. Abstaining is a legitimate outcome, not a failure of the code.
- **Level.** delta = 0.10, the paper's default example `[P1 Def. 1, facered-q2]`.
- **p-values.** For a binary loss, the exact binomial tail P(Bin(n, alpha) <= k), which is "the exact binomial tail bound" `[P1 Sec. 3.2, facered-q2]`; for a loss in [0, 1], the Hoeffding-Bentkus p-value `[P1 Prop. 1 Eq. 1, facered-q2]`.
- **Several risks at once.** Take p_j = max over the risks of p_{j,l}; this is a valid p-value for "some risk exceeds its target" `[P1 Prop. 6, facered-q2]`.
- **Fixed-sequence testing** `[P1 Alg. 1, Sec. 2.3.1, facered-q2]`: walk a fixed path of grid points, keep accepting while p <= delta/|J|, stop at the first failure. With one start (|J| = 1) there is no multiplicity penalty; with |J| starts each path is tested at delta/|J|.
- **Path learned on separate data.** P1 App. D splits the calibration data into a *graph selection set* and a *multiple testing set*, and learns the path on the first only `[P1 App. D, facered-q2]`. Here TUNE plays the graph set and CAL the testing set.
- **Free choice inside the accepted set.** One may "pick any lambda in Lambda-hat as their chosen RCP (even in a data-driven way)" `[P1 Sec. 2.1, facered-q3]`. We pick the cheapest.

**Derived here (not in P1).**
- *Miss risk* R_miss: the probability that a face instance has at least one visible frame uncovered; a binary loss per instance.
- *False-blur risk* R_fb: the mean over episodes of the false-blur fraction of that episode's blur box-frames (zero when the episode draws no box). It is an FDR-style risk in [0, 1] per **episode**; it is *not* pooled precision, and the two can differ (pooled precision is printed alongside). P2 leaves precision to other tools: SeqCRC "only empirically limits false positives" `[P2 Sec. V-D, facered-q6]`, which is why precision is handled by LTT here.
- The path rule: rank all grid points by their TUNE max-p (ties by safety headroom) so that every grid point, including the cheap strided ones, is reachable. P1's own beta-path construction is also implemented for comparison (section 3(c) reports how few grid points it visits here).
- The binomial p-value for R_miss treats face instances as independent, which is the assumption discussed in section 1.

There is no `CHECK` in this cell: correctness of the implementation is tested by simulation in 3(d) (CHECKs 3.2, 3.3, 3.5).
