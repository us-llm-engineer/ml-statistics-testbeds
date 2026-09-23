# BARGAIN: LLM-Powered Data Processing with Guarantees

Zeighami, Shankar, Parameswaran. UC Berkeley. SIGMOD 2026. arXiv:2509.02896.
*"Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees."*

## Elaboration

### Intuitive description

Running every record through an expensive model is safe but costly; routing everything through a cheap
one is cheap but risky. A model cascade tries to get the best of both: use the cheap model's own
confidence to decide which records it can answer on its own, and send only the uncertain ones to the
expensive model. The hard part is choosing that confidence cutoff honestly — pick it by eyeballing a
few examples and you have no idea how often it'll be wrong on data you haven't seen. BARGAIN's
contribution is a way to pick the cutoff that comes with a real probabilistic guarantee: sample a few
records at each candidate cutoff, run a sequential statistical test that can be checked after every new
sample without inflating the error rate (a "testing by betting" construction), and stop scanning
thresholds the moment one is honestly certified. The guarantee is against the *expensive model's own
answers* — not against ground truth, which is a distinction that matters a lot in practice.

### Mathematical formalism

Given a cheap proxy $\mathcal P$, an expensive oracle $\mathcal O$, and a confidence score
$S(x)\in[0,1]$ from the proxy, a cascade threshold $\rho$ routes $x$ to the proxy if $S(x)>\rho$ and to
the oracle otherwise. An **Accuracy-Target** query asks for the fewest oracle calls such that
$P(A(Y)\ge T)\ge 1-\delta$, where $A(Y)=\frac{1}{n}\sum_i \mathbb 1[\mathcal O(x_i)=y_i]$ is agreement
with the oracle over the whole answer set $Y$. Precision/Recall-Target queries are defined analogously
on a binary oracle, e.g. $P(Y)=\frac{\sum_{i\in Y}\mathcal O(x_i)}{|Y|}$.

Certification uses an anytime-valid betting statistic (Lemma B.1, a Krichevsky–Trofimov-style
predictable-plug-in simplification of Waudby-Smith & Ramdas' Theorem 3): for a sequence $Y_1,\dots,Y_i$,

$$K(m,Y)=\prod_{j\le i}\Big(1+\min\big(\lambda_j,\tfrac{3}{4m}\big)(Y_j-m)\Big),\qquad
P\big(\exists i:K(m,Y[{:}i])\ge 1/\alpha\big)\le\alpha\ \text{ whenever the true mean }\mu<m.$$

Algorithm 2 scans candidate thresholds $\mathcal C_M$ from high to low, sampling uniformly from
$D_\rho=\{x:S(x)>\rho\}$ until certification fires, stopping at the first threshold that fails (returning
the previous, larger $\rho$). With tolerance $\eta=0$ (the paper's default, justified empirically by
observing real precision curves are monotone), Lemma 3.5 gives $P(\mathcal P_D(\rho^*)<T)\le(\eta+1)\alpha=\alpha$
with **no union bound needed over the $M$ candidates** — certification happens directly at wealth
$\ge 1/\delta$. A companion impossibility result (Lemma B.11) shows recall-target queries are
structurally hard when positives are rare: any monotone-sampling algorithm meeting a recall target on
all datasets must have precision
$P\big(\mathcal P_D(\rho_S)\le n_+/n\big)\ge(1-n_+/n)^k-\delta$ — i.e. low true positive rates force low
achievable precision, regardless of sampling budget $k$.

Crucially, **all of BARGAIN's metrics are defined relative to the oracle's own labels, not to human
ground truth**, and its guarantee is proven for a static batch of data — the paper gives no rule for
re-certifying a threshold as a stream drifts. Its guarantee also holds even when the proxy score is
poorly calibrated; a badly calibrated score just makes the *utility* (how many oracle calls get avoided)
worse, not the guarantee.

## Practical use cases in this Project

Notebook 1 §§4–6 build the betting statistic from Lemma B.1 (verified by simulation after finding one
malformed variance formula, corrected to the standard textbook plug-in), a simplified re-implementation
of Algorithm 2's threshold certification (with sample reuse and the no-union-bound property both
verified), and the recall-target impossibility bound. Notebook 2's rule-vs-LLM boundary sweep is the
practical motivation: it shows empirically where a cheap keyword-rule "proxy" and an LLM "oracle" agree
and disagree by rule-score bin. Notebook 3 §2 applies BARGAIN-style certification for real: the rule
tier is the proxy, a cached LLM verdict is the oracle, and the notebook certifies how many postings the
rules can answer on their own at a stated accuracy target against the oracle — while being explicit
throughout that this measures agreement with the LLM, not with ground truth (that's PPI's job), and
that the certification is scoped to one fixed batch of months, not a continuously adapting threshold.
