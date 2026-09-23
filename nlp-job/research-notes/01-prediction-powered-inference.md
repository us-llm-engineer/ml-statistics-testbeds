# Prediction-Powered Inference (PPI)

Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. *Science*, Vol. 382, No. 6671, pp. 669–674, 2023.
UC Berkeley. arXiv:2301.09633.

## Elaboration

### Intuitive description

You want to know a population number — what fraction of job postings are genuinely GenAI roles this
month — but only a small sample has been human-labeled, and human labels are expensive. Meanwhile a
cheap model (a rule score, an LLM call) has already labeled *every* posting, but its labels are biased
and you don't know by how much. PPI's move: don't trust the cheap labels outright, and don't throw them
away either. Use the small human-labeled sample to measure exactly how wrong the cheap model is (its
*rectifier*, the average gap between its label and the true one), then apply that correction to the
cheap model's verdict on the whole month. The result is a confidence interval that is provably valid no
matter how bad the cheap model turns out to be — a bad model just makes the interval wider, never wrong.

### Mathematical formalism

For a target parameter $\theta^* = \mathbb{E}[Y_i]$, with $n$ labeled pairs $(X_i, Y_i)$ and $N \gg n$
unlabeled points $\tilde X_i$, and a predictor $f$ independent of both samples, the prediction-powered
estimator is

$$\hat\theta_{\mathrm{PP}} = \underbrace{\frac{1}{N}\sum_{i=1}^N f(\tilde X_i)}_{\tilde\theta_f}
- \underbrace{\frac{1}{n}\sum_{i=1}^n \big(f(X_i)-Y_i\big)}_{\hat\Delta},$$

with the asymptotically valid 95% interval

$$\hat\theta_{\mathrm{PP}} \pm z_{1-\alpha/2}\sqrt{\hat\sigma^2_{f-Y}/n + \hat\sigma^2_f/N}$$

(Algorithm 1; Proposition 1 gives $\liminf_{n,N\to\infty}\mathbb{P}(\theta^*\in\mathcal C_\alpha^{\mathrm{PP}})\ge 1-\alpha$).
The only assumptions are that $(X,Y)$ and $(\tilde X,\tilde Y)$ are i.i.d. draws from a common
distribution $P$, and $f$ is independent of the observed data — nothing is assumed about $f$'s accuracy.

Whether PPI actually beats a classical labeled-only interval is a separate, checkable question
(Appendix G.1): comparing $\mathrm{Var}(\hat\theta_{\mathrm{class}}) = \mathrm{Var}(Y_i)/n$ against
$\mathrm{Var}(\hat\theta_{\mathrm{PP}}) = \mathrm{Var}(f(X_i))/N + \mathrm{Var}(f(X_i)-Y_i)/n$, PPI wins
exactly when $\mathrm{Var}(f(X_i)-Y_i) < \mathrm{Var}(Y_i)$. For a binary outcome with symmetric flip
error $\eta$, the paper works this out to a concrete break-even: at $p=0.5$ prevalence, $\eta$ must be
below 25%; at $p=0.1$, below about 9.5%.

Two extensions used later: **label shift** (Section 4.2.2), when the mix of true labels changes across
a stream but $Q_{X\mid Y}=P_{X\mid Y}$ holds, corrected via a confusion matrix $K$ estimated on labeled
data and $Q_Y = K^{-1}Q_f$ (Theorem 3); and a **power-tuned variant** (not from this paper — it is the
citation "Angelopoulos et al., 2023b" inside a different paper's reference list), which replaces the
fixed rectifier with a tunable weight $\lambda$ minimizing variance,
$\lambda^* = \mathrm{Cov}(Y,f) / ((1+n/N)\,\mathrm{Var}(f))$.

## Practical use cases in this Project

Notebook 1 §§1–3 implements the estimator, its break-even condition (reproducing the paper's own 25%
and ~9.5% figures from the closed-form variance equations), the power-tuned variant, and the label-shift
correction — each verified by Monte Carlo coverage simulation before touching any generated data.
Notebook 2's audit section explicitly checks that its own sampling stays consistent with PPI's i.i.d.
assumption (flagging where it doesn't — near-duplicate reposts — as a limitation Notebook 3 fixes).
Notebook 3 §4 runs PPI for real: a 100-label simulated monthly audit, comparing a classical interval
against PPI with two different predictors (the full pipeline label, and the LLM label alone), showing
the more accurate predictor buys a roughly 2x narrower interval from the same 100 labels — and checking,
because the corpus is synthetic and the truth is known, that every interval's stated coverage actually
holds up over repeated audits.
