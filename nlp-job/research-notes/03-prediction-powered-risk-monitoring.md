# PPRM: Prediction-Powered Risk Monitoring

Zhang, Cai, Yu, Simeone. Zhejiang University / Northeastern University London. ICML 2026.
arXiv:2602.02229. *"Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful
Distribution Shifts."*

## Elaboration

### Intuitive description

A model deployed over time will see its input distribution shift — new phrasing, new topics, seasonal
patterns. Most of that shift is harmless; the model still works fine. The dangerous case is a shift that
actually degrades performance, and the two look identical from a plain "has the data changed?" test.
Labeling every incoming record to check performance directly is too expensive to do continuously. PPRM's
answer: combine a trickle of real human labels with a flood of cheap synthetic labels (from an auxiliary
model, however good or bad it happens to be) into a single running estimate of error rate, wrapped in a
statistical test that can be checked after every new data point without inflating the false-alarm rate —
and, critically, the test is built around the *running average* risk crossing a *harm* threshold, not
around "has anything changed." A quiet content wave that doesn't hurt accuracy should never trigger it;
a real accuracy drop should.

### Mathematical formalism

Define the running risk $\bar R_t=\frac1t\sum_{t'=1}^t R_{t'}$ and the harmful-shift condition
$\bar R_{t^*}>R_0+\epsilon_{\mathrm{tol}}$ for some tolerance $\epsilon_{\mathrm{tol}}>0$ above a
calibrated nominal risk $R_0$. The monitoring problem is the sequential hypothesis test
$H_0:\bar R_t\le R_0+\epsilon_{\mathrm{tol}}\ \forall t$ vs. $H_1:\exists t^*$ violating it, with a
type-I error budget $P_{H_0}(\exists t:\Phi_t{=}1)\le\delta$.

The supervised baseline (SRM, Podkopaev & Ramdas 2022) builds an upper confidence bound $U_0$ on $R_0$
and a lower confidence sequence $L_t$ on $\bar R_t$ from labeled data alone, alarming when
$L_t>U_0+\epsilon_{\mathrm{tol}}$, with combined error $\le\delta_S+\delta_T$ (Lemma 2.2). PPRM
generalizes this to a semi-supervised setting: at each step $t$, given $n_t$ labeled points and $N_t$
unlabeled points scored by an auxiliary predictor $f_p$, the prediction-powered risk estimate (Eq. 17–18)

$$\hat R_t^{\mathrm{PP}} = \underbrace{\frac{\eta_t}{N_t}\sum \ell\big(f(\tilde x_{t,j}),\tilde y_{t,j}\big)}_{\text{unlabeled term}}
+ \underbrace{\frac1{n_t}\sum \ell\big(f(x_{t,i}),y_{t,i}\big) - \frac{\eta_t}{n_t}\sum \ell\big(f(x_{t,i}),\tilde y_{t,i}\big)}_{\text{labeled rectifier}}$$

is unbiased whenever $\{\eta_t\}$ is a predictable sequence (Lemma 3.1), and Theorem 3.2 shows it
inherits the same $\delta_S+\delta_T$ type-I error guarantee as SRM — **with no assumption whatsoever on
the quality of $f_p$**, only that $n_t>0$ labels arrive at every step. Lemma 3.3 gives the
variance-minimizing weight
$\eta_t^*=\mathrm{Cov}(u_t,\tilde u_t^{\mathrm L})/\big((1+n_t/N_t)\mathrm{Var}(\tilde u_t^{\mathrm U})\big)$,
estimated online from a sliding window. The confidence sequence itself (Lemma 2.1, a conjugate-mixture
empirical-Bernstein bound) is anytime-valid: $P\big(\forall t: |\mu_t-\hat\mu_t| < u(V_t)/t\big)\ge 1-2\delta_T$,
where $u(\cdot)$ comes from a betting-style mixture over a grid of the bet size $\lambda$.

A weak $f_p$ doesn't break validity, but it does hurt speed: with a poorly calibrated predictor and a
fixed $\eta_t$, PPRM can detect a real shift *more slowly* than the plain labeled-only SRM baseline; the
adaptive $\eta_t$ (Lemma 3.3) is specifically what prevents that.

## Practical use cases in this Project

Notebook 1 §7 implements PPRM's own Lemma 2.1 confidence sequence directly (discretizing the mixing
distribution $q(\lambda)$ over a 200-point grid, an implementation choice used consistently in Notebook
3), verifies the type-I error guarantee by simulation under a null (no-shift) scenario, and reproduces
the paper's qualitative claims: a good auxiliary predictor detects a real shift faster than SRM, a weak
one with a fixed weight can be slower, and the adaptive weight recovers most of the gap. Notebook 3 §5
runs the monitor on two constructed streams built on top of Notebook 2's corpus: a "buzzword wave" where
the underlying data changes but pipeline accuracy does not (PPRM correctly stays silent, while a naive
two-sample frequency test flags it as a false alarm), and an "LLM-tier outage" where accuracy genuinely
degrades (PPRM alarms, and faster than a labels-only monitor when the synthetic labeler is informative)
— the practical demonstration of "harmful drift, not just any drift" that motivates using PPRM over a
simpler distributional-change detector.
