"""Statistical-guarantee machinery for idp-fintech (checkpoint module).

Sources (formulas transcribed / generalised straight from the papers; see docstrings for the exact equation each
function implements):

  - Bai & Jin, "SCoRE" (arXiv:2603.24704): weighted MDR e-values Eq. (6.1) and Thm 6.2, weighted SDR e-values
    Eq. (6.2) and Thm 6.3, the gamma<=alpha shortcut Prop. A.1, the efficient SDR search Algorithm 4, e-BH
    Theorem 3.3, e-value boosting Theorem 5.5, and the weight-balancing Assumptions A.3 (MDR) / A.5 (SDR).
    The unweighted special case (all weights = 1) reduces to Eq. (4.1) (MDR) and Eq. (5.1) (SDR).
  - Jin & Candes (2023a), "Model-free selective inference under covariate shift via weighted conformal
    p-values" (arXiv:2307.09291): weighted conformal p-values Eq. (4)-(5), Weighted Conformalized Selection
    (WCS) Algorithm 1, Eq. (6)-(9) (auxiliary p-values, first-step selection, hete/homo/dtm pruning).
  - Tibshirani, Foygel Barber, Candes & Ramdas (arXiv:1904.06019), Eq. (12): density-ratio weights from a
    probabilistic classifier, w(x) = odds(x) * n_source / n_target, with clipping and cross-fitting
    (Tibshirani et al.'s own recommendation to avoid overfitting bias in the weight estimate).
  - Gurram (arXiv:2608.14639): document-level ("tier 4") PAC certification via geometric acceptance
    candidates and a Holm-step-down / fixed-sequence delta-split; the Hoeffding macro-functional variant is
    from this paper. The empirical-Bernstein (Maurer & Pontil, 2009) and "binomial share of erroring
    documents" tier-4 variants are DERIVED HERE (not in any source paper) as power/robustness alternatives to
    the Hoeffding bound, using the same candidate/Holm/fixed-sequence scaffolding.

Conventions used throughout (see the coordinator's brief): for SCoRE-style risk control, s = predicted-risk
score and LOW score = trusted (a unit is trusted when s <= threshold); for tier-4 (`doc_tier4_threshold`)
HIGHER score = trusted (accept when score >= tau), the opposite convention, because that machinery inherits
Notebook 1 / Notebook 3's document-trust scores directly. L in [0, 1] is always a bounded loss. max(empty
set) = -inf and 1/0 = +inf are used exactly as in the brief.

All functions are pure (no module-level mutable state) and numpy-vectorised except where a Python loop over
`m` test/target points is unavoidable (each iteration then does O(n) or O(m) vectorised numpy work; see the
docstrings of `weighted_sdr_e_values` and `wcs_select` for the complexity discussion).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "weighted_mdr_e_values", "weighted_mdr_decision", "mdr_e_values", "sdr_e_values",
    "weighted_sdr_e_values", "ebh", "ebh_boosted",
    "domain_classifier_weights", "effective_sample_size", "balance_weights_mdr", "balance_weights_sdr",
    "doc_tier4_threshold", "clopper_pearson_upper",
    "weighted_conformal_pvalues", "wcs_select", "wbh_select",
]


# ======================================================================================================
# 1-2. SCoRE weighted MDR / SDR e-values, e-BH, boosting
# ======================================================================================================
def weighted_mdr_e_values(cal_s, cal_L, cal_w, test_s, test_w, gamma):
    """SCoRE weighted-MDR e-value, Eq. (6.1) (generalises the unweighted Eq. (4.1)):

        E_t = inf_{ell in [0,1]} 1{s_t <= t(ell)} * W / (sum_i w_i L_i 1{s_i<=t(ell)} + w_t ell 1{s_t<=t(ell)})
        t(ell) = max{t in M : F(t;ell) <= gamma},  F(t;ell) = (sum_i w_i L_i 1{s_i<=t} + w_t ell 1{s_t<=t}) / W
        W = sum_i w_i + w_t,  M = {s_1..s_n, s_t} (tie-complete: t ranges over score VALUES).

    Exact infimum (no grid): F(t;ell) is affine and non-decreasing in ell for t>=s_t and constant for t<s_t, so
    t(ell) is non-increasing in ell and E(ell) is piecewise-monotone with candidates ell in {0, 1, breakpoints},
    breakpoint ell_b(t) = (gamma W - sum_i w_i L_i 1{s_i<=t}) / w_t for t in M with t>=s_t (t(ell) switches value
    there). At any true breakpoint the numerator equals gamma*W exactly, so E = 1/gamma there regardless of the
    weights -- this is what collapses the search to `min(E(0), E(1), 1/gamma if a breakpoint exists)`, generalising
    NB3's `b_mdr_e_values` (which assumed all weights = 1, W = n+1) by carrying W, w_t through every step.
    """
    cal_s = np.asarray(cal_s, float); cal_L = np.asarray(cal_L, float); cal_w = np.asarray(cal_w, float)
    test_s = np.asarray(test_s, float); test_w = np.asarray(test_w, float)
    n = len(cal_s)
    o = np.argsort(cal_s, kind="stable"); c = cal_s[o]
    wl = (cal_w * cal_L)[o]
    Sc = np.concatenate([[0.0], np.cumsum(wl)])                     # Sc[k] = sum over first k sorted calib points
    S_el = Sc[np.searchsorted(c, c, side="right")]                  # tie-complete cumulative sum AT each calib point

    Wc = cal_w.sum()
    W = Wc + test_w                                                 # per test point total weight
    B = gamma * W
    p = np.searchsorted(c, test_s, side="left")
    q = np.searchsorted(c, test_s, side="right")
    cnt1 = np.searchsorted(S_el, B, side="right")

    def E_at(ell):
        inc = test_w * ell
        m2 = np.searchsorted(S_el, B - inc, side="right")
        test_feas = (cnt1 >= p) & (Sc[q] + inc <= B)
        k = np.maximum(p, m2)
        cum_k = np.where(k == p, Sc[q], S_el[np.clip(k - 1, 0, n - 1)])
        den = cum_k + inc
        with np.errstate(divide="ignore"):
            val = np.where(den > 1e-9, W / np.maximum(den, 1e-300), np.inf)
        return np.where(test_feas, val, 0.0)

    w_safe = np.where(test_w > 1e-300, test_w, 1e-300)
    lo = np.searchsorted(S_el, B - w_safe - 1e-9, side="left")
    hi = np.searchsorted(S_el, B + 1e-9, side="right")
    has_bp = (hi > np.maximum(lo, p)) | ((Sc[q] >= B - w_safe - 1e-9) & (Sc[q] <= B + 1e-9))
    cand = np.minimum(E_at(0.0), E_at(1.0))
    return np.where(has_bp, np.minimum(cand, 1.0 / gamma), cand)


def weighted_mdr_decision(cal_s, cal_L, cal_w, test_s, test_w, alpha):
    """Prop. A.1 shortcut (valid for gamma = alpha <= alpha, i.e. gamma set to alpha here):

        1{E >= 1/alpha} = 1{ (w_t + sum_i w_i L_i 1{s_i <= s_t}) / (sum_i w_i + w_t) <= alpha }.

    Avoids the full e-value computation; used directly for the MDR trust decision (Theorem 3.2, psi = 1{E>=1/alpha}).
    """
    cal_s = np.asarray(cal_s, float); cal_L = np.asarray(cal_L, float); cal_w = np.asarray(cal_w, float)
    test_s = np.asarray(test_s, float); test_w = np.asarray(test_w, float)
    o = np.argsort(cal_s, kind="stable"); c = cal_s[o]
    wl = (cal_w * cal_L)[o]
    Sc = np.concatenate([[0.0], np.cumsum(wl)])
    S_le = Sc[np.searchsorted(c, test_s, side="right")]             # tie-complete sum_i w_i L_i 1{s_i<=s_t}
    Wc = cal_w.sum()
    lhs = (test_w + S_le) / (Wc + test_w)
    return lhs <= alpha


def mdr_e_values(cal_s, cal_L, test_s, gamma):
    """Unweighted SCoRE-MDR e-values, Eq. (4.1) (= weighted_mdr_e_values with all weights = 1)."""
    cal_s = np.asarray(cal_s, float); test_s = np.asarray(test_s, float)
    return weighted_mdr_e_values(cal_s, cal_L, np.ones(len(cal_s)), test_s, np.ones(len(test_s)), gamma)


def sdr_e_values(cal_s, cal_L, test_s, gamma):
    """Unweighted SCoRE-SDR e-values, Eq. (5.1) (= weighted_sdr_e_values with all weights = 1)."""
    cal_s = np.asarray(cal_s, float); test_s = np.asarray(test_s, float)
    return weighted_sdr_e_values(cal_s, cal_L, np.ones(len(cal_s)), test_s, np.ones(len(test_s)), gamma)


def weighted_sdr_e_values(cal_s, cal_L, cal_w, test_s, test_w, gamma):
    """SCoRE weighted-SDR e-value, Eq. (6.2), for every test unit j among m test units:

        E_j = inf_{ell} 1{s_j<=t_j(ell)} * (w_j + sum_i w_i) / (w_j ell 1{s_j<=t_j(ell)} + sum_i w_i L_i 1{s_i<=t_j(ell)})
        t_j(ell) = max{t in M : FR_j(t;ell) <= gamma},  M = all calibration AND test scores
        FR_j(t;ell) = (w_j ell 1{s_j<=t} + sum_i w_i L_i 1{s_i<=t}) / (1 + sum_{k!=j} 1{s_k<=t}) * m / (w_j + sum_i w_i)

    FR_j is NOT monotone in t (the "1+#others<=t" denominator can jump independently of the numerator), so unlike
    the MDR case the infimum cannot be collapsed to a constant (1/gamma) at a breakpoint: the value there is
    m / (gamma * (1+#others<=t)), which depends on t. We therefore keep NB3's `b_sdr_e_value` exact-search
    structure (breakpoints = {0, 1, ell_b(t) for t>=s_j}, evaluate FR at every (ell, t) pair, take the largest
    feasible t per candidate ell, minimise over candidates), generalised with calibration/test weights. This is
    O(n) work to build the shared tie-complete cumulative sums once, then a Python loop over the m test points,
    each doing O(|M|) candidate generation and an O(|candidates| x |M|) numpy evaluation -- independent of any
    per-test-point O(n) Python loop, i.e. no O(m^2 n) Python-level loop.
    """
    cal_s = np.asarray(cal_s, float); cal_L = np.asarray(cal_L, float); cal_w = np.asarray(cal_w, float)
    test_s = np.asarray(test_s, float); test_w = np.asarray(test_w, float)
    n = len(cal_s); m = len(test_s)
    o = np.argsort(cal_s, kind="stable"); c = cal_s[o]
    wl = (cal_w * cal_L)[o]
    Sc = np.concatenate([[0.0], np.cumsum(wl)])
    Mv = np.unique(np.concatenate([cal_s, test_s]))
    S_le = Sc[np.searchsorted(c, Mv, side="right")]                 # sum_i w_i L_i 1{s_i<=t}, t in Mv
    test_sorted = np.sort(test_s)
    cnt_all = np.searchsorted(test_sorted, Mv, side="right")        # #{test scores <= t} (ALL m test points)
    Wc = cal_w.sum()

    E = np.empty(m)
    for j in range(m):
        s_j = float(test_s[j]); w_j = float(test_w[j]); Wj = Wc + w_j
        ge = Mv >= s_j
        gef = ge.astype(float)
        cnt_others = cnt_all - ge.astype(int)                       # sum_{k!=j} 1{s_k<=t}
        denom_factor = (1.0 + cnt_others).astype(float)
        w_safe = w_j if w_j > 1e-300 else 1e-300
        lb = (gamma * Wj * denom_factor / m - S_le) / w_safe         # ell_b(t), Algorithm 4's formula
        mask = ge & (lb >= -1e-9) & (lb <= 1 + 1e-9)
        ells = np.unique(np.concatenate([[0.0, 1.0], np.clip(lb[mask], 0.0, 1.0)]))
        FR = (ells[:, None] * w_j * gef[None, :] + S_le[None, :]) / denom_factor[None, :] * (m / Wj)
        feas = FR <= gamma + 1e-12
        row_has = feas.any(axis=1)
        if not row_has.all():
            E[j] = 0.0
            continue
        k = feas.shape[1] - 1 - np.argmax(feas[:, ::-1], axis=1)     # last feasible column per row (ell candidate)
        den = ells * w_j + S_le[k]
        ge_k = ge[k]
        with np.errstate(divide="ignore"):
            val = np.where(ge_k, np.where(den > 1e-9, Wj / np.maximum(den, 1e-300), np.inf), 0.0)
        E[j] = float(val.min())
    return E


def ebh(E, alpha):
    """e-BH (SCoRE Theorem 3.3): tau = max{k : #{E_j >= m/(alpha k)} >= k}; select E_j >= m/(alpha tau)."""
    E = np.asarray(E, float); m = len(E)
    k_arr = np.arange(1, m + 1)
    Es = np.sort(E)[::-1]
    ok = Es >= m / (alpha * k_arr)
    if not ok.any():
        return np.zeros(m, bool)
    k = int(k_arr[ok].max())
    return E >= m / (alpha * k)


def ebh_boosted(E, alpha, mode="hete", rng=None):
    """e-value boosting (SCoRE Theorem 5.5): hete = iid xi_j ~ Unif[0,1] per test unit, homo = one shared xi;
    run e-BH on E_j / xi_j (hete) or E_j / xi (homo)."""
    E = np.asarray(E, float); m = len(E)
    if rng is None:
        rng = np.random.default_rng()
    if mode == "hete":
        xi = rng.uniform(size=m)
    elif mode == "homo":
        xi = rng.uniform()
    else:
        raise ValueError(f"unknown boosting mode: {mode!r}")
    return ebh(E / xi, alpha)


# ======================================================================================================
# 3. weights: domain classifier, effective sample size, balancing (Assumptions A.3 / A.5)
# ======================================================================================================
def domain_classifier_weights(X_src, X_tgt, n_folds=5, clip_quantile=None, seed=0, model="lr"):
    """Density-ratio weights from a cross-fitted probabilistic classifier (Tibshirani et al. arXiv:1904.06019,
    Eq. (12)). Fit C=1 (target) vs C=0 (source) on the pooled sample; each point's probability is predicted by a
    model trained on the OTHER folds only (K-fold cross-fitting, avoiding the in-sample overfitting bias the
    paper warns about). p is clipped to [0.01, 0.99]; odds = p/(1-p); w = odds * n_src/n_tgt (the Bayes-rule
    rescaling that turns a class-odds ratio into an estimate of dQ/dP under a pooled two-sample design).
    Optionally cap w_src at its `clip_quantile` quantile (and w_tgt at the same cap), then rescale BOTH arrays
    by the same constant so mean(w_src) = 1.

    model="lr": StandardScaler + LogisticRegression. model="hgb": HistGradientBoostingClassifier(max_depth=3).
    """
    X_src = np.asarray(X_src, float); X_tgt = np.asarray(X_tgt, float)
    n_src, n_tgt = len(X_src), len(X_tgt)
    X = np.vstack([X_src, X_tgt])
    y = np.r_[np.zeros(n_src), np.ones(n_tgt)]
    N = len(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    p = np.zeros(N)
    for tr_idx, te_idx in kf.split(X):
        if model == "lr":
            clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
        elif model == "hgb":
            clf = HistGradientBoostingClassifier(max_depth=3, random_state=seed)
        else:
            raise ValueError(f"unknown model: {model!r}")
        clf.fit(X[tr_idx], y[tr_idx])
        p[te_idx] = clf.predict_proba(X[te_idx])[:, 1]
    p = np.clip(p, 0.01, 0.99)
    odds = p / (1.0 - p)
    w = odds * (n_src / n_tgt)
    w_src, w_tgt = w[:n_src], w[n_src:]
    if clip_quantile is not None:
        cap = np.quantile(w_src, clip_quantile)
        w_src = np.minimum(w_src, cap)
        w_tgt = np.minimum(w_tgt, cap)
    scale = 1.0 / np.mean(w_src)
    return w_src * scale, w_tgt * scale


def effective_sample_size(w):
    """Kish's effective sample size: (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def _exp_tilt(w0, g, rhs):
    """Exponential tilting w_i = w0_i * exp(a + b*g_i) solving mean(w)=1 and mean(w*g)=rhs to numerical precision.

    exp(a) factors out of both moment equations, so the 2-unknown system reduces to 1 unknown b solving
    F(b) := sum_i w0_i g_i exp(b g_i) / sum_i w0_i exp(b g_i) = rhs (a tilted-mean functional, monotone
    increasing in b when g is non-degenerate), then a = -log(mean(w0 * exp(b g))). We use a bracketing
    root-find (`scipy.optimize.brentq`) on this 1-D reduction instead of a raw 2-D Newton step for
    robustness (Newton on the original system can diverge for extreme starting weights; the reduction is
    algebraically identical, so this still delivers the same "Newton, 2 unknowns" solution)."""
    w0 = np.asarray(w0, float); g = np.asarray(g, float)
    if np.allclose(g, g[0]):
        # degenerate: mean(w*g) = g[0]*mean(w) = g[0] automatically once mean(w)=1; just normalise.
        a = -np.log(np.mean(w0))
        return w0 * np.exp(a)

    def F(b):
        wt = w0 * np.exp(b * g)
        return float(np.sum(wt * g) / np.sum(wt))

    lo, hi = -1.0, 1.0
    while F(lo) > rhs and lo > -1e6:
        lo *= 2.0
    while F(hi) < rhs and hi < 1e6:
        hi *= 2.0
    b = brentq(lambda bb: F(bb) - rhs, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=300)
    a = -np.log(np.mean(w0 * np.exp(b * g)))
    return w0 * np.exp(a + b * g)


def balance_weights_mdr(w0, l_cal, s_cal, l_test, s_test, alpha):
    """SCoRE Assumption A.3 weight balancing for MDR.

    t_hat = sup{t in observed test scores : (1/m) sum_j l_test_j 1{s_test_j<=t} <= alpha} (-inf if none).
    g_i = l_cal_i 1{s_cal_i<=t_hat}; return positive weights w = w0 * exp(a+b g) (exponential tilting, solved
    via `_exp_tilt`) matching mean(w)=1 and mean(w g) = (1/m) sum_j l_test_j 1{s_test_j<=t_hat}.
    """
    w0 = np.asarray(w0, float); l_cal = np.asarray(l_cal, float); s_cal = np.asarray(s_cal, float)
    l_test = np.asarray(l_test, float); s_test = np.asarray(s_test, float)
    n = len(w0); m = len(l_test)

    order = np.argsort(s_test, kind="stable"); sv = s_test[order]; lv = l_test[order]
    csum = np.cumsum(lv)
    uniq = np.unique(sv)
    last_idx = np.searchsorted(sv, uniq, side="right") - 1
    functional = csum[last_idx] / m
    feas = functional <= alpha + 1e-12
    t_hat = float(uniq[feas].max()) if feas.any() else -np.inf

    if np.isfinite(t_hat):
        g = l_cal * (s_cal <= t_hat)
        rhs = float(np.mean(l_test * (s_test <= t_hat)))
    else:
        g = np.zeros(n)
        rhs = 0.0
    w = _exp_tilt(w0, g, rhs)
    return w, t_hat


def balance_weights_sdr(w0, l_cal, s_cal, l_test, s_test, alpha):
    """SCoRE Assumption A.5 weight balancing for SDR.

    The paper leaves the exact one-step cutoff choice open; we DERIVE HERE (documented, no test covers this
    function) t_hat = sup{t : [(1/n) sum_i w0_i l_cal_i 1{s_cal_i<=t}] / (1 v #{s_test_j<=t}) <= alpha}, i.e. the
    A.5 functional evaluated with the PRELIMINARY weights w0 (mirroring the SDR e-value's own
    "1 v #test<=t" denominator, Eq. 6.2), searched over the union of calibration and test score values. Given
    t_hat, the same exponential tilting as `balance_weights_mdr` is applied with g_i = l_cal_i 1{s_cal_i<=t_hat}
    and target rhs = (1/m) sum_j l_test_j 1{s_test_j<=t_hat}, matching (1/n) sum_i w_i l_cal_i 1{s_cal_i<=t_hat}
    = rhs and mean(w) = 1.
    """
    w0 = np.asarray(w0, float); l_cal = np.asarray(l_cal, float); s_cal = np.asarray(s_cal, float)
    l_test = np.asarray(l_test, float); s_test = np.asarray(s_test, float)
    n = len(w0); m = len(l_test)

    Mv = np.unique(np.concatenate([s_cal, s_test]))
    order = np.argsort(s_cal, kind="stable"); sc = s_cal[order]; wl = (w0 * l_cal)[order]
    Sc = np.concatenate([[0.0], np.cumsum(wl)])
    num = Sc[np.searchsorted(sc, Mv, side="right")] / n
    cnt = np.searchsorted(np.sort(s_test), Mv, side="right")
    den = np.maximum(1, cnt)
    functional = num / den
    feas = functional <= alpha + 1e-12
    t_hat = float(Mv[feas].max()) if feas.any() else -np.inf

    if np.isfinite(t_hat):
        g = l_cal * (s_cal <= t_hat)
        rhs = float(np.mean(l_test * (s_test <= t_hat)))
    else:
        g = np.zeros(n)
        rhs = 0.0
    w = _exp_tilt(w0, g, rhs)
    return w, t_hat


# ======================================================================================================
# 4. document-level PAC ("tier 4", Gurram arXiv:2608.14639 + derived-here variants) and Clopper-Pearson
# ======================================================================================================
def clopper_pearson_upper(k, n, conf):
    """One-sided Clopper-Pearson upper confidence bound on a binomial proportion: Beta(conf; k+1, n-k) quantile
    (1.0 when k >= n, i.e. all successes)."""
    if k < n:
        return float(stats.beta.ppf(conf, k + 1, n - k))
    return 1.0


def _geometric_candidates(scores, n_candidates=15):
    """15 geometric acceptance-fraction candidates (1%-100% of the scores), snapped to distinct score values,
    most conservative (highest threshold) first (generalises NB3's `b_geometric_candidates`)."""
    scores = np.asarray(scores, float); n = len(scores)
    fracs = np.geomspace(0.01, 1.0, n_candidates)
    ks = np.clip(np.round(fracs * n).astype(int), 1, n)
    return np.unique(np.sort(scores)[::-1][ks - 1])[::-1]


def _holm_reject(pvals, budget):
    """Holm step-down at total budget `budget` (generalises NB3's `b_holm_reject`)."""
    pvals = np.asarray(pvals, float); m = len(pvals)
    order = np.argsort(pvals); sp = pvals[order]
    th = budget / (m - np.arange(m))
    rs = np.zeros(m, bool)
    for i in range(m):
        if sp[i] <= th[i]:
            rs[i] = True
        else:
            break
    r = np.zeros(m, bool); r[order] = rs
    return r


def _fixed_seq(pvals, budget):
    """Fixed-sequence test at budget `budget`, most-conservative-candidate-first (generalises `b_fixed_seq`)."""
    r = np.zeros(len(pvals), bool)
    for i, pv in enumerate(pvals):
        if pv <= budget:
            r[i] = True
        else:
            break
    return r


def _bernstein_pvalue(mean_, var_hat, D, alpha):
    """Smallest beta in (0,1] for which the one-sided empirical-Bernstein bound (Maurer & Pontil, 2009)
    `mean + sqrt(2 var_hat log(2/beta)/D) + 7 log(2/beta)/(3(D-1)) <= alpha` holds (found by bisection on a
    log scale, since the bound is monotone non-increasing in beta); 1.0 if it never holds even at beta=1."""
    def holds(beta):
        bound = mean_ + np.sqrt(2 * var_hat * np.log(2 / beta) / D) + 7 * np.log(2 / beta) / (3 * (D - 1))
        return bound <= alpha

    if not holds(1.0):
        return 1.0
    lo, hi = 1e-300, 1.0
    if holds(lo):
        return lo
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if holds(mid):
            hi = mid
        else:
            lo = mid
    return hi


def doc_tier4_threshold(doc_ids, scores, err, alpha, delta, method):
    """Document-level ("tier 4") PAC certification. HIGHER score = more trusted; accept fields with
    score >= tau. Returns +inf if nothing is certified.

    15 geometric acceptance-fraction candidates (`_geometric_candidates`), most conservative first. Per
    candidate, compute a p-value for H0: "functional > alpha" over the accepting documents, then certify with
    delta/2 Holm step-down UNION delta/2 fixed-sequence (most conservative candidate first, stop at the first
    non-rejection); return the certified candidate with the largest acceptance (n_ts).

    method="hoeffding" (Gurram): functional = macro mean, over accepting documents, of the within-document
    error rate among accepted fields; p = exp(-2 D gap^2), gap = max(0, alpha - mean), D = #accepting documents.
    method="binomial_any_error" (derived here): functional = share of accepting documents with >=1 accepted
    error; p = P[Bin(D, alpha) <= k], k = #accepting documents having >=1 error.
    method="bernstein" (derived here): same functional as hoeffding, but a one-sided empirical-Bernstein
    p-value (`_bernstein_pvalue`) instead of Hoeffding's; requires D >= 2 (p=1 otherwise).
    """
    doc_ids = np.asarray(doc_ids); scores = np.asarray(scores, float); err = np.asarray(err, float)
    docs, inv = np.unique(doc_ids, return_inverse=True)
    D_total = len(docs)
    th = _geometric_candidates(scores)
    p = np.ones(len(th)); n_ts = np.zeros(len(th))
    for i, t in enumerate(th):
        mask = scores >= t
        n_ts[i] = mask.sum()
        if not mask.any():
            continue
        cnt = np.bincount(inv[mask], minlength=D_total)
        serr = np.bincount(inv[mask], weights=err[mask], minlength=D_total)
        has = cnt > 0
        Dn = int(has.sum())
        if Dn == 0:
            continue
        if method in ("hoeffding", "bernstein"):
            per_doc = serr[has] / cnt[has]
            mean_ = float(per_doc.mean())
            if method == "hoeffding":
                gap = max(0.0, alpha - mean_)
                p[i] = np.exp(-2 * Dn * gap ** 2)
            else:
                if Dn < 2:
                    p[i] = 1.0
                else:
                    var_hat = float(per_doc.var(ddof=1))
                    p[i] = _bernstein_pvalue(mean_, var_hat, Dn, alpha)
        elif method == "binomial_any_error":
            k = int((serr[has] > 0).sum())
            p[i] = float(stats.binom.cdf(k, Dn, alpha))
        else:
            raise ValueError(f"unknown tier-4 method: {method!r}")
    cert = _holm_reject(p, delta / 2) | _fixed_seq(p, delta / 2)
    if not cert.any():
        return float(np.inf)
    ci = np.nonzero(cert)[0]
    return float(th[ci[np.argmax(n_ts[ci])]])


# ======================================================================================================
# 5. Weighted conformal p-values and Weighted Conformalized Selection (Jin & Candes 2023a)
# ======================================================================================================
def _sum_less(Vc, wc, targets):
    """sum_i wc_i 1{Vc_i < targets} for a vector of query points, O((n+m) log n)."""
    o = np.argsort(Vc, kind="stable"); v = Vc[o]; wsort = wc[o]
    csum = np.concatenate([[0.0], np.cumsum(wsort)])
    idx = np.searchsorted(v, targets, side="left")
    return csum[idx]


def weighted_conformal_pvalues(Vc, wc, Vhat, wt, deterministic=True, rng=None):
    """Weighted conformal p-values (Jin & Candes 2023a).

    Deterministic, Eq. (5): p_j = (sum_i w_i 1{V_i<Vhat_j} + w_j) / (sum_i w_i + w_j).
    Random, Eq. (4): p_j = (sum_i w_i 1{V_i<Vhat_j} + (w_j + sum_i w_i 1{V_i=Vhat_j}) U_j) / (sum_i w_i + w_j),
    U_j ~ Unif[0,1].
    """
    Vc = np.asarray(Vc, float); wc = np.asarray(wc, float)
    Vhat = np.asarray(Vhat, float); wt = np.asarray(wt, float)
    Wc_total = wc.sum()
    sum_lt = _sum_less(Vc, wc, Vhat)
    if deterministic:
        return (sum_lt + wt) / (Wc_total + wt)
    o = np.argsort(Vc, kind="stable"); v = Vc[o]; wsort = wc[o]
    csum = np.concatenate([[0.0], np.cumsum(wsort)])
    sum_le = csum[np.searchsorted(v, Vhat, side="right")]
    sum_eq = sum_le - sum_lt
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(size=len(Vhat))
    return (sum_lt + (wt + sum_eq) * U) / (Wc_total + wt)


def wcs_select(Vc, wc, Vhat, wt, q, pruning, xi=None, rng=None):
    """Weighted Conformalized Selection (Jin & Candes 2023a, Algorithm 1, Eq. (5)-(9)).

    p_j by Eq. (5) (deterministic). Auxiliary p_l^(j) = (sum_lt[l] + w_j 1{Vhat_j<Vhat_l}) / (sum_i w_i + w_j)
    for l != j (Eq. 6) -- note sum_lt[l] = sum_i w_i 1{V_i<Vhat_l} does NOT depend on j, so it is computed ONCE
    (O((n+m) log n)) and reused for every j; only the O(m^2) pairwise indicator 1{Vhat_j<Vhat_l} and the
    per-column BH-style search over k depend on m, giving O(n log n + m^2 log m) total with no O(m^2 n) term.
    k_j* = max{k : 1 + #{l!=j : p_l^(j) <= q k/m} >= k}; |R_{j->0}| = 1 + that count at k_j*; s_j = q|R_{j->0}|/m;
    first step F = {j : p_j <= s_j}. Pruning (r ranges from 0): hete -- xi_j iid Unif, r* = max{r>=0 :
    #{j in F : xi_j |R_{j->0}| <= r} >= r}; homo -- one shared xi; dtm -- xi = 1 (deterministic, |R_{j->0}| itself).
    R = {j in F : val_j <= r*} where val = |R_{j->0}| (dtm) or xi * |R_{j->0}| (homo/hete).
    """
    Vc = np.asarray(Vc, float); wc = np.asarray(wc, float)
    Vhat = np.asarray(Vhat, float); wt = np.asarray(wt, float)
    m = len(Vhat)
    Wc_total = wc.sum()
    sum_lt = _sum_less(Vc, wc, Vhat)                                # length m, independent of j
    p = (sum_lt + wt) / (Wc_total + wt)                             # main p-values, Eq. (5)

    ind = (Vhat[None, :] < Vhat[:, None]).astype(float)             # ind[l, j] = 1{Vhat_j < Vhat_l}
    P = (sum_lt[:, None] + wt[None, :] * ind) / (Wc_total + wt[None, :])   # P[l, j] = p_l^(j), Eq. (6)
    np.fill_diagonal(P, np.inf)                                     # exclude l == j
    Ps = np.sort(P, axis=0)                                         # each column sorted ascending
    k_arr = np.arange(1, m + 1)
    thresh = q * k_arr / m

    R0 = np.zeros(m, int)
    for j in range(m):
        counts = np.searchsorted(Ps[:, j], thresh, side="right")
        cond = counts >= (k_arr - 1)
        k_star = int(k_arr[cond].max())
        R0[j] = 1 + int(counts[k_star - 1])

    s = q * R0 / m
    first = p <= s

    if pruning == "dtm":
        val = R0.astype(float)
    elif pruning in ("homo", "hete"):
        if xi is None:
            if rng is None:
                rng = np.random.default_rng()
            xi = rng.uniform(size=m) if pruning == "hete" else rng.uniform()
        val = xi * R0
    else:
        raise ValueError(f"unknown pruning mode: {pruning!r}")

    r_arr = np.arange(0, m + 1)
    vs = np.sort(val[first])
    counts_r = np.searchsorted(vs, r_arr, side="right")
    cond_r = counts_r >= r_arr
    r_star = int(r_arr[cond_r].max())
    return first & (val <= r_star)


def wbh_select(Vc, wc, Vhat, wt, q):
    """Plain Benjamini-Hochberg on the deterministic weighted conformal p-values (Eq. 5); asymptotic-only
    comparator (no finite-sample FDR guarantee under weighting/dependence), used as a baseline against WCS."""
    p = weighted_conformal_pvalues(Vc, wc, Vhat, wt, deterministic=True)
    m = len(p)
    order = np.argsort(p); sp = p[order]
    thresh = q * np.arange(1, m + 1) / m
    ok = sp <= thresh
    if not ok.any():
        return np.zeros(m, bool)
    kmax = int(np.nonzero(ok)[0].max())
    cutoff = sp[kmax]
    return p <= cutoff
