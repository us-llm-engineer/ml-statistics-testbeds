"""Spec tests for idp-fintech/lib/guarantees.py (written by the coordinator before the implementation).

Run from the project root:  .venv/bin/python -m pytest tests -q

Every reference implementation below is written straight from the paper formula (slow, brute force); the library must
match it. Formula sources (re-read before changing anything):
  - SCoRE weighted MDR / SDR e-values, Eq. (6.1)/(6.2), Prop. A.1, Alg. 4:      python3 research/reread.py 13
  - SCoRE estimated weights, balancing Assumptions A.3/A.5:                     python3 research/reread.py 14
  - SCoRE boosting (Thm 5.5), risk-reward score:                                python3 research/reread.py 15
  - Weighted conformal p-values / WCS (Jin & Candes 2023a):                     python3 research/reread.py idpshift 1 2
  NOTE: idpfin-q13 mis-transcribes FR_{n+j} (fraction inverted) and drops 1/m in Alg. 4; the references below use the
        paper form, which reduces to q5 Eq. (5.1) when w = 1.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import guarantees as g  # noqa: E402

RNG = np.random.default_rng(20260923)


# ----------------------------------------------------------------------------------------------------------------
# brute-force references (straight from the formulas; O(n^2) or worse, only for small instances)
# ----------------------------------------------------------------------------------------------------------------
def _ref_mdr_e(cal_s, cal_L, cal_w, s_t, w_t, gamma):
    """Eq. (6.1): E = inf_{ell in [0,1]} 1{s_t <= t(ell)} * sum_{i<=n+1} w_i / (sum_i w_i L_i 1{s_i<=t(ell)} + w_t ell 1{s_t<=t(ell)}),
    t(ell) = max{t in M : F(t; ell) <= gamma}, F(t; ell) = (sum_i w_i L_i 1{s_i<=t} + w_t ell 1{s_t<=t}) / sum_{i<=n+1} w_i,
    M = {s_1..s_n, s_t}; E = 0 when inf_ell t(ell) = -inf.  The infimum over ell is evaluated EXACTLY: between breakpoints
    t(ell) is constant and E is non-increasing in ell, so the inf is attained at ell = 1 or just left of / at a breakpoint."""
    M = np.unique(np.r_[cal_s, s_t])
    W = cal_w.sum() + w_t

    def t_of(ell):
        F = np.array([(np.sum(cal_w * cal_L * (cal_s <= t)) + w_t * ell * (s_t <= t)) / W for t in M])
        ok = M[F <= gamma + 1e-15]
        return ok.max() if ok.size else -np.inf

    def E_of(ell):
        t = t_of(ell)
        if not np.isfinite(t) or s_t > t:
            return 0.0
        den = np.sum(cal_w * cal_L * (cal_s <= t)) + w_t * ell
        return np.inf if den <= 0 else W / den

    # breakpoints: F(t; ell) = gamma for some t >= s_t  ->  ell_b = (gamma W - sum_i w_i L_i 1{s_i<=t}) / w_t
    bps = [(gamma * W - np.sum(cal_w * cal_L * (cal_s <= t))) / w_t for t in M if t >= s_t]
    cands = {0.0, 1.0}
    for b in bps:
        for x in (b, b - 1e-10, b + 1e-10):
            if 0.0 <= x <= 1.0:
                cands.add(float(x))
    if min(t_of(e) for e in cands) == -np.inf:
        return 0.0
    return min(E_of(e) for e in cands)


def _ref_sdr_e(cal_s, cal_L, cal_w, test_s, test_w, j, gamma):
    """Eq. (6.2) for test unit j, same exact-infimum argument as above."""
    m = len(test_s)
    s_j, w_j = test_s[j], test_w[j]
    M = np.unique(np.r_[cal_s, test_s])
    Wj = w_j + cal_w.sum()
    others = np.delete(test_s, j)

    def FR(t, ell):
        num = w_j * ell * (s_j <= t) + np.sum(cal_w * cal_L * (cal_s <= t))
        return num / (1 + np.sum(others <= t)) * m / Wj

    def t_of(ell):
        ok = [t for t in M if FR(t, ell) <= gamma + 1e-15]
        return max(ok) if ok else -np.inf

    def E_of(ell):
        t = t_of(ell)
        if not np.isfinite(t) or s_j > t:
            return 0.0
        den = w_j * ell + np.sum(cal_w * cal_L * (cal_s <= t))
        return np.inf if den <= 0 else Wj / den

    bps = []
    for t in M:
        if t >= s_j:
            # FR(t; ell) = gamma  ->  ell = (gamma * Wj * (1 + #others<=t) / m - sum_i w_i L_i 1{s_i<=t}) / w_j
            bps.append((gamma * Wj * (1 + np.sum(others <= t)) / m - np.sum(cal_w * cal_L * (cal_s <= t))) / w_j)
    cands = {0.0, 1.0}
    for b in bps:
        for x in (b, b - 1e-10, b + 1e-10):
            if 0.0 <= x <= 1.0:
                cands.add(float(x))
    if min(t_of(e) for e in cands) == -np.inf:
        return 0.0
    return min(E_of(e) for e in cands)


def _ref_ebh(E, alpha):
    m = len(E)
    ks = [k for k in range(1, m + 1) if np.sum(E >= m / (alpha * k)) >= k]
    if not ks:
        return np.zeros(m, bool)
    k = max(ks)
    return E >= m / (alpha * k)


def _toy(n, m, rng, shift=0.0):
    """Covariate x ~ N(shift, 1); bounded loss L = clip(|x| * u, 0, 1); predicted-risk score s = |x| + small noise (low = trusted)."""
    x_c = rng.normal(0, 1, n); x_t = rng.normal(shift, 1, m)
    L_c = np.clip(np.abs(x_c) * rng.uniform(0, 0.6, n), 0, 1)
    L_t = np.clip(np.abs(x_t) * rng.uniform(0, 0.6, m), 0, 1)
    s_c = np.abs(x_c) + rng.normal(0, 0.05, n); s_t = np.abs(x_t) + rng.normal(0, 0.05, m)
    return x_c, L_c, s_c, x_t, L_t, s_t


def _gauss_w(x, shift):
    """Exact density ratio dQ/dP for N(shift,1) vs N(0,1)."""
    return np.exp(shift * x - shift ** 2 / 2)


# ----------------------------------------------------------------------------------------------------------------
# 1. weighted SCoRE MDR (Eq. 6.1, Thm 6.2, Prop. A.1)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_weighted_mdr_e_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    x_c, L_c, s_c, x_t, _, s_t = _toy(30, 6, rng, shift=0.5)
    w_c, w_t = _gauss_w(x_c, 0.5), _gauss_w(x_t, 0.5)
    gamma = [0.05, 0.1, 0.2][seed % 3]
    E = g.weighted_mdr_e_values(s_c, L_c, w_c, s_t, w_t, gamma)
    assert E.shape == (6,)
    for j in range(6):
        ref = _ref_mdr_e(s_c, L_c, w_c, s_t[j], w_t[j], gamma)
        assert (np.isinf(ref) and np.isinf(E[j])) or np.isclose(E[j], ref, rtol=1e-8, atol=1e-10), (j, E[j], ref)


@pytest.mark.parametrize("seed", range(8))
def test_weighted_mdr_reduces_to_unweighted(seed):
    rng = np.random.default_rng(100 + seed)
    _, L_c, s_c, _, _, s_t = _toy(40, 5, rng)
    ones_c, ones_t = np.ones(40), np.ones(5)
    E_w = g.weighted_mdr_e_values(s_c, L_c, ones_c, s_t, ones_t, 0.1)
    E_u = g.mdr_e_values(s_c, L_c, s_t, 0.1)            # unweighted Eq. (4.1)
    assert np.allclose(E_w, E_u, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_prop_a1_shortcut_equals_threshold_decision(seed):
    """Prop. A.1 (gamma <= alpha): 1{E >= 1/alpha} = 1{(w_t + sum_i w_i L_i 1{s_i <= s_t}) / sum_{i<=n+1} w_i <= gamma}."""
    rng = np.random.default_rng(200 + seed)
    x_c, L_c, s_c, x_t, _, s_t = _toy(50, 20, rng, shift=0.7)
    w_c, w_t = _gauss_w(x_c, 0.7), _gauss_w(x_t, 0.7)
    alpha = 0.15
    dec = g.weighted_mdr_decision(s_c, L_c, w_c, s_t, w_t, alpha)          # gamma = alpha
    E = g.weighted_mdr_e_values(s_c, L_c, w_c, s_t, w_t, alpha)
    assert dec.dtype == bool and np.array_equal(dec, E >= 1 / alpha)


def test_weighted_mdr_control_known_w_monte_carlo():
    """Thm 6.2: under Assumption 6.1 with known w, E_Q[L * psi] <= alpha (finite sample)."""
    rng = np.random.default_rng(7)
    alpha, shift, reps, vals = 0.10, 0.8, 1500, []
    for _ in range(reps):
        x_c, L_c, s_c, x_t, L_t, s_t = _toy(200, 1, rng, shift=shift)
        dec = g.weighted_mdr_decision(s_c, L_c, _gauss_w(x_c, shift), s_t, _gauss_w(x_t, shift), alpha)
        vals.append(L_t[0] * dec[0])
    vals = np.array(vals)
    assert vals.mean() <= alpha + 3 * vals.std(ddof=1) / np.sqrt(reps)


# ----------------------------------------------------------------------------------------------------------------
# 2. weighted SCoRE SDR (Eq. 6.2, Thm 6.3, Alg. 4) + e-BH (Thm 3.3) + boosting (Thm 5.5)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_weighted_sdr_e_matches_bruteforce(seed):
    rng = np.random.default_rng(300 + seed)
    x_c, L_c, s_c, x_t, _, s_t = _toy(25, 7, rng, shift=0.4)
    w_c, w_t = _gauss_w(x_c, 0.4), _gauss_w(x_t, 0.4)
    gamma = [0.1, 0.2][seed % 2]
    E = g.weighted_sdr_e_values(s_c, L_c, w_c, s_t, w_t, gamma)
    for j in range(7):
        ref = _ref_sdr_e(s_c, L_c, w_c, s_t, w_t, j, gamma)
        assert (np.isinf(ref) and np.isinf(E[j])) or np.isclose(E[j], ref, rtol=1e-8, atol=1e-10), (j, E[j], ref)


@pytest.mark.parametrize("seed", range(6))
def test_weighted_sdr_reduces_to_unweighted(seed):
    rng = np.random.default_rng(400 + seed)
    _, L_c, s_c, _, _, s_t = _toy(30, 8, rng)
    E_w = g.weighted_sdr_e_values(s_c, L_c, np.ones(30), s_t, np.ones(8), 0.2)
    E_u = g.sdr_e_values(s_c, L_c, s_t, 0.2)             # unweighted Eq. (5.1)
    assert np.allclose(E_w, E_u, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_ebh_matches_definition(seed):
    rng = np.random.default_rng(500 + seed)
    E = rng.exponential(3.0, 40) * (rng.random(40) < 0.6)
    assert np.array_equal(g.ebh(E, 0.1), _ref_ebh(E, 0.1))


def test_weighted_sdr_control_known_w_monte_carlo():
    rng = np.random.default_rng(9)
    alpha, shift, reps, sdr = 0.2, 0.6, 400, []
    for _ in range(reps):
        x_c, L_c, s_c, x_t, L_t, s_t = _toy(120, 20, rng, shift=shift)
        E = g.weighted_sdr_e_values(s_c, L_c, _gauss_w(x_c, shift), s_t, _gauss_w(x_t, shift), alpha)
        R = g.ebh(E, alpha)
        sdr.append(L_t[R].sum() / max(1, R.sum()))
    sdr = np.array(sdr)
    assert sdr.mean() <= alpha + 3 * sdr.std(ddof=1) / np.sqrt(reps)


def test_boosting_valid_and_not_less_powerful():
    """Thm 5.5: hete/homo boosted e-BH keep SDR <= alpha; they should select at least as much as plain e-BH on average."""
    rng = np.random.default_rng(10)
    alpha, reps = 0.2, 400
    res = {"plain": [], "hete": [], "homo": []}
    size = {"plain": [], "hete": [], "homo": []}
    for _ in range(reps):
        _, L_c, s_c, _, L_t, s_t = _toy(120, 20, rng)
        E = g.sdr_e_values(s_c, L_c, s_t, alpha)
        for mode in res:
            R = g.ebh(E, alpha) if mode == "plain" else g.ebh_boosted(E, alpha, mode=mode, rng=rng)
            res[mode].append(L_t[R].sum() / max(1, R.sum())); size[mode].append(R.sum())
    for mode in ("hete", "homo"):
        v = np.array(res[mode])
        assert v.mean() <= alpha + 3 * v.std(ddof=1) / np.sqrt(reps), mode
        assert np.mean(size[mode]) >= np.mean(size["plain"]) - 1e-9, mode


# ----------------------------------------------------------------------------------------------------------------
# 3. weights: domain classifier, effective sample size, balancing (Assumption A.3)
# ----------------------------------------------------------------------------------------------------------------
def test_effective_sample_size():
    assert np.isclose(g.effective_sample_size(np.ones(50)), 50.0)
    w = np.r_[np.ones(9), 10.0]
    assert np.isclose(g.effective_sample_size(w), w.sum() ** 2 / np.sum(w ** 2))


def test_domain_classifier_weights_recover_gaussian_ratio():
    rng = np.random.default_rng(11)
    Xs, Xt = rng.normal(0, 1, (3000, 1)), rng.normal(0.8, 1, (3000, 1))
    w_s, w_t = g.domain_classifier_weights(Xs, Xt, n_folds=5, clip_quantile=0.99, seed=0)
    assert w_s.shape == (3000,) and w_t.shape == (3000,) and np.all(w_s > 0) and np.all(np.isfinite(w_s))
    assert abs(w_s.mean() - 1.0) < 0.05                                   # normalised so the source mean is 1
    true = _gauss_w(Xs[:, 0], 0.8)
    assert np.corrcoef(np.log(w_s), np.log(true))[0, 1] > 0.9


def test_balancing_satisfies_assumption_a3():
    rng = np.random.default_rng(12)
    n, m, alpha = 400, 300, 0.1
    s_c, s_t = rng.uniform(0, 1, n), rng.uniform(0.2, 1, m)
    l_c, l_t = np.clip(s_c * 0.5, 0, 1), np.clip(s_t * 0.5, 0, 1)
    w0 = rng.uniform(0.5, 2.0, n)
    w, t_hat = g.balance_weights_mdr(w0, l_c, s_c, l_t, s_t, alpha)
    assert np.all(w > 0)
    assert np.isclose(w.mean(), 1.0, atol=1e-6)
    lhs = np.mean(w * l_c * (s_c <= t_hat)); rhs = np.mean(l_t * (s_t <= t_hat))
    assert np.isclose(lhs, rhs, atol=1e-6)
    # t_hat is Assumption A.3's cutoff: sup{t : (1/m) sum_j lhat_j 1{s_j <= t} <= alpha}
    assert np.mean(l_t * (s_t <= t_hat)) <= alpha + 1e-12


# ----------------------------------------------------------------------------------------------------------------
# 4. document-level PAC variants (tier 4, derived here) and Clopper-Pearson helper
# ----------------------------------------------------------------------------------------------------------------
def test_clopper_pearson_upper():
    assert np.isclose(g.clopper_pearson_upper(0, 10, 0.95), 1 - 0.05 ** (1 / 10))
    assert np.isclose(g.clopper_pearson_upper(3, 20, 0.9), stats.beta.ppf(0.9, 4, 17))


@pytest.mark.parametrize("method", ["hoeffding", "binomial_any_error", "bernstein"])
def test_doc_tier4_variants_are_valid(method):
    """With iid documents, P(true functional at the certified threshold > alpha) <= delta (+ CP tolerance).
    'hoeffding'/'bernstein' bound the macro per-document error rate; 'binomial_any_error' bounds the share of accepting
    documents with at least one accepted error (a stricter functional)."""
    rng = np.random.default_rng(13)
    alpha, delta, reps = 0.10, 0.10, 200
    viol = 0
    for _ in range(reps):
        D, k = 300, 6
        doc = np.repeat(np.arange(D), k)
        eff = np.repeat(rng.normal(0, 1.0, D), k)
        score = rng.normal(0, 1, D * k) - 0.8 * eff
        p_err = 1 / (1 + np.exp(-(-2.2 - 1.5 * score + eff)))
        err = (rng.random(D * k) < p_err).astype(float)
        tau = g.doc_tier4_threshold(doc, score, err, alpha, delta, method=method)
        if not np.isfinite(tau):
            continue
        # population functional by a large fresh draw from the same DGP
        D2 = 20000
        doc2 = np.repeat(np.arange(D2), k); eff2 = np.repeat(rng.normal(0, 1.0, D2), k)
        sc2 = rng.normal(0, 1, D2 * k) - 0.8 * eff2
        er2 = (rng.random(D2 * k) < 1 / (1 + np.exp(-(-2.2 - 1.5 * sc2 + eff2)))).astype(float)
        acc = sc2 >= tau
        cnt = np.bincount(doc2[acc], minlength=D2); tot = np.bincount(doc2[acc], weights=er2[acc], minlength=D2)
        has = cnt > 0
        if not has.any():
            continue
        val = (tot[has] / cnt[has]).mean() if method != "binomial_any_error" else (tot[has] > 0).mean()
        viol += val > alpha
    assert viol / reps <= delta + 3 * np.sqrt(delta * (1 - delta) / reps)


def test_doc_tier4_binomial_and_bernstein_not_weaker_than_hoeffding_on_average():
    """Derived-here power claim: on the same data the empirical-Bernstein variant accepts at least as much as Hoeffding on average."""
    rng = np.random.default_rng(14)
    cov = {"hoeffding": [], "bernstein": []}
    for _ in range(60):
        D, k = 400, 6
        doc = np.repeat(np.arange(D), k); eff = np.repeat(rng.normal(0, 1.0, D), k)
        score = rng.normal(0, 1, D * k) - 0.8 * eff
        err = (rng.random(D * k) < 1 / (1 + np.exp(-(-2.5 - 1.5 * score + eff)))).astype(float)
        for mth in cov:
            tau = g.doc_tier4_threshold(doc, score, err, 0.10, 0.10, method=mth)
            cov[mth].append(np.mean(score >= tau) if np.isfinite(tau) else 0.0)
    assert np.mean(cov["bernstein"]) >= np.mean(cov["hoeffding"]) - 1e-9


# ----------------------------------------------------------------------------------------------------------------
# 5. weighted conformal p-values and Weighted Conformalized Selection (Jin & Candes 2023a; idpshift-q1)
# ----------------------------------------------------------------------------------------------------------------
def _ref_wpval_det(Vc, wc, Vhat, wt):
    """Eq. (5): p_j = (sum_i w_i 1{V_i < Vhat_j} + w_j) / (sum_i w_i + w_j)."""
    return np.array([(np.sum(wc * (Vc < v)) + w) / (wc.sum() + w) for v, w in zip(Vhat, wt)])


def _ref_wcs(Vc, wc, Vhat, wt, q, pruning, xi):
    """Algorithm 1 of Jin & Candes 2023a, straight from Eq. (5)-(9). xi: array (hete) or scalar (homo), unused for dtm."""
    m = len(Vhat)
    p = _ref_wpval_det(Vc, wc, Vhat, wt)
    R0 = np.zeros(m, int); s = np.zeros(m)
    for j in range(m):
        den = wc.sum() + wt[j]
        pl = np.array([(np.sum(wc * (Vc < Vhat[l])) + wt[j] * (Vhat[j] < Vhat[l])) / den for l in range(m)])   # Eq. (6)
        pl[j] = 0.0
        ks = [k for k in range(1, m + 1) if 1 + np.sum(np.delete(pl, j) <= q * k / m) >= k]
        k = max(ks)
        R0[j] = 1 + np.sum(np.delete(pl, j) <= q * k / m)
        s[j] = q * R0[j] / m
    first = p <= s
    if pruning == "dtm":
        val = R0.astype(float)
    elif pruning == "homo":
        val = xi * R0
    else:
        val = xi * R0
    rs = [r for r in range(0, m + 1) if np.sum(first & (val <= r)) >= r]
    r = max(rs)
    return first & (val <= r)


def test_weighted_pvalue_reduces_to_cfbh_when_unweighted():
    rng = np.random.default_rng(15)
    Vc, Vhat = rng.normal(size=50), rng.normal(size=10)
    p_w = g.weighted_conformal_pvalues(Vc, np.ones(50), Vhat, np.ones(10), deterministic=True)
    p_cf = (1 + np.array([np.sum(Vc < v) for v in Vhat])) / 51           # cfBH deterministic p-value (Thm 2.6 form)
    assert np.allclose(p_w, p_cf)
    assert np.allclose(g.weighted_conformal_pvalues(Vc, rng.uniform(0.5, 2, 50), Vhat, rng.uniform(0.5, 2, 10), deterministic=True)
                       .shape, (10,))


@pytest.mark.parametrize("seed", range(8))
def test_wcs_matches_reference(seed):
    rng = np.random.default_rng(600 + seed)
    Vc, Vhat = rng.normal(size=40), rng.normal(-0.5, 1, size=12)
    wc, wt = rng.uniform(0.3, 3, 40), rng.uniform(0.3, 3, 12)
    for pruning in ("dtm", "homo", "hete"):
        xi = rng.uniform(size=12) if pruning == "hete" else rng.uniform()
        got = g.wcs_select(Vc, wc, Vhat, wt, q=0.2, pruning=pruning, xi=xi)
        assert np.array_equal(got, _ref_wcs(Vc, wc, Vhat, wt, 0.2, pruning, xi)), pruning


def test_wcs_fdr_known_w_monte_carlo():
    """Theorem 3.1: with known w, WCS (any pruning) keeps FDR <= q in finite samples. Clipped score V(x,y) = M*1{y>0} - mu(x)."""
    rng = np.random.default_rng(16)
    q, shift, reps = 0.2, 0.8, 300
    fdp = {"hete": [], "homo": []}
    for _ in range(reps):
        xc, xt = rng.normal(0, 1, 200), rng.normal(shift, 1, 40)
        yc = (rng.random(200) < 1 / (1 + np.exp(-2 * xc))).astype(float)
        yt = (rng.random(40) < 1 / (1 + np.exp(-2 * xt))).astype(float)
        mu = lambda x: 1 / (1 + np.exp(-2 * x))
        Vc = 100 * (yc > 0) - mu(xc); Vhat = 100 * 0 - mu(xt)             # test side evaluated at c = 0 (select y = 1)
        wc, wt = _gauss_w(xc, shift), _gauss_w(xt, shift)
        for pr in fdp:
            xi = rng.uniform(size=40) if pr == "hete" else rng.uniform()
            R = g.wcs_select(Vc, wc, Vhat, wt, q=q, pruning=pr, xi=xi)
            fdp[pr].append(np.sum(R & (yt == 0)) / max(1, R.sum()))
    for pr, v in fdp.items():
        v = np.array(v)
        assert v.mean() <= q + 3 * v.std(ddof=1) / np.sqrt(reps), pr
