"""Notebook 3, PART B (owned by the part-B agent): sections 5-8.

  5. SCoRE with an amount-weighted document loss (P2)
  6. The validity ladder on our data, per document type (P3)
  7. Month-7 shift: frozen-configuration confirmation + what nobody covers
  8. Deliverable: threshold documentation + cost accounting

Namespace rule: every global defined here is prefixed `B_` / `b_` (loop temporaries: leading underscore).
The machinery for the Mondrian-LTT tiers is COPIED (with attribution) from build_nb1.py and only re-vectorised;
the SCoRE e-value code is a tie-safe, vectorised re-implementation checked against Notebook 1's version.
"""
from nb3_common import M, C


def _s5():
    out = []

    # ------------------------------------------------------------------------------------------------ 5: markdown
    out.append(M(r'''
## 5. SCoRE with an amount-weighted document loss (P2)

**Claim.** Notebook 2's business pain is not "how many fields are wrong" but "how much *money* is wrong". A bounded
amount-weighted loss `L_doc` in [0,1] (Section 2) fits SCoRE directly. Cost-weighted losses fit SCoRE, do **not** fit cfBH, and do
not fit Gurram's tier-3 binomial test: "A loss weighted by monetary amount (or dollar error) fits SCoRE directly, provided the risk
function is normalized/scaled to a bounded interval" (idpfin-q11); cfBH's authors list this as an open problem: "counting the number
of errors may be less sensible if the cost of making an error varies with individuals or depends on the outcomes" (idpfin-q3, q11).

**What is promised (quoted).** Two different risks, two different promises (idpfin-q4):

| name | definition (idpfin-q4) | what it is NOT |
| --- | --- | --- |
| MDR, marginal deployment risk | `E[L_{n+1} * psi_{n+1}] <= alpha` (2.1) | **not** "at most alpha of accepted items are wrong": "a procedure may deploy few but comparatively risky cases yet still controlling the MDR" |
| SDR, selective deployment risk | `E[ sum_j L_{n+j} 1{j in R} / (1 v |R|) ] <= alpha` (2.3), via e-BH | an expectation over batches, not a per-batch or high-probability statement |

Both come from a *risk-adjusted e-value*: "`E >= 0` almost surely and `E[E * L] <= 1`" (Definition 3.1, idpfin-q4); trust decision
`psi = 1{E >= 1/alpha}` (Theorem 3.2) or e-BH (Theorem 3.3). Assumptions, verbatim (idpfin-q4): the loss is bounded ("without loss of
generality, assume L in [0,1]"), data are exchangeable, and "both f(.) and s(.) are trained independently of D_calib and D_test".
We use `gamma = alpha` (the default recommended in idpfin-q10; "the asymptotic power is optimized at gamma = alpha", Theorem 4.6(ii),
idpfin-q5).

**Choices made here (derived here, none of it is in the papers).**

- *Unit = document.* SCoRE "does not formulate a document-level clustered extraction protocol" (idpfin-q11); the document is the
  exchangeable unit (Gurram, idpfin-q11), so we apply SCoRE to whole documents. Calibration and test documents come from
  `resplit(seed)` (a random split of a finite pool, hence exchangeable) restricted to **money documents**
  (invoice / bank statement / compliance report). **KYC forms are excluded**: they have no money field, so `L_doc` is undefined.
- *Score* `s = DA_risk` (optimistic, learned fusion) and `DA_risk_weak` (pessimistic), both *predicted E[L_doc]*, **LOW = trusted**:
  a document is trusted when `s(X) <= t` (the convention of Notebook 1 / P2). Both are fit on `train` documents only.
- *Ties.* The predicted-risk scores are clipped at 0 and are piecewise constant, so they contain exact ties (about 11% of money
  documents sit at exactly 0). The paper's threshold set is `M = {s(X_i)}`, a set of score **values**: `F(t; l)` at a tied value counts
  *every* document with that score. Notebook 1's index-wise scan (fine for continuous toy scores) evaluates `F` at partial tie groups,
  which is anti-conservative; the implementation below scans **distinct values** and we demonstrate the difference in S5.3.
- *SDR algorithm.* The paper states that computing the SDR e-value "necessitates a search over l in [0,1] which can be
  computationally prohibitive" and that Algorithm 3 has complexity `O((n+m)m + (n+m)log(n+m))` (Proposition 5.2, idpfin-q6). **The
  algorithm itself is not quoted in our traces**, so the SDR code below is our own exact implementation (the infimum over l is
  attained at l = 0, l = 1 or a breakpoint where a feasibility constraint switches; Notebook 1's derivation), *not* Algorithm 3.
- *Tolerances.* Monte-Carlo means: `mean <= target + 3 SE` (SE across resplits/batches). Proportions: `<= target +` the exact one-sided
  Clopper-Pearson upper bound minus the estimate (conf 0.9987), never a degenerate zero tolerance. The resplits/batches all reshuffle the
  *same* finite pool, so they estimate the expectation over random splits of this corpus, not a population.
'''))

    # ------------------------------------------------------------------------------------------------ 5: machinery
    out.append(C(r'''
import time as b_time
from scipy.stats import binom as b_binom
from IPython.display import display as b_display, Markdown as b_md

PAL3.setdefault("b_weak", "#ff7f0e"); PAL3.setdefault("b_naive", "#9467bd"); PAL3.setdefault("b_folk", "#7f7f7f")
PAL3.setdefault("b_target", "#000000"); PAL3.setdefault("b_good", "#2ca02c")
B_SCORES = {"opt": DA_risk, "weak": DA_risk_weak}
B_SC_LABEL = {"opt": "optimistic (learned fusion)", "weak": "pessimistic (weak signals)"}
B_SC_COLOR = {"opt": PAL3["hgb"], "weak": PAL3["b_weak"]}
B_POOL = np.where(np.isin(DOC_SPLIT, ["calib", "test"]) & DA_has_money)[0]      # money docs of the calib+test pool


def b_tbl(headers, rows):
    """Markdown pipe table (rendered through IPython.display.Markdown)."""
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def b_mc_tol_mean(x):
    x = np.asarray(x, float); return 3 * np.std(x, ddof=1) / np.sqrt(len(x))


def b_cp_tol(p_hat, reps, conf=0.9987):
    """Exact one-sided Clopper-Pearson upper bound minus the estimate (copied from Notebook 1's mc_tol_prop)."""
    k = min(max(int(round(p_hat * reps)), 0), reps)
    up = 1.0 if k >= reps else stats.beta.ppf(conf, k + 1, reps - k)
    return float(up - p_hat)


def b_mdr_e_values(cal_s, cal_L, test_s, gamma):
    """SCoRE-MDR e-values (4.1)-(4.2) for ALL test points at once. Exact infimum over l (Notebook 1's breakpoint
    argument: candidates l=0, l=1 and, if a breakpoint lies in [0,1], the value 1/gamma), TIE-COMPLETE: the
    threshold t ranges over score VALUES, so F(t; l) counts every calibration document tied at t. O(log n) per point."""
    n = len(cal_s); o = np.argsort(cal_s, kind="stable"); c = cal_s[o]
    Sc = np.concatenate([[0.0], np.cumsum(cal_L[o])])
    S_el = Sc[np.searchsorted(c, c, side="right")]                        # sum of L over calib scores <= c_i (whole tie group)
    t = np.asarray(test_s, float)
    p = np.searchsorted(c, t, side="left"); q = np.searchsorted(c, t, side="right")
    B = gamma * (n + 1)
    cnt1 = np.searchsorted(S_el, B, side="right")

    def E_at(ell):
        m2 = np.searchsorted(S_el, B - ell, side="right")
        test_feas = (cnt1 >= p) & (Sc[q] + ell <= B)
        k = np.maximum(p, m2)
        cum_k = np.where(k == p, Sc[q], S_el[np.clip(k - 1, 0, n - 1)])
        den = cum_k + ell
        with np.errstate(divide="ignore"):
            val = np.where(den > 1e-9, (n + 1) / np.maximum(den, 1e-300), np.inf)
        return np.where(test_feas, val, 0.0)

    lo = np.searchsorted(S_el, B - 1 - 1e-9, side="left"); hi = np.searchsorted(S_el, B + 1e-9, side="right")
    has_bp = (hi > np.maximum(lo, p)) | ((Sc[q] >= B - 1 - 1e-9) & (Sc[q] <= B + 1e-9))
    cand = np.minimum(E_at(0.0), E_at(1.0))
    return np.where(has_bp, np.minimum(cand, 1.0 / gamma), cand)


def b_mdr_indexwise(cal_s, cal_L, test_s, gamma):
    """Notebook 1's `mdr_e_value` logic, vectorised over test points (INDEX-wise scan: F is evaluated at every sorted
    position, i.e. also inside a tie group). Kept only to show why ties matter (S5.3)."""
    n = len(cal_s); o = np.argsort(cal_s, kind="stable"); c = cal_s[o]
    S = np.concatenate([[0.0], np.cumsum(cal_L[o])]); S = np.concatenate([S, [S[-1]]])
    t = np.asarray(test_s, float)
    p = np.searchsorted(c, t, side="left"); q = np.searchsorted(c, t, side="right")
    J = np.arange(n + 1)[None, :]
    cum = np.where(J < q[:, None], S[J + 1], np.where(J == q[:, None], S[q][:, None], S[J]))
    ge = J >= p[:, None]; B = gamma * (n + 1); rows = np.arange(len(t))

    def E_at(ell):
        feas = (cum + ell * ge) <= B
        k = feas.sum(1) - 1; kk = np.clip(k, 0, n)
        ok = (k >= 0) & (kk >= p)
        den = cum[rows, kk] + ell
        with np.errstate(divide="ignore"):
            val = np.where(den > 1e-9, (n + 1) / np.maximum(den, 1e-300), np.inf)
        return np.where(ok, val, 0.0)

    lb = B - cum
    has_bp = (ge & (lb >= -1e-9) & (lb <= 1 + 1e-9)).any(1)
    cand = np.minimum(E_at(0.0), E_at(1.0))
    return np.where(has_bp, np.minimum(cand, 1.0 / gamma), cand)


def b_mdr_grid_ref(cal_s, cal_L, s_t, gamma, grid):
    """Definition (4.1)-(4.2) evaluated literally (distinct score values, dense l-grid): an UPPER bound of the exact infimum."""
    n = len(cal_s); Mv = np.unique(np.concatenate([cal_s, [s_t]]))
    Sle = np.array([cal_L[cal_s <= v].sum() for v in Mv]); ge = Mv >= s_t; best = np.inf
    for ell in grid:
        feas = np.nonzero((Sle + ell * ge) / (n + 1) <= gamma)[0]
        if len(feas) == 0:
            val = 0.0
        else:
            k = feas[-1]
            val = 0.0 if not ge[k] else ((n + 1) / (Sle[k] + ell) if Sle[k] + ell > 1e-9 else np.inf)
        best = min(best, val)
    return best


# ---- SCoRE-SDR: exact infimum over l, tie-complete, one test point at a time (candidates = {0, 1, breakpoints in [0,1]}) ----
def b_sdr_prep(cal_s, cal_L, test_s):
    """Distinct-value grid M of calibration+test scores with tie-complete S_le(t) = sum L_i 1{s_i<=t} and cnt(t) = #{test <= t}."""
    Mv = np.unique(np.concatenate([cal_s, test_s]))
    o = np.argsort(cal_s, kind="stable"); c = cal_s[o]; Sc = np.concatenate([[0.0], np.cumsum(cal_L[o])])
    return Mv, Sc[np.searchsorted(c, Mv, side="right")], np.searchsorted(np.sort(test_s), Mv, side="right")


def b_sdr_e_value(prep, s_j, gamma, n, m):
    Mv, S_le, cnt = prep
    ge = (Mv >= s_j).astype(float)
    lb = gamma * (n + 1) * (1.0 + cnt - ge) / m - S_le                    # l at which FR(t; l) = gamma (only t >= s_j depend on l)
    lb = lb[(ge > 0) & (lb >= -1e-9) & (lb <= 1 + 1e-9)]
    ells = np.unique(np.concatenate([[0.0, 1.0], np.clip(lb, 0.0, 1.0)]))
    FR = (ells[:, None] * ge[None, :] + S_le[None, :]) / (n + 1) * (m / (1.0 + cnt - ge))[None, :]
    feas = FR <= gamma + 1e-12                                            # FR is NOT monotone in t: take the last feasible column
    if not feas.any(1).all():
        return 0.0                                                        # E = 0 when max(empty) = -inf for some l
    k = feas.shape[1] - 1 - np.argmax(feas[:, ::-1], axis=1)
    den = ells + S_le[k]
    val = np.where(ge[k] > 0, np.where(den > 1e-9, (n + 1) / np.maximum(den, 1e-300), np.inf), 0.0)
    return float(val.min())


def b_sdr_e_values(cal_s, cal_L, test_s, gamma, prep=None):
    n, m = len(cal_s), len(test_s); prep = prep or b_sdr_prep(cal_s, cal_L, test_s)
    return np.array([b_sdr_e_value(prep, test_s[j], gamma, n, m) for j in range(m)])


def b_sdr_grid_ref(cal_s, cal_L, test_s, j, gamma, grid):
    n, m = len(cal_s), len(test_s); Mv = np.unique(np.concatenate([cal_s, test_s]))
    S_le = np.array([cal_L[cal_s <= v].sum() for v in Mv]); cnt = np.array([(test_s <= v).sum() for v in Mv]); ge = (Mv >= test_s[j]).astype(float)
    best = np.inf
    for ell in grid:
        feas = np.nonzero((ell * ge + S_le) / (n + 1) * (m / (1 + cnt - ge)) <= gamma)[0]
        if len(feas) == 0:
            return 0.0
        k = feas[-1]
        val = 0.0 if ge[k] == 0 else ((n + 1) / (ell + S_le[k]) if ell + S_le[k] > 1e-9 else np.inf)
        best = min(best, val)
    return best


def b_ebh_select(E, alpha):
    """e-BH (Theorem 3.3, idpfin-q4): k* = max{k : #{E_j >= m/(alpha k)} >= k}; select E_j >= m/(alpha k*)."""
    m = len(E); Es = np.sort(E)[::-1]; k = np.arange(1, m + 1); ok = Es >= m / (alpha * k)
    return E >= m / (alpha * k[ok].max()) if ok.any() else np.zeros(m, bool)


# ---- verification of the implementations ----
_c0, _t0 = resplit(3); _c0, _t0 = _c0[DA_has_money[_c0]], _t0[DA_has_money[_t0]]
_rng = np.random.default_rng(0)

# (i) tie-free comparison against the index-wise scan (= Notebook 1's logic): with all-distinct scores they must agree exactly
_jit = DA_risk + np.random.default_rng(1).uniform(0, 1e-7, N_DOC)          # tie-breaking jitter, only for this comparison
_agree = all(np.allclose(b_mdr_e_values(_jit[_c0], DA_L[_c0], _jit[_t0], g), b_mdr_indexwise(_jit[_c0], DA_L[_c0], _jit[_t0], g))
             for g in (0.05, 0.1, 0.2))
check("S5.1 tie-complete O(log n) MDR e-values equal Notebook 1's index-wise scan when scores are tie-free", _agree,
      "gamma in {0.05, 0.10, 0.20}, one resplit, all test documents")

# (ii) against the literal definition on the REAL (tied) scores: exact inf must not exceed any dense-grid value
_grid = np.linspace(0, 1, 401); _ok, _gap = True, 0.0
for _g in (0.05, 0.1, 0.2):
    for _s in (DA_risk, DA_risk_weak):
        _sub = _rng.choice(len(_t0), 30, replace=False)
        _f = b_mdr_e_values(_s[_c0], DA_L[_c0], _s[_t0][_sub], _g)
        _r = np.array([b_mdr_grid_ref(_s[_c0], DA_L[_c0], _s[_t0][j], _g, _grid) for j in _sub])
        _ok &= bool(np.all(_f <= _r + 1e-9)) and bool(np.all((_f >= 1 / _g) == (_r >= 1 / _g)))
        _gap = max(_gap, float(np.max(np.where(np.isinf(_r), 0, _r - _f))))
check("S5.2 exact-inf MDR e-value <= dense-grid evaluation of the definition (real, tied scores) and the trust decisions agree", _ok,
      f"3 gammas x 2 scores x 30 test docs; max grid-exact gap {_gap:.4f} (a grid can only overestimate an infimum)")

# (iii) why ties matter: coarsen the score to one decimal (10 distinct values) and compare the two scans over resplits
_res_tie = {}
for _g in (0.10, 0.20):
    _a_, _b_ = [], []
    for _sd in range(60):
        _c1, _t1 = resplit(2000 + _sd); _c1, _t1 = _c1[DA_has_money[_c1]], _t1[DA_has_money[_t1]]
        _s = np.round(DA_risk, 1)
        _a_.append(np.mean(DA_L[_t1] * (b_mdr_e_values(_s[_c1], DA_L[_c1], _s[_t1], _g) >= 1 / _g)))
        _b_.append(np.mean(DA_L[_t1] * (b_mdr_indexwise(_s[_c1], DA_L[_c1], _s[_t1], _g) >= 1 / _g)))
    _res_tie[_g] = (np.mean(_a_), np.mean(_b_), b_mc_tol_mean(_a_), b_mc_tol_mean(_b_))
B_TIE_DEMO = _res_tie
print("tie demo (score rounded to 1 decimal, 60 resplits): " + "; ".join(
    f"alpha={g}: tie-complete MDR {v[0]:.4f} | index-wise MDR {v[1]:.4f}" for g, v in _res_tie.items()))
check("S5.3 on heavily tied scores the tie-complete scan keeps MDR <= alpha while the index-wise scan exceeds alpha",
      all(v[0] <= g + v[2] for g, v in _res_tie.items()) and any(v[1] > g + v[3] for g, v in _res_tie.items()),
      "; ".join(f"alpha={g}: tie-complete {v[0]:.4f} (tol {v[2]:.4f}), index-wise {v[1]:.4f} (tol {v[3]:.4f})" for g, v in _res_tie.items()))

# (iv) SDR e-value vs the literal definition (small batch, real tied scores + a rounded score)
_ok, _gap = True, 0.0
for (_n, _m) in ((100, 30), (250, 50)):
    for _rep in range(3):
        _ix = _rng.choice(B_POOL, _n + _m, replace=False); _c, _t = _ix[:_n], _ix[_n:]
        for _s in (DA_risk, np.round(DA_risk_weak, 1)):
            for _g in (0.1, 0.2):
                _f = b_sdr_e_values(_s[_c], DA_L[_c], _s[_t], _g)
                _r = np.array([b_sdr_grid_ref(_s[_c], DA_L[_c], _s[_t], j, _g, np.linspace(0, 1, 201)) for j in range(0, _m, 7)])
                _ok &= bool(np.all(_f[::7] <= _r + 1e-9)); _gap = max(_gap, float(np.max(np.where(np.isinf(_r), 0, _r - _f[::7]))))
check("S5.4 exact-inf SDR e-value <= dense-grid evaluation of definition (5.1) (tied scores, two batch sizes)", _ok, f"max grid-exact gap {_gap:.4f}")

# (v) the SDR feasibility question: time the exact e-value at production scale
_ix = _rng.permutation(B_POOL); _n_full = len(_ix) // 2; _c, _t = _ix[:_n_full], _ix[_n_full:]
_t0_ = b_time.time(); _pr = b_sdr_prep(DA_risk[_c], DA_L[_c], DA_risk[_t]); _E = [b_sdr_e_value(_pr, DA_risk[_t][j], 0.1, len(_c), len(_t)) for j in range(60)]
B_SDR_MS_PER_EVALUE = (b_time.time() - _t0_) / 60 * 1e3
print(f"exact SDR e-value at production scale (n={len(_c)} calibration, m={len(_t)} test documents): {B_SDR_MS_PER_EVALUE:.2f} ms per e-value "
      f"-> {B_SDR_MS_PER_EVALUE * len(_t) / 1e3:.2f} s per (batch, alpha, score)")
check("S5.5 exact SDR is feasible at production scale here (< 1 s per batch and alpha)", B_SDR_MS_PER_EVALUE * len(_t) / 1e3 < 1.0,
      f"{B_SDR_MS_PER_EVALUE * len(_t) / 1e3:.2f} s (Notebook 1 only ran n=80, m=30 because its loop was pure Python over every candidate)")
'''))

    # ------------------------------------------------------------------------------------------------ 5(a): MDR
    out.append(M(r'''
### 5(a). SCoRE-MDR at full scale: an expectation over the money-document pool

For each of 400 resplits the calibration half calibrates, every test-half money document gets an e-value, and
`psi = 1{E >= 1/alpha}` (Algorithm 1 with `gamma = alpha`). We check `E[L * psi] <= alpha` and then, **as a separate quantity**, report
the average loss per *accepted* document. Theorem 3.2 controls the first; the second is what a reader who misreads MDR as a
selective error rate would assume is `<= alpha` (idpfin-q4, q11: MDR "does NOT guarantee that at most alpha of accepted ... are wrong").
'''))

    out.append(C(r'''
B_ALPHAS = (0.05, 0.10, 0.20)
B_N_RESPLITS_MDR = 400


def b_mdr_mc(score, alpha, n_resplits=B_N_RESPLITS_MDR, seed0=1000):
    rec = []
    for sd in range(n_resplits):
        cal, tst = resplit(seed0 + sd); cal, tst = cal[DA_has_money[cal]], tst[DA_has_money[tst]]
        psi = b_mdr_e_values(score[cal], DA_L[cal], score[tst], alpha) >= 1 / alpha
        L = DA_L[tst]
        rec.append((np.mean(L * psi), psi.sum(), len(tst), (L * psi).sum(), (L[psi] > alpha).sum(), np.mean(L)))
    return np.array(rec, float)


B_MDR_RES = {(k, a): b_mdr_mc(s, a) for k, s in B_SCORES.items() for a in B_ALPHAS}
_rows = []
for (_k, _a), _r in B_MDR_RES.items():
    _nz = _r[:, 1] > 0
    _rows.append((B_SC_LABEL[_k].split(" ")[0], f"{_a:.2f}", f"{_r[:, 0].mean():.4f} +/- {b_mc_tol_mean(_r[:, 0]):.4f}", f"{_r[:, 0].mean() / _a:.3f}",
                  f"{_r[:, 1].mean():.0f} / {_r[:, 2].mean():.0f}", f"{(_r[:, 1] / _r[:, 2]).mean():.3f}",
                  f"{_r[:, 3].sum() / _r[:, 1].sum():.4f}", f"{_r[:, 3].sum() / _r[:, 1].sum() / _a:.2f}x",
                  f"{_r[_nz, 4].sum() / _r[_nz, 1].sum():.3f}", f"{np.mean(_r[:, 0] > _a):.2f}", f"{int((~_nz).sum())}"))
b_display(b_md(b_tbl(["score", "alpha", "MDR = E[L psi] (mean +/- 3SE)", "MDR/alpha", "accepted / test money docs", "coverage",
                      "avg loss per ACCEPTED doc (different quantity)", "... as multiple of alpha", "share of accepted docs with L_doc > alpha",
                      "share of splits with MDR > alpha", "splits with 0 accepted"], _rows)))
for _i, ((_k, _a), _r) in enumerate(B_MDR_RES.items()):
    check(f"S5.{6 + _i} MDR <= alpha ({B_SC_LABEL[_k].split(' ')[0]} score, alpha={_a:.2f}, {len(_r)} resplits)", _r[:, 0].mean() <= _a + b_mc_tol_mean(_r[:, 0]),
          f"MDR {_r[:, 0].mean():.4f} <= {_a} + 3SE {b_mc_tol_mean(_r[:, 0]):.4f}")
_sel_gt = {(k, a): B_MDR_RES[(k, a)][:, 3].sum() / B_MDR_RES[(k, a)][:, 1].sum() for (k, a) in B_MDR_RES}
check("S5.12 MDR is NOT a selective rate: the average loss per accepted document exceeds alpha at alpha in {0.05, 0.10} for both scores",
      all(_sel_gt[(k, a)] > a for k in B_SCORES for a in (0.05, 0.10)),
      "; ".join(f"{k} a={a}: {_sel_gt[(k, a)]:.3f}" for k in B_SCORES for a in (0.05, 0.10)))
'''))

    out.append(M(r'''
**How to read this chart.** Left: realized MDR (filled dots, mean over 400 resplits, bars = 3 SE) against the nominal `alpha`; the
dashed line is `y = x`. Dots on or below the line are what Theorem 3.2 promises, and they sit *on* it: MDR is tight, not conservative.
The hollow squares are the **average loss per accepted document**, a different quantity: at small `alpha` they lie well **above** the
line, because MDR spends its whole budget `alpha` on a *large* accepted set of documents that are individually still risky (the
"deploy many, each somewhat risky" reading of "a procedure may deploy few but comparatively risky cases yet still controlling the MDR"). Right: the coverage that buys.
The pessimistic score accepts fewer documents at every `alpha`. Both scores land near coverage 1 at `alpha = 0.20` because the pool's
mean loss is about 0.22, i.e. barely above that budget.
'''))

    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
_ax = _axes[0]
_ax.plot([0, 0.24], [0, 0.24], "--", color=PAL3["diag"], lw=1, label="y = x (Theorem 3.2 target)")
for _k in B_SCORES:
    _mdr = [B_MDR_RES[(_k, a)][:, 0].mean() for a in B_ALPHAS]; _err = [b_mc_tol_mean(B_MDR_RES[(_k, a)][:, 0]) for a in B_ALPHAS]
    _sel = [B_MDR_RES[(_k, a)][:, 3].sum() / B_MDR_RES[(_k, a)][:, 1].sum() for a in B_ALPHAS]
    _dx = -0.002 if _k == "opt" else 0.002
    _ax.errorbar(np.array(B_ALPHAS) + _dx, _mdr, yerr=_err, fmt="o", color=B_SC_COLOR[_k], capsize=3, label=f"MDR, {_k}")
    _ax.plot(np.array(B_ALPHAS) + _dx, _sel, "s", mfc="none", color=B_SC_COLOR[_k], ms=8, label=f"avg loss per accepted doc, {_k}")
_ax.set_xlabel("nominal alpha"); _ax.set_ylabel("realized quantity"); _ax.set_title("MDR is on target; per-accepted loss is not"); _ax.legend(fontsize=7.5)
_ax = _axes[1]
_w = 0.35
for _i, _k in enumerate(B_SCORES):
    _cv = [(B_MDR_RES[(_k, a)][:, 1] / B_MDR_RES[(_k, a)][:, 2]).mean() for a in B_ALPHAS]
    _ax.bar(np.arange(3) + (_i - 0.5) * _w, _cv, _w, color=B_SC_COLOR[_k], label=_k)
    for _x, _v in enumerate(_cv):
        _ax.text(_x + (_i - 0.5) * _w, _v + 0.01, f"{_v:.2f}", ha="center", fontsize=8)
_ax.set_xticks(range(3)); _ax.set_xticklabels([f"alpha={a:.2f}" for a in B_ALPHAS]); _ax.set_ylim(0, 1.1); _ax.set_ylabel("share of test money documents auto-accepted")
_ax.set_title("Coverage of SCoRE-MDR"); _ax.legend()
plt.tight_layout(); plt.show()
'''))

    # ------------------------------------------------------------------------------------------------ 5(b): SDR
    out.append(M(r'''
### 5(b). SCoRE-SDR with e-BH: is it feasible at our batch sizes, and how conservative is it?

Notebook 1 only ran the exact SDR e-value at n=80, m=30 because it looped in pure Python over every candidate. The check S5.5 above
timed the exact e-value (own implementation, **not** the paper's Algorithm 3, whose body is not quoted in our traces) at
production scale, and it is fast enough here, so **SDR is run at full scale** (n=796 calibration, m=797 test money documents per
batch, 100 batches) *and* at a small-batch scale that is closer to a single day's queue (n=250 calibration, m=50 test, 200 batches).
Selection = e-BH at level `alpha` on the e-values (Theorem 3.3, idpfin-q4). The SDR of a batch with no selection is 0 by the
`1 v |R|` convention; we report how many batches select nothing and the SDR conditional on selecting.

Comparators (both **without** any guarantee): the **naive rule** "accept a document if its predicted risk `s(x) <= alpha`" and the
**folklore rule** (Notebook 2's baseline lifted to documents: accept iff every critical field has verbalized confidence >= 0.9 and
passes the business rules). The strict folklore rule accepts almost nothing at document level, so a **coverage-matched folklore** is also reported: rank documents by their
*lowest* raw verbalized confidence over critical fields and accept the same number of documents SCoRE accepted in that batch.
'''))

    out.append(C(r'''
_vm = np.full(N_DOC, 1.0)
for _k in np.where(FA_crit)[0]:
    _vm[FA_doc[_k]] = min(_vm[FA_doc[_k]], FA_verb[_k])
B_DOC_VERBMIN = _vm                                                       # lowest raw verbalized confidence over the doc's critical fields
_fk = np.array([f["accepted"] for f in FIELDS], int)                      # Notebook 2 baseline: verbalized >= 0.9 and rule_failed == 0
B_DOC_FOLK = np.ones(N_DOC, bool)
for _k in np.where(FA_crit & (_fk == 0))[0]:
    B_DOC_FOLK[FA_doc[_k]] = False


def b_sdr_mc(n, m, reps, seed0, alphas=B_ALPHAS):
    res = {(k, a): {q: [] for q in ("sdr", "cov", "nsel", "naive_sdr", "naive_cov", "folk_sdr", "folk_cov", "match_sdr")} for k in B_SCORES for a in alphas}
    for r in range(reps):
        rng = np.random.default_rng([SEED3, seed0, r]); ix = rng.permutation(B_POOL)[:n + m]; c, t = ix[:n], ix[n:]
        Lt = DA_L[t]; order_v = np.argsort(-B_DOC_VERBMIN[t], kind="stable")
        for k, s in B_SCORES.items():
            prep = b_sdr_prep(s[c], DA_L[c], s[t])
            for a in alphas:
                o = res[(k, a)]
                sel = b_ebh_select(np.array([b_sdr_e_value(prep, s[t][j], a, n, m) for j in range(m)]), a)
                o["sdr"].append((Lt * sel).sum() / max(1, sel.sum())); o["cov"].append(sel.mean()); o["nsel"].append(sel.sum())
                nv = s[t] <= a; o["naive_sdr"].append((Lt * nv).sum() / max(1, nv.sum())); o["naive_cov"].append(nv.mean())
                fk = B_DOC_FOLK[t]; o["folk_sdr"].append((Lt * fk).sum() / max(1, fk.sum())); o["folk_cov"].append(fk.mean())
                kk = int(sel.sum()); o["match_sdr"].append(Lt[order_v[:kk]].mean() if kk > 0 else 0.0)
    return {key: {q: np.array(v) for q, v in o.items()} for key, o in res.items()}


_t0_ = b_time.time()
B_SDR_SCALES = {"full (n=796, m=797)": (796, 797, 100, 11), "small batch (n=250, m=50)": (250, 50, 200, 12)}
B_SDR_RES = {sc: b_sdr_mc(*cfg) for sc, cfg in B_SDR_SCALES.items()}
B_SDR_SECONDS = b_time.time() - _t0_
print(f"SDR Monte Carlo runtime: {B_SDR_SECONDS:.0f} s")

_rows, _rows2 = [], []
for _sc, _res in B_SDR_RES.items():
    for (_k, _a), _o in _res.items():
        _nz = _o["nsel"] > 0
        _rows.append((_sc, _k, f"{_a:.2f}", f"{_o['sdr'].mean():.4f} +/- {b_mc_tol_mean(_o['sdr']):.4f}", f"{_o['sdr'].mean() / _a:.2f}", f"{_o['cov'].mean():.3f}",
                      f"{int((~_nz).sum())} / {len(_nz)}", f"{_o['sdr'][_nz].mean() if _nz.any() else float('nan'):.4f}", f"{np.mean(_o['sdr'] > _a):.2f}"))
        _rows2.append((_sc, _k, f"{_a:.2f}", f"{_o['naive_sdr'].mean():.4f} / {_o['naive_cov'].mean():.3f} / {np.mean(_o['naive_sdr'] > _a):.2f}",
                       f"{_o['folk_sdr'].mean():.4f} / {_o['folk_cov'].mean():.3f}", f"{_o['match_sdr'].mean():.4f} / {_o['cov'].mean():.3f}"))
b_display(b_md("**SCoRE-SDR (exact e-values + e-BH, gamma = alpha)**\n\n" + b_tbl(
    ["scale", "score", "alpha", "SDR (mean +/- 3SE)", "SDR/alpha", "coverage", "batches selecting nothing", "SDR given a selection", "batches with SDR > alpha"], _rows)))
b_display(b_md("**Comparators without a guarantee** (each cell: realized avg loss among accepted / coverage / share of batches above alpha)\n\n" + b_tbl(
    ["scale", "score", "alpha", "naive: predicted risk <= alpha", "folklore, strict (avg loss / coverage)", "folklore, coverage-matched to SCoRE (avg loss / coverage)"], _rows2)))
_n = 0
for _sc, _res in B_SDR_RES.items():
    for _k in B_SCORES:
        _ok = all(_res[(_k, a)]["sdr"].mean() <= a + b_mc_tol_mean(_res[(_k, a)]["sdr"]) for a in B_ALPHAS)
        check(f"S5.{13 + _n} SDR <= alpha at every alpha ({_k} score, {_sc}, {len(_res[(_k, 0.1)]['sdr'])} batches)", _ok,
              "; ".join(f"a={a}: {_res[(_k, a)]['sdr'].mean():.4f} (tol {b_mc_tol_mean(_res[(_k, a)]['sdr']):.4f})" for a in B_ALPHAS))
        _n += 1
_full = B_SDR_RES["full (n=796, m=797)"]; _small = B_SDR_RES["small batch (n=250, m=50)"]
check("S5.17 SDR is conservative (well below alpha) for this continuous, non-binary loss, and more so at the full batch size",
      all(_full[(k, 0.10)]["sdr"].mean() < 0.5 * 0.10 for k in B_SCORES) and all(_small[(k, 0.10)]["sdr"].mean() >= _full[(k, 0.10)]["sdr"].mean() for k in B_SCORES),
      "alpha=0.10: " + "; ".join(f"{k}: full {_full[(k, 0.10)]['sdr'].mean():.4f} vs small {_small[(k, 0.10)]['sdr'].mean():.4f}" for k in B_SCORES))
print(f"strict folklore accepts {B_DOC_FOLK[B_POOL].mean():.4f} of money documents (avg loss among them {DA_L[B_POOL][B_DOC_FOLK[B_POOL]].mean():.4f}): "
      "a per-field 0.9 cutoff, demanded of ALL critical fields at once, leaves nothing to auto-accept at document level.")
'''))

    out.append(M(r'''
**How to read this chart.** Left: realized SDR (average loss among accepted documents, mean over batches) against the nominal
`alpha`, dashed `y = x`. SCoRE-SDR (filled markers, two batch scales) stays below the line, often far below: with a continuous loss the
infimum over the unknown test loss "may be slightly conservative" (quoted in idpfin-q6, section 8.3) and at m=797 the e-BH cut-off
`m/(alpha k)` is demanding, so at `alpha = 0.05` most full-size batches select **nothing**. The naive rule (hollow purple) is also
below the line here, **but only because the predicted-risk model happens to be calibrated on data that look like the calibration data**; it
carries no guarantee and Section 7 shows what it does after the month-7 shift. Right: what each rule accepts. The naive rule
accepts more than SCoRE-SDR at every `alpha`; SCoRE's price for a guarantee that does not depend on the risk model being calibrated is
this coverage gap, and it is smaller for small batches.
'''))

    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
_ax = _axes[0]
_ax.plot([0, 0.24], [0, 0.24], "--", color=PAL3["diag"], lw=1, label="y = x")
_mk = {"full (n=796, m=797)": "o", "small batch (n=250, m=50)": "^"}
for _sc, _res in B_SDR_RES.items():
    for _k in B_SCORES:
        _ax.plot(B_ALPHAS, [_res[(_k, a)]["sdr"].mean() for a in B_ALPHAS], _mk[_sc], color=B_SC_COLOR[_k], ms=7, label=f"SCoRE-SDR {_k}, {_sc.split(' (')[0]}")
for _k in B_SCORES:
    _ax.plot(B_ALPHAS, [B_SDR_RES["small batch (n=250, m=50)"][(_k, a)]["naive_sdr"].mean() for a in B_ALPHAS], "s", mfc="none", color=PAL3["b_naive"], ms=8,
             label="naive rule (pred. risk <= alpha)" if _k == "opt" else None)
_ax.set_xlabel("nominal alpha"); _ax.set_ylabel("average loss among accepted documents"); _ax.set_title("SDR: guaranteed rule vs naive rule"); _ax.legend(fontsize=7)
_ax = _axes[1]
_labels, _vals, _cols = [], [], []
for _a in B_ALPHAS:
    for _nm, _q, _res in (("SDR full", "cov", B_SDR_RES["full (n=796, m=797)"]), ("SDR small", "cov", B_SDR_RES["small batch (n=250, m=50)"]), ("naive", "naive_cov", B_SDR_RES["small batch (n=250, m=50)"])):
        _labels.append(f"{_a:.2f}\n{_nm}"); _vals.append(_res[("opt", _a)][_q].mean())
        _cols.append(PAL3["b_naive"] if _nm == "naive" else (PAL3["hgb"] if _nm == "SDR full" else PAL3["lr"]))
_ax.bar(range(len(_vals)), _vals, color=_cols)
_ax.set_xticks(range(len(_vals))); _ax.set_xticklabels(_labels, fontsize=7); _ax.set_ylabel("share of documents accepted (optimistic score)"); _ax.set_title("Coverage: SDR (two batch sizes) vs naive")
plt.tight_layout(); plt.show()
'''))

    # ------------------------------------------------------------------------------------------------ 5(c)
    out.append(M(r'''
### 5(c). Amount-weighted versus count-weighted loss: same documents, different accept sets

A count-weighted document loss treats a wrong cent and a wrong million as one wrong field: `L_cnt` = share of the document's money
fields that are wrong. The amount-weighted `L_doc` counts how much of the document's monetary *value* is wrong (Section 2). To isolate the **loss**, each rule uses its own
train-fitted risk model (same features, same learner) and its own SCoRE-MDR calibration at `alpha = 0.10`, averaged over 40 fixed resplits.
"Value at risk" below is `sum(L_doc * V) / sum(V)` over the accepted documents, with `V = max(sum |true money amounts|, 1)`: the
share of the accepted documents' monetary value that is wrong, pooled across documents (derived here; bounded in [0,1] because `L_doc` already clips at 1).
'''))

    out.append(C(r'''
B_ERR_AMT, B_VAL_AMT, B_NMONEY, B_NWRONG_MONEY = np.zeros(N_DOC), np.zeros(N_DOC), np.zeros(N_DOC), np.zeros(N_DOC)


def b_num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


for _k, _f in enumerate(FIELDS):
    if _f["field"] in MONEY_FIELDS[_f["doc_type"]]:
        _d = FA_doc[_k]; B_NMONEY[_d] += 1; B_NWRONG_MONEY[_d] += 1 - FA_correct[_k]
        _tr, _pr = b_num(_f["truth"]), b_num(_f["predicted"])
        if _tr is None:
            B_ERR_AMT[_d] += abs(_pr) if _pr is not None else 0.0
        else:
            B_VAL_AMT[_d] += abs(_tr); B_ERR_AMT[_d] += abs(_pr - _tr) if _pr is not None else abs(_tr)
B_V = np.maximum(B_VAL_AMT, 1.0)
B_L_CNT = np.where(DA_has_money, B_NWRONG_MONEY / np.maximum(B_NMONEY, 1), np.nan)
check("S5.18 recomputed amount loss equals the prelude's DA_L on every money document",
      np.allclose(np.minimum(1.0, B_ERR_AMT / B_V)[DA_has_money], DA_L[DA_has_money]), f"{int(DA_has_money.sum())} documents")

_trm = np.where((DOC_SPLIT == "train") & DA_has_money)[0]
_mk = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=25, l2_regularization=1.0, random_state=SEED3)
B_RISK_CNT = {"opt": np.clip(_mk().fit(X_doc_oof[_trm], B_L_CNT[_trm]).predict(X_doc), 0, 1),         # same features/learner as DOC_MODEL_L, fit on train only
              "weak": np.clip(_mk().fit(X_doc_weak[_trm], B_L_CNT[_trm]).predict(X_doc_weak), 0, 1)}
B_CMP_ALPHA = 0.10


def b_cmp_split(sd, k):
    cal, tst = resplit(sd); cal, tst = cal[DA_has_money[cal]], tst[DA_has_money[tst]]
    acc_a = b_mdr_e_values(B_SCORES[k][cal], DA_L[cal], B_SCORES[k][tst], B_CMP_ALPHA) >= 1 / B_CMP_ALPHA
    acc_c = b_mdr_e_values(B_RISK_CNT[k][cal], B_L_CNT[cal], B_RISK_CNT[k][tst], B_CMP_ALPHA) >= 1 / B_CMP_ALPHA
    wrong_val = DA_L[tst] * B_V[tst]

    def st(acc):
        n = max(1, acc.sum())
        return [acc.mean(), DA_L[tst][acc].sum() / n, B_L_CNT[tst][acc].sum() / n, (B_NWRONG_MONEY[tst][acc] > 0).sum() / n,
                wrong_val[acc].sum() / max(B_V[tst][acc].sum(), 1e-9), wrong_val[acc].sum() / wrong_val.sum(), np.mean(DA_L[tst] * acc), np.mean(B_L_CNT[tst] * acc)]
    return st(acc_a) + st(acc_c) + [(acc_a & acc_c).sum() / max(1, (acc_a | acc_c).sum()), (acc_a & ~acc_c).mean(), (~acc_a & acc_c).mean()]


B_CMP = {k: np.array([b_cmp_split(sd, k) for sd in range(40)]) for k in B_SCORES}
_rows = []
for _k in B_SCORES:
    _m_ = B_CMP[_k].mean(0)
    for _nm, _o in (("amount-weighted L_doc (SCoRE-MDR)", 0), ("count-weighted L_cnt (SCoRE-MDR)", 8)):
        _rows.append((_k, _nm, f"{_m_[_o]:.3f}", f"{_m_[_o + 1]:.3f}", f"{_m_[_o + 2]:.3f}", f"{_m_[_o + 3]:.3f}", f"{_m_[_o + 4]:.4f}", f"{_m_[_o + 5]:.3f}"))
b_display(b_md(b_tbl(["score", "rule", "coverage", "avg L_doc among accepted", "avg L_cnt among accepted", "accepted docs with >=1 wrong money field",
                      "value at risk in accepted docs", "share of all wrong value that sits in accepted docs"], _rows)))
_rows = [(_k, f"{B_CMP[_k].mean(0)[16]:.3f}", f"{B_CMP[_k].mean(0)[17]:.3f}", f"{B_CMP[_k].mean(0)[18]:.3f}") for _k in B_SCORES]
b_display(b_md(b_tbl(["score", "Jaccard overlap of the two accepted sets", "share of docs accepted ONLY by the amount rule", "share of docs accepted ONLY by the count rule"], _rows)))
for _i, _k in enumerate(B_SCORES):
    _x = B_CMP[_k]
    check(f"S5.{19 + 3 * _i} the two losses select different documents ({_k} score)", _x[:, 16].mean() < 1 and _x[:, 17].mean() > 0 and _x[:, 18].mean() > 0,
          f"Jaccard {_x[:, 16].mean():.3f}; only-amount {_x[:, 17].mean():.3f}, only-count {_x[:, 18].mean():.3f} of test money docs")
    check(f"S5.{20 + 3 * _i} each rule meets its own MDR budget alpha={B_CMP_ALPHA} over 40 resplits ({_k} score)",
          _x[:, 6].mean() <= B_CMP_ALPHA + b_mc_tol_mean(_x[:, 6]) and _x[:, 15].mean() <= B_CMP_ALPHA + b_mc_tol_mean(_x[:, 15]),
          f"amount MDR {_x[:, 6].mean():.4f} (tol {b_mc_tol_mean(_x[:, 6]):.4f}); count MDR {_x[:, 15].mean():.4f} (tol {b_mc_tol_mean(_x[:, 15]):.4f})")
    check(f"S5.{21 + 3 * _i} the amount rule tolerates documents that contain a wrong money field ({_k} score)", _x[:, 3].mean() > 0,
          f"{_x[:, 3].mean():.3f} of amount-accepted docs have >=1 wrong money field vs {_x[:, 11].mean():.3f} for the count rule")
'''))

    out.append(M(r'''
**How to read this chart.** Left: for each score, the two accepted sets as shares of all test money documents: accepted by both
(grey), only under the amount-weighted loss (red) and only under the count-weighted loss (blue). Non-zero red and blue slices are the point: the two
losses do not merely rescale the same ranking. Right: among the documents each rule accepts, the share of monetary value that is wrong (value at risk). The
accepted sets differ by roughly a tenth of the documents and the effect on value at risk is modest for the optimistic score;
do not read more into the gap than the table supports, since each bar is a mean of 40 resplits that reuse the same documents.
'''))

    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
_ax = _axes[0]
for _i, _k in enumerate(B_SCORES):
    _x = B_CMP[_k].mean(0); _both = _x[0] - _x[17]; _oa, _oc = _x[17], _x[18]
    _ax.bar(_i, _both, color=PAL3["raw"], label="both rules" if _i == 0 else None)
    _ax.bar(_i, _oa, bottom=_both, color=PAL3["hgb"], label="only amount-weighted rule" if _i == 0 else None)
    _ax.bar(_i, _oc, bottom=_both + _oa, color=PAL3["lr"], label="only count-weighted rule" if _i == 0 else None)
_ax.set_xticks([0, 1]); _ax.set_xticklabels(["optimistic", "pessimistic"]); _ax.set_ylabel("share of test money documents"); _ax.set_title("Which documents each loss accepts (alpha = 0.10)"); _ax.legend(fontsize=8)
_ax = _axes[1]
_w = 0.35
for _i, (_nm, _o, _c) in enumerate((("amount-weighted rule", 4, PAL3["hgb"]), ("count-weighted rule", 12, PAL3["lr"]))):
    _v = [B_CMP[k].mean(0)[_o] for k in B_SCORES]
    _ax.bar(np.arange(2) + (_i - 0.5) * _w, _v, _w, color=_c, label=_nm)
    for _x_, _vv in enumerate(_v):
        _ax.text(_x_ + (_i - 0.5) * _w, _vv + 0.002, f"{_vv:.3f}", ha="center", fontsize=8)
_ax.set_xticks([0, 1]); _ax.set_xticklabels(["optimistic", "pessimistic"]); _ax.set_ylabel("share of accepted monetary value that is wrong"); _ax.set_title("Value at risk in the accepted set"); _ax.set_ylim(0, _ax.get_ylim()[1] * 1.25); _ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()
'''))

    # ------------------------------------------------------------------------------------------------ 5(d)
    out.append(M(r'''
### 5(d). Covariate shift: what SCoRE offers, and why it is **not** implemented here

SCoRE's shift extension is stated as (idpfin-q5): "**Assumption 6.1.** The labeled data follow `(X_i, Y_i) ~ P` while the test data follow
`(X_{n+j}, Y_{n+j}) ~ Q`, and the two distributions obey `dQ/dP(x, y) = w(x)` for a known or estimable weight function". With **known**
`w` the weighted e-values (Equations 6.1 and 6.2) give "exact finite-sample MDR and SDR control" under `Q`; with **estimated** `w` the
guarantee is only asymptotic and needs `L_2(P_X)`-consistency of the estimated weights (Theorems 6.4 and 6.5, idpfin-q5, q12).

Here the density ratio `w(x)` between the calibration batch and the month-7 batch is **unknown**, and the month-7 shift is not obviously
covariate-only: new vendor templates and heavier OCR noise change how *errors* arise given the same document, i.e. `P(Y | X)` moves too. None
of the three sources says how to certify under such concept drift without new labels ("None of the three sources state how to
perform real-time, online re-calibration or re-certification under unobserved concept drift", idpfin-q12). So **weighted e-values are
not built**; Section 7 instead checks empirically what a frozen configuration does on the shift split and how many freshly labelled
documents restore validity. Among the engineering choices left open: estimating `w(x)` (for example by a domain classifier on the score and template features) and running the weighted e-values.
'''))
    return out



def _s6():
    out = []

    out.append(M(r"""
## 6. The validity ladder on our data, per document type (P3)

**Claim.** Not every "held at alpha" is the same promise. Gurram's ladder separates *expectation* control (tiers 1-2, which sit at the
boundary and exceed `alpha` in a large share of splits) from *probabilistic* certificates (tiers 3-4). We run all rungs on our
fields with the train-fitted scores, `alpha = 0.10`, `delta = 0.10` (headline) and `alpha = 0.20`, over 40 fixed document-level 50/50
resplits (`resplit(seed)`, seeds 0..39; fields of a document never straddle calibration and test).

**Quoted (idpfin-q8).** Table 4 caption: "Vocabulary: tiers 1-2 control expected selective risk; only tiers 3-4 certify." The tier-3 mechanism: "For each
taxonomy group g and candidate threshold t, we test H0: selective risk of t in g > alpha with an exact binomial tail p-value ... half the budget
delta to Holm step-down ..., half to a fixed-sequence pass from the most conservative candidate (the 'mix' rule ...). Candidates are 15 geometric
acceptance-fraction quantiles (1%-100%) of the group's calibration scores, snapped to the nearest distinct-value boundary". Guarantee: "with
probability >= 1 - delta over the calibration draw, the true selective risk among accepted fields in that group is <= alpha, if within-group
accepted-field errors are iid." Tier 4: "the mean per-document error rate among accepting documents is <= alpha (finite-sample bound, documents iid) ...
it bounds a macro per-document functional, not field-level (micro) selective risk." Cluster-corrected `ltt.neff` "replaces n_t by n_t / d_eff ... approximate
(the deff is estimated), and uniformly more conservative". Measured design effects in the paper: "d_eff ranges from 1.84 to 2.45" (idpfin-q11).

**Taxonomy discipline (idpfin-q8, section 5.3).** "Choosing the conditioning taxonomy among K candidates to maximize coverage is a selection
problem." Ours is **document type**, pre-specified before any result was looked at, and it is never changed to raise coverage. Budgets are per group
(`delta` for each of the four groups: four separate statements).

| tier | procedure (ours) | guarantee form | unit | certificate? |
| --- | --- | --- | --- | --- |
| 1 | split add-one, pooled | `E[selective risk] <= alpha` | field | no (expectation) |
| 2 | add-one x document type | `E[selective risk] <= alpha` (per group) | field | no (expectation) |
| 3 | Mondrian LTT, exact binomial tails, 15 candidates, delta/2 Holm + delta/2 fixed-sequence, per document type (and a pooled single-group variant) | `P(true group risk > alpha) <= delta` if within-group accepted-field errors are iid | field | **yes (PAC)** |
| 3b | as 3 with `n_t / d_eff` (`d_eff` estimated on the calibration fields of each group) | same, approximate | field | yes (PAC, approximate, more conservative) |
| 4 | doc-Hoeffding over accepting documents, per document type | `P(macro per-document risk > alpha) <= delta`, documents iid | document | **yes (PAC)**, macro functional |

**Conventions.** Every field is a unit (all 28,032 fields, error = 1 - correct); scores are "higher = more trusted": optimistic
`FA_p_hgb`, pessimistic `FA_p_weak`, and the raw verbalized confidence `FA_verb` (the frozen, uninformed black-box used for the two-regime
demonstration). "Zero-coverage splits contribute risk 0" (Gurram section 3.5, quoted in the cited source text of idpfin-q10); we follow it, **count
those splits, and also report risk conditional on non-zero coverage**, the "report both" of Table 4's footnote (idpfin-q8: "0.020 is the
zero-filled mean ... Report both."). Violation = fraction of splits whose realized test risk exceeds `alpha`. The 40 resplits share documents, so every
spread below is a split-stability statement, not population inference (Gurram's own caveat, idpfin-q9: "Inference is in-corpus").
"""))

    out.append(C(r'''
# ---- Mondrian-LTT machinery: COPIED from build_nb1.py (geometric_candidates, holm_reject, fixed_sequence_reject_conservative_first,
#      mondrian_ltt_certify, estimate_design_effect, mondrian_ltt_doc_hoeffding), renamed with b_ and vectorised where it is pure bookkeeping ----
def b_geometric_candidates(scores, n_candidates=15):
    n = len(scores); fracs = np.geomspace(0.01, 1.0, n_candidates)
    ks = np.clip(np.round(fracs * n).astype(int), 1, n)
    return np.unique(np.sort(scores)[::-1][ks - 1])[::-1]              # distinct values, most conservative (highest) first


def b_holm_reject(pvals, budget):
    m = len(pvals); order = np.argsort(pvals); sp = pvals[order]; th = budget / (m - np.arange(m)); rs = np.zeros(m, bool)
    for i in range(m):
        if sp[i] <= th[i]:
            rs[i] = True
        else:
            break
    r = np.zeros(m, bool); r[order] = rs; return r


def b_fixed_seq(pvals, budget):
    r = np.zeros(len(pvals), bool)
    for i, p in enumerate(pvals):
        if p <= budget:
            r[i] = True
        else:
            break
    return r


def b_deff(doc_ids, err):
    """ANOVA plug-in design effect d_eff = 1 + (k_bar - 1) * rho_hat (derived here in Notebook 1; the paper does not give the formula)."""
    docs, inv, n_d = np.unique(doc_ids, return_inverse=True, return_counts=True); D, N = len(docs), len(err)
    if D < 2 or N <= D:
        return 1.0
    p_d = np.bincount(inv, weights=err, minlength=D) / n_d; p_bar = err.mean()
    MSB = np.sum(n_d * (p_d - p_bar) ** 2) / (D - 1); MSW = np.sum((err - p_d[inv]) ** 2) / max(1, N - D)
    k0 = (N - np.sum(n_d ** 2) / N) / (D - 1)
    rho = (1.0 if MSB > 1e-9 else 0.0) if MSW <= 1e-9 else (MSB - MSW) / (MSB + (k0 - 1) * MSW)
    return max(1.0, 1 + (N / D - 1) * float(np.clip(rho, 0, 1)))


def b_ltt_certify(scores, err, alpha, delta, deff=None):
    """Tier 3 (and 3b when `deff` > 1): returns the certified threshold (accept score >= tau), inf = review everything."""
    th = b_geometric_candidates(scores)
    n_ts = np.array([(scores >= t).sum() for t in th], float); k_ts = np.array([err[scores >= t].sum() for t in th], float)
    d = deff if (deff is not None and deff > 1.0) else 1.0
    n_int = np.maximum(1, np.floor(n_ts / d).astype(int)); k_int = np.minimum(n_int, np.round(k_ts / d).astype(int))
    p = b_binom.cdf(k_int, n_int, alpha)                                   # exact binomial tail Pr[Bin(n_t, alpha) <= k_t]
    cert = b_holm_reject(p, delta / 2) | b_fixed_seq(p, delta / 2)
    if not cert.any():
        return np.inf
    ci = np.nonzero(cert)[0]
    return th[ci[np.argmax(n_ts[ci])]]                                      # certified candidate with the largest calibration acceptance


def b_doc_hoeffding(doc_ids, scores, err, alpha, delta):
    """Tier 4: per-document loss = within-document error rate among accepted fields; Hoeffding p-value over accepting documents."""
    docs, inv = np.unique(doc_ids, return_inverse=True); D = len(docs)
    th = b_geometric_candidates(scores); p = np.ones(len(th)); n_ts = np.zeros(len(th))
    for i, t in enumerate(th):
        m = scores >= t; n_ts[i] = m.sum()
        if not m.any():
            continue
        cnt = np.bincount(inv[m], minlength=D); s = np.bincount(inv[m], weights=err[m], minlength=D); has = cnt > 0
        gap = max(0.0, alpha - (s[has] / cnt[has]).mean()); p[i] = np.exp(-2 * has.sum() * gap ** 2)
    cert = b_holm_reject(p, delta / 2) | b_fixed_seq(p, delta / 2)
    if not cert.any():
        return np.inf
    ci = np.nonzero(cert)[0]
    return th[ci[np.argmax(n_ts[ci])]]


B_ERR = (1 - FA_correct).astype(float)                    # field error indicator
B_GRP = DOC_TYPE_IDX[FA_doc]                              # pre-specified taxonomy: document type
B_FSPLIT = DOC_SPLIT[FA_doc]
B_TIERS = ["t1", "t2", "t3p", "t3", "t3b", "t4"]
B_TIER_LABEL = {"t1": "1 add-one, pooled", "t2": "2 add-one x type", "t3p": "3 LTT, pooled (1 group)", "t3": "3 LTT x type", "t3b": "3b LTT neff x type", "t4": "4 doc-Hoeffding x type"}
B_TIER_SHORT = {"t1": "T1\npooled", "t2": "T2\nx type", "t3p": "T3\npooled", "t3": "T3\nx type", "t3b": "T3b\nneff", "t4": "T4\ndoc"}
B_TIER_FORM = {"t1": "E[risk] <= alpha (expectation)", "t2": "E[risk] <= alpha (expectation)", "t3p": "PAC, P(risk > alpha) <= delta",
               "t3": "PAC per group, P(group risk > alpha) <= delta", "t3b": "PAC per group, approximate (n/d_eff)", "t4": "PAC on macro per-doc risk, docs iid"}
B_SCORE_FIELD = {"hgb": FA_p_hgb, "weak": FA_p_weak, "verb": FA_verb}
B_SCORE_NAME = {"hgb": "optimistic (learned fusion, FA_p_hgb)", "weak": "pessimistic (weak signals, FA_p_weak)", "verb": "raw verbalized confidence (FA_verb)"}


def b_fit_thresholds(score, mask, alpha, delta):
    """All rungs fit on the fields selected by `mask` (calibration fields). Returns {tier: tau (scalar or array(4))}, and d_eff per group."""
    sc, er, gr, dc = score[mask], B_ERR[mask], B_GRP[mask], FA_doc[mask]
    G = range(4); deffs = np.array([b_deff(dc[gr == g], er[gr == g]) for g in G])
    return {"t1": add_one_threshold(sc, er, alpha),
            "t2": np.array([add_one_threshold(sc[gr == g], er[gr == g], alpha) for g in G]),
            "t3p": b_ltt_certify(sc, er, alpha, delta),
            "t3": np.array([b_ltt_certify(sc[gr == g], er[gr == g], alpha, delta) for g in G]),
            "t3b": np.array([b_ltt_certify(sc[gr == g], er[gr == g], alpha, delta, deff=deffs[g]) for g in G]),
            "t4": np.array([b_doc_hoeffding(dc[gr == g], sc[gr == g], er[gr == g], alpha, delta) for g in G])}, deffs


def b_accept(score, mask, tau):
    t = np.asarray(tau, float)
    return score[mask] >= (t[B_GRP[mask]] if t.ndim else t)


B_MASKS = {}


def b_split_masks(seed):
    if seed not in B_MASKS:
        cal, tst = resplit(seed); dm = np.zeros(N_DOC, int); dm[cal] = 1; dm[tst] = 2
        B_MASKS[seed] = (dm[FA_doc] == 1, dm[FA_doc] == 2)
    return B_MASKS[seed]


def b_metrics(acc, mask, alpha):
    """[coverage, micro risk (0 if nothing accepted), n accepted, macro per-doc risk, micro violation, macro violation,
    #groups violating, #groups accepting, risk among accepted CRITICAL fields]"""
    te, tg, td, tc = B_ERR[mask], B_GRP[mask], FA_doc[mask], FA_crit[mask]
    n = int(acc.sum()); risk = te[acc].mean() if n else 0.0
    if n:
        _, inv = np.unique(td[acc], return_inverse=True); macro = float((np.bincount(inv, weights=te[acc]) / np.bincount(inv)).mean())
    else:
        macro = 0.0
    gv = gn = 0
    for g in range(4):
        a = acc & (tg == g)
        if a.any():
            gn += 1; gv += int(te[a].mean() > alpha)
    crit = acc & tc
    return [n / len(acc), risk, n, macro, float(risk > alpha), float(macro > alpha), gv, gn, te[crit].mean() if crit.any() else 0.0]


def b_ladder(score_key, alpha, delta=0.10, n_splits=40):
    score = B_SCORE_FIELD[score_key]; res = {t: [] for t in B_TIERS}; deffs = []
    for sd in range(n_splits):
        cm, tm = b_split_masks(sd)
        assert len(set(FA_doc[cm]) & set(FA_doc[tm])) == 0
        th, d = b_fit_thresholds(score, cm, alpha, delta); deffs.append(d)
        for t in B_TIERS:
            res[t].append(b_metrics(b_accept(score, tm, th[t]), tm, alpha))
    return {t: np.array(v, float) for t, v in res.items()}, np.array(deffs)


_t0_ = b_time.time()
B_LADDER, B_DEFF = {}, {}
for _k in ("hgb", "weak", "verb"):
    for _a in (0.10, 0.20):
        B_LADDER[(_k, _a)], B_DEFF[(_k, _a)] = b_ladder(_k, _a)
print(f"ladder runtime {b_time.time() - _t0_:.0f} s (3 scores x 2 alphas x 40 splits x 6 rungs)")
check("S6.1 in all 40 splits calibration and test fields come from disjoint documents (asserted inside b_ladder)", True, "document-level splits")
print("estimated design effect d_eff per document type (calibration half, mean over 40 splits; errors of ALL fields): " +
      ", ".join(f"{DOC_TYPES[g]} {B_DEFF[('hgb', 0.10)][:, g].mean():.2f}" for g in range(4)))
'''))

    out.append(C(r'''
def b_summ(res, tier):
    r = res[tier]; nz = r[:, 2] > 0
    return dict(cov=r[:, 0].mean(), cov_sd=r[:, 0].std(ddof=1), risk=r[:, 1].mean(), risk_nz=(r[nz, 1].mean() if nz.any() else float("nan")), viol=r[:, 4].mean(),
                zero=int((~nz).sum()), macro=r[:, 3].mean(), macro_viol=r[:, 5].mean(), gviol=(r[:, 6].sum() / r[:, 7].sum() if r[:, 7].sum() > 0 else float("nan")),
                gpairs=int(r[:, 7].sum()), crit_risk=(r[nz, 8].mean() if nz.any() else float("nan")))


for _a in (0.10, 0.20):
    for _k in ("hgb", "weak"):
        _rows = []
        for _t in B_TIERS:
            _s = b_summ(B_LADDER[(_k, _a)], _t)
            _rows.append((B_TIER_LABEL[_t], B_TIER_FORM[_t], f"{_s['cov']:.3f} +/- {_s['cov_sd']:.3f}", f"{_s['risk']:.3f}", "n/a" if np.isnan(_s['risk_nz']) else f"{_s['risk_nz']:.3f}",
                          f"{_s['viol']:.3f}", f"{_s['zero']}/40", "n/a" if np.isnan(_s['gviol']) or _t in ("t1", "t3p") else f"{_s['gviol']:.3f} ({_s['gpairs']})",
                          f"{_s['macro']:.3f}"))
        b_display(b_md(f"**alpha = {_a:.2f}, delta = 0.10, {B_SCORE_NAME[_k]}** (mean over 40 resplits; risk = realized micro selective risk on test fields, zero-filled)\n\n" + b_tbl(
            ["tier", "guarantee form", "coverage (mean +/- sd)", "achieved risk (zero-filled)", "risk given coverage > 0", "share of splits with risk > alpha",
             "zero-coverage splits", "share of (split, group) pairs with group risk > alpha (n pairs)", "macro per-doc risk"], _rows)))

# ---- self-checks ----
_n = 2
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        _r = B_LADDER[(_k, _a)]["t1"][:, 1]
        check(f"S6.{_n} tier 1 add-one meets its EXPECTATION target ({_k}, alpha={_a:.2f})", _r.mean() <= _a + b_mc_tol_mean(_r), f"mean risk {_r.mean():.4f} <= {_a} + 3SE {b_mc_tol_mean(_r):.4f}"); _n += 1
check(f"S6.{_n} ...but tier 1 is NOT a certificate: it exceeds alpha in more than delta of the splits (alpha=0.10, both scores)",
      all(B_LADDER[(k, 0.10)]["t1"][:, 4].mean() > 0.10 for k in ("hgb", "weak")), "; ".join(f"{k}: {B_LADDER[(k, 0.10)]['t1'][:, 4].mean():.3f} of splits" for k in ("hgb", "weak"))); _n += 1
for _t in ("t3p", "t3", "t3b"):
    _ok, _det = True, []
    for _k in ("hgb", "weak", "verb"):
        for _a in (0.10, 0.20):
            _v = B_LADDER[(_k, _a)][_t][:, 4].mean(); _ok &= _v <= 0.10 + b_cp_tol(_v, 40); _det.append(f"{_k}/{_a}: {_v:.3f}")
    check(f"S6.{_n} tier {_t[1:]} (PAC) violation fraction <= delta = 0.10 + Clopper-Pearson tolerance (3 scores x 2 alphas x 40 splits)", _ok, "; ".join(_det)); _n += 1
_ok, _det = True, []
for _t in ("t3", "t3b"):
    for _k in ("hgb", "weak", "verb"):
        for _a in (0.10, 0.20):
            _r = B_LADDER[(_k, _a)][_t]; _np, _vv = _r[:, 7].sum(), _r[:, 6].sum()
            if _np > 0:
                _p = _vv / _np; _ok &= _p <= 0.10 + b_cp_tol(_p, int(_np)); _det.append(f"{_t}/{_k}/{_a}: {int(_vv)}/{int(_np)}")
check(f"S6.{_n} per-GROUP violation (what tier 3 actually bounds) <= delta + CP tolerance over all (split, document-type) pairs that accepted something", _ok, "; ".join(_det)); _n += 1
_ok, _det = True, []
for _k in ("hgb", "weak", "verb"):
    for _a in (0.10, 0.20):
        _v = B_LADDER[(_k, _a)]["t4"][:, 5].mean(); _ok &= _v <= 0.10 + b_cp_tol(_v, 40); _det.append(f"{_k}/{_a}: {_v:.3f}")
check(f"S6.{_n} tier 4 macro per-document violation fraction <= delta + CP tolerance", _ok, "; ".join(_det)); _n += 1
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        _c = [b_summ(B_LADDER[(_k, _a)], t)["cov"] for t in ("t1", "t3", "t3b", "t4")]
        check(f"S6.{_n} coverage falls (weakly) down the ladder 1 >= 3 x type >= 3b >= 4 ({_k}, alpha={_a:.2f})", all(_c[i] >= _c[i + 1] - 1e-9 for i in range(3)), " >= ".join(f"{c:.3f}" for c in _c)); _n += 1
B_RETAINED = {(k, a): b_summ(B_LADDER[(k, a)], "t3")["cov"] / b_summ(B_LADDER[(k, a)], "t1")["cov"] for k in ("hgb", "weak", "verb") for a in (0.10, 0.20)}
check(f"S6.{_n} the rigor cost shrinks as alpha grows for weak scores: share of tier-1 coverage retained by tier 3 x type is larger at alpha=0.20 than 0.10",
      all(B_RETAINED[(k, 0.20)] > B_RETAINED[(k, 0.10)] for k in ("weak", "verb")), "; ".join(f"{k}: {B_RETAINED[(k, 0.10)]:.2f} -> {B_RETAINED[(k, 0.20)]:.2f}" for k in ("weak", "verb"))); _n += 1
print(f"[info] for the learned score the retained share barely moves ({B_RETAINED[('hgb', 0.10)]:.2f} at 0.10 vs {B_RETAINED[('hgb', 0.20)]:.2f} at 0.20): with only 15 geometric "
      "candidates the certified coverage plateaus at the last candidate that certifies (a property of the grid, not something to tune away). "
      "Paper's own numbers for this idea are in idpfin-q9 (retention 28% at alpha=0.10 vs 84% at alpha=0.20, CORD).")
'''))

    out.append(M(r"""
**Two-regime law (idpfin-q9), tested on our data.** Quoted: "With a learned score, covariates belong in the score" (Regime 1: "a tree fusion
splits on the covariate internally and equalizes per-group score scales, so external conditioning only fragments the threshold sample") and "With a weak or frozen score,
covariates belong in the taxonomy - where pooled cannot certify" (Regime 2), scoped sharply: "conditioning rescues certification where the pooled score head cannot
certify at the target alpha ... and fragmentation costs coverage where pooled certifies fine". In our setting: learned score = `FA_p_hgb` (which already
sees document type and field name). The two deliberately weaker contrasts are `FA_p_weak` (a 7-parameter logistic model that still has an additive document-type shift) and raw verbalized confidence
`FA_verb` (no document type at all). Thus `FA_verb` is the closer match to the paper's "score cannot encode the covariate" regime; `FA_p_weak` is a useful counterexample to treating the empirical law as universal. The pooled comparators are add-one pooled (tier 1) and LTT on a single group (tier 3 pooled). We report what we see, including a paired
sign-flip p-value (20,000 flips over the 40 resplit differences; **split-stability only**, the splits share documents).
"""))

    out.append(C(r'''
def b_signflip_p(d, flips=20000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(flips, len(d))); null = np.abs((s * d).mean(1))
    return float((1 + (null >= abs(d.mean()) - 1e-15).sum()) / (flips + 1))


B_TWO_REGIME = {}
_rows = []
for _k in ("hgb", "weak", "verb"):
    for _a in (0.10, 0.20):
        for _name, _pooled, _grp in (("add-one (tier 1 -> 2)", "t1", "t2"), ("LTT (tier 3 pooled -> x type)", "t3p", "t3")):
            _d = B_LADDER[(_k, _a)][_grp][:, 0] - B_LADDER[(_k, _a)][_pooled][:, 0]
            _pz = int((B_LADDER[(_k, _a)][_pooled][:, 2] == 0).sum())
            B_TWO_REGIME[(_k, _a, _pooled)] = dict(diff=_d.mean(), sd=_d.std(ddof=1), better=int((_d > 0).sum()), p=b_signflip_p(_d), pooled_cov=B_LADDER[(_k, _a)][_pooled][:, 0].mean(),
                                                    pooled_zero=_pz, grp_cov=B_LADDER[(_k, _a)][_grp][:, 0].mean())
            _x = B_TWO_REGIME[(_k, _a, _pooled)]
            _rows.append((_k, f"{_a:.2f}", _name, f"{_x['pooled_cov']:.3f}", f"{_x['grp_cov']:.3f}", f"{_x['diff']:+.3f} +/- {_x['sd']:.3f}", f"{_x['better']}/40", f"{_x['p']:.4f}", f"{_pz}/40"))
b_display(b_md(b_tbl(["score", "alpha", "comparison", "coverage pooled", "coverage x document type", "paired diff (x type - pooled)", "splits where x type is better", "sign-flip p (split stability)",
                      "pooled zero-coverage splits"], _rows)))
_n = 17
_se = lambda k, a, t: B_TWO_REGIME[(k, a, t)]["sd"] / np.sqrt(40)
check(f"S6.{_n} regime 1 (learned score): conditioning by document type on top of the learned score does not help LTT coverage (paired diff <= 3 SE)",
      all(B_TWO_REGIME[("hgb", a, "t3p")]["diff"] <= 3 * _se("hgb", a, "t3p") for a in (0.10, 0.20)),
      "; ".join(f"alpha={a}: {B_TWO_REGIME[('hgb', a, 't3p')]['diff']:+.4f} (3SE {3 * _se('hgb', a, 't3p'):.4f})" for a in (0.10, 0.20))); _n += 1
print("[info] the same check with a strict '<= 0' threshold would have FAILED at alpha=0.20 by "
      f"{B_TWO_REGIME[('hgb', 0.20, 't3p')]['diff']:+.4f} (better in {B_TWO_REGIME[('hgb', 0.20, 't3p')]['better']}/40 splits, p={B_TWO_REGIME[('hgb', 0.20, 't3p')]['p']:.2f}): "
      "a noise-level difference on a coverage plateau, which is why every paired-difference claim in this notebook uses the same 3-SE tolerance as the Monte-Carlo means.")
check(f"S6.{_n} regime 1 (learned score): ... and does not help add-one coverage either (paired diff <= 3 SE)",
      all(B_TWO_REGIME[("hgb", a, "t1")]["diff"] <= 3 * _se("hgb", a, "t1") for a in (0.10, 0.20)),
      "; ".join(f"alpha={a}: {B_TWO_REGIME[('hgb', a, 't1')]['diff']:+.4f} (3SE {3 * _se('hgb', a, 't1'):.4f})" for a in (0.10, 0.20))); _n += 1
_x = B_TWO_REGIME[("weak", 0.10, "t3p")]
check(f"S6.{_n} regime 2 scope (weak score, alpha=0.10): the paper's taxonomy rescue does NOT appear for this score",
      not (_x["diff"] > 0 and _x["p"] < 0.05),
      f"diff {_x['diff']:+.4f}, better in {_x['better']}/40 splits, p={_x['p']:.4f} (pooled coverage {_x['pooled_cov']:.4f}, zero-coverage in {_x['pooled_zero']}/40). This weak score already has document-type one-hots, so it is not the paper's pure 'cannot encode the covariate' case."); _n += 1
_x = B_TWO_REGIME[("verb", 0.10, "t3p")]
check(f"S6.{_n} regime 2 (raw verbalized confidence, alpha=0.10, pooled LTT cannot certify): x-type LTT coverage is significantly larger than pooled",
      _x["diff"] > 0 and _x["p"] < 0.05,
      f"diff {_x['diff']:+.4f}, better in {_x['better']}/40 splits, p={_x['p']:.4f} (pooled coverage {_x['pooled_cov']:.4f}, zero-coverage in {_x['pooled_zero']}/40)"); _n += 1
check(f"S6.{_n} regime 2 scoping: where pooled LTT certifies fine (alpha=0.20), fragmenting by type costs coverage (weak and raw scores)",
      all(B_TWO_REGIME[(k, 0.20, "t3p")]["diff"] < 0 and B_TWO_REGIME[(k, 0.20, "t3p")]["p"] < 0.05 for k in ("weak", "verb")),
      "; ".join(f"{k}: {B_TWO_REGIME[(k, 0.20, 't3p')]['diff']:+.3f} (p={B_TWO_REGIME[(k, 0.20, 't3p')]['p']:.4f})" for k in ("weak", "verb"))); _n += 1
print("[info] the weak score FA_p_weak already contains an additive document-type shift (7-parameter logistic model with type one-hots), so it is not a score 'that cannot encode' the covariate; "
      "that is a hypothesis for why the regime-2 rescue is visible for the raw verbalized confidence but not for FA_p_weak, not something we tested.")
'''))

    out.append(M(r"""
**How to read this chart.** Four panels at `alpha = delta = 0.10` unless stated. **Top-left, coverage**: share of test fields auto-accepted per rung,
optimistic score (red) vs pessimistic score (orange); whiskers are the sd over 40 resplits. Each step down the ladder should cost coverage; for the pessimistic score
the certified rungs sit near zero because no candidate threshold certifies at 10% error. **Top-right, achieved risk**: the realized error rate among accepted
fields (zero-filled); the dashed line is `alpha`. Tiers 1-2 sit *at* the line by construction, the PAC rungs sit below it (that gap is the price of a certificate).
**Bottom-left, violation fraction**: the share of splits whose realized test risk exceeds `alpha`; the dashed line is `delta`. Expectation tiers cross it
in about half the splits, PAC tiers do not. **Bottom-right, two-regime law**: the paired coverage difference "x document type minus pooled" for LTT; bars above zero mean that conditioning
on document type helps. The learned score should not gain from it. Here raw verbalized confidence gains where pooled cannot certify (`alpha = 0.10`), whereas the weak logistic fusion does not; both lose where pooled can certify (`alpha = 0.20`).
"""))

    out.append(C(r'''
_fig, _axes = plt.subplots(2, 2, figsize=(13, 8.6))
_x = np.arange(len(B_TIERS)); _w = 0.38
for _ax, _what, _title, _ref in ((_axes[0, 0], "cov", "Coverage (share of test fields auto-accepted)", None), (_axes[0, 1], "risk", "Achieved selective risk (zero-filled)", 0.10),
                                 (_axes[1, 0], "viol", "Share of splits with realized risk > alpha", 0.10)):
    for _i, (_k, _c) in enumerate((("hgb", PAL3["hgb"]), ("weak", PAL3["b_weak"]))):
        _v = [b_summ(B_LADDER[(_k, 0.10)], t)[_what] for t in B_TIERS]
        _e = [b_summ(B_LADDER[(_k, 0.10)], t)["cov_sd"] for t in B_TIERS] if _what == "cov" else None
        _ax.bar(_x + (_i - 0.5) * _w, _v, _w, yerr=_e, capsize=2, color=_c, label="optimistic score" if _k == "hgb" else "pessimistic score")
        for _xx, _vv in zip(_x, _v):
            _ax.text(_xx + (_i - 0.5) * _w, _vv + 0.012, f"{_vv:.2f}", ha="center", fontsize=6.5)
    if _ref is not None:
        _ax.axhline(_ref, ls="--", color=PAL3["diag"], lw=1)
    _ax.set_xticks(_x); _ax.set_xticklabels([B_TIER_SHORT[t] for t in B_TIERS], fontsize=8); _ax.set_title(_title, fontsize=10); _ax.legend(fontsize=8)
_ax = _axes[1, 1]
_labs, _vals, _cols = [], [], []
for _k in ("hgb", "weak", "verb"):
    for _a in (0.10, 0.20):
        _labs.append(f"{_k}\nalpha={_a:.2f}"); _vals.append(B_TWO_REGIME[(_k, _a, "t3p")]["diff"])
        _cols.append({"hgb": PAL3["hgb"], "weak": PAL3["b_weak"], "verb": PAL3["raw"]}[_k])
_ax.bar(range(len(_vals)), _vals, color=_cols); _ax.axhline(0, color=PAL3["diag"], lw=1)
for _i, _v in enumerate(_vals):
    _ax.text(_i, (_v + 0.006) if _v >= 0 else 0.006, f"{_v:+.3f}", ha="center", fontsize=8)
_ax.set_xticks(range(len(_vals))); _ax.set_xticklabels(_labs, fontsize=8); _ax.set_title("LTT coverage: x document type minus pooled", fontsize=10); _ax.set_ylabel("paired coverage difference")
plt.tight_layout(); plt.show()
'''))
    return out


def _s7():
    out = []

    out.append(M(r"""
## 7. Month-7 shift: frozen-configuration confirmation (P3 section 6.4 analogue) and what nobody covers

**What Gurram did (quoted, idpfin-q9).** "the production configuration (learned no-NLI fusion, pooled split-protocol add-one; frozen before the run) was executed **once, with no tuning,** on selection-untouched
genuine captures ... it attained coverage 0.167 at achieved risk 0.093 (viol 0.38) at alpha=0.10 ... Across both untouched captures the risk contract never failed;
what varies is coverage, which tracks the model's signal quality". Note what that confirmation is: a **new extractor on the same corpus family**, so exchangeability
between calibration and test still held; the paper itself says "Inference is in-corpus" (idpfin-q9). It is **not** a shifted batch.

**What our batch does.** The `shift` split (500 documents, month 7) has new vendor templates and heavier OCR noise: base rate of fully-correct documents 0.19 instead of 0.33, field error rate about 0.45
instead of 0.30. That violates exchangeability between calibration and deployment, and **no guarantee in P1-P3 was promised under it**: cfBH's authors list it as a
limitation ("Reliable selection under distribution shift, if not infeasible, may require more involved techniques", idpfin-q3); SCoRE covers only *covariate* shift with a known
or consistently estimated density ratio (Assumption 6.1; not built, Section 5(d)); and "None of the three sources state how to perform real-time, online re-calibration or re-certification under
unobserved concept drift" without new labeled data (idpfin-q12). Also unstated anywhere: a formula for how many labelled calibration units are enough ("None of the three sources state a single universal, closed-form analytical formula",
idpfin-q12), so the labelled-document budget below is **derived here, empirically**.

**Protocol (ours).** (1) Freeze scores and thresholds learned on the `calib` split only (tier 1 pooled add-one, tier 3 Mondrian LTT by document type, `alpha in {0.10, 0.20}`, `delta = 0.10`, both scores)
plus the folklore rule (verbalized confidence >= 0.9 and no rule failure); apply **once** to `test` (control, same distribution) and `shift`. (2) Report the accepted-set field error with a
**document-cluster bootstrap** CI (2,000 resamples of documents) and the share of bootstrap resamples whose risk exceeds `alpha`. (3) Re-calibrate the *thresholds* (scores stay frozen) on `N` labelled shift documents
and evaluate on the remaining shift documents, 40 random draws per `N`. Rule fixed **before** looking at the recalibration results: the *budget* is the smallest `N` in {50, 100, 200, 300} at which tier-3 pooled recalibration
(i) certifies something in all 40 draws, (ii) has violation fraction <= `delta` + Clopper-Pearson tolerance, and (iii) reaches at least 90% of its own coverage at `N = 300`.
"""))

    out.append(C(r'''
def b_doc_arrays(mask, acc):
    docs = np.unique(FA_doc[mask]); inv = np.searchsorted(docs, FA_doc[mask])
    n_acc = np.bincount(inv[acc], minlength=len(docs)).astype(float); e_acc = np.bincount(inv[acc], weights=B_ERR[mask][acc], minlength=len(docs))
    return n_acc, e_acc


def b_boot_ratio(num_d, den_d, thr, B=2000, seed=0):
    """Document-cluster bootstrap of sum(num)/sum(den): (point, lo, hi, share of resamples with ratio > thr)."""
    rng = np.random.default_rng([SEED3, 7, seed]); D = len(num_d); idx = rng.integers(0, D, size=(B, D))
    num, den = num_d[idx].sum(1), den_d[idx].sum(1); ok = den > 0; r = num[ok] / den[ok]
    pt = num_d.sum() / den_d.sum() if den_d.sum() > 0 else float("nan")
    if len(r) == 0:
        return pt, float("nan"), float("nan"), float("nan")
    return pt, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), float(np.mean(r > thr))


B_CALM, B_TESTM, B_SHIFTM = (B_FSPLIT == "calib"), (B_FSPLIT == "test"), (B_FSPLIT == "shift")
B_FOLK_FIELD = np.array([f["accepted"] for f in FIELDS], int).astype(bool)          # Notebook 2 baseline, learned from nothing

B_FROZEN = {}
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        B_FROZEN[(_k, _a)] = b_fit_thresholds(B_SCORE_FIELD[_k], B_CALM, _a, 0.10)[0]
check("S7.1 every threshold was learned from `calib`-split documents only (never test or shift)", set(DOC_SPLIT[FA_doc[B_CALM]]) == {"calib"} and not (B_CALM & (B_TESTM | B_SHIFTM)).any(),
      f"{len(set(FA_doc[B_CALM]))} calibration documents")


def b_eval_frozen(acc_fn, alpha):
    row = {}
    for nm, m in (("test", B_TESTM), ("shift", B_SHIFTM)):
        acc = acc_fn(m); n_d, e_d = b_doc_arrays(m, acc); pt, lo, hi, pg = b_boot_ratio(e_d, n_d, alpha)
        row[nm] = dict(cov=acc.mean(), risk=pt, lo=lo, hi=hi, p_gt=pg, nacc=int(acc.sum()))
    return row


B_FROZEN_RES = {}
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        for _t in ("t1", "t3"):
            B_FROZEN_RES[(_k, _a, _t)] = b_eval_frozen(lambda m, _k=_k, _a=_a, _t=_t: b_accept(B_SCORE_FIELD[_k], m, B_FROZEN[(_k, _a)][_t]), _a)
B_FROZEN_RES[("folk", None, "folk")] = b_eval_frozen(lambda m: B_FOLK_FIELD[m], 0.10)
B_FROZEN_RES[("folk", None, "folk")]["alpha_ref"] = 0.10
_rows = []
for (_k, _a, _t), _r in B_FROZEN_RES.items():
    _tau = "-" if _k == "folk" else (f"{B_FROZEN[(_k, _a)][_t]:.3f}" if _t == "t1" else "[" + ", ".join("review all" if np.isinf(x) else f"{x:.3f}" for x in B_FROZEN[(_k, _a)][_t]) + "]")
    _ref = 0.10 if _a is None else _a
    _rows.append(({"hgb": "optimistic", "weak": "pessimistic", "folk": "folklore (>=0.9 & rules)"}[_k], "-" if _a is None else f"{_a:.2f}", {"t1": "1 add-one pooled", "t3": "3 LTT x type", "folk": "-"}[_t], _tau,
                  f"{_r['test']['cov']:.3f}", f"{_r['test']['risk']:.3f} [{_r['test']['lo']:.3f}, {_r['test']['hi']:.3f}]",
                  f"{_r['shift']['cov']:.3f}", f"{_r['shift']['risk']:.3f} [{_r['shift']['lo']:.3f}, {_r['shift']['hi']:.3f}]", f"{_r['shift']['p_gt']:.2f}",
                  "held" if _r["shift"]["risk"] <= _ref else "BROKEN"))
b_display(b_md("**Frozen configurations applied once** (risk = error rate among accepted fields with document-cluster bootstrap 95% CI; folklore is compared with alpha = 0.10; thresholds apply to the field score, per document type in the order invoice, bank statement, KYC, compliance report)\n\n" + b_tbl(
    ["score", "alpha", "rule", "frozen threshold(s)", "test coverage", "test risk [95% CI]", "shift coverage", "shift risk [95% CI]", "share of bootstrap resamples with shift risk > alpha", "verdict on shift"], _rows)))

check("S7.2 control: on the unshifted `test` split every frozen tier-3 configuration keeps its realized risk at or below alpha (this run: one calibration draw)",
      all(B_FROZEN_RES[(k, a, "t3")]["test"]["risk"] <= a for k in ("hgb", "weak") for a in (0.10, 0.20)),
      "; ".join(f"{k}/{a}: {B_FROZEN_RES[(k, a, 't3')]['test']['risk']:.3f}" for k in ("hgb", "weak") for a in (0.10, 0.20)))
_broken = {key: r["shift"]["risk"] > (0.10 if key[1] is None else key[1]) for key, r in B_FROZEN_RES.items()}
check("S7.3 as theory says (no promise under shift): EVERY frozen configuration, certified or not, has a realized shift risk above its alpha", all(_broken.values()),
      "; ".join(f"{k[0]}/{k[1]}/{k[2]}: {r['shift']['risk']:.3f}" for k, r in B_FROZEN_RES.items()))
check("S7.4 shift degrades every frozen configuration (shift risk > test risk)", all(r["shift"]["risk"] > r["test"]["risk"] for r in B_FROZEN_RES.values()),
      "; ".join(f"{k[0]}/{k[1]}/{k[2]}: {r['test']['risk']:.3f} -> {r['shift']['risk']:.3f}" for k, r in B_FROZEN_RES.items()))
_clear = [k for k, r in B_FROZEN_RES.items() if r["shift"]["lo"] > (0.10 if k[1] is None else k[1])]
print("[info] guarantees on the shift batch: violated with the whole bootstrap CI above alpha for " + (", ".join(f"{k[0]}/{k[1]}/{k[2]}" for k in _clear) or "none")
      + "; point estimate above alpha but CI reaching alpha (few accepted fields) for " + (", ".join(f"{k[0]}/{k[1]}/{k[2]}" for k in B_FROZEN_RES if k not in _clear) or "none") + ".")
'''))

    out.append(C(r'''
# ---- supplementary 1: how often does a frozen config break, across 40 different calibration draws from the calib split? ----
_calib_docs = np.unique(FA_doc[B_CALM]); B_SHIFT_VIOL = {}
for _k in ("hgb", "weak"):
    for _t in ("t1", "t3"):
        _v = {"test": [], "shift": []}
        for _sd in range(40):
            _rng = np.random.default_rng([SEED3, 31, _sd]); _sub = np.zeros(N_DOC, bool); _sub[_rng.choice(_calib_docs, int(0.8 * len(_calib_docs)), replace=False)] = True
            _tau = b_fit_thresholds(B_SCORE_FIELD[_k], _sub[FA_doc] & B_CALM, 0.10, 0.10)[0][_t]
            for _nm, _m in (("test", B_TESTM), ("shift", B_SHIFTM)):
                _acc = b_accept(B_SCORE_FIELD[_k], _m, _tau); _v[_nm].append(B_ERR[_m][_acc].mean() > 0.10 if _acc.any() else False)
        B_SHIFT_VIOL[(_k, _t)] = {nm: float(np.mean(x)) for nm, x in _v.items()}
b_display(b_md("**Violation across 40 calibration draws** (80% of the `calib` documents each; alpha = 0.10, delta = 0.10; share of draws whose realized risk exceeds alpha)\n\n" + b_tbl(
    ["score", "rule", "on `test` (no shift)", "on `shift`"], [({"hgb": "optimistic", "weak": "pessimistic"}[k], {"t1": "1 add-one pooled", "t3": "3 LTT x type"}[t], f"{v['test']:.2f}", f"{v['shift']:.2f}") for (k, t), v in B_SHIFT_VIOL.items()])))
check("S7.5 across calibration draws the certified rule stays within delta on `test` but not on `shift`",
      all(B_SHIFT_VIOL[(k, "t3")]["test"] <= 0.10 + b_cp_tol(B_SHIFT_VIOL[(k, "t3")]["test"], 40) for k in ("hgb", "weak")) and all(B_SHIFT_VIOL[(k, "t3")]["shift"] > 0.10 + b_cp_tol(0.10, 40) for k in ("hgb", "weak")),
      "; ".join(f"{k}: test {B_SHIFT_VIOL[(k, 't3')]['test']:.2f}, shift {B_SHIFT_VIOL[(k, 't3')]['shift']:.2f}" for k in ("hgb", "weak")))

# ---- supplementary 2: the document-level rules of Section 5 under shift (calibrated on calib money docs, applied once) ----
_cm = np.where((DOC_SPLIT == "calib") & DA_has_money)[0]; B_DOC_SHIFT = {}
for _k, _s in B_SCORES.items():
    for _a in (0.10, 0.20):
        _psi_cal = None
        for _nm, _sp in (("test", "test"), ("shift", "shift")):
            _td = np.where((DOC_SPLIT == _sp) & DA_has_money)[0]
            _psi = b_mdr_e_values(_s[_cm], DA_L[_cm], _s[_td], _a) >= 1 / _a
            _nv = _s[_td] <= _a
            _rng = np.random.default_rng([SEED3, 41]); _ix = _rng.integers(0, len(_td), size=(2000, len(_td)))
            _mdr = (DA_L[_td] * _psi)[_ix].mean(1); _nsel = _nv[_ix].sum(1); _nvl = (DA_L[_td] * _nv)[_ix].sum(1) / np.maximum(_nsel, 1)
            B_DOC_SHIFT[(_k, _a, _nm)] = dict(mdr=float(np.mean(DA_L[_td] * _psi)), mdr_ci=np.percentile(_mdr, [2.5, 97.5]), cov=float(_psi.mean()),
                                              naive=float((DA_L[_td] * _nv).sum() / max(1, _nv.sum())), naive_ci=np.percentile(_nvl, [2.5, 97.5]), naive_cov=float(_nv.mean()))
_rows = []
for (_k, _a, _nm), _v in B_DOC_SHIFT.items():
    _rows.append(({"opt": "optimistic", "weak": "pessimistic"}[_k], f"{_a:.2f}", _nm, f"{_v['cov']:.3f}", f"{_v['mdr']:.4f} [{_v['mdr_ci'][0]:.4f}, {_v['mdr_ci'][1]:.4f}]",
                  f"{_v['naive_cov']:.3f}", f"{_v['naive']:.4f} [{_v['naive_ci'][0]:.4f}, {_v['naive_ci'][1]:.4f}]"))
b_display(b_md("**Document-level rules from Section 5, calibrated once on `calib` money documents** (bootstrap over test documents with the calibration fixed)\n\n" + b_tbl(
    ["score", "alpha", "applied to", "SCoRE-MDR coverage", "realized MDR E[L psi] [95% CI]", "naive coverage (pred. risk <= alpha)", "naive avg loss among accepted [95% CI]"], _rows)))
_mdr_shift_breaks = [(k, a) for k in B_SCORES for a in (0.10, 0.20) if B_DOC_SHIFT[(k, a, "shift")]["mdr"] > a]
check("S7.6 without SCoRE's required density-ratio weighting, the shifted batch breaks the observed MDR target in at least one configuration (not universally)",
      bool(_mdr_shift_breaks),
      "; ".join(f"{k}/{a}: {B_DOC_SHIFT[(k, a, 'shift')]['mdr']:.3f} (test {B_DOC_SHIFT[(k, a, 'test')]['mdr']:.3f})" for k in B_SCORES for a in (0.10, 0.20))
      + f"; observed shift overshoots: {_mdr_shift_breaks}. Any non-overshoot is a finite sample result, not evidence that the unweighted guarantee survived shift.")
_naive_shift_breaks = [(k, a) for k in B_SCORES for a in (0.10, 0.20) if B_DOC_SHIFT[(k, a, "shift")]["naive"] > a]
check("S7.7 the old-data naive risk rule also has at least one observed shifted-batch overshoot; it has no shift guarantee",
      bool(_naive_shift_breaks),
      "; ".join(f"{k}/{a}: {B_DOC_SHIFT[(k, a, 'shift')]['naive']:.3f} (test {B_DOC_SHIFT[(k, a, 'test')]['naive']:.3f})" for k in B_SCORES for a in (0.10, 0.20))
      + f"; observed shift overshoots: {_naive_shift_breaks}. Non-overshoot cells are reported, not promoted to a guarantee.")
'''))

    out.append(C(r'''
# ---- re-calibration on N labelled shift documents (scores stay frozen; thresholds re-learned), 40 random draws per N ----
B_SHIFT_DOCS = np.where(DOC_SPLIT == "shift")[0]
B_RECAL_NS = (50, 100, 200, 300)
B_RECAL_METHODS = ("stale_t1", "stale_t3", "recal_t1", "recal_t3p", "recal_t3")


def b_recal(score_key, alpha, N, R=40, delta=0.10):
    score = B_SCORE_FIELD[score_key]; fz = B_FROZEN[(score_key, alpha)]; rows = {m: [] for m in B_RECAL_METHODS}
    for r in range(R):
        rng = np.random.default_rng([SEED3, 77, N, r]); perm = rng.permutation(B_SHIFT_DOCS); dm = np.zeros(N_DOC, int); dm[perm[:N]] = 1; dm[perm[N:]] = 2
        lm, tm = dm[FA_doc] == 1, dm[FA_doc] == 2; sc_l, er_l, gr_l = score[lm], B_ERR[lm], B_GRP[lm]
        taus = {"stale_t1": fz["t1"], "stale_t3": fz["t3"], "recal_t1": add_one_threshold(sc_l, er_l, alpha), "recal_t3p": b_ltt_certify(sc_l, er_l, alpha, delta),
                "recal_t3": np.array([b_ltt_certify(sc_l[gr_l == g], er_l[gr_l == g], alpha, delta) for g in range(4)])}
        for m_, tau in taus.items():
            acc = b_accept(score, tm, tau); te = B_ERR[tm]
            rows[m_].append((acc.mean(), te[acc].mean() if acc.any() else 0.0, float(te[acc].mean() > alpha) if acc.any() else 0.0, float(not acc.any())))
    return {m_: np.array(v) for m_, v in rows.items()}


B_RECAL = {(k, a, N): b_recal(k, a, N) for k in ("hgb", "weak") for a in (0.10, 0.20) for N in B_RECAL_NS}
_lab = {"stale_t1": "stale tier 1 (frozen)", "stale_t3": "stale tier 3 x type (frozen)", "recal_t1": "recal. tier 1 add-one", "recal_t3p": "recal. tier 3 pooled", "recal_t3": "recal. tier 3 x type"}
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        _rows = []
        for _m in B_RECAL_METHODS:
            for _N in B_RECAL_NS:
                _v = B_RECAL[(_k, _a, _N)][_m]
                _rows.append((_lab[_m], _N, f"{_v[:, 0].mean():.3f}", f"{_v[:, 1].mean():.3f}", f"{_v[:, 2].mean():.2f}", f"{int(_v[:, 3].sum())}/40"))
        b_display(b_md(f"**Re-calibration, {'optimistic' if _k == 'hgb' else 'pessimistic'} score, alpha = {_a:.2f}, delta = 0.10** (mean over 40 draws; evaluated on the remaining 500 - N shift documents)\n\n" + b_tbl(
            ["rule", "labelled shift docs N", "coverage", "realized risk (zero-filled)", "share of draws with risk > alpha", "draws with zero coverage"], _rows)))

_ok, _det = True, []
for (_k, _a, _N), _res in B_RECAL.items():
    for _m in ("recal_t3p", "recal_t3"):
        _v = _res[_m][:, 2].mean(); _ok &= _v <= 0.10 + b_cp_tol(_v, 40); _det.append(f"{_k}/{_a}/{_N}/{_m[6:]}: {_v:.2f}")
check("S7.8 recalibrated PAC tiers (3 pooled, 3 x type) are valid again on the shifted distribution: violation fraction <= delta + CP tolerance in every cell", _ok, "; ".join(_det[:8]) + " ...")
check("S7.9 recalibrated tier 1 (expectation) meets its target on average but is not a certificate (hgb, alpha=0.10, N=200: risk <= alpha + 3SE, violation > delta)",
      B_RECAL[("hgb", 0.10, 200)]["recal_t1"][:, 1].mean() <= 0.10 + b_mc_tol_mean(B_RECAL[("hgb", 0.10, 200)]["recal_t1"][:, 1]) and B_RECAL[("hgb", 0.10, 200)]["recal_t1"][:, 2].mean() > 0.10,
      f"risk {B_RECAL[('hgb', 0.10, 200)]['recal_t1'][:, 1].mean():.4f}, violation {B_RECAL[('hgb', 0.10, 200)]['recal_t1'][:, 2].mean():.2f}")
check("S7.10 stale (frozen) thresholds keep violating on the shifted distribution regardless of how many labelled shift documents exist elsewhere (hgb, alpha=0.10: violation in >= 90% of draws)",
      all(B_RECAL[("hgb", 0.10, N)]["stale_t3"][:, 2].mean() >= 0.90 for N in B_RECAL_NS), "; ".join(f"N={N}: {B_RECAL[('hgb', 0.10, N)]['stale_t3'][:, 2].mean():.2f}" for N in B_RECAL_NS))

# labelled-document budget by the pre-fixed rule
B_BUDGET = {}
for _k in ("hgb", "weak"):
    for _a in (0.10, 0.20):
        _cov300 = B_RECAL[(_k, _a, 300)]["recal_t3p"][:, 0].mean(); _found = None
        for _N in B_RECAL_NS:
            _v = B_RECAL[(_k, _a, _N)]["recal_t3p"]
            if _v[:, 3].sum() == 0 and _v[:, 2].mean() <= 0.10 + b_cp_tol(_v[:, 2].mean(), 40) and _cov300 > 0 and _v[:, 0].mean() >= 0.9 * _cov300:
                _found = _N; break
        B_BUDGET[(_k, _a)] = dict(N=_found, cov300=_cov300)
print("labelled-shift-document budget (derived here, rule fixed in advance; tier-3 pooled recalibration): " + "; ".join(
    f"{k}/alpha={a}: " + (f"N={v['N']} (coverage at N=300: {v['cov300']:.3f})" if v["N"] else f"not reached within N<=300 (coverage at N=300: {v['cov300']:.3f})") for (k, a), v in B_BUDGET.items()))
check("S7.11 a labelled-document budget within the tested grid exists for the optimistic score (both alphas); none is claimed for the pessimistic score unless the rule found one",
      all(B_BUDGET[("hgb", a)]["N"] is not None for a in (0.10, 0.20)), "; ".join(f"{k}/{a}: {v['N']}" for (k, a), v in B_BUDGET.items()))
_pre = {k: b_summ(B_LADDER[(k, 0.10)], "t3p")["cov"] for k in ("hgb", "weak")}
print(f"[info] for scale: pre-shift tier-3 pooled coverage at alpha=0.10 on the resplit protocol was {_pre['hgb']:.3f} (optimistic) / {_pre['weak']:.3f} (pessimistic); after recalibration on the shifted batch it is "
      f"{B_RECAL[('hgb', 0.10, 300)]['recal_t3p'][:, 0].mean():.3f} / {B_RECAL[('weak', 0.10, 300)]['recal_t3p'][:, 0].mean():.3f} at N=300 - validity comes back, coverage does not, because the shifted documents are harder to separate.")
'''))

    out.append(M(r"""
**How to read this chart.** Left: the accepted-set field error on the `shift` batch for each frozen configuration (dots, with document-cluster bootstrap 95% CI) next to its error on the unshifted `test`
split (grey squares); the tick on each group marks its own `alpha`. Every dot sits above its tick: whatever was promised on calibration-like data does not carry over. Middle and right: recalibrating the
thresholds on `N` labelled shift documents (optimistic score, `alpha = 0.10`; 40 draws per point). Middle, coverage: the stale thresholds accept a lot but on a batch where they are wrong; the recalibrated certified rule
accepts less. Right, realized risk: the stale rules sit far above `alpha` (dashed), while the recalibrated certified rules come back **below** it. The expectation-only recalibrated add-one hovers at `alpha` and,
as in Section 6, exceeds it in a large share of draws. Note the recovered coverage is far below the pre-shift coverage: the shifted documents are harder to separate.
"""))

    out.append(C(r'''
_fig, _axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
_ax = _axes[0]; _keys = [k for k in B_FROZEN_RES]; _y = np.arange(len(_keys))[::-1]
for _yy, _key in zip(_y, _keys):
    _r = B_FROZEN_RES[_key]; _ref = 0.10 if _key[1] is None else _key[1]
    _c = {"hgb": PAL3["hgb"], "weak": PAL3["b_weak"], "folk": PAL3["b_folk"]}[_key[0]]
    _ax.errorbar(_r["shift"]["risk"], _yy, xerr=[[_r["shift"]["risk"] - _r["shift"]["lo"]], [_r["shift"]["hi"] - _r["shift"]["risk"]]], fmt="o", color=_c, capsize=3)
    _ax.plot(_r["test"]["risk"], _yy, "s", color=PAL3["raw"], ms=6); _ax.plot([_ref, _ref], [_yy - 0.3, _yy + 0.3], color=PAL3["diag"], lw=2)
_ax.set_yticks(_y); _ax.set_yticklabels([("folklore" if k[0] == "folk" else f"{k[0]} a={k[1]:.2f} T{k[2][1:]}") for k in _keys], fontsize=8)
_ax.set_xlabel("error rate among accepted fields"); _ax.set_title("Frozen rules: shift (dot, 95% CI) vs test (square); tick = alpha", fontsize=9)
for _ax, _q, _title in ((_axes[1], 0, "Coverage vs labelled shift docs N"), (_axes[2], 1, "Realized risk vs N")):
    for _m, _c, _ls in (("stale_t3", PAL3["b_folk"], "--"), ("recal_t1", PAL3["lr"], "-"), ("recal_t3p", PAL3["b_good"], "-"), ("recal_t3", PAL3["hgb"], "-")):
        _ax.plot(B_RECAL_NS, [B_RECAL[("hgb", 0.10, N)][_m][:, _q].mean() for N in B_RECAL_NS], "o" + _ls, color=_c, label=_lab[_m])
    _ax.set_xlabel("labelled shift documents N"); _ax.set_title(_title + " (optimistic, alpha=0.10)", fontsize=9)
_axes[2].axhline(0.10, ls=":", color=PAL3["diag"]); _axes[2].set_ylabel("realized risk"); _axes[1].set_ylabel("coverage"); _axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
'''))
    return out


def cells():
    out = []
    out += _s5()
    out += _s6()
    out += _s7()
    return out
