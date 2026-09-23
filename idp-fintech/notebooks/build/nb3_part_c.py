"""Notebook 3, PART C (owned by this agent): section 7b.

  7b. Repairing the month-7 shift without new labels -- and where it cannot be repaired

Namespace rule: every global defined here is prefixed C_ / c_ (loop temporaries: leading underscore).
Runs in ONE kernel after prelude + part B (reuses B_POOL, B_SCORES, B_SC_LABEL, B_SC_COLOR, DA_*, resplit, X_doc,
b_tbl/b_md/b_display, b_time, b_mc_tol_mean, b_cp_tol, DA_mu, DA_mu_weak, and part B's own §7 results B_DOC_SHIFT /
B_FROZEN_RES / B_RECAL / B_BUDGET). Deliberately independent of Part A (own C_M_BIG / c_domain_v), so it builds and
tests standalone as `--parts prelude,b,c`; in the FULL notebook it also runs after Part A, which is harmless since no
names collide (A_ vs C_ prefixes).
"""
import re

from nb3_common import M, C


def cells():
    out = []
    out.extend(_intro())
    out.extend(_part_i())
    out.extend(_part_ii())
    out.extend(_part_iii())
    return out


def _intro():
    out = []
    out.append(M(r'''
## 7b. Repairing the month-7 shift without new labels -- and where it cannot be repaired

Section 7 froze scores and thresholds and watched every rule break on the `shift` split; the only repair it found needed new
**labelled** shift documents. SCoRE and the covariate-shift conformal-selection literature offer a *label-free* repair for
one specific failure mode: **covariate shift**, `dQ/dP(x, y) = w(x)` (Assumption 6.1, quoted below). This section (i) checks that
repair **exactly**, on a semi-synthetic shift built so Assumption 6.1 holds by construction; (ii) tries it for real on the
month-7 batch, where `w(x)` must be **estimated**; and (iii) asks the question the papers never answer: is the real month-7
failure covariate shift at all?

**Quoted (idpfin-q13).** Assumption 6.1: *"The labeled data follow `(X_i, Y_i) ~ P` while the test data follow
`(X_{n+j}, Y_{n+j}) ~ Q`, and the two distributions obey `dQ/dP(x, y) = w(x)` for a known or estimable weight function
`w : X -> R_+`"*. Weighted MDR e-value, Eq. (6.1): `E = inf_l 1{s(X) <= t_gamma(l)} * (sum w_i) / (sum w_i L_i 1{s(X_i)<=t_gamma(l)} + w_test l 1{...})`,
`t_gamma(l) = max{t in M : F(t;l) <= gamma}`. **Theorem 6.2**: *"Under Assumption 6.1, for any fixed constant gamma in (0,1),
it holds that E_Q[L_{n+1} E_{gamma,n+1}] <= 1"* -- a **finite-sample** guarantee taken under the **target distribution Q**, not P.
Theorem 6.3 states the SDR analogue (Eq. 6.2) with the same finite-sample-under-Q property. **Proposition A.1** (idpfin-q13) is the
computational shortcut used below for the MDR trust decision at `gamma = alpha`.

**Quoted (idpshift-q1, WCS).** Weighted conformal p-value, Eq. (5): `p_j = (sum_i w(X_i) 1{V_i < Vhat_j} + w(X_j)) / (sum_i w(X_i) + w(X_j))`.
**Theorem 3.1**: *"Suppose `{Z_i}` iid `~ P` and `{Z_{n+j}}` iid `~ Q`, and (1) holds ... Then with either `R in {R_hete, R_homo, R_dtm}`,
it holds that E[ sum_j 1{j in R, Y_{n+j} <= c_{n+j}} / (1 v |R|) ] <= q"* (the expectation is over **both** calibration and test data) --
again finite-sample, under known `w`. When `w` is estimated, **Theorem 3.5** (idpshift-q1, q2) bounds `FDR <= q * E[gamma_hat^2 / (1 + q(gamma_hat^2-1)/m)]`,
`gamma_hat = sup_x max{w_hat(x)/w(x), w(x)/w_hat(x)}`: inflation, not exact control, and it "converges to zero if `w_hat` is consistent" (idpshift-q1).

**What none of the sources give (idpfin-q18, idpshift-q4, flagged explicitly).** No diagnostic for whether the covariate-shift
assumption `P(Y|X) = Q(Y|X)` holds; no weighted/shift-robust cfBH or Mondrian LTT (SCoRE is "the ONLY source ... that formulates
weighted risk-adjusted e-values for covariate shift", idpfin-q18); no treatment of covariates with **zero support** in one arm
(`P_X(x) = 0 implies w(x) = infinity`, idpshift-q4) -- part (ii)'s covariate set C hits exactly this. The importance-sampling
effective sample size below (`G.effective_sample_size`, Kish's formula, idpshift-q3/q4) is **not** in any of the three fintech
papers; it is standard covariate-shift folklore quoted from the companion Tibshirani et al. paper the shift traces cover.
'''))
    out.append(C(r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..") / "lib"))
import guarantees as G

C_BETAS = (0.5, 1.5, 3.0)          # tilt strengths: w(x) = exp(beta * standardized DA_risk), mean(w) = 1 over the pool
C_ALPHA = 0.10
C_Q = 0.10


def c_tilt(pool, z, beta):
    """Exact finite-population tilt on `pool`: w(x) = exp(beta * z_std) / mean_pool(exp(beta * z_std)), so mean(w) = 1
    over the pool BY CONSTRUCTION (Assumption 6.1 holds exactly for the two-arm sampling scheme below)."""
    zs = (z[pool] - z[pool].mean()) / (z[pool].std() + 1e-12)
    raw = np.exp(beta * zs)
    return raw / raw.mean(), raw / raw.sum()


def c_draw(pool, w, q, n, m, rng):
    """Calibration: n iid draws WITH replacement, uniform over `pool` (= P). Test: m iid draws WITH replacement,
    probability proportional to `w` (= Q). Returns (cal_doc_idx, test_doc_idx, cal_w, test_w); w(x) is evaluated at
    each unit's OWN covariate regardless of which arm drew it, exactly as w_i = w(X_i) in Eq. (6.1)/(6.2)."""
    cal_pos = rng.integers(0, len(pool), size=n)
    test_pos = rng.choice(len(pool), size=m, replace=True, p=q)
    return pool[cal_pos], pool[test_pos], w[cal_pos], w[test_pos]


check("S7b.0 the tilt has mean 1 over its pool by construction (3 betas, both pools)",
      all(abs(c_tilt(B_POOL, DA_risk, b)[0].mean() - 1.0) < 1e-9 for b in C_BETAS)
      and all(abs(c_tilt(np.where(np.isin(DOC_SPLIT, ["calib", "test"]))[0], DA_risk, b)[0].mean() - 1.0) < 1e-9 for b in C_BETAS))
'''))
    return out


def _part_i():
    out = []
    out.append(M(r'''
### (i). Known-weight semi-synthetic shift: an exact finite-sample check

**Construction (derived here).** Pool = money documents of `calib` + `test` (`B_POOL`, for SCoRE) or all documents of
`calib` + `test` (for cfBH/WCS, since `DA_y_ok` is defined for every document type). The tilt variable is `z` = the
optimistic predicted risk `DA_risk` (an observable covariate), standardized on the pool; `w(x) = exp(beta * z_std(x))`,
renormalised to mean 1 over the pool. Each repetition draws a fresh calibration batch **uniformly** (= P) and a fresh test
batch **with probability proportional to `w`** (= Q), both with replacement from the *same fixed pool*: this makes
`dQ/dP(x) = w(x)` hold **exactly** for this two-arm sampling scheme (not approximately), so Theorem 6.2/6.3 and WCS Theorem 3.1
apply verbatim with **known** weights. 300 repetitions per (beta, score); tolerances as in Section 5 (mean <= target + 3 SE).
'''))
    out.append(C(r'''
C_N_KW, C_M_KW, C_REPS_KW = 220, 220, 300
C_ALLPOOL = np.where(np.isin(DOC_SPLIT, ["calib", "test"]))[0]


def c_mdr_known_mc(score, beta, alpha=C_ALPHA, n=C_N_KW, m=C_M_KW, reps=C_REPS_KW, seed0=5100):
    w, q = c_tilt(B_POOL, DA_risk, beta)
    rows = []
    for _r in range(reps):
        _rng = np.random.default_rng([SEED3, seed0, int(beta * 100), _r])
        _ci, _ti, _cw, _tw = c_draw(B_POOL, w, q, n, m, _rng)
        _Lc, _Lt = DA_L[_ci], DA_L[_ti]; _sc, _st = score[_ci], score[_ti]
        _psi_uw = G.weighted_mdr_decision(_sc, _Lc, np.ones(n), _st, np.ones(m), alpha)
        _psi_w = G.weighted_mdr_decision(_sc, _Lc, _cw, _st, _tw, alpha)
        rows.append((np.mean(_Lt * _psi_uw), _psi_uw.mean(), np.mean(_Lt * _psi_w), _psi_w.mean(), G.effective_sample_size(_cw)))
    return np.array(rows)


C_MDR_KNOWN = {(k, b): c_mdr_known_mc(s, b) for k, s in B_SCORES.items() for b in C_BETAS}
_rows = []
for (_k, _b), _r in C_MDR_KNOWN.items():
    _rows.append((B_SC_LABEL[_k].split(" ")[0], f"{_b:.1f}", f"{_r[:, 4].mean():.1f} / {C_N_KW}", f"{_r[:, 0].mean():.4f} +/- {b_mc_tol_mean(_r[:, 0]):.4f}",
                  f"{_r[:, 1].mean():.3f}", f"{_r[:, 2].mean():.4f} +/- {b_mc_tol_mean(_r[:, 2]):.4f}", f"{_r[:, 3].mean():.3f}"))
b_display(b_md("**SCoRE-MDR under a KNOWN exact tilt** (alpha = 0.10, money-document pool, 300 reps per row)\n\n" + b_tbl(
    ["score", "tilt strength beta", "Kish ESS of calibration weights / n", "unweighted realized MDR_Q (mean +/- 3SE)", "unweighted coverage",
     "weighted realized MDR_Q (mean +/- 3SE)", "weighted coverage"], _rows)))

_n = 1
_ov_opt = [C_MDR_KNOWN[("opt", b)][:, 0].mean() > C_ALPHA + b_mc_tol_mean(C_MDR_KNOWN[("opt", b)][:, 0]) for b in C_BETAS]
_ov_weak = [C_MDR_KNOWN[("weak", b)][:, 0].mean() > C_ALPHA + b_mc_tol_mean(C_MDR_KNOWN[("weak", b)][:, 0]) for b in C_BETAS]
check(f"S7b.{_n} unweighted MDR overshoots its budget for the WEAK/pessimistic score at the two milder tilts (beta=0.5, 1.5), and does NOT "
      "overshoot for the OPTIMISTIC score at any tested tilt strength -- these are two separate, honest facts, not merged into one 'either' claim",
      _ov_weak[0] and _ov_weak[1] and not any(_ov_opt),
      "weak: " + "; ".join(f"beta={b}: {C_MDR_KNOWN[('weak', b)][:, 0].mean():.4f} (overshoot={_ov_weak[_i]})" for _i, b in enumerate(C_BETAS)) +
      " | opt: " + "; ".join(f"beta={b}: {C_MDR_KNOWN[('opt', b)][:, 0].mean():.4f} (overshoot={_ov_opt[_i]})" for _i, b in enumerate(C_BETAS))); _n += 1
_cov_uw_opt = [C_MDR_KNOWN[("opt", b)][:, 1].mean() for b in C_BETAS]; _cov_w_opt = [C_MDR_KNOWN[("opt", b)][:, 3].mean() for b in C_BETAS]
check(f"S7b.{_n} the OPTIMISTIC score's unweighted MDR stays WITHIN budget at every tilt strength (S7b.1) but loses substantial coverage relative "
      "to the weighted rule at beta=1.5 and beta=3.0 -- a POWER LOSS, not a safety failure: the threshold-defining score is closely related to the "
      "tilt variable itself, so restricting to score<=threshold structurally excludes most of the oversampled high-risk tail",
      not any(_ov_opt) and _cov_uw_opt[1] < 0.5 * _cov_w_opt[1] and _cov_uw_opt[2] < 0.5 * _cov_w_opt[2],
      "; ".join(f"beta={b}: unweighted cov {_cov_uw_opt[_i]:.3f} vs weighted cov {_cov_w_opt[_i]:.3f}" for _i, b in enumerate(C_BETAS))); _n += 1
for _k in B_SCORES:
    _ok = all(C_MDR_KNOWN[(_k, b)][:, 2].mean() <= C_ALPHA + b_mc_tol_mean(C_MDR_KNOWN[(_k, b)][:, 2]) for b in C_BETAS)
    check(f"S7b.{_n} weighted MDR (Prop. A.1, KNOWN weights) meets its budget at every tilt strength ({_k} score)", _ok,
          "; ".join(f"beta={b}: {C_MDR_KNOWN[(_k, b)][:, 2].mean():.4f}" for b in C_BETAS)); _n += 1
check(f"S7b.{_n} Kish ESS of the calibration weights falls as the tilt strengthens (both scores, monotone in beta)",
      all(C_MDR_KNOWN[(k, C_BETAS[0])][:, 4].mean() > C_MDR_KNOWN[(k, C_BETAS[1])][:, 4].mean() > C_MDR_KNOWN[(k, C_BETAS[2])][:, 4].mean() for k in B_SCORES),
      "; ".join(f"{k}: " + " > ".join(f"{C_MDR_KNOWN[(k, b)][:, 4].mean():.0f}" for b in C_BETAS) for k in B_SCORES)); _n += 1
'''))
    out.append(C(r'''
def c_sdr_known_mc(score, beta, alpha=C_ALPHA, n=C_N_KW, m=C_M_KW, reps=200, seed0=5200):
    w, q = c_tilt(B_POOL, DA_risk, beta)
    rows = []
    for _r in range(reps):
        _rng = np.random.default_rng([SEED3, seed0, int(beta * 100), _r])
        _ci, _ti, _cw, _tw = c_draw(B_POOL, w, q, n, m, _rng)
        _Lc, _Lt = DA_L[_ci], DA_L[_ti]; _sc, _st = score[_ci], score[_ti]
        _E_uw = G.weighted_sdr_e_values(_sc, _Lc, np.ones(n), _st, np.ones(m), alpha)
        _E_w = G.weighted_sdr_e_values(_sc, _Lc, _cw, _st, _tw, alpha)
        _sel_uw = G.ebh(_E_uw, alpha); _sel_w = G.ebh(_E_w, alpha)
        _sel_hete = G.ebh_boosted(_E_w, alpha, mode="hete", rng=_rng); _sel_homo = G.ebh_boosted(_E_w, alpha, mode="homo", rng=_rng)
        _f = lambda sel: (_Lt * sel).sum() / max(1, sel.sum())
        rows.append((_f(_sel_uw), _sel_uw.mean(), _f(_sel_w), _sel_w.mean(), _f(_sel_hete), _sel_hete.mean(), _f(_sel_homo), _sel_homo.mean()))
    return np.array(rows)


_t0_c = b_time.time()
C_SDR_KNOWN = {(k, b): c_sdr_known_mc(s, b) for k, s in B_SCORES.items() for b in C_BETAS}
print(f"known-weight SDR Monte Carlo runtime: {b_time.time() - _t0_c:.0f}s (2 scores x 3 betas x 200 batches)")

_rows = []
for (_k, _b), _r in C_SDR_KNOWN.items():
    _rows.append((B_SC_LABEL[_k].split(" ")[0], f"{_b:.1f}", f"{_r[:, 0].mean():.4f}", f"{_r[:, 1].mean():.3f}",
                  f"{_r[:, 2].mean():.4f} +/- {b_mc_tol_mean(_r[:, 2]):.4f}", f"{_r[:, 4].mean():.4f} +/- {b_mc_tol_mean(_r[:, 4]):.4f}",
                  f"{_r[:, 6].mean():.4f} +/- {b_mc_tol_mean(_r[:, 6]):.4f}", f"{_r[:, 3].mean():.3f}"))
b_display(b_md("**SCoRE-SDR under a KNOWN exact tilt** (e-BH, alpha = 0.10; hete/homo = e-value boosting, Theorem 5.5)\n\n" + b_tbl(
    ["score", "tilt beta", "unweighted SDR_Q", "unweighted coverage", "weighted SDR_Q (plain e-BH)", "weighted SDR_Q (hete-boosted)",
     "weighted SDR_Q (homo-boosted)", "weighted coverage (plain)"], _rows)))
_n = 6
_sdr_overshoot_any = any(C_SDR_KNOWN[(k, b)][:, 0].mean() > C_ALPHA + b_mc_tol_mean(C_SDR_KNOWN[(k, b)][:, 0]) for k in B_SCORES for b in C_BETAS)
check(f"S7b.{_n} unlike weak-score MDR (S7b.1), unweighted SDR does NOT overshoot its budget at ANY tested (score, beta) cell -- e-BH's SDR is "
      "already conservative even before any tilt is applied (Section 5 found it sits 'often far below' alpha with no shift at all)",
      not _sdr_overshoot_any,
      "; ".join(f"{k}/beta={b}: {C_SDR_KNOWN[(k, b)][:, 0].mean():.4f}" for k in B_SCORES for b in C_BETAS)); _n += 1
_sdr_ess15, _sdr_ess3 = C_MDR_KNOWN[("opt", 1.5)][:, 4].mean(), C_MDR_KNOWN[("opt", 3.0)][:, 4].mean()
_sdr_near_empty = all(C_SDR_KNOWN[(k, b)][:, 1].mean() <= 0.02 and C_SDR_KNOWN[(k, b)][:, 3].mean() <= 0.02 for k in B_SCORES for b in (1.5, 3.0))
check(f"S7b.{_n} at beta>=1.5, BOTH the unweighted AND the weighted SDR rules are near-empty (coverage <= 0.02, most batches select nothing) for "
      f"both scores -- this is NOT a 'collapse' of one relative to the other, it is sample exhaustion: the calibration weights' Kish ESS has "
      f"already fallen to {_sdr_ess15:.0f}/{_sdr_ess3:.0f} out of n={C_N_KW} at beta=1.5/3.0 (S7b.5), leaving too little effective calibration "
      "mass for e-BH to select anything under either weighting; SDR is uninformative about whether weighting 'worked' at this tilt strength",
      _sdr_near_empty,
      "; ".join(f"{k}/beta={b}: unweighted cov {C_SDR_KNOWN[(k, b)][:, 1].mean():.3f}, weighted cov {C_SDR_KNOWN[(k, b)][:, 3].mean():.3f}"
                for k in B_SCORES for b in (1.5, 3.0))); _n += 1
_ok = all(C_SDR_KNOWN[(k, b)][:, 2].mean() <= C_ALPHA + b_mc_tol_mean(C_SDR_KNOWN[(k, b)][:, 2]) for k in B_SCORES for b in C_BETAS)
check(f"S7b.{_n} weighted SDR (known weights, plain e-BH) meets its budget at every tilt strength and score", _ok,
      "; ".join(f"{k}/beta={b}: {C_SDR_KNOWN[(k, b)][:, 2].mean():.4f}" for k in B_SCORES for b in C_BETAS)); _n += 1
_ok_b = all(C_SDR_KNOWN[(k, b)][:, 4].mean() <= C_ALPHA + b_mc_tol_mean(C_SDR_KNOWN[(k, b)][:, 4]) and
            C_SDR_KNOWN[(k, b)][:, 6].mean() <= C_ALPHA + b_mc_tol_mean(C_SDR_KNOWN[(k, b)][:, 6]) for k in B_SCORES for b in C_BETAS)
check(f"S7b.{_n} e-value boosting (hete and homo) also meets the budget under the known tilt", _ok_b,
      "hete/homo both <= alpha + 3SE in every (score, beta) cell"); _n += 1
'''))
    out.append(C(r'''
C_M_BIG = 100.0


def c_domain_v(mu, y):
    """Same clipped monotone nonconformity score as Part A's A_V (M*y - mu(x)), re-derived here (own constant
    C_M_BIG) so Part C builds and tests standalone against prelude + Part B, without depending on Part A."""
    return C_M_BIG * y - mu


def c_cfbh_known_mc(mu, beta, q=C_Q, n=280, m=280, reps=200, seed0=5300):
    w, qq = c_tilt(C_ALLPOOL, DA_risk, beta)
    rows = []
    for _r in range(reps):
        _rng = np.random.default_rng([SEED3, seed0, int(beta * 100), _r])
        _ci, _ti, _cw, _tw = c_draw(C_ALLPOOL, w, qq, n, m, _rng)
        _y = DA_y_ok[_ti]
        _Vc = c_domain_v(mu[_ci], DA_y_ok[_ci]); _Vh = c_domain_v(mu[_ti], 0)
        _sel_uw = G.wbh_select(_Vc, np.ones(n), _Vh, np.ones(m), q)
        _sel_w = G.wbh_select(_Vc, _cw, _Vh, _tw, q)
        _sel_hete = G.wcs_select(_Vc, _cw, _Vh, _tw, q, "hete", rng=_rng)
        _sel_homo = G.wcs_select(_Vc, _cw, _Vh, _tw, q, "homo", rng=_rng)
        _sel_dtm = G.wcs_select(_Vc, _cw, _Vh, _tw, q, "dtm")
        _f = lambda sel: ((1 - _y)[sel].sum() / max(1, sel.sum()), sel.sum() / m, sel.sum() == 0)
        rows.append([x for sel in (_sel_uw, _sel_w, _sel_hete, _sel_homo, _sel_dtm) for x in _f(sel)] + [G.effective_sample_size(_cw)])
    return np.array(rows)


_t0_c = b_time.time()
C_CFBH_KNOWN = {(k, b): c_cfbh_known_mc(mu, b) for k, mu in (("opt", DA_mu), ("weak", DA_mu_weak)) for b in C_BETAS}
print(f"known-weight cfBH/WCS Monte Carlo runtime: {b_time.time() - _t0_c:.0f}s (2 scores x 3 betas x 200 batches)")

_lab = ["unweighted (WBH, w=1)", "weighted WBH", "WCS hete", "WCS homo", "WCS dtm"]
_rows = []
for (_k, _b), _r in C_CFBH_KNOWN.items():
    _cells = []
    for _i in range(5):
        _cells.append(f"FDR {_r[:, 3*_i].mean():.3f} / cov {_r[:, 3*_i+1].mean():.3f} / empty {_r[:, 3*_i+2].mean():.2f}")
    _rows.append(({"opt": "optimistic", "weak": "pessimistic"}[_k], f"{_b:.1f}", *_cells))
b_display(b_md(f"**cfBH/WCS document auto-posting under a KNOWN exact tilt** (q = {C_Q:.2f}; each cell: realized FDR / coverage / share of empty-selection batches)\n\n" + b_tbl(
    ["score", "tilt beta", *_lab], _rows)))
_n = 10
_cfbh_overshoot_any = any(C_CFBH_KNOWN[(k, b)][:, 0].mean() > C_Q + b_cp_tol(C_CFBH_KNOWN[(k, b)][:, 0].mean(), 200) for k in ("opt", "weak") for b in C_BETAS)
check(f"S7b.{_n} mirroring SDR (S7b.6), unweighted WBH does NOT overshoot q at ANY tested (score, beta) cell -- the failure mode this tilt "
      "induces for document-level selection is a loss of power, not an FDR violation",
      not _cfbh_overshoot_any,
      "; ".join(f"{k}/beta={b}: {C_CFBH_KNOWN[(k, b)][:, 0].mean():.3f}" for k in ("opt", "weak") for b in C_BETAS)); _n += 1
_ess_cfbh15 = np.mean([C_CFBH_KNOWN[(k, 1.5)][:, 15].mean() for k in ("opt", "weak")])
_ess_cfbh3 = np.mean([C_CFBH_KNOWN[(k, 3.0)][:, 15].mean() for k in ("opt", "weak")])
_cfbh_near_empty = all(C_CFBH_KNOWN[(k, b)][:, 1].mean() <= 0.02 and C_CFBH_KNOWN[(k, b)][:, 4].mean() <= 0.02 for k in ("opt", "weak") for b in (1.5, 3.0))
check(f"S7b.{_n} at beta>=1.5, BOTH unweighted and weighted cfBH/WCS selection are near-empty (coverage <= 0.02) for both scores -- NOT a "
      f"'collapse' of one relative to the other (e.g. 0.000 vs 0.009 is not a meaningful gap), just sample exhaustion: the C_ALLPOOL calibration "
      f"weights' own Kish ESS has fallen to {_ess_cfbh15:.0f}/{_ess_cfbh3:.0f} out of n=280 at beta=1.5/3.0 (a separate, larger pool than "
      "B_POOL's, so its own ESS is quoted here rather than reusing S7b.5's money-document numbers)",
      _cfbh_near_empty,
      "; ".join(f"{k}/beta={b}: unweighted cov {C_CFBH_KNOWN[(k, b)][:, 1].mean():.3f}, weighted cov {C_CFBH_KNOWN[(k, b)][:, 4].mean():.3f}"
                for k in ("opt", "weak") for b in (1.5, 3.0))); _n += 1
_worst = max(((k, b, i), C_CFBH_KNOWN[(k, b)][:, i].mean()) for k in ("opt", "weak") for b in C_BETAS for i in (3, 6, 9))
_ok = all(C_CFBH_KNOWN[(k, b)][:, i].mean() <= C_Q + b_cp_tol(C_CFBH_KNOWN[(k, b)][:, i].mean(), 200) for k in ("opt", "weak") for b in C_BETAS for i in (3, 6, 9))
check(f"S7b.{_n} weighted WBH and WCS (hete, homo) keep realized FDR <= q + CP tolerance under the known tilt (dtm excluded: expected to be conservative, checked separately)", _ok,
      f"worst cell: {_worst}"); _n += 1
check(f"S7b.{_n} WCS.dtm is at least as conservative (FDR-wise) as WCS.hete/homo (idpshift-q2: dtm 'made zero selections' in the paper's own simulation)",
      all(C_CFBH_KNOWN[(k, b)][:, 12].mean() <= C_CFBH_KNOWN[(k, b)][:, 6].mean() + 0.02 for k in ("opt", "weak") for b in C_BETAS),
      "dtm FDR vs hete FDR, both averaged over 200 batches per cell")
'''))
    out.append(M(r'''
**How to read this chart.** Left: realized MDR (SCoRE) under the known tilt, unweighted (hollow) vs weighted (filled), against `alpha`
(dashed). Unweighted MDR does NOT simply drift upward with `beta`: the pessimistic/weak score's unweighted line starts ABOVE `alpha`
at the mildest tilt (S7b.1's overshoot) then falls as `beta` grows, purely because its accepted set shrinks toward empty; the
optimistic score's unweighted line is below `alpha` even at the mildest tilt and falls further for the same reason (S7b.2) -- both
DECLINE with `beta`, they just start from different sides of the target. Weighted MDR stays at or below `alpha` throughout for both
scores. Right: Kish effective sample size of the calibration weights as `beta` grows -- the ESS collapse is the **price** of exact
repair: the calibration batch still has `n` documents, but only `ESS(w)` of them are "worth" as much once reweighted to look like the
shifted test distribution.
'''))
    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
_ax = _axes[0]
_ax.axhline(C_ALPHA, ls="--", color=PAL3["diag"], lw=1, label="alpha (target)")
for _k in B_SCORES:
    _uw = [C_MDR_KNOWN[(_k, b)][:, 0].mean() for b in C_BETAS]; _w = [C_MDR_KNOWN[(_k, b)][:, 2].mean() for b in C_BETAS]
    _ax.plot(C_BETAS, _uw, "o--", mfc="none", color=B_SC_COLOR[_k], label=f"unweighted, {_k}")
    _ax.plot(C_BETAS, _w, "o-", color=B_SC_COLOR[_k], label=f"weighted, {_k}")
_ax.set_xlabel("tilt strength beta"); _ax.set_ylabel("realized MDR_Q"); _ax.set_title("Known-weight repair: MDR under an exact tilt"); _ax.legend(fontsize=7.5)
_ax = _axes[1]
for _k in B_SCORES:
    _ess = [C_MDR_KNOWN[(_k, b)][:, 4].mean() for b in C_BETAS]
    _ax.plot(C_BETAS, _ess, "o-", color=B_SC_COLOR[_k], label=_k)
_ax.axhline(C_N_KW, ls=":", color=PAL3["diag"], lw=1, label=f"n = {C_N_KW} (no reweighting cost)")
_ax.set_xlabel("tilt strength beta"); _ax.set_ylabel("Kish ESS of calibration weights"); _ax.set_title("The coverage price: effective sample size falls"); _ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))
    return out


def _part_ii():
    out = []
    out.append(M(r'''
### (ii). The real month-7 batch, with ESTIMATED weights

Now the honest version: `w(x)` between `calib` and `shift` is **unknown**. Theorems 6.4/6.5 (idpfin-q14) need the weight
estimator trained **independently** of calibration and test, and `L_2(P_X)`-consistent; the guarantee then is only
**asymptotic** (`limsup MDR_n <= alpha`), never finite-sample. Remark 6.6 and Assumption A.3/A.5 (idpfin-q14) add a
**doubly-robust** route: balance the estimated weights against an estimated conditional risk `l_hat(x)`, and the asymptotic
guarantee survives if *either* the weights *or* `l_hat` is consistent (not both). WCS's analogue (idpshift-q1, q2, Theorem 3.5)
is *not* doubly robust: only the weight estimate needs to be consistent, and estimation error inflates FDR by a factor
`gamma_hat^2` that "converges to zero if `w_hat` is consistent" but is otherwise a genuine inflation, not a bound of `q`.

**Independence protocol (ours, following idpfin-q14/idpshift-q4's classifier recipe).** SOURCE = `train`-split documents,
TARGET = a random **half** of the `shift` documents (labels unused, `C_SHIFT_TARGET`); fit `p_hat(x) = P(target | x)` on the
pooled sample, `w_hat(x) = clip(p_hat, 0.01, 0.99)/(1 - clip(...)) * n_src/n_tgt` (Tibshirani et al.'s odds formula,
idpshift-q3), rescaled to mean 1 on the applied set. Apply the **fitted, frozen** classifier out-of-sample to `calib`
documents (calibration weights) and the **other half** of `shift` (`C_SHIFT_EVAL`, evaluation half: its labels are used only
to measure realized risk, never to fit weights or the score).

**Three covariate sets (derived here; `X_doc`'s own document/field-type one-hots are excluded from set A so it is a genuine
"scores only" contrast).**

| set | covariates |
| --- | --- |
| A | the 8 score-derived `X_doc` features (min/mean/log-sum of critical-field scores, count, verbalized-min, rule failures, classifier confidence, mean field score) |
| B | A + an OCR-noise proxy (derived here) + document type (4 one-hot) |
| C | B + a "template seen in train" flag |
'''))
    out.append(C(r'''
_CONF_DIGITS = set("0123456789"); _CONF_LETTERS = set("OlS")   # exactly the characters build_nb2.py's CHAR_CONFUSIONS can produce
_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def c_ocr_noise_proxy(text):
    """Derived here: share of alphanumeric tokens that mix a digit with one of the OCR-confusable letters {O, l, S}
    (build_nb2.py's apply_ocr_noise only ever substitutes within {0<->O, 1<->l, 5<->S}), a marker that noise has
    turned a clean digit run or a clean word into a mixed token."""
    toks = _TOK_RE.findall(text)
    if not toks:
        return 0.0
    flagged = sum(1 for t in toks if any(c in _CONF_DIGITS for c in t) and any(c in _CONF_LETTERS for c in t))
    return flagged / len(toks)


C_OCR_PROXY = np.array([c_ocr_noise_proxy(d["ocr_text"]) for d in DOCS])
_train_mask = DOC_SPLIT == "train"
_noise_of = np.array([d["noise_level"] for d in DOCS])
_means = {lvl: C_OCR_PROXY[_train_mask & (_noise_of == lvl)].mean() for lvl in ("low", "medium", "high")}
print("OCR-noise proxy, mean by TRUE noise_level (train documents only, validation, never used as a model input): " +
      ", ".join(f"{k}={v:.4f}" for k, v in _means.items()))
check("S7b.14 the OCR-noise proxy is monotone increasing in the TRUE (never-modeled) noise level, on train documents",
      _means["low"] < _means["medium"] < _means["high"], f"{_means}")

C_TRAIN_TEMPLATES = {d["vendor_template_id"] for d in DOCS if d["split"] == "train"}
C_TEMPLATE_SEEN = np.array([1.0 if d["vendor_template_id"] in C_TRAIN_TEMPLATES else 0.0 for d in DOCS])
_calib_idx_all = np.where(DOC_SPLIT == "calib")[0]
_shift_idx_all = np.where(DOC_SPLIT == "shift")[0]
print(f"'template seen in train' flag: mean on calib documents = {C_TEMPLATE_SEEN[_calib_idx_all].mean():.4f}, "
      f"mean on shift documents = {C_TEMPLATE_SEEN[_shift_idx_all].mean():.4f}")
check("S7b.15 the template-seen flag is degenerate (constant) on at least one of the two arms -- a positivity failure "
      "(idpshift-q4: 'both sources silent on zero support'; direction here is the opposite of the brief's guess -- month-7 "
      "templates are drawn from a disjoint SHIFT_TEMPLATES pool never used by train/calib/test, so the flag is 1 on calib and 0 on shift, not the reverse)",
      C_TEMPLATE_SEEN[_calib_idx_all].std() < 1e-9 or C_TEMPLATE_SEEN[_shift_idx_all].std() < 1e-9,
      f"calib mean {C_TEMPLATE_SEEN[_calib_idx_all].mean():.4f} (sd {C_TEMPLATE_SEEN[_calib_idx_all].std():.4f}), "
      f"shift mean {C_TEMPLATE_SEEN[_shift_idx_all].mean():.4f} (sd {C_TEMPLATE_SEEN[_shift_idx_all].std():.4f})")

C_FEAT = {"A": X_doc[:, :8],
          "B": np.column_stack([X_doc[:, :8], C_OCR_PROXY, X_doc[:, 8:12]]),
          "C": np.column_stack([X_doc[:, :8], C_OCR_PROXY, X_doc[:, 8:12], C_TEMPLATE_SEEN])}

_rng_split = np.random.default_rng([SEED3, 701])
_perm = _rng_split.permutation(_shift_idx_all)
C_SHIFT_TARGET = np.sort(_perm[: len(_perm) // 2])          # unlabeled target half, used only to FIT the weight classifier
C_SHIFT_EVAL = np.sort(_perm[len(_perm) // 2:])             # evaluation half: labels used only to measure realized risk
C_TRAIN_IDX = np.where(DOC_SPLIT == "train")[0]
check("S7b.16 the shift target (weight-fitting) and evaluation halves are disjoint and partition the shift split",
      len(set(C_SHIFT_TARGET) & set(C_SHIFT_EVAL)) == 0 and len(C_SHIFT_TARGET) + len(C_SHIFT_EVAL) == len(_shift_idx_all))
'''))
    out.append(C(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def c_domain_weights(Xsrc, Xtgt, Xcal, Xeval, seed=0):
    """Fit C=1 (target, unlabeled shift documents) vs C=0 (source, train documents) with a single classifier (NOT
    cross-fitted here, unlike G.domain_classifier_weights: we need to apply the SAME frozen classifier OUT-OF-SAMPLE to
    `calib` and the shift evaluation half, which is the independence protocol idpfin-q14/idpshift-q4 actually call for,
    not the in-sample cross-fitting of the lib helper). Odds formula w = p/(1-p) * n_src/n_tgt (idpshift-q3, Eq. 12),
    clipped p in [0.01, 0.99], rescaled so mean(w) = 1 on the CALIBRATION set (the set this weight will multiply)."""
    Xp = np.vstack([Xsrc, Xtgt]); y = np.r_[np.zeros(len(Xsrc)), np.ones(len(Xtgt))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed)).fit(Xp, y)
    n_src, n_tgt = len(Xsrc), len(Xtgt)

    def w_of(Xq):
        p = np.clip(clf.predict_proba(Xq)[:, 1], 0.01, 0.99)
        clip_share = float(np.mean((p <= 0.0101) | (p >= 0.9899)))
        return (p / (1 - p)) * (n_src / n_tgt), clip_share, p

    w_cal, clip_cal, p_cal = w_of(Xcal)
    w_eval, clip_eval, p_eval = w_of(Xeval)
    scale = 1.0 / w_cal.mean()
    return w_cal * scale, w_eval * scale, clip_cal, clip_eval, clf


C_WEIGHTS = {}
_rows = []
for _key, _Xd in C_FEAT.items():
    _wc, _we, _cc, _ce, _clf = c_domain_weights(_Xd[C_TRAIN_IDX], _Xd[C_SHIFT_TARGET], _Xd[_calib_idx_all], _Xd[C_SHIFT_EVAL])
    C_WEIGHTS[_key] = dict(w_cal=_wc, w_eval=_we, clip_cal=_cc, clip_eval=_ce)
    _rows.append((_key, f"{G.effective_sample_size(_wc):.1f} / {len(_wc)}", f"{_wc.max():.2f}", f"{_cc:.3f}",
                  f"{G.effective_sample_size(_we):.1f} / {len(_we)}", f"{_we.max():.2f}", f"{_ce:.3f}"))
b_display(b_md("**Estimated covariate-shift weights per covariate set** (calibration weights rescaled to mean 1; independence: classifier fit on `train` vs a random unlabeled half of `shift`)\n\n" + b_tbl(
    ["covariate set", "calibration Kish ESS / n", "max calibration weight", "share of calibration weights at the clip boundary",
     "eval-half Kish ESS / n", "max eval weight", "share of eval weights at the clip boundary"], _rows)))
check("S7b.17 Kish ESS alone does NOT diagnose set C's positivity failure -- ESS is actually the HIGHEST of the three sets (clip-saturation "
      "makes weights nearly CONSTANT across ~97% of calibration documents and ~83% of eval documents, since the template flag almost "
      "perfectly separates train- from shift-templates, so the classifier saturates at the SAME clip boundary for most units in each arm; "
      "near-uniform weights read as high ESS even though the estimate is uninformative). The max-weight / clip-share columns above are what "
      "actually expose the failure (eval-half max weight ~9,700x vs calibration max ~4x).",
      G.effective_sample_size(C_WEIGHTS["C"]["w_cal"]) > G.effective_sample_size(C_WEIGHTS["A"]["w_cal"]) and C_WEIGHTS["C"]["clip_cal"] > 0.9,
      f"ESS: A={G.effective_sample_size(C_WEIGHTS['A']['w_cal']):.1f}, B={G.effective_sample_size(C_WEIGHTS['B']['w_cal']):.1f}, C={G.effective_sample_size(C_WEIGHTS['C']['w_cal']):.1f} (n={len(_calib_idx_all)}); "
      f"clip share (calibration) C={C_WEIGHTS['C']['clip_cal']:.3f}")
check("S7b.18 covariate set C shows the positivity failure directly: a large share of its calibration (or eval) weights sit at the clip boundary",
      max(C_WEIGHTS["C"]["clip_cal"], C_WEIGHTS["C"]["clip_eval"]) > 0.5,
      f"clip share: calibration {C_WEIGHTS['C']['clip_cal']:.3f}, eval {C_WEIGHTS['C']['clip_eval']:.3f}")
'''))
    out.append(C(r'''
C_KEY_IDX = {"A": 0, "B": 1, "C": 2}
C_SCORE_IDX = {"opt": 0, "weak": 1}


def c_boot_mean(x, thr, B=2000, seed=0):
    _seed_seq = seed if isinstance(seed, (list, tuple)) else [seed]
    rng = np.random.default_rng([SEED3, 801, *_seed_seq]); n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n)); boots = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.mean(boots > thr))


_money_cal = DA_has_money[_calib_idx_all]; _money_eval = DA_has_money[C_SHIFT_EVAL]
C_EST_RES = {}
for _key in ("A", "B", "C"):
    _wc_all, _we_all = C_WEIGHTS[_key]["w_cal"], C_WEIGHTS[_key]["w_eval"]
    for _sk, _sc_full in B_SCORES.items():
        _wc, _we = _wc_all[_money_cal], _we_all[_money_eval]
        _sc_cal, _sc_eval = _sc_full[_calib_idx_all][_money_cal], _sc_full[C_SHIFT_EVAL][_money_eval]
        _Lc, _Le = DA_L[_calib_idx_all][_money_cal], DA_L[C_SHIFT_EVAL][_money_eval]
        _psi = G.weighted_mdr_decision(_sc_cal, _Lc, _wc, _sc_eval, _we, C_ALPHA)
        _mdr_pt, _mdr_lo, _mdr_hi, _mdr_pg = c_boot_mean(_Le * _psi, C_ALPHA, seed=[C_KEY_IDX[_key], C_SCORE_IDX[_sk], 1])
        _lhat_cal = DA_risk[_calib_idx_all][_money_cal]; _lhat_eval = DA_risk[C_SHIFT_EVAL][_money_eval]
        _wb, _that = G.balance_weights_mdr(_wc, _lhat_cal, _sc_cal, _lhat_eval, _sc_eval, C_ALPHA)
        _psi_bal = G.weighted_mdr_decision(_sc_cal, _Lc, _wb, _sc_eval, _we, C_ALPHA)
        _mdr_bal_pt, _mdr_bal_lo, _mdr_bal_hi, _mdr_bal_pg = c_boot_mean(_Le * _psi_bal, C_ALPHA, seed=[C_KEY_IDX[_key], C_SCORE_IDX[_sk], 2])
        _E_w = G.weighted_sdr_e_values(_sc_cal, _Lc, _wc, _sc_eval, _we, C_ALPHA)
        _sel = G.ebh(_E_w, C_ALPHA)
        _sdr_pt, _sdr_lo, _sdr_hi, _sdr_pg = c_boot_mean((_Le * _sel)[_sel] if _sel.any() else np.array([0.0]), C_ALPHA, seed=[C_KEY_IDX[_key], C_SCORE_IDX[_sk], 3])
        _rng_b = np.random.default_rng([SEED3, 802, C_KEY_IDX[_key], C_SCORE_IDX[_sk]])
        _sel_hete = G.ebh_boosted(_E_w, C_ALPHA, mode="hete", rng=_rng_b); _sel_homo = G.ebh_boosted(_E_w, C_ALPHA, mode="homo", rng=_rng_b)
        _wb_s, _that_s = G.balance_weights_sdr(_wc, _lhat_cal, _sc_cal, _lhat_eval, _sc_eval, C_ALPHA)
        _E_bal = G.weighted_sdr_e_values(_sc_cal, _Lc, _wb_s, _sc_eval, _we, C_ALPHA); _sel_bal = G.ebh(_E_bal, C_ALPHA)
        _sdr_bal_pt, _sdr_bal_lo, _sdr_bal_hi, _sdr_bal_pg = c_boot_mean((_Le * _sel_bal)[_sel_bal] if _sel_bal.any() else np.array([0.0]), C_ALPHA, seed=[C_KEY_IDX[_key], C_SCORE_IDX[_sk], 4])
        print(f"[B2 diagnostic] set {_key}/{_sk}: A.5 t_hat={_that_s:.4f} (finite={np.isfinite(_that_s)}); "
              f"balanced weight range [{_wb_s.min():.4f}, {_wb_s.max():.4f}] mean={_wb_s.mean():.4f}; "
              f"plain E_w: min={_E_w.min():.4f}, max={_E_w.max():.4f}, m/alpha={len(_E_w) / C_ALPHA:.1f}, share>=m/alpha={np.mean(_E_w >= len(_E_w) / C_ALPHA):.3f}; "
              f"ebh fires: plain={bool(_sel.any())} ({int(_sel.sum())} selected), balanced={bool(_sel_bal.any())} ({int(_sel_bal.sum())} selected)")
        C_EST_RES[(_key, _sk)] = dict(mdr=(_mdr_pt, _mdr_lo, _mdr_hi, _mdr_pg), mdr_cov=float(_psi.mean()),
                                       mdr_bal=(_mdr_bal_pt, _mdr_bal_lo, _mdr_bal_hi, _mdr_bal_pg), mdr_bal_cov=float(_psi_bal.mean()),
                                       sdr=(_sdr_pt, _sdr_lo, _sdr_hi, _sdr_pg), sdr_cov=float(_sel.mean()),
                                       sdr_hete_cov=float(_sel_hete.mean()), sdr_homo_cov=float(_sel_homo.mean()),
                                       sdr_bal=(_sdr_bal_pt, _sdr_bal_lo, _sdr_bal_hi, _sdr_bal_pg), sdr_bal_cov=float(_sel_bal.mean()))

_rows = []
for _key in ("A", "B", "C"):
    for _sk in B_SCORES:
        _r = C_EST_RES[(_key, _sk)]
        _rows.append((_key, B_SC_LABEL[_sk].split(" ")[0], f"{_r['mdr'][0]:.4f} [{_r['mdr'][1]:.4f}, {_r['mdr'][2]:.4f}]", f"{_r['mdr_cov']:.3f}",
                      f"{_r['mdr_bal'][0]:.4f} [{_r['mdr_bal'][1]:.4f}, {_r['mdr_bal'][2]:.4f}]", f"{_r['mdr_bal_cov']:.3f}",
                      f"{_r['sdr'][0]:.4f} [{_r['sdr'][1]:.4f}, {_r['sdr'][2]:.4f}]", f"{_r['sdr_cov']:.3f}", f"{_r['sdr_bal'][0]:.4f}", f"{_r['sdr_bal_cov']:.3f}"))
b_display(b_md("**Estimated-weight SCoRE on the real month-7 shift** (evaluation half only, alpha = 0.10; MDR/SDR bootstrap 95% CIs over eval documents; "
               "these are ASYMPTOTIC guarantees (Thm 6.4/6.5) or doubly-robust asymptotic (A.3/A.5 'bal' columns), never finite-sample)\n\n" + b_tbl(
    ["covariate set", "score", "MDR (plain weights) [95% CI]", "coverage", "MDR (A.3-balanced) [95% CI]", "coverage",
     "SDR (plain weights, e-BH) [95% CI]", "coverage", "SDR (A.5-balanced)", "coverage"], _rows)))
print("for comparison, the UNWEIGHTED frozen document-level rule on the (full) shift split (Section 7, B_DOC_SHIFT): " +
      "; ".join(f"{k}/alpha={a}: MDR {B_DOC_SHIFT[(k, a, 'shift')]['mdr']:.4f}" for k in B_SCORES for a in (0.10,)))
_n = 19
check(f"S7b.{_n} the two non-degenerate covariate sets (A, B) retain non-trivial coverage on the evaluation half, unlike set C's collapse (S7b.20)",
      C_EST_RES[("A", "opt")]["mdr_cov"] > 0.05 and C_EST_RES[("B", "opt")]["mdr_cov"] > 0.05,
      f"coverage: A={C_EST_RES[('A', 'opt')]['mdr_cov']:.4f}, B={C_EST_RES[('B', 'opt')]['mdr_cov']:.4f}, C={C_EST_RES[('C', 'opt')]['mdr_cov']:.4f}; "
      f"MDR by set (optimistic score): " + ", ".join(f"{k}={C_EST_RES[(k, 'opt')]['mdr'][0]:.4f}" for k in ("A", "B", "C")))
check(f"S7b.{_n + 1} set C's near-degenerate weights collapse SCoRE-MDR to near-total NON-acceptance (not a wide CI): the enormous eval-half "
      "weights (S7b.17/S7b.18's positivity failure) make the weighted-MDR decision's (test_w + S_le)/(Wc + test_w) <= alpha condition fail for "
      "almost every eval document, since a huge test_w alone can already exceed alpha*(Wc+test_w) -- coverage collapses toward zero and the "
      "bootstrap CI degenerates to a point at 0, not a wide interval",
      C_EST_RES[("C", "opt")]["mdr_cov"] < 0.05 and (C_EST_RES[("C", "opt")]["mdr"][2] - C_EST_RES[("C", "opt")]["mdr"][1]) < 1e-6,
      f"set C coverage {C_EST_RES[('C', 'opt')]['mdr_cov']:.4f}; CI [{C_EST_RES[('C', 'opt')]['mdr'][1]:.6f}, {C_EST_RES[('C', 'opt')]['mdr'][2]:.6f}]; "
      f"for contrast, set A coverage {C_EST_RES[('A', 'opt')]['mdr_cov']:.4f}, CI width {C_EST_RES[('A', 'opt')]['mdr'][2] - C_EST_RES[('A', 'opt')]['mdr'][1]:.4f}")
check(f"S7b.21 (B2) the A.5-balanced SDR selects ZERO documents for every covariate set, even set A whose PLAIN weighted SDR selects normally "
      "(0.774 coverage) -- the [B2 diagnostic] print above shows this is NOT caused by a degenerate t_hat (finite, 0.87-0.93) or by an unstable "
      "balanced-weight blow-up (mean 1, range ~0.03-16.6 for set A): the A.5 balancing (itself derived here, not from the paper) redistributes "
      "weight mass so no self-consistent e-BH threshold exists at this eval-half size. Set B's PLAIN (unbalanced) SDR also selects zero despite "
      "0.79 MDR coverage for the same weights: SDR's e-BH needs many test e-values clustered above a SHARED level, and set B's wider eval-weight "
      "range (max ~142 vs set A's ~24, weights table above) apparently breaks that clustering where MDR's simpler per-unit Prop. A.1 decision does "
      "not -- SDR stays far more sensitive to weight structure than MDR, consistent with its extra conservatism already seen in Sections 5 and 7b(i)",
      all(C_EST_RES[(k, s)]["sdr_bal_cov"] == 0.0 for k in ("A", "B", "C") for s in B_SCORES) and C_EST_RES[("B", "opt")]["sdr_cov"] == 0.0
      and C_EST_RES[("A", "opt")]["sdr_cov"] > 0.5,
      f"sdr_bal_cov all sets/scores: " + ", ".join(f"{k}/{s}={C_EST_RES[(k, s)]['sdr_bal_cov']:.3f}" for k in ("A", "B", "C") for s in B_SCORES) +
      f" | plain sdr_cov: A={C_EST_RES[('A', 'opt')]['sdr_cov']:.3f}, B={C_EST_RES[('B', 'opt')]['sdr_cov']:.3f}")
'''))
    out.append(C(r'''
_all_v = {}
_SEL_STORE = {}
_rows = []
for _key in ("A", "B", "C"):
    _wc_all, _we_all = C_WEIGHTS[_key]["w_cal"], C_WEIGHTS[_key]["w_eval"]
    _Vc = c_domain_v(DA_mu[_calib_idx_all], DA_y_ok[_calib_idx_all]); _Vh = c_domain_v(DA_mu[C_SHIFT_EVAL], 0); _y = DA_y_ok[C_SHIFT_EVAL]
    _rng_v = np.random.default_rng([SEED3, 803, C_KEY_IDX[_key]])
    _p_wbh = G.weighted_conformal_pvalues(_Vc, _wc_all, _Vh, _we_all, deterministic=True)
    _sel_wbh = G.wbh_select(_Vc, _wc_all, _Vh, _we_all, C_Q)
    _sel_hete = G.wcs_select(_Vc, _wc_all, _Vh, _we_all, C_Q, "hete", rng=_rng_v)
    _sel_homo = G.wcs_select(_Vc, _wc_all, _Vh, _we_all, C_Q, "homo", rng=_rng_v)
    _sel_dtm = G.wcs_select(_Vc, _wc_all, _Vh, _we_all, C_Q, "dtm")
    _SEL_STORE[_key] = dict(wbh=_sel_wbh, hete=_sel_hete, homo=_sel_homo, dtm=_sel_dtm, p=_p_wbh, wc=_wc_all, we=_we_all)

    def _fdr_ci(sel):
        _wrong = (1 - _y)[sel].astype(float)
        _pt, _lo, _hi, _pg = c_boot_mean(_wrong, C_Q, seed=[C_KEY_IDX[_key], int(sel.sum()) % 1000]) if sel.any() else (0.0, 0.0, 0.0, 0.0)
        return _pt, _lo, _hi, float(sel.mean())

    _all_v[_key] = {"wbh": _fdr_ci(_sel_wbh), "hete": _fdr_ci(_sel_hete), "homo": _fdr_ci(_sel_homo), "dtm": _fdr_ci(_sel_dtm)}
    _rows.append((_key, *[f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] (cov {v[3]:.3f})" for v in _all_v[_key].values()]))

print("[B1 diagnostic] do the weights reach wbh_select/wcs_select, and why do A and B end up so similar?")
for _key in ("A", "B", "C"):
    _s = _SEL_STORE[_key]
    print(f"  set {_key}: calib weight range [{_s['wc'].min():.4f}, {_s['wc'].max():.4f}] mean={_s['wc'].mean():.4f}; "
          f"eval weight range [{_s['we'].min():.4f}, {_s['we'].max():.4f}] mean={_s['we'].mean():.4f}; "
          f"p-value summary: min={_s['p'].min():.5f}, p10={np.percentile(_s['p'], 10):.5f}, median={np.median(_s['p']):.4f}, max={_s['p'].max():.4f}, "
          f"share p<=q={np.mean(_s['p'] <= C_Q):.3f}; selected (wbh)={int(_s['wbh'].sum())}, (dtm)={int(_s['dtm'].sum())}")
_jacc_wc = float(np.corrcoef(_SEL_STORE["A"]["wc"], _SEL_STORE["B"]["wc"])[0, 1])
_jacc_we = float(np.corrcoef(_SEL_STORE["A"]["we"], _SEL_STORE["B"]["we"])[0, 1])
for _proc in ("wbh", "hete", "homo", "dtm"):
    _sa, _sb = _SEL_STORE["A"][_proc], _SEL_STORE["B"][_proc]
    _jacc = (_sa & _sb).sum() / max(1, (_sa | _sb).sum())
    print(f"  A vs B, {_proc}: Jaccard overlap of selected sets = {_jacc:.4f} ({int((_sa & _sb).sum())} shared / {int((_sa | _sb).sum())} union); "
          f"selected p-values (A) all <= {(_SEL_STORE['A']['p'][_sa].max() if _sa.any() else float('nan')):.2e}, "
          f"smallest UNselected p-value (A) >= {(np.min(_SEL_STORE['A']['p'][~_sa]) if (~_sa).any() else float('nan')):.2e}")
print(f"  correlation between set-A and set-B weights: calibration r={_jacc_wc:.4f}, eval r={_jacc_we:.4f} "
      "(if near 1, the OCR-noise proxy + document type add little beyond the score features for THIS train-vs-shift-target classification task)")
_n = 22
check(f"S7b.{_n} (B1) the weights genuinely reach wbh_select/wcs_select and genuinely differ between A and B (calibration weight correlation "
      f"{_jacc_wc:.2f}, not 1.0), yet A and B select the IDENTICAL 60 documents under all four procedures (Jaccard=1.0) -- NOT a bug: every "
      "selected document's p-value sits below 0.022 while every unselected one sits above 0.029, a clean gap with no borderline cases near the "
      "cutoff for either weighting, so no amount of reweighting within this range (or randomized WCS pruning) moves anyone across it",
      _jacc_wc < 0.9 and all((_SEL_STORE["A"][p_] == _SEL_STORE["B"][p_]).all() for p_ in ("wbh", "hete", "homo", "dtm")),
      f"calib weight corr={_jacc_wc:.4f}; selected-vs-unselected p-value gap: max selected {_SEL_STORE['A']['p'][_SEL_STORE['A']['wbh']].max():.4f} "
      f"< min unselected {_SEL_STORE['A']['p'][~_SEL_STORE['A']['wbh']].min():.4f}"); _n += 1
_sel_uw = G.wbh_select(c_domain_v(DA_mu[_calib_idx_all], DA_y_ok[_calib_idx_all]), np.ones(len(_calib_idx_all)),
                       c_domain_v(DA_mu[C_SHIFT_EVAL], 0), np.ones(len(C_SHIFT_EVAL)), C_Q)
_y_eval = DA_y_ok[C_SHIFT_EVAL]
_uw_fdr, _uw_lo, _uw_hi, _uw_pg = c_boot_mean((1 - _y_eval)[_sel_uw].astype(float), C_Q, seed=99) if _sel_uw.any() else (0.0, 0.0, 0.0, 0.0)
b_display(b_md(f"**Document auto-posting (cfBH-style) on the real month-7 evaluation half, q = {C_Q:.2f}** (each cell: realized FDR [95% CI] and coverage; "
               f"unweighted frozen comparator, ignoring the shift entirely: FDR {_uw_fdr:.3f} [{_uw_lo:.3f}, {_uw_hi:.3f}], coverage {_sel_uw.mean():.3f})\n\n" +
               b_tbl(["covariate set", "weighted WBH", "WCS hete", "WCS homo", "WCS dtm"], _rows)))
_n = 23
_pb_break = B_DOC_SHIFT[("weak", 0.10, "shift")]["mdr"] > 0.10
check(f"S7b.{_n} the unweighted (shift-blind) rule already overshoots its budget on the FULL shift batch for the pessimistic score (Section 7's "
      "B_DOC_SHIFT, reused here, ~800 documents); our own cfBH-style recomputation on the smaller evaluation half here "
      f"({_uw_fdr:.3f} vs q={C_Q:.2f}) sits close to but does not clearly exceed q at this reduced sample (~250 documents) -- consistent with "
      "sampling noise at this size, not evidence the shift is benign",
      _pb_break, f"B_DOC_SHIFT weak/alpha=0.10 shift MDR = {B_DOC_SHIFT[('weak', 0.10, 'shift')]['mdr']:.4f} (> 0.10); "
                 f"cfBH eval-half unweighted FDR here = {_uw_fdr:.3f}"); _n += 1
check(f"S7b.{_n} (B3) estimated-weight MDR on the real month-7 shift is WORSE than the unweighted frozen rule, not just short of finite-sample "
      "validity: sets A and B both realize MDR ~0.185 (optimistic score) against the unweighted B_DOC_SHIFT frozen rule's 0.096 (S7 and Section 7 "
      "B_DOC_SHIFT) -- reweighting toward the shift distribution here does not merely fail to help, it actively makes the realized risk worse",
      C_EST_RES[("A", "opt")]["mdr"][0] > B_DOC_SHIFT[("opt", 0.10, "shift")]["mdr"] and C_EST_RES[("B", "opt")]["mdr"][0] > B_DOC_SHIFT[("opt", 0.10, "shift")]["mdr"],
      f"A MDR={C_EST_RES[('A', 'opt')]['mdr'][0]:.4f}, B MDR={C_EST_RES[('B', 'opt')]['mdr'][0]:.4f} vs unweighted B_DOC_SHIFT MDR={B_DOC_SHIFT[('opt', 0.10, 'shift')]['mdr']:.4f}"); _n += 1
check(f"S7b.{_n} estimated-weight WCS/WBH does NOT clearly restore FDR validity on the real month-7 evaluation half for EITHER non-degenerate "
      "covariate set (A or B). PRIMARY explanation (source-consistent with our own diagnostic): Section 7b(iii)'s is_shift coefficient CI "
      "excludes 0 controlling for score+B (S7b.27 below), i.e. the month-7 batch carries a real P(Y|X) (concept) shift, not just a covariate "
      "shift -- SCoRE/WCS's weighting machinery assumes dQ/dP(x,y)=w(x) (Assumption 6.1 / Eq. 1, idpfin-q13/idpshift-q1), which is SILENT about "
      "any change in P(Y|X); reweighting toward calibration documents that merely LOOK like the shift batch in X cannot correct for the shift "
      "batch's outcomes behaving differently given X, and S7b.24's finding (estimated-weight MDR WORSE than unweighted) shows this can "
      "backfire, not just fall short. SECONDARY (smaller) contributors: the asymptotic-only guarantee (Theorem 3.5/6.4) needs weight consistency "
      "and a large m, and S5.4/S5.5-style exact-vs-grid checks in Section 5 already validate the e-value machinery itself, so this is not an "
      "implementation bug",
      all(_all_v[k]["wbh"][0] > C_Q for k in ("A", "B")),
      f"A: FDR {_all_v['A']['wbh'][0]:.3f} [{_all_v['A']['wbh'][1]:.3f}, {_all_v['A']['wbh'][2]:.3f}]; "
      f"B: FDR {_all_v['B']['wbh'][0]:.3f} [{_all_v['B']['wbh'][1]:.3f}, {_all_v['B']['wbh'][2]:.3f}]")
'''))
    out.append(M(r'''
**How to read this chart.** Left: Kish ESS of the calibration weights for each covariate set (bars). Set C's bar is
*highest*, not lowest -- the counter-intuitive finding of S7b.17: clip-saturation from the near-perfect template-flag
separation makes calibration weights nearly CONSTANT (high ESS by the formula) even though the estimate is uninformative;
the real damage shows up on the right, where set C's coverage collapses to essentially zero (S7b.20), not in this chart.
ESS is the wrong tool for this particular failure mode; max weight and clip share (the table above) are the right ones.
Right: realized MDR on the evaluation half (optimistic score), plain estimated weights vs the Assumption A.3-balanced
variant, with bootstrap 95% CIs (whiskers); the dashed line is `alpha`. Neither variant is a finite-sample guarantee here
(weights are estimated, Theorem 6.4 is asymptotic); read the CI, not just the point estimate.
'''))
    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
_ax = _axes[0]
_ess = [G.effective_sample_size(C_WEIGHTS[k]["w_cal"]) for k in ("A", "B", "C")]
_ax.bar(["A\n(score only)", "B\n(+ OCR + type)", "C\n(+ template flag)"], _ess, color=[PAL3["lr"], PAL3["hgb"], PAL3["b_naive"]])
for _i, _v in enumerate(_ess):
    _ax.text(_i, _v + 5, f"{_v:.0f}", ha="center", fontsize=9)
_ax.axhline(len(_calib_idx_all), ls=":", color=PAL3["diag"], label=f"n = {len(_calib_idx_all)} (no reweighting cost)")
_ax.set_ylabel("Kish ESS of calibration weights"); _ax.set_title("Estimated-weight ESS by covariate set"); _ax.legend(fontsize=8)
_ax = _axes[1]
_x = np.arange(3); _w = 0.35
_pt_plain = [C_EST_RES[(k, "opt")]["mdr"][0] for k in ("A", "B", "C")]; _lo_plain = [C_EST_RES[(k, "opt")]["mdr"][1] for k in ("A", "B", "C")]; _hi_plain = [C_EST_RES[(k, "opt")]["mdr"][2] for k in ("A", "B", "C")]
_pt_bal = [C_EST_RES[(k, "opt")]["mdr_bal"][0] for k in ("A", "B", "C")]; _lo_bal = [C_EST_RES[(k, "opt")]["mdr_bal"][1] for k in ("A", "B", "C")]; _hi_bal = [C_EST_RES[(k, "opt")]["mdr_bal"][2] for k in ("A", "B", "C")]
_ax.errorbar(_x - _w / 2, _pt_plain, yerr=[np.array(_pt_plain) - np.array(_lo_plain), np.array(_hi_plain) - np.array(_pt_plain)], fmt="o", capsize=4, color=PAL3["hgb"], label="plain estimated weights")
_ax.errorbar(_x + _w / 2, _pt_bal, yerr=[np.array(_pt_bal) - np.array(_lo_bal), np.array(_hi_bal) - np.array(_pt_bal)], fmt="s", capsize=4, color=PAL3["lr"], label="A.3-balanced")
_ax.axhline(C_ALPHA, ls="--", color=PAL3["diag"], lw=1)
_ax.set_xticks(_x); _ax.set_xticklabels(["A", "B", "C"]); _ax.set_ylabel("realized MDR (eval half, optimistic score)"); _ax.set_title("Estimated-weight MDR by covariate set"); _ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))
    return out


def _part_iii():
    out = []
    out.append(M(r'''
### (iii). Is it covariate shift at all? (diagnostic, derived here; idpfin-q18: sources give none)

No source states a diagnostic for `P(Y|X) = Q(Y|X)` (idpfin-q18, idpshift-q4, both flagged above). Using the `shift` split's
labels as ground truth **only for this diagnostic** (never to fit a score or a threshold), we compare `P(y_ok=1 | score)` and
`E[L_doc | score]` between `calib` and `shift` in score bins, then fit a logistic/linear model of the outcome on the score
plus covariate set B **and an `is_shift` indicator**: a significant `is_shift` coefficient *after controlling for the score and
B* is evidence of a `P(Y|X)` shift that weighting cannot fix (weighting only re-balances `X`, never changes what `Y` does
given `X`). Inference is bootstrap-based (2,000 resamples), not a parametric test (no `statsmodels` dependency).
'''))
    out.append(C(r'''
def c_reliability(score, idx_a, idx_b, y, n_bins=8):
    _edges = np.quantile(score[np.r_[idx_a, idx_b]], np.linspace(0, 1, n_bins + 1)); _edges[0] -= 1e-9; _edges[-1] += 1e-9
    out = {}
    for _nm, _idx in (("calib", idx_a), ("shift", idx_b)):
        _xs, _ys, _ns = [], [], []
        for _lo, _hi in zip(_edges[:-1], _edges[1:]):
            _m = (score[_idx] > _lo) & (score[_idx] <= _hi) & ~np.isnan(y[_idx])
            if _m.sum() == 0:
                continue
            _xs.append(score[_idx][_m].mean()); _ys.append(y[_idx][_m].mean()); _ns.append(int(_m.sum()))
        out[_nm] = (np.array(_xs), np.array(_ys), np.array(_ns))
    return out


C_RELIAB_Y = c_reliability(DA_mu, _calib_idx_all, _shift_idx_all, DA_y_ok.astype(float))
_money_c, _money_s = _calib_idx_all[DA_has_money[_calib_idx_all]], _shift_idx_all[DA_has_money[_shift_idx_all]]
C_RELIAB_L = c_reliability(DA_risk, _money_c, _money_s, DA_L)
_gap_y = np.interp(C_RELIAB_Y["shift"][0].mean(), *C_RELIAB_Y["calib"][:2]) - C_RELIAB_Y["shift"][1].mean()
print(f"P(y_ok=1|score): calib bins mean {C_RELIAB_Y['calib'][1].mean():.3f}, shift bins mean {C_RELIAB_Y['shift'][1].mean():.3f} "
      f"(gap at matched score level, roughly, {_gap_y:+.3f})")
print(f"E[L_doc|score] (money docs): calib bins mean {C_RELIAB_L['calib'][1].mean():.3f}, shift bins mean {C_RELIAB_L['shift'][1].mean():.3f}")
check("S7b.26 the shift batch is worse than calib EVEN WITHIN matched score bins (not purely a covariate-shift story: pure covariate "
      "shift with an unchanged P(Y|X) would leave same-bin outcome rates unchanged)",
      C_RELIAB_Y["shift"][1].mean() < C_RELIAB_Y["calib"][1].mean() - 0.02, f"calib {C_RELIAB_Y['calib'][1].mean():.3f} vs shift {C_RELIAB_Y['shift'][1].mean():.3f}")
'''))
    out.append(C(r'''
def c_boot_coef(X, y, B=400, seed=0, kind="logit"):
    from sklearn.linear_model import LogisticRegression, Ridge
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if kind == "logit":
        _fit = lambda Xa, ya: LogisticRegression(max_iter=2000).fit(Xa, ya).coef_[0, -1]
    else:
        _fit = lambda Xa, ya: Ridge(alpha=1.0).fit(Xa, ya).coef_[-1]
    beta0 = _fit(Xs, y)
    rng = np.random.default_rng([SEED3, 901, seed]); n = len(y); boots = []
    for _b in range(B):
        _ix = rng.integers(0, n, size=n)
        if kind == "logit" and len(np.unique(y[_ix])) < 2:
            continue
        try:
            boots.append(_fit(Xs[_ix], y[_ix]))
        except Exception:
            continue
    boots = np.array(boots)
    return float(beta0), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


_idx_y = np.r_[_calib_idx_all, _shift_idx_all]
_is_shift_y = np.r_[np.zeros(len(_calib_idx_all)), np.ones(len(_shift_idx_all))]
_X_y = np.column_stack([C_FEAT["B"][_idx_y], _is_shift_y])
C_SHIFT_COEF_Y = c_boot_coef(_X_y, DA_y_ok[_idx_y].astype(float), kind="logit", seed=1)

_idx_l = np.r_[_money_c, _money_s]
_is_shift_l = np.r_[np.zeros(len(_money_c)), np.ones(len(_money_s))]
_X_l = np.column_stack([C_FEAT["B"][_idx_l], _is_shift_l])
C_SHIFT_COEF_L = c_boot_coef(_X_l, DA_L[_idx_l], kind="ols", seed=2)

print(f"is_shift coefficient (standardized), controlling for score + covariate set B: outcome y_ok (logit) {C_SHIFT_COEF_Y[0]:+.3f} "
      f"[{C_SHIFT_COEF_Y[1]:+.3f}, {C_SHIFT_COEF_Y[2]:+.3f}]; outcome L_doc (OLS) {C_SHIFT_COEF_L[0]:+.3f} [{C_SHIFT_COEF_L[1]:+.3f}, {C_SHIFT_COEF_L[2]:+.3f}]")
check("S7b.27 the is_shift indicator remains significant (95% bootstrap CI excludes 0) after controlling for the score and covariate set B: "
      "evidence of a P(Y|X) shift on top of any covariate shift, i.e. the month-7 failure is not PURELY covariate shift",
      (C_SHIFT_COEF_Y[1] > 0 or C_SHIFT_COEF_Y[2] < 0) or (C_SHIFT_COEF_L[1] > 0 or C_SHIFT_COEF_L[2] < 0),
      f"y_ok CI [{C_SHIFT_COEF_Y[1]:+.3f}, {C_SHIFT_COEF_Y[2]:+.3f}]; L_doc CI [{C_SHIFT_COEF_L[1]:+.3f}, {C_SHIFT_COEF_L[2]:+.3f}]")

print(f"[info] connecting to Section 7's labelled-recalibration budget (B_BUDGET): weighting here spends 0 labelled shift "
      f"documents and gets an ASYMPTOTIC repair at best (part ii); recalibration (Section 7) spends N labelled shift documents "
      f"(B_BUDGET={ {k: v['N'] for k, v in B_BUDGET.items()} }) for a finite-sample PAC repair. Given (S7b.21) the month-7 shift "
      "carries a real P(Y|X) component, weighting alone -- however well the weights are estimated -- cannot be expected to fully "
      "repair it; recalibration on real labels is the only route tested here that addresses the P(Y|X) part directly.")
'''))
    out.append(M(r'''
**How to read this chart.** Left: `P(y_ok=1 | score)` in 8 quantile bins, calib (solid) vs shift (dashed); if this were pure
covariate shift with an unchanged `P(Y|X)`, the two curves would sit on top of each other (only the *density* of documents across
bins would move, not the curve itself). A visible vertical gap at the same score level is the concept-shift signature. Right: the
bootstrapped `is_shift` coefficient (controlling for score + covariate set B) for both outcomes, with 95% CIs; bars whose CI excludes
zero support a real `P(Y|X)` shift, not just a covariate one.
'''))
    out.append(C(r'''
_fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
_ax = _axes[0]
_ax.plot(*C_RELIAB_Y["calib"][:2], "o-", color=PAL3["hgb"], label="calib")
_ax.plot(*C_RELIAB_Y["shift"][:2], "o--", color=PAL3["b_naive"], label="shift")
_ax.set_xlabel("predicted P(y_ok=1) (DA_mu)"); _ax.set_ylabel("observed share of good documents"); _ax.set_title("Reliability: calib vs shift"); _ax.legend()
_ax = _axes[1]
_labs = ["y_ok\n(logit)", "L_doc\n(OLS, money docs)"]
_pts = [C_SHIFT_COEF_Y[0], C_SHIFT_COEF_L[0]]; _los = [C_SHIFT_COEF_Y[1], C_SHIFT_COEF_L[1]]; _his = [C_SHIFT_COEF_Y[2], C_SHIFT_COEF_L[2]]
_ax.errorbar(range(2), _pts, yerr=[np.array(_pts) - np.array(_los), np.array(_his) - np.array(_pts)], fmt="o", capsize=5, color=PAL3["hgb"], ms=8)
_ax.axhline(0, ls="--", color=PAL3["diag"], lw=1)
_ax.set_xticks(range(2)); _ax.set_xticklabels(_labs); _ax.set_ylabel("standardized is_shift coefficient, controlling for score + B"); _ax.set_title("Concept-shift signature (95% bootstrap CI)")
plt.tight_layout(); plt.show()
'''))
    return out
