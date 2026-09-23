"""Notebook 3, PART A (owned by the part-A agent): section 3 (why the folklore / add-one rule breaks on OUR documents) and
section 4 (cfBH at the document unit).  Runs in ONE kernel with the prelude and part B, so EVERY global defined here carries
an A_ / a_ prefix (loop temporaries start with an underscore)."""
from nb3_common import M, C


def cells():
    out = []

    # =====================================================================================================================
    # SECTION 3
    # =====================================================================================================================
    out.append(M(r'''
## 3. Why the folklore / add-one rule breaks on OUR documents (P3 section 4, reproduced *qualitatively* on the Notebook-2 data)

**What Gurram's paper says** (re-read from `idpfin-q7`, verbatim). The add-one rule:

> "The add-one rule picks the smallest threshold $\tau$ whose smoothed empirical selective risk on calibration is $\le \alpha$:"
> $$\tau = \min \left\{ t : \frac{1 + \#\{i: c_i \ge t, \text{err}_i\}}{1 + \#\{i: c_i \ge t\}} \le \alpha \right\}, \quad \tau = \infty \text{ (review everything) if none qualifies.}$$ (idpfin-q7)

and the three diagnosed failure modes (Table 3 of the paper, "each pinned by a counterfactual", idpfin-q7):

1. **Document clustering.** "At the add-one threshold the estimated design effect is 2.15 (CORD), 1.84 (FUNSD), 2.04 (XFUND-de) -- up to 2.45 at other thresholds -- so the effective calibration sample is roughly half its nominal size." (idpfin-q7)
2. **Score-refit leakage.** "Fitting a high-capacity score and its threshold on the same calibration fields transfers the score's optimism into the threshold. [...] A 5-parameter logistic fusion barely overfits (risk 0.105)..." (idpfin-q7)
3. **Tie-mass pathology.** "A threshold accepts a tie mass whole or not at all: the smallest reachable candidate accepted $n=245$ fields at empirical risk $0.114 > \alpha$, so no certificate existed at any confidence level." (idpfin-q7)

And the repair for tiers 1-2, the fit/val protocol: "Fit the score model and every data-dependent transform [...] on the fit half only; compute the add-one threshold and all Mondrian bin edges on the untouched val half; never touch test. This restores score-threshold independence -- it does not restore exchangeability, so document clustering remains and tiers 1-2 stay marginal, on-average guarantees." (idpfin-q7)

**What we do here.** Apply the add-one rule (`add_one_threshold`, from the prelude) to the Notebook-2 pipeline at $\alpha = 0.10$ with the
**optimistic** score (`FA_p_hgb`) and the **pessimistic** score (`FA_p_weak`), on **fields** (all fields of the calibration documents), and see
which of the three failure modes are visible on *our* synthetic data. Rules of engagement: paper numbers appear only in the quotes above;
every number printed below is ours; where an effect is *not* visible we say so, and no threshold below was adjusted after looking at results.
The folklore rule of Notebook 2 (verbalized confidence $\ge 0.9$ and no rule failure) is a fixed, un-calibrated cousin of this rule and is
compared in section 4.

**Vocabulary.** The add-one rule is a **tier-1 expectation** procedure (Table 4 of the paper: "tiers 1-2 control expected selective risk; only tiers 3-4 certify", idpfin-q8). Nothing in this section is a certificate.
'''))

    out.append(M(r'''
### 3a. Failure mode 1 -- document clustering: 40 (and 200) document-level resplits versus a field-level-split control

Protocol (each split): take `resplit(seed)` (a 50/50 **document** split of the calib+test pool, stratified by document type); calibrate the add-one threshold on **all fields of the
calibration documents**; evaluate on **all fields of the test documents**. Reported per score: mean coverage (fraction of test fields accepted), mean achieved
selective risk (error rate among accepted test fields), the fraction of splits whose risk exceeds $\alpha$ ("violation"), and the count of zero-coverage splits.

**Control.** The same pool of fields split at random **by field** (stratified by document type too, so only the *clustering* differs), seeds paired with the document splits.
If document clustering matters, the document splits should be more variable than the field control at the same mean; and the **design effect**
(Kish; Notebook 2's `design_effect`) on the pool and on the accepted set says how much smaller the effective sample is than the nominal one:
$\text{deff} = 1 + (\bar n - 1)\rho$, $n_{\text{eff}} = n / \text{deff}$.

**Zero-coverage convention (disclosed).** When the rule returns $\tau=\infty$ (nothing accepted) the split's risk is set to **0**, so the plain mean risk is a *zero-filled mean*.
The paper's Table 4 does the same and says so: "0.020 is the zero-filled mean -- conditional on certifying anything, achieved risk is 0.038. Report both." (idpfin-q8).
We therefore also print the mean risk **conditional on non-zero coverage**. (The exact phrase "zero-coverage splits contribute risk 0" is not in the traces; the footnote above is the closest wording.)
'''))

    out.append(C(r'''
# ---- Part A shared helpers (all A_-prefixed) --------------------------------------------------------------------------
A_ALPHA = 0.10
A_ERR = (1 - FA_correct).astype(float)                       # 1 = the field is wrong
A_POOL_FIELDS = np.where(np.isin(DOC_SPLIT, ["calib", "test"])[FA_doc])[0]   # every field of every calib+test document
A_FTYPE = DOC_TYPE_IDX[FA_doc]
A_SCORES = {"opt": FA_p_hgb, "weak": FA_p_weak}
A_LABEL = {"opt": "optimistic (learned HGB fusion)", "weak": "pessimistic (weak-signal LR)"}
A_WEAK_COLS = [0, 4, 5, 6, 7, 8, 9]                          # same columns as the prelude's weak-signal score
PAL3.update({"a_opt": "#d62728", "a_weak": "#ff7f0e", "a_doc": "#1f77b4", "a_field": "#7f7f7f", "a_ok": "#2ca02c", "a_bad": "#9467bd", "a_folk": "#8c564b"})


def A_cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound for a proportion k/n."""
    return 1.0 if k >= n else float(stats.beta.ppf(conf, k + 1, n - k))


def A_cp_lower(k, n, conf=0.95):
    return 0.0 if k <= 0 else float(stats.beta.ppf(1 - conf, k, n - k + 1))


def A_deff(doc_ids, err):
    """Vectorised one-way-ANOVA (Kish) design effect, identical formula to Notebook 2's design_effect; returns (deff, icc, avg_cluster_size)."""
    _u, _inv = np.unique(doc_ids, return_inverse=True)
    _ns = np.bincount(_inv).astype(float)
    _means = np.bincount(_inv, weights=err) / _ns
    _n, _k = _ns.sum(), len(_ns)
    if _k < 2 or _n <= _k:
        return 1.0, 0.0, _n / max(_k, 1)
    _gm = err.sum() / _n
    _n0 = (_n - (_ns ** 2).sum() / _n) / (_k - 1)
    _msb = (_ns * (_means - _gm) ** 2).sum() / (_k - 1)
    _msw = ((err - _means[_inv]) ** 2).sum() / (_n - _k)
    _den = _msb + (_n0 - 1) * _msw
    _icc = max(0.0, (_msb - _msw) / _den) if _den > 0 else 0.0
    return max(1.0, 1 + (_n / _k - 1) * _icc), _icc, _n / _k


def A_pipe(header, rows):
    """Print a Markdown pipe table."""
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for _r in rows:
        print("| " + " | ".join(str(_x) for _x in _r) + " |")


def A_doc_split_fields(seed):
    _c, _t = resplit(seed)
    return np.where(np.isin(FA_doc, _c))[0], np.where(np.isin(FA_doc, _t))[0]


def A_field_split_fields(seed):
    """Control: random 50/50 split of the SAME pool of fields, stratified by document type, ignoring which document a field belongs to."""
    _rng = np.random.default_rng([SEED3, 777, int(seed)])
    _ic, _it = [], []
    for _j in range(4):
        _ids = A_POOL_FIELDS[A_FTYPE[A_POOL_FIELDS] == _j].copy()
        _rng.shuffle(_ids)
        _h = int(round(len(_ids) * 0.5))
        _ic += list(_ids[:_h]); _it += list(_ids[_h:])
    return np.array(sorted(_ic)), np.array(sorted(_it))


def A_addone_eval(score_cal, ic, score_test, it, alpha=A_ALPHA):
    """Add-one threshold on calibration fields ic (scores score_cal), evaluated on test fields it (scores score_test)."""
    _tau = add_one_threshold(score_cal, A_ERR[ic], alpha)
    _acc = score_test >= _tau
    _na = int(_acc.sum())
    _ac = score_cal >= _tau
    return dict(tau=_tau, n_acc=_na, cov=_na / len(it), risk=float(A_ERR[it][_acc].mean()) if _na else 0.0, zero=(_na == 0),
                cal_risk=float(A_ERR[ic][_ac].mean()) if _ac.any() else 0.0, cal_cov=float(_ac.mean()),
                acc_idx=it[_acc])


def A_arr(rows, key):
    return np.array([_r[key] for _r in rows], dtype=float)


A_N_LONG, A_N_SHORT = 200, 40
A_res3a = {}
_t0 = time.time()
for _k, _sc in A_SCORES.items():
    for _mode, _fn in (("doc", A_doc_split_fields), ("field", A_field_split_fields)):
        _rows = []
        for _seed in range(A_N_LONG):
            _ic, _it = _fn(_seed)
            _r = A_addone_eval(_sc[_ic], _ic, _sc[_it], _it)
            # z-statistic: (test risk - calib risk) on the accepted sets, scaled by the plain binomial standard error
            _at = _r["acc_idx"]
            _ac = _ic[_sc[_ic] >= _r["tau"]]
            if len(_ac) and len(_at):
                _p = (A_ERR[_ac].sum() + A_ERR[_at].sum()) / (len(_ac) + len(_at))
                _r["z"] = (A_ERR[_at].mean() - A_ERR[_ac].mean()) / np.sqrt(_p * (1 - _p) * (1 / len(_ac) + 1 / len(_at)))
            else:
                _r["z"] = np.nan
            if _mode == "doc":
                _r["deff_pool"], _r["icc_pool"], _r["nbar_pool"] = A_deff(FA_doc[_it], A_ERR[_it])
                _r["deff_acc"], _r["icc_acc"], _r["nbar_acc"] = A_deff(FA_doc[_at], A_ERR[_at]) if len(_at) > 1 else (1.0, 0.0, 1.0)
            _rows.append(_r)
        A_res3a[(_k, _mode)] = _rows
print(f"add-one runs: 2 scores x 2 split types x {A_N_LONG} splits in {time.time() - _t0:.1f}s\n")


def A_summ(rows, n):
    _r = rows[:n]
    _risk, _cov, _zero = A_arr(_r, "risk"), A_arr(_r, "cov"), A_arr(_r, "zero").astype(bool)
    _nz = ~_zero
    return dict(n=n, cov=_cov.mean(), cov_sd=_cov.std(ddof=1), risk=_risk.mean(), risk_sd=_risk.std(ddof=1),
                risk_nz=_risk[_nz].mean() if _nz.any() else float("nan"), viol=float((_risk > A_ALPHA).mean()), nviol=int((_risk > A_ALPHA).sum()),
                zero=int(_zero.sum()))


A_tab3a = {(k, m, n): A_summ(A_res3a[(k, m)], n) for k in A_SCORES for m in ("doc", "field") for n in (A_N_SHORT, A_N_LONG)}
_rows = []
for _k in A_SCORES:
    for _m in ("doc", "field"):
        for _n in (A_N_SHORT, A_N_LONG):
            _s = A_tab3a[(_k, _m, _n)]
            _rows.append([A_LABEL[_k].split(" (")[0], "by DOCUMENT" if _m == "doc" else "by field (control)", _n, f"{_s['cov']:.3f}", f"{_s['risk']:.4f}", f"{_s['risk_sd']:.4f}",
                          f"{_s['risk_nz']:.4f}", f"{_s['viol']:.3f} ({_s['nviol']}/{_n})", _s["zero"]])
print("Add-one rule at alpha=0.10, calibrate on calib fields, evaluate on test fields (zero-filled mean risk, and mean risk conditional on non-zero coverage):")
A_pipe(["score", "split", "#splits", "mean coverage", "mean risk (zero-filled)", "sd of risk", "mean risk given coverage>0", "violation frac (risk>alpha)", "zero-coverage splits"], _rows)

# ---- design effects and the Kish relation ---------------------------------------------------------------------------------
print("\nDesign effects across the first 40 document splits (Kish: deff = 1 + (nbar-1)*icc, n_eff = n/deff):")
_rows, A_deff3a = [], {}
for _k in A_SCORES:
    _r = A_res3a[(_k, "doc")][:A_N_SHORT]
    _dp, _da = A_arr(_r, "deff_pool"), A_arr(_r, "deff_acc")
    A_deff3a[_k] = dict(deff_pool=_dp.mean(), deff_acc=_da.mean(), icc_acc=A_arr(_r, "icc_acc").mean(), nbar_acc=A_arr(_r, "nbar_acc").mean(),
                        n_pool=np.mean([len(_x["acc_idx"]) / max(_x["cov"], 1e-9) for _x in _r]), n_acc=A_arr(_r, "n_acc").mean())
    _d = A_deff3a[_k]
    _rows.append([A_LABEL[_k].split(" (")[0], f"{_d['deff_pool']:.2f}", f"{_d['n_pool']:.0f} -> {_d['n_pool'] / _d['deff_pool']:.0f}", f"{_d['deff_acc']:.2f}", f"{_d['icc_acc']:.3f}",
                  f"{_d['nbar_acc']:.2f}", f"{1 + (_d['nbar_acc'] - 1) * _d['icc_acc']:.2f}", f"{_d['n_acc']:.0f} -> {_d['n_acc'] / _d['deff_acc']:.0f}"])
A_pipe(["score", "deff, pool of test fields", "n -> n_eff (pool)", "deff, accepted set", "icc (accepted)", "mean accepted fields per accepted doc", "1+(nbar-1)*icc", "n -> n_eff (accepted)"], _rows)

# does clustering show up in split-to-split variability?  z = (test risk - calib risk) / plain binomial SE:  sd(z) ~ 1 under field-iid, ~ sqrt(deff) under clustering
print("\nSplit-to-split variability of the calib-vs-test risk gap (z, 200 splits; ~1.0 if fields were exchangeable, inflated by clustering; Kish ceiling sqrt(deff_accepted) shown):")
_rows = []
A_z3a = {}
for _k in A_SCORES:
    _zd, _zf = A_arr(A_res3a[(_k, "doc")], "z"), A_arr(A_res3a[(_k, "field")], "z")
    _zd, _zf = _zd[~np.isnan(_zd)], _zf[~np.isnan(_zf)]
    A_z3a[_k] = (_zd.std(ddof=1), _zf.std(ddof=1))
    _rd, _rf = A_arr(A_res3a[(_k, "doc")], "risk"), A_arr(A_res3a[(_k, "field")], "risk")
    _rows.append([A_LABEL[_k].split(" (")[0], f"{_zd.std(ddof=1):.2f}", f"{_zf.std(ddof=1):.2f}", f"{_zd.std(ddof=1) / _zf.std(ddof=1):.2f}", f"{np.sqrt(A_deff3a[_k]['deff_acc']):.2f}",
                  f"{_rd.std(ddof=1):.4f} vs {_rf.std(ddof=1):.4f}"])
A_pipe(["score", "sd(z) by document", "sd(z) by field (control)", "ratio", "Kish ceiling sqrt(deff_acc)", "sd of risk: document vs field"], _rows)

# ---- naive binomial CI versus document-cluster bootstrap CI on the accepted set -------------------------------------------
def A_cluster_boot_ci(idx, B=400, seed=0):
    _u, _inv = np.unique(FA_doc[idx], return_inverse=True)
    _n = np.bincount(_inv).astype(float)
    _e = np.bincount(_inv, weights=A_ERR[idx])
    _rng = np.random.default_rng([SEED3, 31337, seed])
    _cnt = _rng.multinomial(len(_u), np.full(len(_u), 1 / len(_u)), size=B).astype(float)
    _rk = (_cnt @ _e) / (_cnt @ _n)
    return float(np.quantile(_rk, 0.025)), float(np.quantile(_rk, 0.975))


print("\nNaive Wilson 95% CI (treats fields as independent) vs document-cluster bootstrap 95% CI for the ACCEPTED-set risk, mean over the first 20 splits:")
_rows, A_ci3a = [], {}
for _k in A_SCORES:
    _wr = []
    for _s in range(20):
        _a = A_res3a[(_k, "doc")][_s]["acc_idx"]
        _w = wilson_ci(int(A_ERR[_a].sum()), len(_a)); _b = A_cluster_boot_ci(_a, seed=_s)
        _wr.append(((_b[1] - _b[0]) / (_w[1] - _w[0]), _w[1] - _w[0], _b[1] - _b[0]))
    _wr = np.array(_wr)
    A_ci3a[_k] = _wr.mean(axis=0)
    _rows.append([A_LABEL[_k].split(" (")[0], f"{_wr[:, 1].mean():.4f}", f"{_wr[:, 2].mean():.4f}", f"{_wr[:, 0].mean():.2f}", f"{np.sqrt(A_deff3a[_k]['deff_acc']):.2f}"])
A_pipe(["score", "Wilson width (naive)", "cluster-bootstrap width", "width ratio", "sqrt(deff_accepted)"], _rows)

# ---- cross-check of the vectorised design effect against Notebook 2's design_effect -----------------------------------------
_a0 = A_res3a[("opt", "doc")][0]["acc_idx"]
_recs = [{"doc_id": FIELDS[_i]["doc_id"], "error": int(A_ERR[_i])} for _i in _a0]
_de_nb2, _icc_nb2 = design_effect(_recs)
_de_mine, _icc_mine, _ = A_deff(FA_doc[_a0], A_ERR[_a0])

# ---- self-checks ----------------------------------------------------------------------------------------------------------
print()
for _k in A_SCORES:
    _s = A_tab3a[(_k, "doc", A_N_LONG)]
    check(f"S3.1[{_k}] add-one on fields, document resplits: mean risk within 0.01 (=10% of alpha) of alpha=0.10",
          abs(_s["risk"] - A_ALPHA) <= 0.01, f"mean risk {_s['risk']:.4f}, coverage {_s['cov']:.3f}, {A_N_LONG} splits")
    _s = A_tab3a[(_k, "field", A_N_LONG)]
    check(f"S3.2[{_k}] the field-level-split control also lands within 0.01 of alpha (so the rule is fine ON AVERAGE either way)",
          abs(_s["risk"] - A_ALPHA) <= 0.01, f"mean risk {_s['risk']:.4f}")
    _nv, _n = A_tab3a[(_k, "doc", A_N_LONG)]["nviol"], A_N_LONG
    check(f"S3.3[{_k}] an expectation guarantee is NOT a per-batch guarantee: one-sided 95% Clopper-Pearson LOWER bound on the violation fraction > 0.25",
          A_cp_lower(_nv, _n) > 0.25, f"violation {_nv}/{_n} = {_nv / _n:.3f}, CP lower {A_cp_lower(_nv, _n):.3f} (resplits share one pool, so the bound is optimistic)")
    _d = A_deff3a[_k]
    check(f"S3.4[{_k}] errors are materially clustered by document: mean deff > 1.1 on the pool AND on the accepted set (40 splits)",
          _d["deff_pool"] > 1.1 and _d["deff_acc"] > 1.1, f"deff pool {_d['deff_pool']:.2f}, accepted {_d['deff_acc']:.2f}")
    check(f"S3.5[{_k}] a naive binomial (Wilson) CI on the accepted set is too narrow: cluster-bootstrap CI is wider",
          A_ci3a[_k][0] > 1.0, f"width ratio {A_ci3a[_k][0]:.2f} (Kish sqrt(deff) = {np.sqrt(A_deff3a[_k]['deff_acc']):.2f})")
for _k in A_SCORES:
    check(f"S3.5b[{_k}] Kish relation for intervals: cluster-bootstrap / Wilson width ratio is within 15% of sqrt(deff_accepted)",
          abs(A_ci3a[_k][0] / np.sqrt(A_deff3a[_k]["deff_acc"]) - 1) <= 0.15, f"ratio {A_ci3a[_k][0]:.2f} vs sqrt(deff) {np.sqrt(A_deff3a[_k]['deff_acc']):.2f}")
check("S3.6 the vectorised A_deff equals Notebook 2's design_effect on a real accepted set", abs(_de_nb2 - _de_mine) < 1e-9 and abs(_icc_nb2 - _icc_mine) < 1e-9,
      f"deff {_de_mine:.4f} vs {_de_nb2:.4f}, icc {_icc_mine:.4f} vs {_icc_nb2:.4f}")
check("S3.7 Kish identity deff = 1 + (nbar-1)*icc holds for the accepted-set estimates (optimistic score, split 0)",
      abs(_de_mine - (1 + (A_res3a[('opt', 'doc')][0]['nbar_acc'] - 1) * A_res3a[('opt', 'doc')][0]['icc_acc'])) < 1e-9)
'''))

    out.append(M(r'''
**How to read this chart.** Top row, one panel per score: the histogram of the **realized selective risk** of the add-one rule across the 200 splits, for document-level resplits (blue) and for the
field-level control (grey), with the target $\alpha = 0.10$ as a vertical line. Look at *where the mass sits relative to the line*: both distributions are centred on $\alpha$ (the rule is right **on average**) but
roughly half of the individual batches land to the right of it, which is exactly what "tiers 1-2 are expectation guarantees" means for the review queue: a single batch that "was calibrated to 10%" is a coin flip on whether
it is above or below 10%. Any *extra width* of the blue histogram over the grey one is the effect of document clustering; compare it with the printed $z$ ratio, not by eye. Bottom-left: the design effect of the **pool** of test fields (all fields, including
non-critical ones) and of each **accepted set**, i.e. how many times smaller the effective sample is than the nominal one. Bottom-right: the mean split-to-split spread of the calib-vs-test risk gap ($z$, 1.0 for exchangeable fields) for documents versus the field control, next to the Kish ceiling
$\sqrt{\text{deff}}$. Honest reading of *our* data: the design effects are clearly above 1 (fields of one document fail together) and the naive Wilson interval on the accepted set is too narrow by about the Kish factor $\sqrt{\text{deff}}$ (printed), but the extra batch-to-batch variability of the realized risk is **modest**, below the Kish ceiling, because the
splits are stratified by document type and draw from one finite pool; the paper's "grouped variant overshoots at 78% of splits" is **not** something these data show, and we do not claim it.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
for _i, _k in enumerate(["opt", "weak"]):
    _ax = axes[0, _i]
    _rd, _rf = A_arr(A_res3a[(_k, "doc")], "risk"), A_arr(A_res3a[(_k, "field")], "risk")
    _bins = np.linspace(min(_rd.min(), _rf.min()) - 1e-3, max(_rd.max(), _rf.max()) + 1e-3, 24)
    _ax.hist(_rf, bins=_bins, color=PAL3["a_field"], alpha=0.55, label=f"split by FIELD (control), viol {(_rf > A_ALPHA).mean():.2f}")
    _ax.hist(_rd, bins=_bins, color=PAL3["a_doc"], alpha=0.55, label=f"split by DOCUMENT, viol {(_rd > A_ALPHA).mean():.2f}")
    _ax.axvline(A_ALPHA, color="k", ls="--", lw=1.2, label="target alpha = 0.10")
    _ax.set_title(f"{A_LABEL[_k]}\nmean coverage {A_tab3a[(_k, 'doc', A_N_LONG)]['cov']:.3f}", fontsize=10)
    _ax.set_xlabel("realized selective risk on test fields"); _ax.set_ylabel("number of splits (of 200)"); _ax.legend(fontsize=8)
_ax = axes[1, 0]
_labels = ["pool of\ntest fields", "accepted set\n(optimistic)", "accepted set\n(pessimistic)"]
_vals = [A_deff3a["opt"]["deff_pool"], A_deff3a["opt"]["deff_acc"], A_deff3a["weak"]["deff_acc"]]
_ax.bar(_labels, _vals, color=[PAL3["a_field"], PAL3["a_opt"], PAL3["a_weak"]])
_ax.axhline(1.0, color="k", lw=1)
for _j, _v in enumerate(_vals):
    _ax.text(_j, _v + 0.05, f"{_v:.2f}", ha="center")
_ax.set_ylabel("design effect (Kish, per-document clustering)"); _ax.set_title("Effective sample = nominal / deff")
_ax = axes[1, 1]
_x = np.arange(2); _w = 0.35
_ax.bar(_x - _w / 2, [A_z3a["opt"][0], A_z3a["weak"][0]], _w, color=PAL3["a_doc"], label="split by DOCUMENT")
_ax.bar(_x + _w / 2, [A_z3a["opt"][1], A_z3a["weak"][1]], _w, color=PAL3["a_field"], label="split by field (control)")
_ax.scatter(_x, [np.sqrt(A_deff3a["opt"]["deff_acc"]), np.sqrt(A_deff3a["weak"]["deff_acc"])], marker="_", s=900, color="k", zorder=5, label="Kish ceiling sqrt(deff)")
_ax.axhline(1.0, color="k", ls=":", lw=1)
_ax.set_xticks(_x); _ax.set_xticklabels(["optimistic", "pessimistic"]); _ax.set_ylabel("sd of z (calib-vs-test risk gap)")
_ax.set_ylim(0, 1.55); _ax.set_title("Batch-to-batch variability: clustering effect"); _ax.legend(fontsize=8, loc="upper right", ncol=1)
plt.tight_layout(); plt.show()
'''))

    # ------------------------------------------------------------------------------------------------------------------
    out.append(M(r'''
### 3b. Failure mode 2 -- score-refit leakage on real data (and the fit/val repair)

Same 40 document splits. The **same** learned score family (the prelude's depth-3 HGB fusion, same hyper-parameters) is used three ways and thresholded with the add-one rule at $\alpha = 0.10$; all are evaluated on the test documents' fields:

| protocol | where the score is fit | where the threshold is set |
| --- | --- | --- |
| **train-fit / calib-threshold** (the prelude's protocol) | `train` documents only | all fields of the calibration documents |
| **same-half** (the failure mode) | the calibration documents' fields *themselves* | the same fields (in-sample scores) |
| **fit/val** (P3 section 5.1 variant) | a random half of the calibration *documents* (stratified by type) | the *other* half of the calibration documents |

Contrast: a **low-capacity** logistic fusion on the six raw signals, refit same-half (the paper: "a 5-parameter logistic fusion barely overfits", idpfin-q7), and the **pessimistic** weak-signal logistic score, train-fit versus same-half. Paired paired-difference tests are one-sided
(same-half risk exceeds train-fit risk). The resplits share one pool, so the violation-fraction bounds are optimistic; tolerances are stated in each check. Same-half refitting is *not* something the prelude does: it is the counterfactual.
'''))

    out.append(C(r'''
def A_hgb():
    """Same configuration as the prelude's HGB fusion (re-stated here so that later parts cannot shadow it)."""
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED3)


def A_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


A_PROTO = ["train_fit", "same_half", "fit_val", "lr_same_half", "weak_train_fit", "weak_same_half"]
A_res3b = {p: [] for p in A_PROTO}
_t0 = time.time()
for _seed in range(A_N_SHORT):
    _ic, _it = A_doc_split_fields(_seed)
    _cal_docs = np.unique(FA_doc[_ic])
    A_res3b["train_fit"].append(A_addone_eval(FA_p_hgb[_ic], _ic, FA_p_hgb[_it], _it))
    _m = A_hgb().fit(X_field[_ic], FA_correct[_ic])                                   # same-half: fit on the calibration fields themselves
    A_res3b["same_half"].append(A_addone_eval(_m.predict_proba(X_field[_ic])[:, 1], _ic, _m.predict_proba(X_field[_it])[:, 1], _it))
    _rng = np.random.default_rng([SEED3, 555, _seed])                                  # fit/val: split the calibration DOCUMENTS
    _fit_docs = []
    for _j in range(4):
        _ids = _cal_docs[DOC_TYPE_IDX[_cal_docs] == _j].copy(); _rng.shuffle(_ids); _fit_docs += list(_ids[:len(_ids) // 2])
    _fm = np.isin(FA_doc[_ic], _fit_docs)
    _i_fit, _i_val = _ic[_fm], _ic[~_fm]
    _m2 = A_hgb().fit(X_field[_i_fit], FA_correct[_i_fit])
    A_res3b["fit_val"].append(A_addone_eval(_m2.predict_proba(X_field[_i_val])[:, 1], _i_val, _m2.predict_proba(X_field[_it])[:, 1], _it))
    _l = A_lr().fit(X_field[_ic][:, :6], FA_correct[_ic])
    A_res3b["lr_same_half"].append(A_addone_eval(_l.predict_proba(X_field[_ic][:, :6])[:, 1], _ic, _l.predict_proba(X_field[_it][:, :6])[:, 1], _it))
    A_res3b["weak_train_fit"].append(A_addone_eval(FA_p_weak[_ic], _ic, FA_p_weak[_it], _it))
    _w = A_lr().fit(X_field[_ic][:, A_WEAK_COLS], FA_correct[_ic])
    A_res3b["weak_same_half"].append(A_addone_eval(_w.predict_proba(X_field[_ic][:, A_WEAK_COLS])[:, 1], _ic, _w.predict_proba(X_field[_it][:, A_WEAK_COLS])[:, 1], _it))
print(f"leakage protocols: {A_N_SHORT} splits x 6 fits in {time.time() - _t0:.1f}s\n")

A_desc3b = {"train_fit": "HGB fusion, fit on train docs, threshold on calib fields", "same_half": "HGB fusion refit on the calib fields, threshold on the same fields (LEAKAGE)",
            "fit_val": "HGB fusion fit on half the calib docs, threshold on the other half", "lr_same_half": "6-signal logistic fusion, same-half",
            "weak_train_fit": "weak-signal LR fit on train docs, threshold on calib", "weak_same_half": "weak-signal LR refit same-half"}
_rows, A_tab3b = [], {}
for _p in A_PROTO:
    _s = A_summ(A_res3b[_p], A_N_SHORT)
    _s["cal_risk"] = A_arr(A_res3b[_p], "cal_risk").mean(); _s["cal_cov"] = A_arr(A_res3b[_p], "cal_cov").mean()
    A_tab3b[_p] = _s
    _rows.append([_p, A_desc3b[_p], f"{_s['cal_cov']:.3f} / {_s['cal_risk']:.4f}", f"{_s['cov']:.3f}", f"{_s['risk']:.4f}", f"{_s['viol']:.3f} ({_s['nviol']}/{A_N_SHORT})", _s["zero"]])
print("Add-one at alpha=0.10 under different score-fitting protocols (40 document splits; 'in-sample' = coverage / risk on the calibration fields at the chosen threshold):")
A_pipe(["protocol", "what it is", "in-sample cov / risk", "test coverage", "test risk", "violation frac", "zero-cov splits"], _rows)

_d_same = A_arr(A_res3b["same_half"], "risk") - A_arr(A_res3b["train_fit"], "risk")
_tt = stats.ttest_1samp(_d_same, 0.0, alternative="greater")
_d_lr = A_arr(A_res3b["lr_same_half"], "risk") - A_arr(A_res3b["train_fit"], "risk")
print(f"\npaired same-half minus train-fit test risk: mean {_d_same.mean():+.4f}, one-sided paired t p = {_tt.pvalue:.2e}; positive in {(_d_same > 0).sum()}/{A_N_SHORT} splits")
print(f"paired 6-signal-LR same-half minus train-fit test risk: mean {_d_lr.mean():+.4f} (a 6-parameter fusion has almost nothing to overfit)")
print(f"coverage change of fit/val vs train-fit: {A_tab3b['fit_val']['cov'] - A_tab3b['train_fit']['cov']:+.3f} (fit/val spends half of the calibration documents on the score fit)")

print()
_TOL = 1.05 * A_ALPHA
check("S3.8 same-half refit is visibly worse than train-fit: paired one-sided t-test p < 0.01", _tt.pvalue < 0.01, f"mean gap {_d_same.mean():+.4f}, p={_tt.pvalue:.1e}")
check("S3.9 same-half overshoots alpha materially: mean test risk > 1.05*alpha (=0.105, a 5% relative tolerance fixed in advance)", A_tab3b["same_half"]["risk"] > _TOL, f"mean risk {A_tab3b['same_half']['risk']:.4f}")
_nv = A_tab3b["same_half"]["nviol"]
check("S3.10 same-half violates in a clear majority of splits: one-sided 95% CP lower bound on the violation fraction > 0.5 (the boundary-rider rate)",
      A_cp_lower(_nv, A_N_SHORT) > 0.5, f"{_nv}/{A_N_SHORT} splits violate, CP lower {A_cp_lower(_nv, A_N_SHORT):.3f}")
check("S3.11 the train-fit / calib-threshold protocol holds on average: mean test risk <= 1.05*alpha", A_tab3b["train_fit"]["risk"] <= _TOL, f"mean risk {A_tab3b['train_fit']['risk']:.4f}")
check("S3.12 the fit/val protocol holds on average: mean test risk <= 1.05*alpha", A_tab3b["fit_val"]["risk"] <= _TOL, f"mean risk {A_tab3b['fit_val']['risk']:.4f}")
check("S3.13 fit/val does not restore per-batch validity (P3: it 'does not restore exchangeability'): violation fraction still > 0.25",
      A_cp_lower(A_tab3b['fit_val']['nviol'], A_N_SHORT) > 0.25, f"{A_tab3b['fit_val']['nviol']}/{A_N_SHORT}, CP lower {A_cp_lower(A_tab3b['fit_val']['nviol'], A_N_SHORT):.3f}")
check("S3.14 a low-capacity 6-signal LR refit same-half barely overfits: mean test risk <= 1.05*alpha and |gap to train-fit| < 0.005 (absolute tolerance fixed in advance)",
      A_tab3b["lr_same_half"]["risk"] <= _TOL and abs(_d_lr.mean()) < 0.005, f"risk {A_tab3b['lr_same_half']['risk']:.4f}, gap {_d_lr.mean():+.4f}")
check("S3.15 pessimistic (weak, 7-parameter) score refit same-half also stays within 1.05*alpha on average", A_tab3b["weak_same_half"]["risk"] <= _TOL,
      f"risk {A_tab3b['weak_same_half']['risk']:.4f} vs train-fit {A_tab3b['weak_train_fit']['risk']:.4f}")
'''))

    out.append(M(r'''
**How to read this chart.** Left: for each protocol, the distribution (box plot, one point per split) of the realized selective risk on the test documents at $\alpha = 0.10$ (dashed line). Look for the box that sits **above** the line: the same-half refit of the high-capacity
score. Its threshold is set on scores the model has already memorized on those very fields, so the calibration risk looks fine while the test risk is higher: the score's optimism has been transferred into the threshold (failure mode 2). The train-fit and fit/val boxes straddle the line
(about half of the batches above it, which is the marginal-guarantee signature from 3a), and the low-capacity 6-signal LR refit same-half is indistinguishable from train-fit because it has almost no capacity to memorize. Right: the same splits in coverage-versus-risk space. The same-half
cloud is shifted up and slightly to the right: a little more coverage, bought with hidden risk, so nothing honest is gained by leaking. **For the review queue:** never let the model that ranks fields and the calibration set that certifies the threshold be the same data; the prelude already follows this (score fit on `train`, threshold on `calib`).
Caveat that stays: fit/val restores score-threshold independence, not exchangeability, so it is still a tier-1 expectation procedure and about half its batches exceed $\alpha$.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
_ax = axes[0]
_order = ["train_fit", "fit_val", "same_half", "lr_same_half", "weak_train_fit", "weak_same_half"]
_names = ["HGB\ntrain-fit", "HGB\nfit/val", "HGB\nsame-half\n(LEAK)", "6-sig LR\nsame-half", "weak LR\ntrain-fit", "weak LR\nsame-half"]
_cols = [PAL3["a_opt"], PAL3["a_ok"], PAL3["a_bad"], PAL3["a_doc"], PAL3["a_weak"], PAL3["a_weak"]]
_data = [A_arr(A_res3b[_p], "risk") for _p in _order]
_bp = _ax.boxplot(_data, tick_labels=_names, patch_artist=True, widths=0.55)
for _patch, _c in zip(_bp["boxes"], _cols):
    _patch.set_facecolor(_c); _patch.set_alpha(0.45)
_rng = np.random.default_rng(0)
for _j, _d in enumerate(_data):
    _ax.scatter(np.full(len(_d), _j + 1) + _rng.normal(0, 0.05, len(_d)), _d, s=10, color="k", alpha=0.5, zorder=3)
_ax.axhline(A_ALPHA, color="k", ls="--", lw=1.2)
_ax.set_ylim(_ax.get_ylim()[0], _ax.get_ylim()[1] + 0.006)
for _j, _p in enumerate(_order):
    _ax.text(_j + 1, 0.995, f"viol {A_tab3b[_p]['viol']:.2f}", ha="center", va="top", fontsize=8, transform=_ax.get_xaxis_transform())
_ax.set_ylabel("realized selective risk on test fields"); _ax.set_title("Score-refit leakage: which protocol keeps the risk at alpha?")
_ax = axes[1]
for _p, _c, _mk in (("train_fit", PAL3["a_opt"], "o"), ("fit_val", PAL3["a_ok"], "s"), ("same_half", PAL3["a_bad"], "^")):
    _ax.scatter(A_arr(A_res3b[_p], "cov"), A_arr(A_res3b[_p], "risk"), s=26, color=_c, marker=_mk, alpha=0.75, label=_p.replace("_", "-"))
_ax.axhline(A_ALPHA, color="k", ls="--", lw=1.2)
_ax.set_xlabel("coverage (fraction of test fields accepted)"); _ax.set_ylabel("realized selective risk"); _ax.set_title("Same splits in coverage / risk space (HGB fusion)")
_ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
'''))

    # ------------------------------------------------------------------------------------------------------------------
    out.append(M(r'''
### 3c. Failure mode 3 -- tie mass

Notebook 2's simulated signals are continuous, so our fused scores have thousands of distinct values and the add-one rule never meets a tie. Real extractors often expose only coarse
signals (a verbalized "0.9"/"0.95", a binary grounded flag). To see what ties do **without touching the data**, we round each score to $L$ equal-width levels (`floor(score*L)/L`) and re-run the add-one rule
on the same 40 document splits. This is a *derived-here* counterfactual, not a re-creation of the paper's dump: the paper's "1,702 to 257 distinct scores" (idpfin-q7) came from dropping a signal from a real extractor. What we can honestly test is the mechanism quoted in section 3:
"A threshold accepts a tie mass whole or not at all" (idpfin-q7), so the smallest reachable candidate is the top level's whole block; if that block's smoothed risk is above $\alpha$ the rule returns $\tau=\infty$ and coverage is exactly zero.
We report both the **zero-filled** mean risk and the mean risk **conditional on non-zero coverage**, and the size and error rate of the top block on the calibration fields of split 0.
'''))

    out.append(C(r'''
A_LEVELS = [None, 20, 10, 5, 3, 2]
_pool_mask = np.isin(FA_doc, np.where(np.isin(DOC_SPLIT, ["calib", "test"]))[0])
A_res3c, A_tab3c = {}, {}
for _k, _sc in A_SCORES.items():
    for _L in A_LEVELS:
        _sr = _sc if _L is None else np.floor(_sc * _L * (1 - 1e-12)) / _L
        _rows_ = []
        for _seed in range(A_N_SHORT):
            _ic, _it = A_doc_split_fields(_seed)
            _rows_.append(A_addone_eval(_sr[_ic], _ic, _sr[_it], _it))
        A_res3c[(_k, _L)] = _rows_
        _s = A_summ(_rows_, A_N_SHORT)
        _ic0, _ = A_doc_split_fields(0)
        _top = _sr[_ic0] == _sr[_ic0].max()
        _s["distinct"] = int(len(np.unique(_sr[_pool_mask]))); _s["top_n"] = int(_top.sum()); _s["top_err"] = float(A_ERR[_ic0][_top].mean())
        _s["top_smoothed"] = (1 + A_ERR[_ic0][_top].sum()) / (1 + _top.sum())
        A_tab3c[(_k, _L)] = _s

_rows = []
for _k in A_SCORES:
    for _L in A_LEVELS:
        _s = A_tab3c[(_k, _L)]
        _rows.append([A_LABEL[_k].split(" (")[0], "continuous" if _L is None else _L, _s["distinct"], f"{_s['cov']:.3f}", f"{_s['risk']:.4f}", "n/a" if np.isnan(_s["risk_nz"]) else f"{_s['risk_nz']:.4f}",
                      f"{_s['viol']:.3f}", f"{_s['zero']}/{A_N_SHORT}", f"{_s['top_n']} / {_s['top_err']:.3f} / {_s['top_smoothed']:.3f}"])
print("Add-one at alpha=0.10 on the rounded scores (40 document splits; top block = calibration fields of split 0 at the top level: size / error rate / add-one smoothed risk):")
A_pipe(["score", "levels L", "distinct values (pool)", "mean coverage", "mean risk (zero-filled)", "mean risk given cov>0", "violation frac", "zero-coverage splits", "top block n / err / smoothed"], _rows)

print()
for _k in A_SCORES:
    check(f"S3.16[{_k}] rounding to L levels leaves exactly L (or fewer) distinct score values on the pool",
          all(A_tab3c[(_k, _L)]["distinct"] <= _L for _L in A_LEVELS if _L is not None), f"distinct counts {[A_tab3c[(_k, _L)]['distinct'] for _L in A_LEVELS if _L is not None]}")
    check(f"S3.17[{_k}] tie mass makes add-one more conservative, never an overshoot: mean risk (zero-filled) <= 1.05*alpha at every L",
          all(A_tab3c[(_k, _L)]["risk"] <= 1.05 * A_ALPHA for _L in A_LEVELS), f"max mean risk {max(A_tab3c[(_k, _L)]['risk'] for _L in A_LEVELS):.4f}")
_c_w, _c_w5 = A_tab3c[("weak", None)]["cov"], A_tab3c[("weak", 5)]["cov"]
check("S3.18 tie mass COLLAPSES coverage for the pessimistic score: coverage at L=5 is below half its continuous coverage", _c_w5 < 0.5 * _c_w,
      f"{_c_w5:.3f} vs {_c_w:.3f}; zero-coverage splits at L=5: {A_tab3c[('weak', 5)]['zero']}/{A_N_SHORT}")
_c_o, _c_o5 = A_tab3c[("opt", None)]["cov"], A_tab3c[("opt", 5)]["cov"]
check("S3.19 the optimistic (bimodal, well-separated) score is robust to ties: coverage at L=5 keeps at least 90% of its continuous coverage", _c_o5 >= 0.9 * _c_o,
      f"{_c_o5:.3f} vs {_c_o:.3f}")
'''))

    out.append(M(r'''
**How to read this chart.** x-axis: how coarse the score is (number of levels $L$; "cont." = the original continuous score). Left: mean coverage (fraction of test fields accepted) of the add-one rule. Right: the share of the 40 splits in which the rule accepted **nothing** (zero coverage; $\tau=\infty$).
For a score that already separates good from bad fields cleanly (**optimistic**, red), coarse levels cost little down to about five levels and cost visibly (but never collapse) at two or three levels: the top block is still pure enough. For the **pessimistic** weak-signal score (orange), the top level's tie block contains too many errors to pass the add-one bar, the rule can only take the block whole or leave it,
and coverage collapses to zero: the review queue is *everything* although the continuous score could have auto-accepted a slice. This is the paper's mechanism reproduced on our data, but only for the weaker score; with the optimistic score the effect is absent, and we say so. Practical reading: if the real extractor only exposes a few discrete confidence levels,
prefer the fused score (many distinct values) over a raw coarse signal, and check the size of the top tie block before promising any coverage.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
_xs = np.arange(len(A_LEVELS))
_xl = ["cont." if _L is None else str(_L) for _L in A_LEVELS]
for _k, _c in (("opt", PAL3["a_opt"]), ("weak", PAL3["a_weak"])):
    axes[0].plot(_xs, [A_tab3c[(_k, _L)]["cov"] for _L in A_LEVELS], "o-", color=_c, label=A_LABEL[_k])
    axes[1].plot(_xs, [A_tab3c[(_k, _L)]["zero"] / A_N_SHORT for _L in A_LEVELS], "o-", color=_c, label=A_LABEL[_k])
for _ax, _yl, _t in zip(axes, ["mean coverage of the add-one rule", "share of splits with ZERO coverage"], ["Coverage as the score gets coarser", "Splits where the top tie block cannot be accepted"]):
    _ax.set_xticks(_xs); _ax.set_xticklabels(_xl); _ax.set_xlabel("number of score levels L (fewer levels = more tie mass)"); _ax.set_ylabel(_yl); _ax.set_title(_t)
    _ax.set_ylim(-0.03, 1.03 if _ax is axes[1] else 0.9); _ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
'''))

    out.append(M(r'''
**Section 3 in one paragraph (what our data does and does not show).** On the Notebook-2 pipeline the add-one rule is right **on average** at $\alpha = 0.10$ under every honest protocol, and wrong **per batch** about half the time; *document clustering* is real
(design effect clearly above 1, naive binomial intervals too narrow) but adds only modest extra batch-to-batch variability here; *score-refit leakage* is clearly visible for the high-capacity fusion and absent for low-capacity ones; *tie mass* collapses coverage for the weak score and not for the strong one.
None of these is a guarantee failure of the rule *in expectation*: they are reasons a single batch, and a report that says "calibrated at 10%", can mislead. All of this is tier-1 vocabulary. The next section changes the unit from the field to the **document** and swaps the add-one rule for cfBH.
'''))

    # =====================================================================================================================
    # SECTION 4
    # =====================================================================================================================
    out.append(M(r'''
## 4. cfBH at the DOCUMENT unit (derived here)

**What the paper says** (re-read from `idpfin-q2` and `idpfin-q11`). The selection problem: from test documents $X_{n+1},\dots,X_{n+m}$ pick a set $R$ whose members all have $Y_{n+j} > c_j$ except for a fraction controlled by the FDR, $\text{FDR} = \mathbb{E}[\text{FDP}]$ with
$\text{FDP} = \sum_j \mathbf{1}\{j \in R, Y_{n+j} \le c_j\} / (1 \vee |R|)$ (idpfin-q11). Ingredients, verbatim from `idpfin-q2`:

> **Definition 2.1.** "A nonconformity score $V(\cdot,\cdot)$ is monotone if $V(x, y) \le V(x, y')$ holds for any $x$ and any $y, y'$ obeying $y \le y'$." (idpfin-q2)
>
> Conformal p-value (4): $p_j = \dfrac{\sum_{i=1}^n \mathbf{1}\{V_i < \hat V_{n+j}\} + \left(1 + \sum_{i=1}^n \mathbf{1}\{V_i = \hat V_{n+j}\}\right) U_j}{n+1}$, "where $U_j \sim \text{Unif}(0,1)$ are i.i.d. random variables to break ties", with $V_i = V(X_i, Y_i)$ and $\hat V_{n+j} = V(X_{n+j}, c_j)$. (idpfin-q2)
>
> **Algorithm 1 (cfBH).** "Compute $k^* = \max\{k : \sum_{j=1}^m \mathbf{1}\{p_j \le qk/m\} \ge k\}$. **Output:** Selection set $R = \{j : p_j \le qk^*/m\}$." (idpfin-q2)
>
> **Theorem 2.3.** "Suppose $V$ is monotone, the calibration data and test data are i.i.d., and data in $\{Z_i\}_{i=1}^n \cup \{\tilde Z_{n+\ell}\}_{\ell \ne j} \cup \{Z_{n+j}\}$ are mutually independent for any $j$. Then, for any $q \in (0,1)$, the output $R$ of Algorithm 1 satisfies $\text{FDR} \le q$." (idpfin-q2)
>
> Warning (Section 2.4): "$\tau$ should not depend on the calibration data, because this would potentially break the mutual independence condition [...] and invalidate FDR control." (idpfin-q2)
>
> Validity is **joint** only: "$\mathbb{P}(p_j \le \alpha \text{ and } j \in H_0) \le \alpha$" (5); the conditional version does **not** hold (footnote 3). (idpfin-q2)

**Mapping to our documents (all of this is derived here; the paper is "silent on multi-field hierarchical or document-level clustering", idpfin-q11).**

| cfBH object | at the document unit |
| --- | --- |
| unit $Z = (X, Y)$ | one document; $X$ its features, $Y$ = `DA_y_ok` in {0,1} (1 = every critical field correct) |
| select $Y > c$ with $c = 0$ | select documents with $Y = 1$; "wrong" = the document has at least one wrong critical field |
| nonconformity score | clipped score $V(x,y) = M\,y - \hat\mu(x)$, $M = 100$, $\hat\mu$ = predicted $P(Y=1 \mid x)$ = `DA_mu` (optimistic) or `DA_mu_weak` (pessimistic); monotone in $y$ for every $x$ |
| calibration / test | the calib / test documents of `resplit(seed)` |
| $\hat V_{n+j}$ | $V(x_j, c=0) = -\hat\mu(x_j)$ |

**Assumptions, and which hold.** (1) $V$ monotone: yes by construction. (2) The score model is independent of calibration and test documents: yes, `DOC_MODEL_Y*` are fit on `train` documents only (prelude section 2), and the threshold $c = 0$ does not depend on calibration data.
(3) Calibration and test documents exchangeable / i.i.d.: `resplit(seed)` is a uniformly random split of a fixed pool, so exchangeable **by construction**, which is what the Monte Carlo below checks. **Caveat, stated plainly:** the *real* claim needs the client's incoming documents to be exchangeable with the calibration documents.
Documents from the same vendor template, month or batch are **not** literally i.i.d.; the shift split is a deliberate violation and is handled by Part B. (4) A random split of a fixed pool means the Monte Carlo below estimates FDR *conditional on this pool*; it does not tell us what the FDR would be on a different population.
'''))

    out.append(M(r'''
**Design of the check.** For each of $B = 1000$ resplits (seeds 0..999) and each $q \in \{0.05, 0.10, 0.20\}$:

- **cfBH** (Algorithm 1) with the clipped score, for `DA_mu` (optimistic) and `DA_mu_weak` (pessimistic); **realized FDR** = mean FDP over batches (FDP $=0$ when nothing is selected, the $1 \vee |R|$ convention); **power** = fraction of good documents in the test batch that are auto-posted; also the fraction of *all* documents auto-posted; the number of **zero-selection batches**; and the mean FDP *conditional on a non-empty selection* (which is *not* promised to be $\le q$).
- **Bonferroni** on the same p-values: select $p_j \le q/m$.
- **Folklore document rule** (Notebook 2's rule lifted to documents): auto-post iff *every* critical field has verbalized confidence $\ge 0.9$ **and** no business rule failed on it. It has no target $q$; we report its realized error among auto-posted documents.
- **Naive per-field cfBH** aggregated to documents: run cfBH on the **critical fields** (calibration = critical fields of calib documents, test = critical fields of test documents; $V = M\cdot\text{correct} - p(x)$ with the field score `FA_p_hgb` / `FA_p_weak`), then auto-post a document iff *all* its critical fields were selected. This is what happens if you ignore the document unit; we measure the FDR **at the document level**.

**Tolerances (fixed in advance).** FDR check: PASS iff mean FDP $\le q + 1.645\,\text{SE}$ with SE = sd(FDP)/$\sqrt{B}$ (one-sided 95%; the resplits share a pool, so SE is optimistic). Proportions (zero-selection batches, joint validity (5)) use one-sided 95% Clopper-Pearson upper bounds.
'''))

    out.append(C(r'''
A_M_BIG = 100.0
A_B = 1000
A_QS = [0.05, 0.10, 0.20]
A_MU = {"opt": DA_mu, "weak": DA_mu_weak}


def A_cfbh_pvalues(V_cal, V_hat, rng):
    """Conformal p-values, eq. (4) of Jin & Candes, with uniform tie-breaking U_j."""
    _Vs = np.sort(V_cal)
    _lo = np.searchsorted(_Vs, V_hat, side="left")            # #{V_i <  V_hat}
    _hi = np.searchsorted(_Vs, V_hat, side="right")           # #{V_i <= V_hat}
    _U = rng.uniform(size=len(V_hat))
    return (_lo + (1 + (_hi - _lo)) * _U) / (len(V_cal) + 1)


def A_bh(p, q):
    """Algorithm 1, steps 3-4 (BH on the p-values): k* = max{k : #{p_j <= q k / m} >= k}; R = {j : p_j <= q k* / m}."""
    _m = len(p)
    _o = np.argsort(p)
    _ok = p[_o] <= q * np.arange(1, _m + 1) / _m
    _k = int(np.where(_ok)[0].max()) + 1 if _ok.any() else 0
    _sel = np.zeros(_m, bool); _sel[_o[:_k]] = True
    return _sel


def A_V(mu, y):
    """Clipped monotone nonconformity score V(x, y) = M*y - mu(x)."""
    return A_M_BIG * y - mu


# ---- implementation self-checks against brute force -------------------------------------------------------------------------
_rng = np.random.default_rng(5)
_Vc = np.round(_rng.normal(size=60), 1); _Vh = np.round(_rng.normal(size=25), 1)         # rounded => genuine ties
_p_fast = A_cfbh_pvalues(_Vc, _Vh, np.random.default_rng(99))
_U = np.random.default_rng(99).uniform(size=25)
_p_brute = np.array([(sum(_Vc[i] < _Vh[j] for i in range(60)) + (1 + sum(_Vc[i] == _Vh[j] for i in range(60))) * _U[j]) / 61 for j in range(25)])
check("S4.1 vectorised conformal p-value equals a brute-force transcription of eq. (4) (with ties)", np.allclose(_p_fast, _p_brute), f"max abs diff {np.abs(_p_fast - _p_brute).max():.2e}")
_ok_bh = True
for _t in range(200):
    _pp = np.random.default_rng(_t).uniform(size=40) ** 2; _q = [0.05, 0.1, 0.2][_t % 3]
    _ks = max([k for k in range(0, 41) if k == 0 or (_pp <= _q * k / 40).sum() >= k])
    _brute = _pp <= _q * _ks / 40 if _ks > 0 else np.zeros(40, bool)
    _ok_bh &= bool(np.array_equal(A_bh(_pp, _q), _brute))
check("S4.2 A_bh equals a brute-force transcription of Algorithm 1 (k*, R) on 200 random p-vectors", _ok_bh)
check("S4.3 the clipped score is monotone in y for every document (Definition 2.1): V(x,0) <= V(x,1)", bool(np.all(A_V(DA_mu, 0) <= A_V(DA_mu, 1)) and np.all(A_V(DA_mu_weak, 0) <= A_V(DA_mu_weak, 1))))

# ---- folklore document rule -------------------------------------------------------------------------------------------------
_rule = np.array([f["rule_failed"] for f in FIELDS], dtype=float)
_field_pass = (FA_verb >= 0.9) & (_rule == 0)
_n_fail = np.bincount(FA_doc[FA_crit], weights=(~_field_pass[FA_crit]).astype(float), minlength=N_DOC)
A_FOLK = _n_fail == 0                                                    # per document: every critical field passes the folklore rule

# ---- Monte Carlo over document resplits ------------------------------------------------------------------------------------
A_mc = {}      # (method, score, q) -> array of rows (fdp, n_sel, power, share_of_all, empty)
A_pv = {"opt": [], "weak": []}   # p-values for the validity property (5)
A_batch = []   # (tst, y) per split for later per-type breakdowns
_t0 = time.time()
_rows_by = defaultdict(list)
_crit_idx = np.where(FA_crit)[0]
_crit_doc = FA_doc[_crit_idx]
A_sel_store = {}   # (score, q, seed) -> selected document ids, for the type breakdown (q=0.10 only)
for _seed in range(A_B):
    _cal, _tst = resplit(_seed)
    _y = DA_y_ok[_tst]; _m = len(_tst); _good = max(1, int(_y.sum()))
    for _sk, _mu in A_MU.items():
        _rng = np.random.default_rng([SEED3, 4242, _seed])
        _p = A_cfbh_pvalues(A_V(_mu[_cal], DA_y_ok[_cal]), A_V(_mu[_tst], 0), _rng)
        if _seed < 400:
            A_pv[_sk].append((_p, _y))
        for _q in A_QS:
            _sel = A_bh(_p, _q); _R = int(_sel.sum())
            _rows_by[("cfbh", _sk, _q)].append(((1 - _y)[_sel].sum() / max(1, _R), _R, _y[_sel].sum() / _good, _R / _m, _R == 0))
            if _q == 0.10:
                A_sel_store[(_sk, _seed)] = _tst[_sel]
            _selb = _p <= _q / _m; _Rb = int(_selb.sum())
            _rows_by[("bonf", _sk, _q)].append(((1 - _y)[_selb].sum() / max(1, _Rb), _Rb, _y[_selb].sum() / _good, _Rb / _m, _Rb == 0))
    _sf = A_FOLK[_tst]; _Rf = int(_sf.sum())
    _rows_by[("folk", "opt", 0.0)].append(((1 - _y)[_sf].sum() / max(1, _Rf), _Rf, _y[_sf].sum() / _good, _Rf / _m, _Rf == 0))
    # naive per-field cfBH -> document aggregation (critical fields only)
    for _sk, _fs in (("opt", FA_p_hgb), ("weak", FA_p_weak)):
        _ic = _crit_idx[np.isin(_crit_doc, _cal)]; _it = _crit_idx[np.isin(_crit_doc, _tst)]
        _rng = np.random.default_rng([SEED3, 9191, _seed])
        _pf = A_cfbh_pvalues(A_V(_fs[_ic], FA_correct[_ic]), A_V(_fs[_it], 0), _rng)
        _nct = np.bincount(FA_doc[_it], minlength=N_DOC)
        for _q in A_QS:
            _sf_ = A_bh(_pf, _q); _Rf_ = int(_sf_.sum())
            _fdp_field = A_ERR[_it][_sf_].sum() / max(1, _Rf_)
            _nsel = np.bincount(FA_doc[_it][_sf_], minlength=N_DOC)
            _post = (_nct > 0) & (_nsel == _nct)
            _post_t = _post[_tst]; _Rd = int(_post_t.sum())
            _rows_by[("field_cfbh_doc", _sk, _q)].append(((1 - _y)[_post_t].sum() / max(1, _Rd), _Rd, _y[_post_t].sum() / _good, _Rd / _m, _Rd == 0, _fdp_field, _Rf_ / len(_it)))
A_mc = {k: np.array(v, dtype=float) for k, v in _rows_by.items()}
print(f"Monte Carlo over {A_B} resplits done in {time.time() - _t0:.1f}s\n")


def A_mc_summary(key):
    _r = A_mc[key]; _n = len(_r)
    _fdp = _r[:, 0]; _se = _fdp.std(ddof=1) / np.sqrt(_n); _emp = _r[:, 4].astype(bool)
    return dict(fdr=_fdp.mean(), se=_se, fdr_nz=_fdp[~_emp].mean() if (~_emp).any() else float("nan"), n_sel=_r[:, 1].mean(), power=_r[:, 2].mean(), share=_r[:, 3].mean(),
                empty=int(_emp.sum()), empty_ub=A_cp_upper(int(_emp.sum()), _n), viol=float((_fdp > key[2]).mean()) if key[2] > 0 else float("nan"))


A_sum = {k: A_mc_summary(k) for k in A_mc}
for _sk in ("opt", "weak"):
    print(f"--- {A_LABEL[_sk]} ---")
    _rows = []
    for _q in A_QS:
        for _meth, _nm in (("cfbh", "cfBH (doc unit)"), ("bonf", "Bonferroni p<=q/m")):
            _s = A_sum[(_meth, _sk, _q)]
            _rows.append([_q, _nm, f"{_s['fdr']:.4f} (+{1.645 * _s['se']:.4f})", f"{_s['fdr_nz']:.4f}", f"{_s['n_sel']:.1f}", f"{_s['share']:.3f}", f"{_s['power']:.3f}", f"{_s['empty']}/{A_B} (CP ub {_s['empty_ub']:.3f})"])
        _s = A_sum[("field_cfbh_doc", _sk, _q)]
        _rows.append([_q, "naive per-field cfBH -> doc (all crit fields selected)", f"{_s['fdr']:.4f} (+{1.645 * _s['se']:.4f})", f"{_s['fdr_nz']:.4f}", f"{_s['n_sel']:.1f}", f"{_s['share']:.3f}", f"{_s['power']:.3f}", f"{_s['empty']}/{A_B} (CP ub {_s['empty_ub']:.3f})"])
    A_pipe(["q", "method", "realized DOC-level FDR (+1.645 SE)", "mean FDP given non-empty selection", "docs auto-posted per batch", "share of all docs", "power (share of good docs)", "empty batches"], _rows)
_s = A_sum[("folk", "opt", 0.0)]
print(f"\nfolklore document rule (score-free): docs auto-posted per batch {_s['n_sel']:.1f} (share {_s['share']:.4f}, power {_s['power']:.4f}); realized error among auto-posted docs {_s['fdr']:.4f}; empty batches {_s['empty']}/{A_B}")
_nf = np.array([[A_mc[('field_cfbh_doc', _sk, _q)][:, 5].mean() for _q in A_QS] for _sk in ('opt', 'weak')])
print("naive per-field cfBH, FIELD-level realized FDR (critical fields) for q=0.05/0.10/0.20: optimistic " + " / ".join(f"{_v:.4f}" for _v in _nf[0]) + " ; pessimistic " + " / ".join(f"{_v:.4f}" for _v in _nf[1]))
print(f"documents in a test batch: about {len(resplit(0)[1])}; good documents about {DA_y_ok[resplit(0)[1]].sum()}")

print()
for _sk in ("opt", "weak"):
    for _q in A_QS:
        _s = A_sum[("cfbh", _sk, _q)]
        check(f"S4.4[{_sk}, q={_q:.2f}] cfBH at the DOCUMENT unit: realized FDR <= q + 1.645*SE", _s["fdr"] <= _q + 1.645 * _s["se"], f"FDR {_s['fdr']:.4f}, SE {_s['se']:.4f}, mean |R| {_s['n_sel']:.1f}, empty batches {_s['empty']}/{A_B}")
for _sk in ("opt", "weak"):
    for _q in A_QS:
        _s = A_sum[("bonf", _sk, _q)]
        check(f"S4.5[{_sk}, q={_q:.2f}] Bonferroni also respects q (it is valid, just powerless): FDR <= q + 1.645*SE", _s["fdr"] <= _q + 1.645 * _s["se"], f"FDR {_s['fdr']:.4f}, mean |R| {_s['n_sel']:.1f}")
_c, _b = A_sum[("cfbh", "opt", 0.10)], A_sum[("bonf", "opt", 0.10)]
check("S4.6 cfBH auto-posts at least 4x as many documents as Bonferroni at q=0.10 (optimistic score)", _c["n_sel"] >= 4 * _b["n_sel"], f"{_c['n_sel']:.1f} vs {_b['n_sel']:.1f} per batch")
# validity property (5): joint, not conditional.  For the clipped score the guarantee is TIGHT (good documents get p ~ 1, so p <= alpha
# happens almost only for bad ones and the joint probability equals alpha up to Monte-Carlo noise); the honest test of "<= alpha" is
# therefore "no significant evidence of a violation": the one-sided 95% Clopper-Pearson LOWER bound must not exceed alpha.
for _sk in ("opt", "weak"):
    _P = np.concatenate([_a for _a, _ in A_pv[_sk]]); _Y = np.concatenate([_b for _, _b in A_pv[_sk]])
    for _a in (0.05, 0.10, 0.20):
        _k_joint, _n_tot = int(((_P <= _a) & (_Y == 0)).sum()), len(_P)
        _cond = ((_P <= _a) & (_Y == 0)).sum() / max(1, (_Y == 0).sum())
        check(f"S4.7[{_sk}, alpha={_a:.2f}] validity (5), JOINT: P(p_j<=alpha and document bad) <= alpha: the 95% CP LOWER bound does not exceed alpha (no evidence of violation)",
              A_cp_lower(_k_joint, _n_tot) <= _a, f"joint {_k_joint / _n_tot:.4f} (CP lower {A_cp_lower(_k_joint, _n_tot):.4f}, upper {A_cp_upper(_k_joint, _n_tot):.4f}); tight, so the estimate sits ON alpha")
    for _a in (0.05, 0.10):
        _kc, _nb = int(((_P <= _a) & (_Y == 0)).sum()), int((_Y == 0).sum())
        check(f"S4.7c[{_sk}, alpha={_a:.2f}] the CONDITIONAL statement P(p_j<=alpha | document bad) <= alpha does NOT hold (footnote 3 reproduced): CP lower bound > alpha",
              A_cp_lower(_kc, _nb) > _a, f"P(p<=alpha | bad) = {_kc / _nb:.4f} vs alpha {_a}")
_dfo = A_sum[("field_cfbh_doc", "opt", 0.10)]
for _q in A_QS:
    _s = A_sum[("field_cfbh_doc", "opt", _q)]
    check(f"S4.8[opt, q={_q:.2f}] IGNORING the document unit (per-field cfBH aggregated) inflates the DOCUMENT-level FDR above q: FDR - 1.645*SE > q", _s["fdr"] - 1.645 * _s["se"] > _q,
          f"doc-level FDR {_s['fdr']:.4f} vs q; the same runs' FIELD-level FDR is {A_mc[('field_cfbh_doc', 'opt', _q)][:, 5].mean():.4f}")
_s = A_sum[("field_cfbh_doc", "weak", 0.20)]
check("S4.9[weak, q=0.20] the same inflation shows for the pessimistic score at q=0.20", _s["fdr"] - 1.645 * _s["se"] > 0.20, f"doc-level FDR {_s['fdr']:.4f}")
_sw = A_sum[("field_cfbh_doc", "weak", 0.10)]
print(f"[info] pessimistic score at q=0.10: naive per-field -> doc FDR is {_sw['fdr']:.4f} with only {_sw['n_sel']:.1f} docs posted per batch; at q=0.05 it posts {A_sum[('field_cfbh_doc', 'weak', 0.05)]['n_sel']:.1f} docs (empty {A_sum[('field_cfbh_doc', 'weak', 0.05)]['empty']}/{A_B}). The inflation needs enough field coverage to appear, so we assert it only where it is visible.")
check("S4.10 the folklore document rule is safe but nearly empty: realized error among auto-posted <= 0.10 while it posts < 2% of documents",
      A_sum[("folk", "opt", 0.0)]["fdr"] <= 0.10 and A_sum[("folk", "opt", 0.0)]["share"] < 0.02, f"error {A_sum[('folk', 'opt', 0.0)]['fdr']:.4f}, share posted {A_sum[('folk', 'opt', 0.0)]['share']:.4f}")
'''))

    out.append(M(r'''
**How to read this chart.** Left: realized document-level FDR (mean FDP over 1000 resplits, error bars $\pm 1.645$ SE) against the target $q$; the diagonal is "FDR = q", and **being on or below the diagonal is the promise**. cfBH (red, optimistic; orange, pessimistic) should hug the diagonal from below: with the clipped score the guarantee is tight (the joint probability in check S4.7 equals $\alpha$), so the budget $q$ is used up.
Bonferroni is also below it but posts almost nothing (see the middle panel). The **naive per-field cfBH aggregated to documents** (blue, optimistic score) is *above* the diagonal at every $q$: it kept its promise about *fields* (printed above) but the client cares about *documents*, and a document needs all of ~5 critical fields to be right, so the field-level FDR is not the document-level FDR. With the pessimistic score (cyan) the inflation only appears at the largest $q$, where the field coverage is high enough for many documents to be posted; at small $q$ it posts almost nothing.
Middle: power, the fraction of genuinely good documents that get auto-posted. Right: the distribution of the per-batch FDP at $q = 0.10$ for cfBH; the mean sits at $q$ but individual batches scatter on both sides, i.e. an FDR guarantee, like tier 1, is an **expectation** over batches and not a per-batch certificate; the pessimistic score also has empty batches (FDP $=0$ by convention). For the review queue:
at $q=0.10$ the optimistic score lets cfBH auto-post a large share of documents while the folklore rule (dotted) posts almost none; the pessimistic score posts far fewer, which is the honest range to quote to a client until a real extractor's signals are measured.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
_qs = np.array(A_QS)
_series = [("cfbh", "opt", PAL3["a_opt"], "o", "-", "cfBH doc unit (optimistic)"), ("cfbh", "weak", PAL3["a_weak"], "o", "-", "cfBH doc unit (pessimistic)"),
           ("field_cfbh_doc", "opt", PAL3["a_doc"], "s", "--", "per-field cfBH -> docs (optimistic)"), ("field_cfbh_doc", "weak", "#17becf", "s", "--", "per-field cfBH -> docs (pessimistic)"),
           ("bonf", "opt", PAL3["a_field"], "^", ":", "Bonferroni (optimistic)")]
_ax = axes[0]
_ax.plot([0, 0.22], [0, 0.22], "k-", lw=1, label="FDR = q")
for _m_, _sk, _c, _mk, _ls, _lb in _series:
    _f = np.array([A_sum[(_m_, _sk, _q)]["fdr"] for _q in A_QS]); _e = np.array([1.645 * A_sum[(_m_, _sk, _q)]["se"] for _q in A_QS])
    _ax.errorbar(_qs, _f, yerr=_e, color=_c, marker=_mk, ls=_ls, capsize=3, label=_lb)
_ax.axhline(A_sum[("folk", "opt", 0.0)]["fdr"], color=PAL3["a_folk"], ls=":", lw=1.5, label="folklore doc rule (no q)")
_ax.set_xlabel("target q"); _ax.set_ylabel("realized document-level FDR"); _ax.set_title("FDR at the document level"); _ax.legend(fontsize=7.5); _ax.set_xlim(0, 0.22); _ax.set_ylim(0, 0.55)
_ax = axes[1]
for _m_, _sk, _c, _mk, _ls, _lb in _series:
    _ax.plot(_qs, [A_sum[(_m_, _sk, _q)]["power"] for _q in A_QS], color=_c, marker=_mk, ls=_ls, label=_lb)
_ax.axhline(A_sum[("folk", "opt", 0.0)]["power"], color=PAL3["a_folk"], ls=":", lw=1.5, label="folklore doc rule")
_ax.set_xlabel("target q"); _ax.set_ylabel("power: share of good documents auto-posted"); _ax.set_title("Power"); _ax.legend(fontsize=7.5)
_ax = axes[2]
_bins = np.linspace(0, 0.3, 31)
for _sk, _c in (("opt", PAL3["a_opt"]), ("weak", PAL3["a_weak"])):
    _fd = A_mc[("cfbh", _sk, 0.10)][:, 0]
    _ax.hist(_fd, bins=_bins, color=_c, alpha=0.55, label=f"{A_LABEL[_sk].split(' (')[0]}: mean {_fd.mean():.3f}, empty {int(A_mc[('cfbh', _sk, 0.10)][:, 4].sum())}")
_ax.axvline(0.10, color="k", ls="--", lw=1.2, label="q = 0.10")
_ax.set_xlabel("per-batch false discovery proportion"); _ax.set_ylabel("number of batches (of 1000)"); _ax.set_title("cfBH per-batch FDP at q = 0.10"); _ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''))

    out.append(M(r'''
### 4b. Which documents does cfBH certify? (per type, and "how many do you auto-post at $q$")

cfBH here is **pooled over document types**. Its guarantee is **marginal**: the FDR of the pooled selected set is $\le q$. **A type-conditional statement is NOT promised** (nothing in `idpfin-q2` / `idpfin-q11` gives one). The type-wise FDPs printed below are therefore *descriptive*; a type with a higher base rate of bad documents can be
over-represented among the false discoveries while the pooled FDR is still $\le q$. If the client needs "at most 10% wrong within each document type", that is what Part B's Mondrian machinery is for, and it costs coverage.
'''))

    out.append(C(r'''
# ---- per-type breakdown at q = 0.10 (mean over the 1000 batches) ------------------------------------------------------------
A_type3 = {}
for _sk in ("opt", "weak"):
    _rows_t = []
    for _j, _t in enumerate(DOC_TYPES):
        _n_t = np.zeros(A_B); _post_t = np.zeros(A_B); _bad_t = np.zeros(A_B); _good_t = np.zeros(A_B)
        for _seed in range(A_B):
            _tst = resplit(_seed)[1]; _sel = A_sel_store[(_sk, _seed)]
            _in_t = DOC_TYPE_IDX[_tst] == _j; _sel_t = DOC_TYPE_IDX[_sel] == _j
            _n_t[_seed] = _in_t.sum(); _good_t[_seed] = DA_y_ok[_tst][_in_t].sum()
            _post_t[_seed] = _sel_t.sum(); _bad_t[_seed] = (1 - DA_y_ok[_sel])[_sel_t].sum()
        A_type3[(_sk, _t)] = dict(n=_n_t.mean(), good=_good_t.mean(), post=_post_t.mean(), bad=_bad_t.mean(), fdp_type=_bad_t.sum() / max(1.0, _post_t.sum()))
        _rows_t.append([_t, f"{_n_t.mean():.0f}", f"{_good_t.mean() / _n_t.mean():.3f}", f"{_post_t.mean():.1f}", f"{_post_t.mean() / _n_t.mean():.3f}", f"{_post_t.mean() / max(1e-9, _good_t.mean()):.3f}",
                        f"{A_type3[(_sk, _t)]['fdp_type']:.3f}"])
    _tot_post = sum(A_type3[(_sk, _t)]["post"] for _t in DOC_TYPES)
    print(f"--- {A_LABEL[_sk]}, q = 0.10 (pooled over types) ---")
    A_pipe(["document type", "docs per test batch", "base rate good", "auto-posted per batch", "share of type posted", "share of type's good docs posted", "pooled-over-batches FDP within type (descriptive)"], _rows_t)
    print(f"total auto-posted per batch {_tot_post:.1f}; pooled FDR (from the MC above) {A_sum[('cfbh', _sk, 0.10)]['fdr']:.4f}\n")

# ---- 'how many documents do you auto-post at q' ---------------------------------------------------------------------------
_rows = []
for _q in A_QS:
    _rows.append([_q] + [f"{A_sum[('cfbh', _sk, _q)]['n_sel']:.0f} ({100 * A_sum[('cfbh', _sk, _q)]['share']:.1f}% of docs; FDR {A_sum[('cfbh', _sk, _q)]['fdr']:.3f})" for _sk in ("opt", "weak")]
                 + [f"{A_sum[('bonf', 'opt', _q)]['n_sel']:.1f} ({100 * A_sum[('bonf', 'opt', _q)]['share']:.1f}%)", f"{A_sum[('field_cfbh_doc', 'opt', _q)]['n_sel']:.0f} (doc FDR {A_sum[('field_cfbh_doc', 'opt', _q)]['fdr']:.3f})",
                    f"{A_sum[('folk', 'opt', 0.0)]['n_sel']:.1f} ({100 * A_sum[('folk', 'opt', 0.0)]['share']:.2f}%; error {A_sum[('folk', 'opt', 0.0)]['fdr']:.3f})"])
print(f"Documents auto-posted per test batch of ~{len(resplit(0)[1])} documents (mean over {A_B} resplits):")
A_pipe(["q", "cfBH, optimistic score", "cfBH, pessimistic score", "Bonferroni (optimistic)", "per-field cfBH -> docs (optimistic)", "folklore doc rule (no q)"], _rows)

print()
_tot = sum(A_type3[("opt", _t)]["post"] for _t in DOC_TYPES)
check("S4.11 per-type auto-post counts add up to the pooled |R| (optimistic score, q=0.10)", abs(_tot - A_sum[("cfbh", "opt", 0.10)]["n_sel"]) < 1e-6, f"{_tot:.2f} vs {A_sum[('cfbh', 'opt', 0.10)]['n_sel']:.2f}")
_tot = sum(A_type3[("weak", _t)]["post"] for _t in DOC_TYPES)
check("S4.12 per-type auto-post counts add up to the pooled |R| (pessimistic score, q=0.10)", abs(_tot - A_sum[("cfbh", "weak", 0.10)]["n_sel"]) < 1e-6, f"{_tot:.2f} vs {A_sum[('cfbh', 'weak', 0.10)]['n_sel']:.2f}")
_fd_ty = {_sk: {_t: A_type3[(_sk, _t)]["fdp_type"] for _t in DOC_TYPES} for _sk in ("opt", "weak")}
_over = {_sk: [_t for _t in DOC_TYPES if _fd_ty[_sk][_t] > 0.10] for _sk in ("opt", "weak")}
print(f"[info] types whose descriptive within-type FDP exceeds q=0.10 (not promised): optimistic {_over['opt'] or 'none'}; pessimistic {_over['weak'] or 'none'}. Pooled FDR <= q does not imply per-type FDR <= q.")
check("S4.13 pooled FDR at q=0.10 <= q while the type-conditional FDPs are reported without a promise (marginal guarantee only)",
      all(A_sum[("cfbh", _sk, 0.10)]["fdr"] <= 0.10 + 1.645 * A_sum[("cfbh", _sk, 0.10)]["se"] for _sk in ("opt", "weak")),
      "; ".join(f"{_sk}: pooled {A_sum[('cfbh', _sk, 0.10)]['fdr']:.4f}, per-type " + "/".join(f"{_fd_ty[_sk][_t]:.3f}" for _t in DOC_TYPES) for _sk in ("opt", "weak")))
'''))

    out.append(M(r'''
**How to read this chart.** Left: for each document type, the share of that type's documents that cfBH (pooled, $q = 0.10$) auto-posts, for the optimistic (red) and the pessimistic (orange) score. Types differ in base rate of good documents (printed above), so a pooled selection favours the easy types.
Right: the descriptive false-discovery proportion **within** each type against the pooled target $q = 0.10$ (dashed). The pooled FDR is at or below $q$ by the theorem; a bar above the line means that type is being auto-posted with a higher error rate than the pooled average, which cfBH does **not** forbid.
(A bar of 0.000, e.g. `kyc_form` with the pessimistic score, means nothing was auto-posted for that type, not that it is perfect.) If a bar for a type is above the line in your own version, that type needs its own threshold, i.e. Mondrian-by-type calibration (Part B), at the price of fewer documents auto-posted.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
_x = np.arange(len(DOC_TYPES)); _w = 0.38
_ax = axes[0]
for _o, (_sk, _c) in enumerate((("opt", PAL3["a_opt"]), ("weak", PAL3["a_weak"]))):
    _sh = [A_type3[(_sk, _t)]["post"] / A_type3[(_sk, _t)]["n"] for _t in DOC_TYPES]
    _ax.bar(_x + (_o - 0.5) * _w, _sh, _w, color=_c, label=A_LABEL[_sk])
    for _j, _v in enumerate(_sh):
        _ax.text(_j + (_o - 0.5) * _w, _v + 0.01, f"{_v:.2f}", ha="center", fontsize=8)
_ax.set_xticks(_x); _ax.set_xticklabels([_t.replace("_", "\n") for _t in DOC_TYPES]); _ax.set_ylabel("share of the type's documents auto-posted (q = 0.10)")
_ax.set_title("Who gets auto-posted (pooled cfBH)"); _ax.legend(fontsize=8)
_ax = axes[1]
for _o, (_sk, _c) in enumerate((("opt", PAL3["a_opt"]), ("weak", PAL3["a_weak"]))):
    _fd = [A_type3[(_sk, _t)]["fdp_type"] for _t in DOC_TYPES]
    _ax.bar(_x + (_o - 0.5) * _w, _fd, _w, color=_c, label=A_LABEL[_sk])
    for _j, _v in enumerate(_fd):
        _ax.text(_j + (_o - 0.5) * _w, _v + 0.003, f"{_v:.3f}", ha="center", fontsize=8)
_ax.axhline(0.10, color="k", ls="--", lw=1.2, label="pooled target q = 0.10")
_ax.set_xticks(_x); _ax.set_xticklabels([_t.replace("_", "\n") for _t in DOC_TYPES]); _ax.set_ylabel("within-type false discovery proportion (descriptive)")
_ax.set_ylim(0, 0.24); _ax.set_title("cfBH is marginal, not type-conditional"); _ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()
'''))

    out.append(M(r'''
**Section 4 in one paragraph.** cfBH, implemented exactly as Algorithm 1 at the **document** unit (derived here), holds its expectation guarantee on our synthetic pool for both the optimistic and the pessimistic score: realized FDR is at or below $q$ within the stated Monte Carlo tolerance, at $q \in \{0.05, 0.10, 0.20\}$. It is valid but **not** a per-batch bound and **not** type-conditional.
The optimistic score auto-posts a large share of documents; the pessimistic score posts far fewer, and some batches select nothing at small $q$: read the range, not the optimistic end. The folklore rule is safe only because it posts almost nothing at the document level. Ignoring the document unit (per-field cfBH aggregated to documents) keeps a *field*-level promise
and breaks the *document*-level one. Caveats that carry to the next parts: i.i.d./exchangeable documents is a design assumption (vendor template and month are not i.i.d.; the shift split breaks it on purpose); synthetic labels are exact whereas real calibration labels are not (Gurram's audit found automatic labels err one-sidedly pessimistic).

**Part A exports (for the closing cell; nothing here is needed by Part B).** Variables: `A_res3a`, `A_tab3a`, `A_deff3a`, `A_z3a`, `A_ci3a` (section 3a), `A_res3b`, `A_tab3b` (3b), `A_res3c`, `A_tab3c` (3c), `A_mc`, `A_sum`, `A_type3`, `A_sel_store`, `A_FOLK`, `A_pv` (section 4).
Helpers: `A_cfbh_pvalues`, `A_bh`, `A_V`, `A_deff`, `A_addone_eval`, `A_pipe`, `A_cp_upper`, `A_cp_lower`. Constants: `A_ALPHA`, `A_QS`, `A_B`, `A_M_BIG`. Every global defined in Part A begins with `A_` or `a_`.
'''))

    return out
