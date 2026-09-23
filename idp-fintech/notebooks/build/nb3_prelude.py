"""Notebook 3 prelude (coordinator-owned): title, reload of Notebook 2, shared document-level definitions.

After the prelude runs, the kernel namespace holds Notebook 2's objects (DOCS, FIELDS, SPLITS, CLASSIFIER, USAGE_LEDGER,
LLM_STATS, helpers such as check/wilson_ci/design_effect/cluster_bootstrap_ci) PLUS the shared NB3 objects documented in
the last markdown cell of section 2 (CONTRACT).
"""
from nb3_common import M, C


def cells():
    out = []

    out.append(M(r'''
# Notebook 3 of 3 -- Project Walkthrough, Part 2: Certify what gets auto-accepted (cfBH, SCoRE, Mondrian LTT) on the Notebook-2 pipeline

This is the third notebook of a three-part portfolio series built around an Upwork job for a fintech SaaS that wants a
document pipeline that classifies, extracts and validates financial documents and **"flags low-confidence extractions for
human review instead of failing silently"**, with **"documentation of extraction accuracy and confidence thresholds"**.

**Framing (read first).** This is a **draft scaffold for a human to extend into their own profile project**, not a record of
production work. Everything runs on the **synthetic** corpus of Notebook 2 (exact latent truth, a *simulated* extractor for
the full corpus, and only a 40-document real-LLM smoke test), so every coverage/risk number below describes this synthetic
setup. The simulated confidence signals are, by Notebook 2's own honest assessment, probably **more informative than a real
extractor's**, so read coverage numbers as optimistic. The "Limitations and open engineering choices" list at the end says what is missing.

**What this notebook adds.** Notebook 1 derived and Monte-Carlo-verified three papers' guarantees on toy data. Here they are
applied to the Notebook-2 pipeline (reloaded, not rebuilt):

| paper | tool | what it promises | unit used here |
| --- | --- | --- | --- |
| P1 cfBH (Jin & Candes 2023) | conformal p-values + BH | E[fraction of auto-accepted items that are wrong] <= q | **document** (derived here) |
| P2 SCoRE (Bai & Jin 2026) | risk-adjusted e-values | MDR / SDR of a bounded, amount-weighted loss, in expectation | **document** (derived here) |
| P3 Gurram 2026 | add-one rule, fit/val protocol, Mondrian LTT tiers | tiers 1-2 expectation; tiers 3-4 PAC (probability >= 1-delta) | field (tiers 1-3) and document (tier 4) |

Rules every section follows (from `research/notebook-research-context.md`, re-read per section via `research/reread.py`):

- "Certificate" is reserved for the PAC tiers (Gurram tiers 3-4). cfBH FDR, SCoRE SDR/MDR and tiers 1-2 are expectation guarantees.
- SCoRE **MDR** is a marginal budget, **not** "at most alpha of accepted items are wrong".
- cfBH and SCoRE are silent about documents; applying them at the **document** unit is our (safe) choice, marked **derived here**.
- Any score or fusion model is trained on documents disjoint from the calibration and test documents.
- Paper numbers appear only in quoted markdown with their `idpfin-qN` slug; every printed number below is ours.
- Synthetic labels are exact, real calibration labels are not (Gurram's audit found automatic labels err one-sidedly pessimistic).

Cell shape per section: markdown claim, computation cell(s) printing numbered self-checks (never plotting), a
"How to read this chart" markdown, then a visualization-only cell.
'''))

    out.append(M(r'''
## 1. Reload Notebook 2's pipeline (not rebuilt)

Notebook 2's cells tagged `core` are executed verbatim in this kernel (they rebuild the corpus, classifier, extractor,
rules and signals from fixed seeds; the DeepSeek smoke test is served entirely from the disk cache, so this makes **zero
live API calls**). Nothing is re-implemented here.
'''))

    out.append(C(r'''
%matplotlib inline
import contextlib, io, json, math, time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

NB2_PATH = Path("02_project_walkthrough_part1.ipynb")
_nb2 = json.load(open(NB2_PATH))
_buf, _t0, _ran = io.StringIO(), time.time(), 0
for _i, _c in enumerate(_nb2["cells"]):
    if _c["cell_type"] == "code" and "core" in _c.get("metadata", {}).get("tags", []):
        with contextlib.redirect_stdout(_buf):
            _r = get_ipython().run_cell("".join(_c["source"]), silent=True)
        _ran += 1
        if not _r.success:
            raise RuntimeError(f"Notebook 2 core cell {_i} failed: {_r.error_in_exec!r}")
_out = _buf.getvalue()
print(f"re-executed {_ran} Notebook-2 core cells in {time.time() - _t0:.1f}s ({len(_out.splitlines())} lines of NB2 output suppressed)")
print(f"DOCS={len(DOCS)}  FIELDS={len(FIELDS)}  splits={ {k: len(v) for k, v in SPLITS.items()} }")
print(f"LLM_STATS after reload: {LLM_STATS}")

_exp_rows = sum(1 for _ in open(Path("../data/nb2/fields.csv"))) - 1
check("S1.1 reloaded FIELDS matches the row count Notebook 2 exported to data/nb2/fields.csv",
      len(FIELDS) == _exp_rows, f"{len(FIELDS)} vs {_exp_rows}")
check("S1.2 the reload made ZERO live LLM calls (smoke test served from cache)",
      LLM_STATS["live_calls"] == 0, f"live_calls={LLM_STATS['live_calls']}")
check("S1.3 document-level splits are disjoint and cover every document",
      sum(len(v) for v in SPLITS.values()) == len(DOCS) and len({d for v in SPLITS.values() for d in v}) == len(DOCS))
'''))

    out.append(M(r'''
## 2. Shared definitions: the document-level outcome, a bounded amount-weighted loss, and independently-fit score models

The three papers need three ingredients that Notebook 2 does not define. They are defined **once, here**, so every later
section uses the same objects. All of it is **derived here** (none of it is in the papers).

**Critical fields.** The fields whose error would matter to the client's core system: amounts and IBAN/currency for
invoices and statements, identity fields for KYC, and flag/amount fields for compliance reports.

**Document outcome `y_ok`.** A document is *good* if **every critical field is correct**. This is the unit for cfBH: "auto-post
the document" is only safe if all its critical fields are right, and documents (not fields) are the exchangeable unit
(Gurram's Failure Mode 1, quoted in Notebook 2 §7 from `idpfin-q7`).

**Bounded document loss `L_doc` in [0,1] (amount-weighted).** Notebook 2's `amount_loss` is in raw currency units (up to 1.6e8)
and is null for some money fields, so it cannot be used by SCoRE, which needs a bounded loss. Instead, for documents with at
least one money field, `L_doc = min(1, sum_i |pred_i - truth_i| / max(sum_i |truth_i|, 1))` over the money fields, where a null
prediction counts as an error of `|truth_i|` and a filled prediction for an absent field counts as `|pred_i|`. This is the
fraction of the document's monetary value that is wrong, clipped at 1. KYC forms have no money fields, so amount-weighted
sections exclude them.

**Independence (P1 section 2.4, P2 assumption, P3 section 5.1; quoted in `idpfin-q12`).** Every score model below is fit on `train`
documents only. Thresholds are set on `calib`, and everything is evaluated on `test` (and `shift`). For the document-level
models the per-field score fed in for *train* documents is **cross-fitted** (5-fold, grouped by document) so the second-stage
model does not train on the first stage's in-sample optimism (Gurram's "score-refit leakage", Failure Mode 2).
'''))

    out.append(C(r'''
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED3 = 20260920
DOC_TYPES = ["invoice", "bank_statement", "kyc_form", "compliance_report"]
CRITICAL_FIELDS = {
    "invoice": ["subtotal", "tax_amount", "total_amount", "iban", "currency"],
    "bank_statement": ["iban", "opening_balance", "total_credits", "total_debits", "closing_balance"],
    "kyc_form": ["full_name", "date_of_birth", "id_number", "id_expiry", "pep_flag"],
    "compliance_report": ["risk_rating", "sar_filed", "total_flagged_amount"],
}
MONEY_FIELDS = {
    "invoice": ["subtotal", "tax_amount", "total_amount"],
    "bank_statement": ["opening_balance", "total_credits", "total_debits", "closing_balance"],
    "kyc_form": [],
    "compliance_report": ["total_flagged_amount"],
}
DOC_IDS = [d["doc_id"] for d in DOCS]
DIDX = {i: k for k, i in enumerate(DOC_IDS)}
DOC_SPLIT = np.array([d["split"] for d in DOCS])
DOC_TYPE_ARR = np.array([d["doc_type"] for d in DOCS])
DOC_TYPE_IDX = np.array([DOC_TYPES.index(d["doc_type"]) for d in DOCS])
DOC_IS_SHIFT = np.array([bool(d["is_shift"]) for d in DOCS])

# ---- field-level arrays aligned with FIELDS -------------------------------------------------
FIELD_NAMES = sorted({f["field"] for f in FIELDS})
FA_doc = np.array([DIDX[f["doc_id"]] for f in FIELDS])
FA_correct = np.array([int(f["correct"]) for f in FIELDS])
FA_crit = np.array([f["field"] in CRITICAL_FIELDS[f["doc_type"]] for f in FIELDS])
FA_verb = np.array([f["verbalized_confidence"] for f in FIELDS], dtype=float)


def _feat_field(f):
    row = [f["verbalized_confidence"], f["self_consistency"], f["grounded"], f["support_score"],
           f["rule_failed"], f["classifier_confidence"]]
    row += [1.0 if f["doc_type"] == t else 0.0 for t in DOC_TYPES]
    row += [1.0 if f["field"] == n else 0.0 for n in FIELD_NAMES]
    row += [1.0 if f["field"] in CRITICAL_FIELDS[f["doc_type"]] else 0.0]
    return row


X_field = np.array([_feat_field(f) for f in FIELDS], dtype=float)
X_field_5sig = X_field[:, :6]  # the raw signals Notebook 2 computed (no doc-type / field one-hots)
_train_docs = DOC_SPLIT == "train"
_is_train_field = _train_docs[FA_doc]


def _hgb():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=SEED3)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


# first-stage models, fit on TRAIN documents' fields only
FUSION_HGB = _hgb().fit(X_field[_is_train_field], FA_correct[_is_train_field])          # "learned fusion" (P3 tier-1 analogue)
FUSION_LR = _lr().fit(X_field_5sig[_is_train_field], FA_correct[_is_train_field])       # "shared low-capacity fusion" (P3 tier-2 analogue)
# "weak-signal" pessimistic score: only what a real LLM extractor plausibly exposes (verbalized confidence, rule failure,
# classifier confidence, doc type). Notebook 2 found the simulated grounding/self-consistency signals more informative than DeepSeek's.
_WEAK_COLS = [0, 4, 5, 6, 7, 8, 9]
FUSION_WEAK = _lr().fit(X_field[_is_train_field][:, _WEAK_COLS], FA_correct[_is_train_field])
FA_p_weak = FUSION_WEAK.predict_proba(X_field[:, _WEAK_COLS])[:, 1]
FA_p_hgb = FUSION_HGB.predict_proba(X_field)[:, 1]   # deployment-time score for calib/test/shift (and in-sample for train)
FA_p_lr = FUSION_LR.predict_proba(X_field_5sig)[:, 1]

# cross-fitted score for TRAIN fields only (used to train the document-level models without in-sample optimism)
FA_p_hgb_oof = FA_p_hgb.copy()
_tr_idx = np.where(_is_train_field)[0]
for _a, _b in GroupKFold(n_splits=5).split(_tr_idx, groups=FA_doc[_tr_idx]):
    _m = _hgb().fit(X_field[_tr_idx[_a]], FA_correct[_tr_idx[_a]])
    FA_p_hgb_oof[_tr_idx[_b]] = _m.predict_proba(X_field[_tr_idx[_b]])[:, 1]
for _k, _f in enumerate(FIELDS):
    _f["p_hgb"], _f["p_lr"], _f["p_weak"] = float(FA_p_hgb[_k]), float(FA_p_lr[_k]), float(FA_p_weak[_k])

FIELD_IDX_BY_DOC = defaultdict(list)
for _k, _di in enumerate(FA_doc):
    FIELD_IDX_BY_DOC[int(_di)].append(_k)


# ---- document table: outcome, bounded loss, document-level features -------------------------------
def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


N_DOC = len(DOCS)
DA_y_ok = np.ones(N_DOC, dtype=int)         # 1 iff every critical field correct
DA_n_crit = np.zeros(N_DOC, dtype=int)
DA_has_money = np.zeros(N_DOC, dtype=bool)
DA_L = np.full(N_DOC, np.nan)               # bounded amount-weighted loss (money docs only)
_num_err, _num_den = np.zeros(N_DOC), np.zeros(N_DOC)
for _k, _f in enumerate(FIELDS):
    _di, _t = FA_doc[_k], _f["doc_type"]
    if FA_crit[_k]:
        DA_n_crit[_di] += 1
        if FA_correct[_k] == 0:
            DA_y_ok[_di] = 0
    if _f["field"] in MONEY_FIELDS[_t]:
        DA_has_money[_di] = True
        tr, pr = _num(_f["truth"]), _num(_f["predicted"])
        if tr is None:
            _num_err[_di] += abs(pr) if pr is not None else 0.0
        else:
            _num_den[_di] += abs(tr)
            _num_err[_di] += abs(pr - tr) if pr is not None else abs(tr)
DA_L[DA_has_money] = np.minimum(1.0, _num_err[DA_has_money] / np.maximum(_num_den[DA_has_money], 1.0))


def _feat_doc(p_field):
    rows = []
    for di in range(N_DOC):
        ix = FIELD_IDX_BY_DOC[di]
        crit = [k for k in ix if FA_crit[k]]
        pc = p_field[crit] if crit else np.array([1.0])
        rows.append([pc.min(), pc.mean(), float(np.sum(np.log(np.clip(pc, 1e-6, 1)))), len(crit),
                     FA_verb[crit].min() if crit else 1.0,
                     float(sum(FIELDS[k]["rule_failed"] for k in crit)),
                     float(DOCS[di]["classifier_confidence"]), float(p_field[ix].mean())]
                    + [1.0 if DOC_TYPE_IDX[di] == j else 0.0 for j in range(4)])
    return np.array(rows, dtype=float)


X_doc = _feat_doc(FA_p_hgb)                # what deployment sees (train-fitted first stage)
X_doc_oof = _feat_doc(FA_p_hgb_oof)        # cross-fitted first stage, used ONLY to train on train docs
_tr_docs = np.where(DOC_SPLIT == "train")[0]
_tr_money = np.array([i for i in _tr_docs if DA_has_money[i]])

DOC_MODEL_Y = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=25,
                                             l2_regularization=1.0, random_state=SEED3).fit(X_doc_oof[_tr_docs], DA_y_ok[_tr_docs])
DOC_MODEL_L = HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=25,
                                            l2_regularization=1.0, random_state=SEED3).fit(X_doc_oof[_tr_money], DA_L[_tr_money])
X_doc_weak = _feat_doc(FA_p_weak)          # weak first stage is a 7-parameter LR: in-sample ~ out-of-sample, so no cross-fit
DOC_MODEL_Y_WEAK = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=25,
                                                  l2_regularization=1.0, random_state=SEED3).fit(X_doc_weak[_tr_docs], DA_y_ok[_tr_docs])
DOC_MODEL_L_WEAK = HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=25,
                                                 l2_regularization=1.0, random_state=SEED3).fit(X_doc_weak[_tr_money], DA_L[_tr_money])
DA_mu_weak = DOC_MODEL_Y_WEAK.predict_proba(X_doc_weak)[:, 1]
DA_risk_weak = np.clip(DOC_MODEL_L_WEAK.predict(X_doc_weak), 0.0, 1.0)
DA_mu = DOC_MODEL_Y.predict_proba(X_doc)[:, 1]                       # P(all critical fields correct | x)
DA_risk = np.clip(DOC_MODEL_L.predict(X_doc), 0.0, 1.0)              # predicted E[L_doc | x] (a RISK score: low = trusted)


def resplit(seed, frac=0.5):
    """Random document-level split of the calib+test pool (stratified by document type) into (cal_idx, test_idx)."""
    rng = np.random.default_rng([SEED3, int(seed)])
    pool = np.where(np.isin(DOC_SPLIT, ["calib", "test"]))[0]
    cal, tst = [], []
    for j in range(4):
        ids = pool[DOC_TYPE_IDX[pool] == j].copy()
        rng.shuffle(ids)
        h = int(round(len(ids) * frac))
        cal += list(ids[:h]); tst += list(ids[h:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def add_one_threshold(scores, errors, alpha):
    """P3 section 4 add-one rule on a HIGHER-IS-MORE-TRUSTED score: smallest t whose smoothed accepted-set error
    (1 + #err) / (1 + #accepted) is <= alpha; returns +inf (review everything) if none qualifies."""
    order = np.argsort(-scores, kind="stable")
    s, e = scores[order], errors[order]
    cum_err, cum_n = np.cumsum(e), np.arange(1, len(s) + 1)
    ok = (1 + cum_err) / (1 + cum_n) <= alpha
    # candidate thresholds must sit on distinct-value boundaries (ties are accepted or rejected whole)
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    ok &= last_of_tie
    return float(s[np.where(ok)[0].max()]) if ok.any() else float("inf")


# ---- self-checks ---------------------------------------------------------------------------------
_used_docs = set(DOC_SPLIT[FA_doc[_is_train_field]])
check("S2.1 first-stage fusion models were fit ONLY on train-split documents", _used_docs == {"train"}, f"{_used_docs}")
check("S2.2 cross-fitted train scores differ from in-sample ones (leakage guard is doing something)",
      float(np.mean(np.abs(FA_p_hgb_oof[_tr_idx] - FA_p_hgb[_tr_idx]))) > 1e-3,
      f"mean |oof-insample| = {np.mean(np.abs(FA_p_hgb_oof[_tr_idx] - FA_p_hgb[_tr_idx])):.4f}")
check("S2.3 L_doc is bounded in [0,1] on every money document",
      np.nanmin(DA_L) >= 0 and np.nanmax(DA_L) <= 1, f"min={np.nanmin(DA_L):.3f} max={np.nanmax(DA_L):.3f}, n_money_docs={DA_has_money.sum()}")
check("S2.4 every document has at least one critical field and a defined outcome", bool((DA_n_crit > 0).all()))
_ev = np.isin(DOC_SPLIT, ["calib", "test"])
_gauc = lambda y, s: stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / (np.sum(y == 1) * np.sum(y == 0))
_f_ev = _ev[FA_doc]
_auc_v = _gauc(FA_correct[_f_ev], FA_verb[_f_ev]); _auc_lr = _gauc(FA_correct[_f_ev], FA_p_lr[_f_ev]); _auc_h = _gauc(FA_correct[_f_ev], FA_p_hgb[_f_ev])
print(f"field AUROC on calib+test (never seen by the fusion models): verbalized {_auc_v:.3f} | LR fusion {_auc_lr:.3f} | HGB fusion {_auc_h:.3f}")
check("S2.5 the learned fusion beats the raw verbalized confidence out-of-sample", _auc_h > _auc_v, f"{_auc_h:.3f} vs {_auc_v:.3f}")
_auc_w = _gauc(FA_correct[_f_ev], FA_p_weak[_f_ev])
print(f"weak-signal (pessimistic) score field AUROC on calib+test: {_auc_w:.3f}")
check("S2.5b the weak score is informative (>0.5) but clearly weaker than the full fusion", 0.5 < _auc_w < _auc_h - 0.05, f"{_auc_w:.3f} vs {_auc_h:.3f}")
_auc_doc = _gauc(DA_y_ok[_ev], DA_mu[_ev])
print(f"document-level: P(y_ok=1) by split: " + ", ".join(f"{s} {DA_y_ok[DOC_SPLIT == s].mean():.3f}" for s in ['train', 'calib', 'test', 'shift'])
      + f" | doc-score AUROC on calib+test {_auc_doc:.3f}")
check("S2.6 the document score separates good from bad documents out-of-sample", _auc_doc > 0.6, f"AUROC={_auc_doc:.3f}")
_auc_doc_w = _gauc(DA_y_ok[_ev], DA_mu_weak[_ev])
print(f"document-level weak-score AUROC on calib+test: {_auc_doc_w:.3f}")
check("S2.6b the weak document score is informative but weaker than the full one", 0.5 < _auc_doc_w < _auc_doc, f"{_auc_doc_w:.3f} vs {_auc_doc:.3f}")
_c, _t = resplit(7)
check("S2.7 resplit() is a disjoint 50/50 document split of the calib+test pool, deterministic per seed",
      len(set(_c) & set(_t)) == 0 and len(_c) + len(_t) == _ev.sum() and np.array_equal(resplit(7)[0], _c),
      f"{len(_c)} calib / {len(_t)} test docs")
print("base rates by doc type (calib+test): " + ", ".join(f"{t} y_ok={DA_y_ok[_ev & (DOC_TYPE_ARR == t)].mean():.3f} (n={(_ev & (DOC_TYPE_ARR == t)).sum()})" for t in DOC_TYPES))
print("money docs by doc type (calib+test): " + ", ".join(f"{t}={(_ev & DA_has_money & (DOC_TYPE_ARR == t)).sum()}" for t in DOC_TYPES))
print("mean L_doc on money docs (calib+test): %.3f ; share of money docs with L_doc > 0: %.3f" % (
    np.nanmean(DA_L[_ev & DA_has_money]), np.mean(DA_L[_ev & DA_has_money] > 0)))
'''))

    out.append(M(r'''
**How to read this chart.** Left: how well each field score separates correct from incorrect fields on documents the models
never saw (calib+test); a higher bar is a better ranking of fields for the review queue. The learned fusion combines Notebook 2's
signals with document/field type and should beat the raw verbalized confidence the "current system" uses. Right: the document
score's reliability on calib+test documents. Points near the diagonal mean the predicted probability that *all critical fields
are right* is roughly honest. **The guarantees in later sections do not depend on this calibration** (cfBH and SCoRE stay valid for
any score); calibration only changes how many documents get certified. Remember the caveat above: the simulated signals are
probably easier to rank than a real extractor's.
'''))

    out.append(C(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
PAL3 = {"raw": "#7f7f7f", "lr": "#1f77b4", "hgb": "#d62728", "diag": "#444444", "doc": "#2ca02c"}
ax = axes[0]
names, vals, cols = ["verbalized\n(raw, folklore)", "weak fusion\n(realistic signals)", "LR fusion\n(6 signals)", "HGB fusion\n(+doc/field type)"], [_auc_v, _auc_w, _auc_lr, _auc_h], [PAL3["raw"], "#ff7f0e", PAL3["lr"], PAL3["hgb"]]
ax.bar(names, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.5, max(vals) + 0.08); ax.set_ylabel("field AUROC (calib+test)"); ax.set_title("Ranking fields for review (never-seen documents)")
ax = axes[1]
_m, _y = DA_mu[_ev], DA_y_ok[_ev]
_edges = np.quantile(_m, np.linspace(0, 1, 9)); _edges[0] -= 1e-9; _edges[-1] += 1e-9
_bx, _by, _bn = [], [], []
for lo, hi in zip(_edges[:-1], _edges[1:]):
    sel = (_m > lo) & (_m <= hi)
    _bx.append(_m[sel].mean()); _by.append(_y[sel].mean()); _bn.append(sel.sum())
ax.plot([0, 1], [0, 1], "--", color=PAL3["diag"], lw=1)
ax.scatter(_bx, _by, s=np.array(_bn) / 3, color=PAL3["doc"], edgecolor="k")
ax.set_xlabel("predicted P(all critical fields correct)"); ax.set_ylabel("observed share of good documents")
ax.set_title("Document score reliability (calib+test, 8 quantile bins)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()
'''))

    out.append(M(r'''
**CONTRACT: shared objects available to every later section** (defined above; do not redefine or shadow them).

- From Notebook 2: `DOCS`, `FIELDS`, `SPLITS`, `CLASSIFIER`, `USAGE_LEDGER`, `LLM_STATS`, and helpers `check`, `wilson_ci`, `fmt_ci`, `design_effect`, `cluster_bootstrap_ci`, `llm_cost_usd`, `ledger_cost_usd`, `score_field`.
- Constants: `SEED3`, `DOC_TYPES`, `CRITICAL_FIELDS`, `MONEY_FIELDS`, `PAL3` (palette; sections may add keys under their own prefix).
- Document arrays (length `N_DOC`, index = position in `DOCS`): `DOC_SPLIT`, `DOC_TYPE_ARR`, `DOC_TYPE_IDX`, `DOC_IS_SHIFT`, `DA_y_ok` (all critical fields correct), `DA_n_crit`, `DA_has_money`, `DA_L` (bounded amount-weighted loss, `nan` for KYC), `DA_mu` (predicted P(y_ok=1)), `DA_risk` (predicted E[L_doc], low = trusted), and the pessimistic-score twins `DA_mu_weak`, `DA_risk_weak`.
- Field arrays (length `len(FIELDS)`): `FA_doc` (document index), `FA_correct`, `FA_crit`, `FA_verb` (raw verbalized confidence), `FA_p_hgb`, `FA_p_lr`, `FA_p_weak` (train-fitted fusion scores: optimistic learned / low-capacity / pessimistic weak-signal), `FA_p_hgb_oof` (cross-fitted, train docs only), `X_field`, `FIELD_IDX_BY_DOC`. Each `FIELDS` row also carries `p_hgb`, `p_lr`, `p_weak`.
- Models: `FUSION_HGB`, `FUSION_LR`, `FUSION_WEAK`, `DOC_MODEL_Y`, `DOC_MODEL_L`, `DOC_MODEL_Y_WEAK`, `DOC_MODEL_L_WEAK` (all fit on `train` documents only).
- Helpers: `resplit(seed)` returns `(cal_doc_idx, test_doc_idx)`, a 50/50 document-level split of the calib+test pool stratified by document type; `add_one_threshold(scores, errors, alpha)` is the P3 section 4 rule on a higher-is-more-trusted score.
'''))
    return out
