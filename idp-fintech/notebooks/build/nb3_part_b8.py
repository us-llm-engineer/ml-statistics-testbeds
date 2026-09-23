"""Notebook 3 section 10 (coordinator-written after the Part B agent hit a session limit; renumbered from 8 to 10 to
make room for Part C's section 7b and Part D's sections 8-9): threshold documentation deliverable + cost accounting.
Reads Part B's exports (B_FROZEN, b_accept, ...)."""
from nb3_common import M, C


def cells():
    out = []
    out.append(M(r'''
## 10. Deliverable: documentation of extraction accuracy and confidence thresholds, and what it costs

The job asks for "documentation of extraction accuracy and confidence thresholds". This section turns Sections 4-7 into that
document. Everything is **derived here** and describes the **synthetic** corpus.

**Recommended configuration (chosen from Section 6, stated with its limits).** Tier 3, Mondrian Learn-then-Test with the
pre-specified taxonomy *document type*, `delta = 0.10`, thresholds fit on the `calib` split (1,049 documents) and evaluated once
on the `test` split. It is the strongest guarantee that still leaves useful coverage in Section 6: a PAC statement per document
type ("with probability >= 1 - delta over the calibration draw, the error rate among auto-accepted fields of this type is <= alpha"),
valid **if fields within a type are iid**, a premise the paper calls "load-bearing" (`idpfin-q8`) and that our clustering breaks,
although Section 6 found it held empirically at this size. Two operating points are shown because the answer depends on how
informative your real confidence signals are:

- **optimistic score** (learned fusion, `FA_p_hgb`) at `alpha = 0.10`;
- **pessimistic score** (weak signals, `FA_p_weak`) at `alpha = 0.20`. At `alpha = 0.10` the pessimistic score certifies almost
  nothing (Section 6), which is itself the finding: with weak signals you cannot promise 10% at this calibration size.

The score, thresholds and their meaning ("auto-accept a field if its score is at least the threshold for its document type;
otherwise route it to human review") are the *configuration object* printed below. A threshold of `null` means "review every
field of that type" (nothing could be certified).
'''))
    out.append(C(r'''
import json as b8_json
from IPython.display import Markdown, display

B8_CONFIGS = [("hgb", 0.10, "optimistic (learned fusion)"), ("weak", 0.20, "pessimistic (weak signals)"), ("weak", 0.10, "pessimistic (weak signals)")]
B8_FIELDS_PER_DOC = np.array([np.mean([len(FIELD_IDX_BY_DOC[d]) for d in np.where((DOC_SPLIT == "test") & (DOC_TYPE_IDX == g))[0]]) for g in range(4)])
B8_ROWS, B8_TOT = [], {}
for _k, _a, _nm in B8_CONFIGS:
    _tau = B_FROZEN[(_k, _a)]["t3"]                       # tier 3 x type, fit on calib documents only (Section 7)
    _sc = B_SCORE_FIELD[_k]
    _acc = b_accept(_sc, B_TESTM, _tau); _tm_grp = B_GRP[B_TESTM]; _te_err = B_ERR[B_TESTM]
    for _g in range(4):
        _in_g = _tm_grp == _g; _a_g = _acc & _in_g
        _cov = _a_g.sum() / _in_g.sum(); _risk = _te_err[_a_g].mean() if _a_g.any() else float("nan")
        _ncal_f, _ncal_d = int((B_CALM & (B_GRP == _g)).sum()), len(set(FA_doc[B_CALM & (B_GRP == _g)]))
        B8_ROWS.append(dict(score=_k, alpha=_a, name=_nm, type=DOC_TYPES[_g], tau=float(_tau[_g]), ncal_f=_ncal_f, ncal_d=_ncal_d,
                            cov=float(_cov), risk=float(_risk), review_fields_per_1000=float((1 - _cov) * B8_FIELDS_PER_DOC[_g] * 1000)))
    B8_TOT[(_k, _a)] = dict(cov=float(_acc.mean()), risk=float(_te_err[_acc].mean()) if _acc.any() else float("nan"), nacc=int(_acc.sum()), ntest=int(_acc.size))

for _k, _a, _nm in B8_CONFIGS:
    _rows = [r for r in B8_ROWS if r["score"] == _k and r["alpha"] == _a]
    _tbl = b_tbl(["document type", "threshold (auto-accept if score >=)", "calibration docs / fields", "coverage on test", "error among accepted (test)", "fields to review per 1,000 docs"],
                 [[r["type"], ("review all" if not np.isfinite(r["tau"]) else f"{r['tau']:.3f}"), f"{r['ncal_d']} / {r['ncal_f']}", f"{r['cov']:.3f}",
                   ("n/a (nothing accepted)" if np.isnan(r["risk"]) else f"{r['risk']:.3f}"), f"{r['review_fields_per_1000']:.0f}"] for r in _rows])
    display(Markdown(f"**{_nm}, alpha = {_a:.2f}, delta = 0.10** (overall coverage on test {B8_TOT[(_k, _a)]['cov']:.3f}, "
                     f"error among accepted {B8_TOT[(_k, _a)]['risk']:.3f})\n\n" + _tbl))

B8_CONFIG_OBJ = {"version": "synthetic-demo-0", "guarantee": "PAC, Mondrian Learn-then-Test tier 3 (Gurram 2026) over document type; premise: fields iid within type",
                 "delta": 0.10, "decision_rule": "auto-accept a field iff score >= thresholds[document_type]; else route to human review; null = review everything",
                 "calibration": "fit on 1,049 `calib` documents; evaluated once on 1,051 `test` documents; synthetic data",
                 "operating_points": {}}
for _k, _a, _nm in B8_CONFIGS[:2]:
    B8_CONFIG_OBJ["operating_points"][f"{_k}_alpha_{_a:.2f}"] = {
        "score": "FA_p_hgb (learned fusion of 6 signals + document/field type)" if _k == "hgb" else "FA_p_weak (verbalized confidence, rule failure, classifier confidence, document type)",
        "alpha": _a, "thresholds": {r["type"]: (round(r["tau"], 4) if np.isfinite(r["tau"]) else None) for r in B8_ROWS if r["score"] == _k and r["alpha"] == _a}}
print(b8_json.dumps(B8_CONFIG_OBJ, indent=2))

check("S8.1 the configuration object round-trips through JSON and every threshold is a finite number or null",
      b8_json.loads(b8_json.dumps(B8_CONFIG_OBJ)) == B8_CONFIG_OBJ)
check("S8.2 all thresholds shown were fit on calibration documents only", set(DOC_SPLIT[FA_doc[B_CALM]]) == {"calib"})
check("S8.3 the table coverage equals a direct recomputation on the test split (hgb, alpha=0.10)",
      abs(B8_TOT[("hgb", 0.10)]["cov"] - float(np.mean(b_accept(B_SCORE_FIELD["hgb"], B_TESTM, B_FROZEN[("hgb", 0.10)]["t3"])))) < 1e-12)
check("S8.4 single-draw sanity (NOT a guarantee): the realized test error stays at or below alpha for the two recommended operating points",
      B8_TOT[("hgb", 0.10)]["risk"] <= 0.10 and B8_TOT[("weak", 0.20)]["risk"] <= 0.20,
      f"hgb/0.10: {B8_TOT[('hgb', 0.10)]['risk']:.3f}; weak/0.20: {B8_TOT[('weak', 0.20)]['risk']:.3f}")
check("S8.5 one fixed calibration draw can exceed the 5% low-coverage proxy for the pessimistic score at alpha=0.10",
      B8_TOT[("weak", 0.10)]["cov"] >= 0.05,
      f"coverage {B8_TOT[('weak', 0.10)]['cov']:.3f}; Section 6 mean across resplits is 0.014")
'''))
    out.append(M(r'''
**How to read the tables.** Each row is one document type with its own threshold. "Coverage on test" is the share of that type's
fields the rule would auto-accept; "error among accepted" should sit **below** `alpha` (a single draw, so this is a sanity check, not
the guarantee: the guarantee is the PAC statement over calibration draws, verified over 40 resplits in Section 6). "Fields to review
per 1,000 docs" is the human workload the rule implies. Compare the three blocks: the optimistic score certifies a large share at
`alpha = 0.10`; the pessimistic score needs `alpha = 0.20` to certify anything useful; at `alpha = 0.10` it sends almost everything to
review. Which block describes your real extractor is an **empirical question this notebook cannot answer**: Notebook 2 found real
DeepSeek signals less informative than the simulated ones, so plan for something between the two.

**Two things these single-draw tables show.**

- **Check S8.5 records the counterexample to a tempting proxy.** The 40-resplit mean coverage for the pessimistic score at
  `alpha = 0.10` was 0.014, but this one fixed calibration split certified the compliance-report type and reached 0.061 overall.
  A low mean is not a per-draw upper bound. The useful finding is the draw-to-draw variability when signals are weak, not a claim
  that the rule accepts literally nothing.
- **A PAC statement allows the odd realized excess.** In the `alpha = 0.20` pessimistic block, the bank-statement row shows 0.213 > 0.20
  on this draw. Tier 3 promises, per type, that this happens with probability at most `delta = 0.10` over calibration draws (Section 6
  counted these per-group violations), not that a single draw never exceeds `alpha`.
'''))
    out.append(M(r'''
### 10b. Cost accounting (derived here, from Notebook 2's usage ledger)

Notebook 2's `USAGE_LEDGER` records the token usage of **every** DeepSeek request of its smoke test, live or cached, so the cost
below reflects what the requests cost when they were made (this notebook makes no calls). It covers 40 documents x 3 samples plus the
retries for truncated calls. Pricing is the peak DeepSeek rate used in Notebook 2 (input cache hit / miss / output = $0.006 / $0.30 /
$1.20 per 1M tokens; off-peak is half).

Human review cost is **not data**: `B8_REVIEW_SECONDS_PER_FIELD` and `B8_REVIEW_USD_PER_HOUR` are placeholders **you must set**. The
policy table shows how the three levers (LLM calls per document, share of fields auto-accepted, wrong fields auto-posted) trade off.
Policies that use the self-consistency signal need `k = 3` LLM samples per document; policies that use only the verbalized confidence,
rule failures, classifier confidence and document type need `k = 1`.
'''))
    out.append(C(r'''
B8_REVIEW_SECONDS_PER_FIELD = 10.0      # ASSUMPTION: placeholder, set from a real time-and-motion sample
B8_REVIEW_USD_PER_HOUR = 30.0           # ASSUMPTION: placeholder, set from your reviewer cost

_ext = [e for e in USAGE_LEDGER if e["section"].endswith("_extraction")]
_first = [e for e in _ext if e["kind"].startswith("call0")]         # the temperature-0 call per document, incl. its retry
B8_N_SMOKE_DOCS = sum(1 for e in _ext if e["kind"] == "call0_t0.0")
B8_COST = {}
for _peak in (True, False):
    B8_COST[_peak] = dict(k1=ledger_cost_usd(_first, peak=_peak) / B8_N_SMOKE_DOCS, k3=ledger_cost_usd(_ext, peak=_peak) / B8_N_SMOKE_DOCS)
_n_retry = sum(1 for e in _ext if "retry" in e["kind"])
print(f"smoke sample: {B8_N_SMOKE_DOCS} documents, {len(_ext)} extraction requests of which {_n_retry} were retries of truncated calls")
print(f"LLM cost per document (peak pricing): k=1 ${B8_COST[True]['k1']:.4f} | k=3 ${B8_COST[True]['k3']:.4f}    (off-peak: k=1 ${B8_COST[False]['k1']:.4f} | k=3 ${B8_COST[False]['k3']:.4f})")
print(f"extrapolated (derived here, smoke-test sized sample so wide uncertainty): per 1,000 documents k=3 ${1000 * B8_COST[True]['k3']:.2f}; per 100,000 documents k=3 ${100000 * B8_COST[True]['k3']:,.0f}")

_fpd = float(np.mean([len(FIELD_IDX_BY_DOC[d]) for d in np.where(DOC_SPLIT == "test")[0]]))
_folk = B_FOLK_FIELD[B_TESTM]; _folk_err = B_ERR[B_TESTM]
_pol = []
def b8_policy(label, k, cov, risk):
    llm = 1000 * B8_COST[True]["k1" if k == 1 else "k3"]
    fields_rev = (1 - cov) * _fpd * 1000
    review = fields_rev * B8_REVIEW_SECONDS_PER_FIELD / 3600 * B8_REVIEW_USD_PER_HOUR
    wrong_auto = cov * _fpd * 1000 * (0.0 if not np.isfinite(risk) else risk)
    _pol.append(dict(label=label, k=k, cov=cov, risk=risk, llm=llm, review=review, total=llm + review, fields_rev=fields_rev, wrong=wrong_auto))

b8_policy("review everything (no auto-accept)", 1, 0.0, float("nan"))
b8_policy("folklore rule (verbalized >= 0.9, no rule failure)", 1, float(_folk.mean()), float(_folk_err[_folk].mean()))
_t1 = b_accept(B_SCORE_FIELD["hgb"], B_TESTM, B_FROZEN[("hgb", 0.10)]["t1"])
b8_policy("tier 1 add-one, optimistic, alpha=0.10 (expectation only)", 3, float(_t1.mean()), float(_folk_err[_t1].mean()))
b8_policy("tier 3 x type, optimistic, alpha=0.10 (PAC)", 3, B8_TOT[("hgb", 0.10)]["cov"], B8_TOT[("hgb", 0.10)]["risk"])
b8_policy("tier 3 x type, pessimistic, alpha=0.20 (PAC)", 1, B8_TOT[("weak", 0.20)]["cov"], B8_TOT[("weak", 0.20)]["risk"])
display(Markdown("**Cost and error per 1,000 documents** (LLM cost from the ledger; review cost from the placeholder parameters; the last column counts fields auto-posted that are wrong)\n\n"
    + b_tbl(["policy", "LLM samples k", "share of fields auto-accepted", "LLM $", "human review $ (assumed)", "total $", "fields to review", "wrong fields auto-posted"],
            [[p["label"], p["k"], f"{p['cov']:.3f}", f"{p['llm']:.2f}", f"{p['review']:.2f}", f"{p['total']:.2f}", f"{p['fields_rev']:.0f}", f"{p['wrong']:.0f}"] for p in _pol])))

check("S8.6 every ledger entry is a cache hit (this run paid nothing) yet carries usage, so the priced cost reflects the work, not this session",
      all(e["cache_hit"] for e in USAGE_LEDGER) and ledger_cost_usd(USAGE_LEDGER) > 0, f"{len(USAGE_LEDGER)} entries, priced total ${ledger_cost_usd(USAGE_LEDGER):.4f}")
check("S8.7 k=3 costs more per document than k=1, and off-peak is half of peak", B8_COST[True]["k3"] > B8_COST[True]["k1"] > 0 and abs(B8_COST[False]["k3"] * 2 - B8_COST[True]["k3"]) < 1e-9)
check("S8.8 the certified optimistic rule auto-accepts more fields than the folklore rule", _pol[3]["cov"] > _pol[1]["cov"], f"{_pol[3]['cov']:.3f} vs {_pol[1]['cov']:.3f}")
check("S8.9 this notebook made no live LLM calls", LLM_STATS["live_calls"] == 0)
'''))
    out.append(M(r'''
**How to read this chart.** Left: total cost per 1,000 documents split into LLM cost (blue) and human review cost (orange), for each
policy; a shorter bar is cheaper. Right: how many **wrong fields are auto-posted** per 1,000 documents (red); the dashed line marks
what a strict "no wrong field posted" business rule would demand (zero). Reading the two together: reviewing everything is the safest and
most expensive; the folklore rule barely reduces review work (it accepts few fields); the certified rules cut review cost sharply while
bounding, not eliminating, the wrong fields that get through. Note that the count of wrong auto-posted fields grows with `alpha`: the
pessimistic rule at `alpha = 0.20` lets through the most (roughly twice the optimistic rule at `alpha = 0.10`), so a looser budget buys
review savings by accepting more errors. **The review-cost axis is driven entirely by the two placeholder
parameters**, so change them before drawing a business conclusion.
'''))
    out.append(C(r'''
PAL3.update({"b8_llm": "#1f77b4", "b8_rev": "#ff7f0e", "b8_wrong": "#d62728"})
_fig, _axes = plt.subplots(1, 2, figsize=(13, 4.8))
_lab = ["review\neverything", "folklore\nrule", "tier 1 add-one\n(optimistic,\nexpectation)", "tier 3 x type\n(optimistic,\nPAC, a=0.10)", "tier 3 x type\n(pessimistic,\nPAC, a=0.20)"]; _x = np.arange(len(_pol))
_axes[0].bar(_x, [p["llm"] for p in _pol], color=PAL3["b8_llm"], label="LLM")
_axes[0].bar(_x, [p["review"] for p in _pol], bottom=[p["llm"] for p in _pol], color=PAL3["b8_rev"], label="human review (assumed)")
for _i, _p in enumerate(_pol):
    _axes[0].text(_i, _p["total"] + 8, f"${_p['total']:.0f}", ha="center", fontsize=9)
_axes[0].set_ylim(0, max(p["total"] for p in _pol) * 1.12); _axes[0].set_xticks(_x); _axes[0].set_xticklabels(_lab, fontsize=8); _axes[0].set_ylabel("USD per 1,000 documents"); _axes[0].legend(fontsize=8)
_axes[0].set_title("Cost per 1,000 documents", fontsize=10)
_axes[1].bar(_x, [p["wrong"] for p in _pol], color=PAL3["b8_wrong"])
for _i, _p in enumerate(_pol):
    _axes[1].text(_i, _p["wrong"] + 20, f"{_p['wrong']:.0f}", ha="center", fontsize=9)
_axes[1].axhline(0, ls="--", color=PAL3["diag"]); _axes[1].set_ylim(0, max(p["wrong"] for p in _pol) * 1.12); _axes[1].set_xticks(_x); _axes[1].set_xticklabels(_lab, fontsize=8)
_axes[1].set_ylabel("wrong fields auto-posted per 1,000 documents"); _axes[1].set_title("What gets through (bounded, not zero)", fontsize=10)
plt.tight_layout(); plt.show()
'''))
    return out
