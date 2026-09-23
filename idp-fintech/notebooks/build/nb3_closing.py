"""Notebook 3 closing section (coordinator-owned): recap, gaps the sources leave open, not-built-yet list.
Renumbered "## 9." -> "## 11." (and its check "S9.1" -> "S11.1") to make room for Part C's section 7b and Part D's
sections 8-9; content updated to reflect what sections 7b-9 now build and what they show still cannot be done."""
from nb3_common import M, C


def cells():
    out = []
    out.append(M(r'''
## 11. Recap, gaps the papers leave open, and limitations and open engineering choices

**What this notebook did (all on the synthetic Notebook-2 corpus).** Reloaded Notebook 2's pipeline from its `core` cells (zero live
LLM calls), defined a document-level outcome and a bounded amount-weighted loss (**derived here**), fit every score model on `train`
documents only, and then applied the three papers -- including, in sections 7b-9, machinery aimed straight at the "Frozen threshold
after a month-7 vendor shift" and several other "not met" checkpoints this portfolio's own README used to list:

| step | paper | guarantee form | unit |
| --- | --- | --- | --- |
| folklore/add-one rule audited under document-level resplits | P3 sections 4-5.1 | expectation (tiers 1-2) | field |
| cfBH auto-post rule | P1 | expectation (FDR) | document (derived here) |
| SCoRE MDR / SDR with an amount-weighted loss | P2 | expectation | document (derived here) |
| Mondrian LTT ladder per document type | P3 section 5.2 | PAC (tiers 3-4) | field / document |
| frozen-configuration check on the month-7 shift batch | P3 section 6.4 (analogue) | none promised under shift | field |
| **7b(i): known-weight covariate-shift repair** (weighted MDR/SDR, e-BH boosting, WCS/WBH) | P2 section 6, Jin & Candes 2023a | **finite-sample**, under Q | document (derived here) |
| **7b(ii): estimated-weight repair on the real shift batch** (3 covariate sets, A.3/A.5 balancing) | P2 Thm 6.4/6.5, Remark 6.6/App. A.4-A.5 | **asymptotic** (or doubly-robust asymptotic) | document (derived here) |
| **7b(iii): covariate- vs concept-shift diagnostic** | none (fully derived here) | diagnostic, not a guarantee | document |
| **8a-8b: SDR e-value boosting, risk/reward score** | P2 Thm 5.5, 4.6(iii)/5.8(iii) | expectation (SDR/MDR) | document |
| **8c: tier-4 power variants** (hoeffding / binomial_any_error / bernstein) | Gurram's open problem + derived-here variants | PAC | document |
| **8d-8e: pure-weak regime-2 test, grouped-taxonomy "lucky-zero"** | P3 section 7, section 5.3 | expectation/PAC | field |
| **9: label-noise simulation + audit-style check** | Gurram section 6.4 (pessimistic quote) + derived-here optimistic arm | diagnostic | field |

**What changed vs. the README's old verdict.** "Frozen threshold after a month-7 vendor shift: no -- violates every configuration" is
now more precise, not reversed: Section 7b shows the shift CAN be repaired **without new labels** if it is (or is well-approximated by)
covariate shift with **known** weights (7b(i), finite-sample, exact by construction on a semi-synthetic check) or **estimated** weights
of a well-behaved covariate set (7b(ii), asymptotic only, and covariate set C's template flag shows the estimator can degenerate).
Section 7b(iii)'s diagnostic finds a real `P(Y|X)` (concept-shift) component in the actual month-7 batch, which **no** weighting scheme
repairs -- so the honest updated verdict is: *"repairable in part, without new labels, if the shift is covariate-only and the weights are
well estimated; the real month-7 batch has a concept-shift component that only new labels (Section 7's recalibration) fixes."*

**Gaps the sources leave open** (each flagged in the NotebookLM traces; nothing below is filled in by us):

- **Refitting a score on calibration data.** None of the three sources gives a way to re-fit the score on the calibration data itself
  while keeping finite-sample validity; every one relies on a separate training set or a fit/val split (`idpfin-q12`).
- **Concept drift, even after Section 7b.** SCoRE's weighted e-values repair only **covariate** shift (`dQ/dP(x,y)=w(x)`); Section
  7b(iii)'s own diagnostic found the real month-7 batch is not purely covariate shift, and "none of the three sources state how to
  perform real-time, online re-calibration or re-certification under unobserved concept drift" without new labeled data (`idpfin-q12`).
  No source gives a **weighted or shift-robust cfBH or Mondrian LTT** either -- SCoRE is "the ONLY source ... that formulates weighted
  risk-adjusted e-values for covariate shift" (`idpfin-q18`); Section 7b's document-level WCS/WBH is as far as that machinery reaches.
- **No pre-deployment covariate-shift diagnostic in any source.** `idpfin-q18`/`idpshift-q4` flag this explicitly; Section 7b(iii)'s
  reliability-gap and `is_shift`-coefficient checks are **derived here**, not a validated general-purpose test.
- **Weight positivity failures are undiagnosed by the papers.** None discusses what to do when an estimated weight is (numerically)
  infinite because a covariate has zero support in one arm (`idpshift-q4`); Section 7b(ii)'s covariate set C hit exactly this and the
  fix here (excluding the offending covariate) is a judgment call, not a citable method.
- **Noisy calibration labels: diagnosed, not corrected.** Gurram found automatic labels err one-sidedly pessimistic against blind human
  review ("a certificate against noisy labels is a certificate about those labels", `idpfin-q9`, `idpfin-q16`); Section 9 reproduces the
  pessimistic case (conservative, as the paper predicts) and simulates the **optimistic** case the paper never studies (real violation,
  worse as the noise share grows) -- but no source, and nothing built here, gives a way to **correct** a threshold fit on noisy labels
  back toward the true target; a 150-field audit is shown to resolve only gross violations, not a margin of a few points.
- **Powered document-level PAC is still an open problem, not solved here.** Gurram names it explicitly; Section 8c's bernstein and
  binomial_any_error variants are **derived-here** alternatives to the paper's Hoeffding bound, and they help somewhat (Section 8c) but
  remain "honest but near-vacuous" at this calibration size -- they are candidate directions, not the paper's missing answer.
- **Minimum calibration size.** None gives a closed-form minimum number of calibration units for a non-empty certificate (`idpfin-q12`).
- **Documents and money.** cfBH and SCoRE never mention document extraction (`idpfin-q11`). cfBH names non-uniform error costs as an
  open problem (`idpfin-q3`), addressed partially here by Section 8b's risk-reward score (derived here, not in the papers as a document
  extraction concept). SCoRE lists data-driven score choice and the online setting as open (`idpfin-q6`).

**Engineering choices left open:** This is a starting point, not a finished system. In rough priority order:

1. **Real documents and real labels.** Replace the synthetic corpus with a real sample (CORD, FUNSD and XFUND are what Gurram used; the
   client's own redacted documents are better) and real human labels. This is the single most valuable step: it would show real label
   noise and real error correlation, and it may shrink the coverage numbers here, which come from simulated signals that Notebook 2
   found more informative than a real extractor's.
2. **A real extractor at scale, with a cost policy.** Only a 40-document DeepSeek smoke test exists. Decide the retry / escalation
   policy for truncated or failed calls, batching, caching by template, routing easy fields away from the LLM, and a spend cap.
3. **OCR and layout.** Notebook 2's "OCR text layer" is a stand-in. A real pipeline needs a layout-aware model or vendor OCR with
   page/bounding-box grounding, and a real NLI entailment signal (the fifth signal in Gurram's list, not computed here).
4. **The unit of certification and the score.** Decide whether the business promise is per field (tiers 1-3), per document (tier 4,
   cfBH and SCoRE at the document unit) or per dollar (SCoRE with an amount-weighted loss, or Section 8b's risk-reward-ratio score if
   auto-posted *value* matters more than auto-posted *count*), and set alpha, q and delta with the client. MDR and SDR answer different
   questions.
5. **Calibration data supply.** Decide how many labeled documents you can afford, who labels them (the human review queue is the
   natural source), and how the review feedback loop refreshes calibration without contaminating the score-training data.
6. **Shift handling, now with a real (partial) option.** Section 7b gives a label-free repair for covariate shift IF you can estimate
   `w(x)` on well-behaved covariates -- decide which covariates to track, monitor their Kish effective sample size in production as an
   early-warning signal (a collapsing ESS, as in Section 7b(ii)'s covariate set C, is a symptom of a positivity failure before it becomes
   an outage), and budget for labelled recalibration (Section 7's `B_BUDGET`) for the concept-shift component that weighting cannot fix.
7. **The REST API and the integration guide** (a stated deliverable): nothing here exposes an endpoint. The threshold JSON in section 10
   is a starting point for the config object.
8. **A human-in-the-loop review UI**, queueing and reviewer disagreement handling for the "review" branch.
9. **Audit logging and compliance.** Record, per decision, the model and score versions, the calibration set id, alpha/delta, the
   threshold, and the seed, to support a SOC 2-style audit trail. No compliance work was done here.
10. **Optional extensions from the job posting:** vector search / RAG for document retrieval; KYC/AML-specific rules.

**Using this for the proposal (your words, your experience).** The posting asks for past projects, an approach to low-confidence
extractions, and a timeline. The honest framing of this project is: a research-grounded prototype on synthetic data, with the
approach *"score each field, certify a threshold on held-out documents at the unit the business cares about, route everything else to
review, weight for detected covariate shift where the weights are well-behaved, and recalibrate on new labels for whatever weighting
cannot fix"*. It is not a claim of production experience; describe only what you have actually built or extend it before claiming more.
'''))
    out.append(C(r'''
_ledger_usd = ledger_cost_usd(USAGE_LEDGER)
print(f"Notebook 2's DeepSeek smoke test, priced from its usage ledger (peak pricing): ${_ledger_usd:.4f} for {len(USAGE_LEDGER)} recorded requests")
print(f"Live API calls made by THIS notebook run: {LLM_STATS['live_calls']} (everything served from the disk cache)")
check("S11.1 Notebook 3 made zero live LLM calls", LLM_STATS["live_calls"] == 0, f"live_calls={LLM_STATS['live_calls']}")
'''))
    return out
