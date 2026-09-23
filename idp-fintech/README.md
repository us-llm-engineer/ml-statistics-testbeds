# idp-fintech: "Document Processing Automation Developer for Fintech SaaS Platform"

Upwork job ~022100186467707705409. The client is a fintech SaaS with high daily volumes of invoices, bank statements, KYC forms and
compliance reports. Review is manual today, and they want a pipeline that works as follows:
1. classify each document;
2. extract its fields;
3. validate the fields against business rules;
4. **flag low-confidence extractions for human review instead of failing silently**;
5. document extraction accuracy and confidence thresholds.

This project is a **starting point**, not a finished product. It is research grounding plus scaffold notebooks that the human
extends into their own profile project. Every result on synthetic data is labelled as such.

One story runs through all three papers: *certify what gets auto-accepted → price the errors → make it hold on real documents*.

- **Certify (foundational):** Jin & Candès, "Selection by Prediction with Conformal p-values", *JMLR* 24(244), 2023, arXiv:2210.01408.
  cfBH selects the units to trust and keeps the FDR of the selected set ≤ q, in finite samples.
- **Price the errors:** Bai & Jin, "Conformal Selective Prediction with General Risk Control" (SCoRE), arXiv:2603.24704, 2026.
  It controls bounded, continuous risks such as amount-weighted errors, either as a marginal budget (MDR) or as a per-accepted-unit
  average (SDR), using e-values. It also covers covariate shift.
- **Make it hold on documents:** Gurram, "Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a
  Validity Ladder, and When Conditioning Pays", arXiv:2608.14639, 2026. It covers:
  - three failure modes: document clustering, score-refit leakage, tie mass;
  - the fit/val protocol;
  - Mondrian Learn-then-Test PAC tiers.

Runners-up and why each lost:
- Kotte 2606.29054 (CRC feasibility floor): generic CRC rather than selection-conditional, and unreviewed.
- SCORC 2606.08517: its joint certificate is evaluated on vision benchmarks.
- Trust or Escalate 2407.18370: a judge cascade that overlaps BARGAIN, which the nlp-job project already uses.
- Conformal Alignment 2405.10301: subsumed by #1 and #2.
- ExtractConf 2606.24420 and ConfBench 2608.01792: confidence signals only, no guarantee.
- CRC 2208.02814: marginal rather than selection-conditional.

## Status: all three notebooks built, executed and reviewed (round 2: 2026-09-23)

| notebook | what it does | verified |
| --- | --- | --- |
| `01_research_foundations.ipynb` | derives each paper's estimator/test and Monte-Carlo-verifies every guarantee on toy data; round 2 adds continuous-Y clip vs residual, cfBH0, known-weight covariate shift (weighted SCoRE, WCS) and where weighting cannot help (concept shift, missing support) | 76 cells, 0 errors, 16 plots, 86/86 self-checks PASS |
| `02_project_walkthrough_part1.ipynb` | synthetic fintech corpus with latent truth (3,500 docs), classifier, real DeepSeek smoke test (40 docs) + simulated extractor, business rules, per-field signals, folklore baseline audit | 0 errors, 0 live calls on re-run, all checks PASS |
| `03_project_walkthrough_part2.ipynb` | applies the papers: add-one failure modes, cfBH at the document unit, SCoRE with an amount-weighted loss, the validity ladder per document type, month-7 shift; round 2 adds label-free shift repair (section 7b), power checkpoints (section 8: boosting, risk-reward score, tier-4 variants, regime 2, grouped taxonomy) and label noise with a 150-field audit (section 9) | 118 cells, 0 errors, 21 plots, 192/192 self-checks PASS |

Everything is a **draft scaffold on synthetic data**, not production experience. The simulated extractor's signals are probably
more informative than a real extractor's (Notebook 2 compares them with DeepSeek's), so every NB3 result is reported for an
**optimistic** and a **pessimistic** score.

Headline findings (all ours, on the synthetic corpus; see the notebooks for tolerances):

- The add-one rule meets its expectation target (risk about 0.10) but exceeds alpha in about half of document-level batches; refitting a
  flexible score on the calibration data itself overshoots in 38 of 40 splits.
- cfBH applied **per field and then aggregated to documents** inflates document-level FDR (0.244 at q = 0.10); applied at the document
  unit it holds (0.099).
- SCoRE-MDR at alpha = 0.10 is met (0.099), but the average loss per accepted document is 0.129: MDR is a budget, not a selective rate.
  Exact SDR e-values are feasible at n = 796 (well under a second per batch) but very conservative for a continuous loss.
- Mondrian LTT (PAC) certifies 0.62 of fields at alpha = 0.10 with the optimistic score. The pessimistic score averages 0.014
  coverage over 40 resplits but reaches 0.061 on one draw, demonstrating that a low mean is not a per-draw upper bound; it needs
  alpha = 0.20 for useful average coverage.
- A frozen configuration violates on the month-7 shift batch in every configuration tested; re-calibrating on about 200 labelled
  shift documents restores validity but not coverage.
- Label-free repair (round 2): with KNOWN covariate-shift weights, weighted SCoRE (MDR/SDR) and weighted conformal selection stay within
  budget exactly, as the theorems promise. On the real month-7 batch the weights must be estimated, and month 7 is partly a concept shift
  (P(y | score) moved; bootstrap CI of the shift coefficient excludes 0), so estimated weights make MDR worse (0.185 vs 0.096 unweighted).
  A 'template seen before' covariate separates the months perfectly: a positivity failure no source handles.
- Tier 4 (per-document PAC) per document type: Gurram's Hoeffding bound certifies nothing at alpha = 0.10; a derived-here empirical-Bernstein
  bound certifies 0.45-0.49 of fields per type, with violations within the delta budget over 40 resplits.
- Label noise: optimistic automatic labels push the add-one rule over budget (true risk 0.126 at 20% noise), and a 150-field audit cannot
  tell 0.126 from 0.10 (Wilson CI [0.099, 0.212]).
- Cost (derived from the smoke-test ledger): about $0.003 per document at k = 1 and $0.009 at k = 3 LLM samples; human review cost is a
  placeholder parameter.

Real API spend: about $0.37 is what the final referenced smoke-test cache reflects (139 requests: 40 documents x 3 samples plus
18 retries of truncated calls). Development iterations that were superseded cost more; the sum of the individually reported figures
is roughly $0.65 in total. All notebook re-runs make zero live calls.

Papers, key results, and 6 chosen figures are in the [portfolio root README](../README.md#idp-fintech).
## Contents

- `lib/guarantees.py` + `tests/test_checkpoints.py` (round 2): weighted SCoRE e-values, e-BH boosting, weighted conformal selection (WCS),
  domain-classifier weights with balancing, tier-4 variants; 77 spec tests against brute-force paper formulas (`.venv/bin/python -m pytest tests -q`).
- `nlm/` (private, not committed): NotebookLM queues `idpfin` (18 answers) and `idpshift` (Jin & Candes 2023a, Tibshirani et al. 2019; 4 answers).
- `research/idp-fintech-research-log.md`: all 12 answers, each with a design note.
- `research/notebook-research-context.md`: the consistency contract (lookup only) and the per-section re-read map.
- `research/reread.py`: prints only the requested queries (server history first, local files as fallback).
- `notebooks/01_research_foundations.ipynb`, `02_project_walkthrough_part1.ipynb`, `03_project_walkthrough_part2.ipynb`.
- `notebooks/build/`: the builder scripts (`build_nb1.py`, `build_nb2.py`, `build_nb3.py` plus its parts). Rebuild with
  `.venv/bin/python notebooks/build/build_nb3.py`, then execute.
- `data/llm_cache/`: disk cache of DeepSeek responses, so re-runs cost nothing. `data/nb2/`: Notebook 2's exported corpus.
- `.venv/`: local virtualenv providing the Jupyter kernel `idp-fintech` (not meant to be committed).

Run a notebook from the project root:
`.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=idp-fintech --ExecutePreprocessor.timeout=3600 notebooks/<nb>.ipynb`
(run 01, 02, 03 in any order; 03 reloads 02's tagged cells itself.)

## Limitations and open engineering choices

See the "Limitations and open engineering choices" sections of Notebooks 2 and 3: real documents and labels, a real extractor with a cost and
retry policy, OCR/layout, the unit of certification (field, document or dollar), calibration-data supply, shift handling, the REST API
and integration guide, the review UI, and audit logging.
