# notebooks/ — internal technical notes

This is the detailed companion to the [repo-level README](../README.md): what each notebook actually
builds, cell by cell, and the self-checks that back every number. For the statistics themselves
(intuition + formalism + where they're applied), see [`../research-notes/`](../research-notes/).

## Shared conventions

Every section in every notebook follows the same cell shape: markdown (claim, quoted where it's from a
paper) → one or two computation cells that print numbered self-checks and never plot → a "How to read
this chart" note → one visualization cell that only plots what the computation cell already computed.
One colour palette per notebook, defined once and reused throughout. Every number is labeled as either
quoted from a paper (with section/theorem/equation numbers) or a toy/synthetic simulation — the two are
never allowed to blur together.

## `01_research_foundations.ipynb` — 39 cells, 8 sections

Builds and verifies, by Monte Carlo simulation, every statistical tool the later notebooks use:

1. **Wald vs. Wilson intervals** — Wilson stays within 0.028 of nominal 95% coverage across the tested
   grid; Wald sags to 0.634 at n=50, p=0.98. Not from any of the three papers (standard references:
   Wilson 1927; Brown, Cai & DasGupta 2001).
2. **PPI's estimator (Algorithm 1)** — reproduces the paper's own break-even accuracy figures (25% flip
   error at 50% prevalence, ~9.6% at 10% prevalence) from the closed-form variance comparison, plus a
   power-tuned variant (PPI++, not one of the three source papers).
3. **PPI under label shift** — shows naive cross-month application collapsing coverage to near 0, and
   the label-shift-corrected estimator restoring it to ~0.94–0.95.
4. **The BARGAIN/PPRM betting statistic (Lemma B.1)** — implements the standard Krichevsky–Trofimov
   predictable plug-in (the source paper's own rendering of the formula didn't check out numerically;
   documented and corrected in the cell). Verifies Ville's inequality holds under repeated checking,
   and that naive repeated-CI "peeking" inflates false rejections from ~0.035 at one look to ~0.19 after
   40 looks while the betting statistic stays flat.
5. **BARGAIN-style threshold certification** — two scenarios: a smooth precision curve (naive one-shot
   selection fails 7.0% of the time against a 10% budget; the certified method fails 3.75%) and a
   stressed one with many near-target thresholds (naive fails 12.0%, exceeding the 10% budget; certified
   stays at 1.0%). The paper's own headline evidence (SUPG missing its target >75% of the time) is
   quoted separately.
6. **Recall-target impossibility (Lemma B.11)** — reproduces the closed-form bound and a direct
   simulation of how often a fixed sampling budget finds zero positives when they're rare.
7. **PPRM's confidence sequence (Lemma 2.1, CM-EB)** — implemented with the mixing distribution
   discretized to a 200-point grid (an implementation choice, documented). Verifies the type-I error
   guarantee under a null stream and reproduces the paper's qualitative claims: a good auxiliary
   predictor detects a real shift faster than a labels-only monitor (median step 870.5 vs. 958.5 in this
   notebook's simulation), a weak one with a fixed weight can be slower (1,239.5 vs. 967.5), and the
   adaptive weight recovers most of the gap (962.5).
8. **MinHash/LSH** — unbiasedness and variance of the Jaccard estimator, and the LSH banding S-curve;
   not from the three papers (Broder 1997; Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*).

## `02_project_walkthrough_part1.ipynb` — 36 cells

- **Corpus generator** (`generate_corpus`): 4,800 synthetic postings, 6 months, 9 latent classes
  (including 4 deliberately definition-hard ones: minor/major mixed GenAI duties, classic non-generative
  ML/NLP, and vague AI-strategy roles), a scripted month-5/6 "buzzword wave," and ~20% near-duplicate
  reposts. Tuned so rule scores for true-positive and true-negative postings genuinely overlap
  (26.3% unresolved) rather than being trivially separable.
- **Rule tier** (`rule_score`, `rule_decision`): a weighted keyword-taxonomy scorer over
  title/responsibilities/requirements/skills/blurb, thresholded into `pos`/`neg`/`unresolved`.
- **LLM tier**: real HTTP calls to DeepSeek (`deepseek-flash`), disk-cached by request-body hash, with a
  hard live-call budget, a validated + auto-repaired batch classifier (pydantic), and a usage ledger that
  prices every response — live or cache-replayed — so per-section cost stays meaningful across re-runs.
  A mapping table documents what would change if the same client were pointed at Claude instead.
- **Batching sweep**: k ∈ {1, 10, 25, 50, 100} on a stratified 225-posting sample. Accuracy stays in
  94–97% with overlapping Wilson CIs and zero failures of any kind at any batch size — large batches are
  both safe and cheaper here.
- **End-to-end pipeline and audit**: the hybrid pipeline's coverage by decision path, plus a simulated
  n=300 human audit (precision 0.576 [0.456, 0.688], recall 0.745 [0.611, 0.845]) checked against the
  full-population true values (0.606, 0.793) — both intervals covered the truth.
- **Boundary diagnostics**: an exact (not sampled) rules-vs-LLM comparison over all 4,800 postings by
  rule-score bin and by latent class, plus an empirical sweep of the routing threshold from `hi=1` to
  `hi=12` trading pipeline error against the share of postings sent to the LLM.

## `03_project_walkthrough_part2.ipynb` — 24 cells

Reuses Notebook 2's generator, rules, LLM client, and classify functions verbatim (re-executing its
cells so every prompt is byte-identical and every call is a cache hit), then applies the three papers:

- **BARGAIN-style routing** (months 1–4, a fixed batch): certifies the rule tier's agreement with the
  cached LLM verdict at T=0.9, δ=0.1. The guarantee holds (empirical failure rate 1.5% across 200 runs),
  but because the rule score is a poorly calibrated confidence signal, the certified policy only lets
  the rules answer ~8% of postings on their own before routing the rest to the LLM.
- **MinHash-LSH dedup**: k=64 hashes, 16 bands of 4, confirmed at Jaccard ≥ 0.6 — 78.8% precision, 99.3%
  recall against the generator's own repost ground truth, removing ~999 near-duplicates before audit
  sampling.
- **PPI evaluation**: a simulated 100-label audit per month, comparing a classical Wilson interval
  against PPI with two different predictors. Using the LLM's own label as the PPI predictor shrinks mean
  interval width to about 0.48x of classical; using the noisier end-to-end pipeline label only gets to
  about 1.06x (right around PPI's own break-even, as the theory predicts).
- **PPRM monitoring**: two constructed monthly streams on top of Notebook 2's corpus. A benign
  "buzzword wave" (data changes, accuracy doesn't) correctly produces no PPRM alarms while a naive
  frequency test does flag it as a change. A simulated LLM-tier outage from month 3 (accuracy genuinely
  degrades) is caught in every simulated stream.
- **Cost accounting**: per-1,000-posting cost for each routing policy, from the same usage ledger
  Notebook 2 built, next to each policy's accuracy against the known truth.

## Reproducing this

```bash
cd notebooks
uvx --python 3.12 --from nbconvert --with ipykernel --with numpy --with matplotlib --with pydantic \
  jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  01_research_foundations.ipynb 02_project_walkthrough_part1.ipynb 03_project_walkthrough_part2.ipynb
```

Every notebook is committed already executed — outputs, self-checks, and figures included — so cloning
and reading is enough; re-running is only needed to change something.
