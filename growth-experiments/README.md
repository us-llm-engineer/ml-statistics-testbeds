# growth-experiments — "AI Growth Engineer — Build & Ship SaaS Growth Experiments"

Upwork job ~022100358403496207283. The client is a founder-led B2B SaaS (Next.js/TypeScript, low traffic) that wants
*idea → working experiment → real users → data*: landing-page and onboarding experiments, programmatic SEO, AI
workflows, experiment tracking.

> **This is a research-grounded starting point**, not finished or production work. The walkthrough notebooks use
> synthetic data with known latent truth; the human extends them into their own shipped experiment.

This file and [`notebooks/README.md`](notebooks/README.md) are text-only technical notes. The papers, the key
results table and every figure with a caption are in the portfolio root [`README.md`](../README.md#growth-experiments).

## One story: screen → measure → iterate

Run the three notebooks in order (see [`notebooks/README.md`](notebooks/README.md) for a per-notebook synopsis).

| Role | Paper |
|---|---|
| **Screen** AI-generated variants before they spend traffic | Persson, Schultzberg, Ankargren (Spotify), *Statistical Foundations of LLM-based A/B Testing*, arXiv:2606.17165 |
| **Measure** with daily peeking allowed (foundational) | Lindon, Ham, Tingley, Bojinov, *Anytime-Valid Inference in Linear Models and Regression-Adjusted Causal Inference*, arXiv:2210.08589 (JASA 2026) |
| **Iterate** across variants while shifting traffic to winners | Liang & Bojinov, *An Experimental Design for Anytime-Valid Causal Inference on Multi-Armed Bandits* (MAD), arXiv:2311.05794 |

Runners-up and why they lost: Molitor & Gold 2506.20523 (covariate-adjusted MAD; proof copied from MAD, arXiv
only), Arbour et al. (KDD 2026) 2606.08853 ("no new estimators", light theory), Lee et al. (Meta) 2607.23696
(generative creatives + bandits; empirical, no theorems), Bojinov–Simchi-Levi–Zhao 2009.00148 (switchbacks; fits
only the programmatic-SEO bullet), Netflix's surrogate-index study 2311.11922 (needs long-term outcome history the
client lacks).

## Limitations

Every result is on synthetic data with a known ground truth chosen for this notebook. The three papers say nothing
about page-level (programmatic SEO) experiments, error control across many experiments over time, or a
covariate-adjusted Mixture Adaptive Design; the papers themselves do not prove a composition guarantee across the
screen/measure/iterate steps either, so anything on those topics is derived here, not sourced.

## Run it

```bash
cd growth-experiments
/usr/local/bin/python3 tools/run_notebook.py notebooks/01_research_foundations.ipynb
/usr/local/bin/python3 tools/run_notebook.py notebooks/02_project_walkthrough_part1.ipynb
/usr/local/bin/python3 tools/run_notebook.py notebooks/03_project_walkthrough_part2.ipynb --timeout 1800
```

Python 3.12 with numpy + matplotlib only (scipy/sklearn/pandas unavailable in this environment). Fixed seeds make
the Monte Carlo runs reproducible; Notebook 2 caches DeepSeek responses in `data/llm_cache/` (created on first run,
safe to delete), and a re-run makes zero live API calls. A `DEEPSEEK_KEY` is needed only to regenerate that cache.

## Layout

```
notebooks/         the three executed notebooks, plus notebooks/README.md
figures/           6 PNGs extracted from the executed notebooks, embedded in the portfolio root README, plus captions.json
tools/run_notebook.py   executes a notebook in place (no nbconvert needed)
README.md
```

Local NotebookLM traces, research notes and response caches are intentionally excluded from the published project.
