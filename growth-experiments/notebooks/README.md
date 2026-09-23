# Notebooks

| Notebook | Shows | Runtime (warm cache) |
|---|---|---|
| [`01_research_foundations.ipynb`](01_research_foundations.ipynb) | Derives and Monte-Carlo-verifies every guarantee the pipeline relies on. | ~1-3 min |
| [`02_project_walkthrough_part1.ipynb`](02_project_walkthrough_part1.ipynb) | The synthetic SaaS funnel, event tracking, assignment/SRM, and an audit of the founder's daily-peeking baseline. | ~15 s |
| [`03_project_walkthrough_part2.ipynb`](03_project_walkthrough_part2.ipynb) | Screen, measure and iterate applied to Notebook 2's world, plus cost and mock production entry points. | ~4-5 min |

See the portfolio root [`README.md`](../../README.md#growth-experiments) for the papers, the key results table and every figure embedded with
a caption; this file is a per-notebook table of contents.

## 01 — Research foundations

- **Why daily peeking breaks a fixed-n test** (derived here). A/A false-positive rate under daily peeking vs. a single look at day 28.
- **Measure** (Lindon, Ham, Tingley, Bojinov). The exact sequential t-test and confidence sequence: the g-prior e-process, the delayed-start
  minimum sample size (247 at g=1e4, 27 at g=1e2, both reproduced), and a regression-adjusted confidence sequence on a derived binary-outcome
  setting the paper itself does not simulate.
- **Iterate** (Liang & Bojinov). The Mixture Adaptive Design's asymptotic confidence sequence, reproducing its coverage and its regret
  decomposition, with the standard (unmixed) bandit shown failing coverage for contrast.
- **Screen** (Persson, Schultzberg, Ankargren). The attenuation-bias formula (Proposition 3), the surrogacy falsification test, and the
  Proposition 4 sensitivity bound, each checked by simulation against its closed form.
- All self-checks reported PASS or an honest NOTE; no check is scored PASS by loosening its threshold.

## 02 — Project walkthrough, part 1: build

- **The synthetic SaaS population.** A visitor funnel with a *known* latent truth: bots, duplicate-domain signups, existing-customer
  second workspaces, and a real activation-delay tail past the 7-day window.
- **Event tracking vs. ground truth.** What a real analytics pipeline would under-count, using the same population.
- **Assignment and sample-ratio-mismatch checks.** A hash-bucket assignment layer, a uniformity/independence/covariate-balance audit, and a
  worked SRM demo showing a lossy client-side "exposure" event can look statistically wrong even when assignment itself is not.
- **The founder's current practice: a baseline audit.** "Peek daily, stop at first p<0.05" against four synthetic experiments, including one
  where a naive signups-only guardrail cannot see that shipped "winner" truly lowers activated users per visitor.
- **AI headline variants.** A cost-tracked, cache-backed DeepSeek generator plus 24 historical headlines with hidden latent-truth conversion
  lifts, ready for Notebook 3's screen.

## 03 — Project walkthrough, part 2: screen, measure, iterate

- **Screen** (Persson et al.). Calibrates DeepSeek's predicted signup rate against the 24 historical headlines, runs the surrogacy
  falsification test on two calibrators, and ranks the AI-generated variants -- honestly, including where the screen turns out weak.
- **Measure** (Lindon et al.). The regression-adjusted anytime-valid confidence sequence applied to Notebook 2's four experiments, compared
  against the naive daily-peek baseline's false-positive rate and against a fixed-horizon test's sample size.
- **Iterate** (Liang & Bojinov). A 4-arm Mixture Adaptive Design over a screened, a random, and an oracle slate of headlines, plus a
  decaying-novelty scenario showing what the confidence sequence's estimand actually tracks.
- **Cost.** What the LLM screen cost in dollars, and what each measurement method costs in days of traffic, side by side.
- **Mock production.** Three independent, API-shaped entry points (`llm_screen`, `fixed_ab`, `mad`) over evidence already computed above,
  with an embedded Next.js/TypeScript contract -- not an automated pipeline, and not a claim that the papers prove a composition guarantee
  across methods.

## Limitations

See the root [`README.md`](../README.md#growth-experiments) for the full statement. In short: every result here is on synthetic data with a
known ground truth chosen for this notebook; the three papers say nothing about page-level (programmatic SEO) experiments, multi-experiment
error control, or a covariate-adjusted Mixture Adaptive Design, so anything on those topics is derived here, not sourced from the papers.
