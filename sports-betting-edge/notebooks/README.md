# Notebooks

| Notebook | Shows | Runtime (cold cache) |
|---|---|---|
| [`01_research_foundations.ipynb`](01_research_foundations.ipynb) | Derives and Monte-Carlo-verifies every guarantee the pipeline relies on. | ~40 s |
| [`02_mock_project_part1_build.ipynb`](02_mock_project_part1_build.ipynb) | The project brief, the synthetic-data simulator, the naive baseline and its audit. | ~70 s |
| [`03_mock_project_part2_prove_screen_combine.ipynb`](03_mock_project_part2_prove_screen_combine.ipynb) | The certified registry, combiner, monitor, money view and scorecard. | ~5.5 min |

See the portfolio root [`README.md`](../../README.md#sports-betting-edge) for the papers, the key results table and every figure embedded with a caption;
this file is a per-notebook table of contents.

## 01 — Research foundations

- **Why naive backtests lie** (derived here). With $K$ null signals, the chance of at least one false positive at $p<0.05$ is $1-0.95^K$ — 87% by $K=40$ —
  and the best of $K$ test statistics grows like $\sqrt{2\ln K}$, not the single-test threshold 1.96.
- **Prove** (Choe and Ramdas). Derives $\Delta_t$, Ville's inequality, the Hoeffding-style and empirical-Bernstein confidence sequences (Thm 1, Thm 2) with
  the gamma-exponential mixture boundary, and shows a fixed-time interval checked at every round is not valid under repeated looking.
- **Prove, continued.** The weak null $H_0^w$ vs. the strong null $H_0^s$ (the paper's Eq. 22 example), the sub-exponential e-process (Thm 3), the exact
  duality between the confidence sequence and the e-process, and Appendix F.1's predictable-subsequence inference.
- **Screen** (Xu and Ramdas). LOND, r-LOND and e-LOND; Theorem 1's FDR guarantee under arbitrary dependence, checked under independent, banded and
  adversarial dependence; the Appendix C sharpness construction, reached almost exactly.
- **Combine** (Chen, Huang, Jordan and Luo). Bin-wise calibeating (Algorithm 1) and multi-calibeating with Hedge (Algorithm 2); regret against the
  external forecaster's refinement score grows like $\log T$, matching the paper's rate. The paper has no experiments of its own — every number here is
  a simulation built to test its theorems.
- **Composition** (derived here). Feeds stopped e-processes from 40 dependent signals into e-LOND, and a confidence sequence around a calibeated
  forecast — the exact pipeline Notebooks 2 and 3 run at project scale.
- 57 self-checks, all met.

## 02 — Mock project, part 1: build

- **The brief.** A subscription betting business with 40 candidate signals and four acceptance criteria (AC1–AC4, defined here and scored in Notebook 3).
- **The simulator** (derived here). 12 historical seasons plus live seasons, a 4.5% overround, a planted longshot mispricing, and 40 signals with a known
  truth: 4 real edges, 1 edge smaller than the margin, 1 decaying edge, 6 correlated proxy signals, 1 look-ahead bug, 27 nulls.
- **Forecasters.** Frozen after 3 training seasons, so every forecast is predictable — the assumption every guarantee in Notebook 1 depends on.
- **The naive pipeline.** Screen at $p<0.05$, stack the winners, report flat-stake ROI in-sample and forward.
- **Baseline audit**, 200 simulated worlds. The naive screen's false-discovery proportion is 18.7% against a 10% target, and it publishes the look-ahead
  signal in 95.5% of worlds — this is the baseline Notebook 3's certified pipeline is measured against.
- 36 self-checks, all met.

## 03 — Mock project, part 2: prove, screen, combine

- **Prove.** A confidence sequence and e-process per signal against the de-margined market; the cost of proof (games needed until the interval excludes
  zero) by signal class.
- **Screen.** A two-stage registry (a backtest shortlist, confirmed on live seasons where the look-ahead bug is absent) driven by e-LOND; this is what
  takes AC1's false-discovery proportion to 0.000 and AC2's look-ahead publication rate to 0 of 200 worlds.
- **Combine.** A Hedge pool over the market and the certified signals, evaluated on fresh seasons the registry never saw; clears AC3, while calibeating
  the market alone does not recover the planted mispricing at this data size.
- **Monitor.** No valid test flags the decaying edge within the AC4 target (3,690 games) — the notebook reports the achievable detection-time
  distribution instead of loosening the criterion — and demonstrates the restart trap: retesting from a data-chosen start inflates false alarms on
  stable edges.
- **Money and the picks feed** (derived here). Brier gain converted to expected ROI at margined odds; a sample picks table built only from certified
  signals.
- **The scorecard.** AC1–AC4 computed by code for both the naive baseline and this pipeline: AC1–AC3 met, AC4 not met by either pipeline.
- 29 self-checks, all met, plus one reported finding (calibeating the market alone does not recover the planted mispricing).

## Limitations

See the root [`README.md`](../README.md#limitations) for the full statement. In short: the simulator's effect sizes and class layout are choices; the
three papers say nothing about profit, margin, staking, costs or the favourite-longshot bias, so every money figure is derived here, not sourced from the
papers; confidence sequences describe performance achieved so far, not the future.
