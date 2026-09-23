# Sportsbook Edge Lab

**Screen, prove, combine: a simulation-based mock project on finding betting edges without fooling yourself.**

A subscription betting business has 40 candidate signals (referee tendencies, rest and travel, recent form), a folder of backtests that all look good, and a
question it cannot answer: *which of these edges are real, which are luck, and which are bugs?* This project builds the whole engagement end to end, on a
simulator where the truth is known, so every claim can be scored against reality.

> **This is a simulation study on synthetic data.** It contains no real bets, no real odds and no claim about real sportsbooks. Every result describes the
> simulator, not real markets. It is not betting advice. Money figures (flat stakes, a 4.5% overround) are simple illustrations, not a trading system.

This file and [`notebooks/README.md`](notebooks/README.md) are text-only technical notes. The papers, the key results table and every figure with a
caption are in the portfolio root [`README.md`](../README.md#sports-betting-edge).

## The three notebooks

Run them in order (see [`notebooks/README.md`](notebooks/README.md) for a per-notebook synopsis). Each one is executed and saved with its outputs, so it
renders on GitHub.

| Notebook | What it does | Checks |
|---|---|---|
| [`01_research_foundations.ipynb`](notebooks/01_research_foundations.ipynb) | Derives each paper's estimator or test from first principles and verifies every guarantee by Monte Carlo (coverage, error rates, regret, FDR). Shows why naive backtests lie, then composes the three methods. | 57 self-checks, all met |
| [`02_mock_project_part1_build.ipynb`](notebooks/02_mock_project_part1_build.ipynb) | The project brief and acceptance criteria, a simulator standing in for the data feeds (40 signals with a hidden truth table), the frozen forecasters, the client's naive pipeline, and an audit of that pipeline over 200 simulated worlds. | 36 self-checks, all met |
| [`03_mock_project_part2_prove_screen_combine.ipynb`](notebooks/03_mock_project_part2_prove_screen_combine.ipynb) | Applies the three papers: a two-stage certified registry, a combined forecaster, a decay monitor, a money view, a sample picks feed, and a scorecard against the acceptance criteria. | 29 self-checks, all met, plus reported findings |

## What the simulator plants

Each world has 12 historical seasons plus live seasons, 40 candidate signals and a known truth: 4 real edges (2 to 3.5 points of win probability), 1 real
edge smaller than the bookmaker margin, 1 edge that decays, 6 correlated proxy signals, 1 **look-ahead bug** (the historical feature contains part of the
outcome), 27 pure nulls, and a small mispricing of longshots. Because the truth is known, false discoveries can be counted exactly.

## Limitations

The simulator is the world: effect sizes, class layout and the overround are choices, and power by class follows from them. The three papers score forecasts with
proper scoring rules and say nothing about profit, margin, staking (Kelly), costs, bet limits, three-way outcomes or the favourite-longshot bias; all money
figures here are derived in the notebooks. Confidence sequences describe performance achieved so far, not the future. The calibeating paper has no experiments and
its bounds depend on the number of distinct forecast values, so forecasts are rounded to a 20-bin grid.

## Run it

Python 3.12. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name edgelab --display-name "Python (edgelab)"
cd notebooks
for nb in 01_research_foundations 02_mock_project_part1_build 03_mock_project_part2_prove_screen_combine; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=edgelab \
    --ExecutePreprocessor.timeout=3600 "$nb.ipynb"
done
```

or open them in Jupyter in that order. Fixed seeds make the runs reproducible. Notebooks 2 and 3 cache their multi-world audits in `data/` (created on first
run, safe to delete); Notebook 3 reads the exported cells of Notebooks 1 and 2, so run them in order. Measured cold-cache times on a 12-core workstation, from a fresh
virtual environment: about 40 seconds, 70 seconds and 5.5 minutes (roughly 8 minutes in total); with warm caches Notebooks 2 and 3 take under a minute each.
On Windows the audits run in one process instead of a pool and take longer.

## Layout

```
notebooks/         the three executed notebooks, plus notebooks/README.md
figures/           8 PNGs extracted from the executed notebooks, embedded in the portfolio root README, plus captions.json
requirements.txt
README.md
```
