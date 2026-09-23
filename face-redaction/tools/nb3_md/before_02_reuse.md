### Reload Notebook 2's pipeline, verbatim

**Claim.** The larger corpus and every evaluation below use exactly the same simulator, tracker, evaluator and cost function as Notebook 2, not a re-typed copy. The cell executes Notebook 2's six cells marked `# REUSE` in order and then reproduces Notebook 2's printed numbers from the same seed.

**CHECKs (a-priori thresholds).**
- 0.1: Notebook 2 has exactly 6 `# REUSE` cells and all execute.
- 0.2: `generate_corpus(SimConfig(seed=SEED))` gives NB2's episode and track counts exactly.
- 0.3: `evaluate()` on NB2's naive configuration reproduces NB2's printed instance recall, frame recall, precision and instance count to its 3 printed decimals (|difference| <= 5e-4; the reproduction is in fact exact, so the tolerance is only a rounding allowance).

This is **derived here** bookkeeping: none of the three papers concerns it, but a certificate about a pipeline is only worth something if the pipeline being certified is the one that ships.
