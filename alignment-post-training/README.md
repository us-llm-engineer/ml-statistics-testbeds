# alignment-post-training — "AI Alignment / Post-Training Researcher"

This profile project is built around an AI-alignment/post-training research brief: given human-written reasoning
traces, rubrics, and preference data for SFT, DPO, or GRPO, predict whether a datum will create a useful gradient
signal.

This project is a **starting point for a profile project**, not a record of production work. The walkthroughs run on a
synthetic corpus with known latent truth, and every limitations list is a starting point for further extension.

One story in three papers: *will a datum move the model, in the right direction where it counts, without side effects?*

- **Magnitude — reward variance.** Razin, Wang, Strauss, Wei, Lee, Arora, *What Makes a Reward Model a Good Teacher? An
  Optimization Perspective*, NeurIPS 2025, arXiv:2503.15477. Low reward variance under the policy flattens the objective
  whatever the accuracy (Theorem 4; tabular Proposition 3).
- **Direction in the tail — rubric rewards.** Zhang, Wang, Gui, et al., *Chasing the Tail: Effective Rubric-based Reward
  Modeling for LLM Post-Training*, ICLR 2026, arXiv:2509.21500. Misspecification among the best responses dominates
  (Theorem 1); rubrics should separate excellent from great.
- **Side effects — likelihood displacement.** Razin, Malladi, Bhaskar, Chen, Arora, Hanin, *Unintentional Unalignment:
  Likelihood Displacement in Direct Preference Optimization*, ICLR 2025, arXiv:2410.08847. Preference pairs with similar
  hidden embeddings (high CHES) push probability off the preferred response.

The three papers were selected after a broader sweep over DPO/GRPO gradient signal, reward-model quality, rubric rewards,
reasoning-trace selection, and RLVR prompt selection. The strongest runners-up were:
- *Learning Dynamics of LLM Finetuning* (ICLR 2025): first-order decomposition under a stable-eNTK assumption, and no
  per-datum score.
- *Which Reasoning Trajectories Teach Students…* (Rank-Surprisal Ratio): strong empirical trace metric, but its authors
  state it has no theoretical framework.
- *GRPO's policy gradient is a U-statistic*: about group size and estimator MSE, not data value.
- *SFT Overtraining Predicts Rank Inversion…*: a workshop paper; its p(1−p) algebra is re-derived here instead.

Papers, key results, and 9 chosen figures are in the [portfolio root README](../README.md#alignment-post-training).

## Contents

- `notebooks/01_research_foundations.ipynb`: derivations, with every guarantee checked by simulation.
- `notebooks/02_project_walkthrough_part1.ipynb`: synthetic replication of a trace/rubric/preference-pair collection
  framework and its accuracy-centric QA baseline, plus a capped DeepSeek verifier smoke test.
- `notebooks/03_project_walkthrough_part2.ipynb`: the three papers applied to that corpus (reward-variance audit, rubric
  tail audit, CHES and toy DPO on a small real model), cost accounting, and limitations and open engineering choices.

The generated corpus and caches are written under `data/` when the notebooks run and are deliberately not versioned.

## Environment

Python 3.12 was used for the recorded run. Create an environment with Jupyter, NumPy, pandas, SciPy, Matplotlib,
PyTorch, and Transformers, then execute the notebooks in order. Notebook 2's 24-call DeepSeek smoke test needs a
`DEEPSEEK_KEY` only when its local response cache is cold; all other data is generated deterministically.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  notebooks/01_research_foundations.ipynb \
  notebooks/02_project_walkthrough_part1.ipynb \
  notebooks/03_project_walkthrough_part2.ipynb
```
