# Notebook 3 of 3 -- Decide, Certify, Re-verify: applying the three papers to the face-redaction pipeline

**What this is.** A draft scaffold and starting point that you extend into your own profile project. It is not a finished product and it is not production experience. Everything below runs on a **synthetic** generator (the one from Notebook 2, with more episodes); a number is only called **measured** when it comes from a real run on this machine (the smoke-test timings), and every price, speed-up and footage-density figure is marked **assumed** and must be replaced by you. The job needs no language model, so none is used anywhere in this notebook.

**The client's job.**
- Blur every face of a bystander in 4K head-mounted household video, and nothing else that matters.
- Requirement: recall >= 99% per face instance, precision >= 95%.
- Minimise the processing cost per video-hour among configurations that meet the requirement.
- Problem: the client cannot measure recall today, because their sample contains no faces (Notebook 2 showed why "no misses on our sample" proves nothing).
- So the proposal must show *how* 99% would be proven, not merely claim it.

**The three papers and the role each plays here.**

| Paper | Role in this notebook |
|---|---|
| P1 Learn then Test (LTT), arXiv:2110.01052 | **Decide.** Choose a configuration with a high-probability (1 - delta) guarantee on several risks at once, or abstain. |
| P2 Sequential conformal risk control (SeqCRC), arXiv:2505.24038 | **Contrast.** Shows what an *in-expectation* guarantee is, and why it is not a 99%-with-confidence claim; it does not control precision. |
| P3 Finite-sample coverage audits, arXiv:2607.21480 | **Certify.** A sealed audit that turns labelled face instances into a lower bound on recall, and rules for keeping design and certification data apart. |

**What this notebook does.**
1. Reloads Notebook 2's pipeline and builds a larger synthetic corpus with three episode-level splits.
2. Sizes the labelled audit set and the staged capture the client would need.
3. Uses LTT to look for a configuration that certifies the client's targets (it finds none), revises a demonstration target set in the open, verifies the procedure by simulation, and asks what detector quality would change the answer.
4. Contrasts SeqCRC with LTT.
5. Certifies on a sealed audit split.
6. Prices the options.
7. Hands the client a re-verification harness, then summarises results and decisions.

**Reading guide.** Every number carries one of five labels.

| Label | Meaning |
|---|---|
| **PAPER NUMBER** | Reproduced from a paper (with section or theorem and the research-trace slug, e.g. `[P3 Sec. 4.6, facered-q7]`). |
| **derived here** | Our own construction or computation; the papers do not cover it (tracking, per-instance units on video, costs, the false-blur risk definition). |
| **synthetic** | Produced by the simulator; a modelling choice, never a real-world finding. |
| **measured** | A real timing from this machine's smoke test. |
| **assumed** | A placeholder parameter (price, speed-up, capture rate) that you must replace. |

Each computation is preceded by the claim it tests and the *a-priori* threshold its `CHECK` uses. A `CHECK` is marked PASS only if it met that stated threshold; "no certified configuration" is a valid, reportable outcome.

<!-- cell -->
## Setup

The next cell fixes the master seed (the same one Notebook 2 used), the colour palette, and two small pieces of bookkeeping that matter for honesty:

- `check(id, description, ok, detail)` prints one numbered self-check. The description states the threshold *before* the number is seen, and PASS or FAIL is never adjusted afterwards.
- `log_event(name)` keeps an ordered event log. P3's Assumption 1 (design-certification separation) is about *order*: the configuration being certified must be fixed before the certification labels are opened, so we record when targets are fixed and when each split is opened.
