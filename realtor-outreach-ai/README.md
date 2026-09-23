# realtor-outreach-ai -- in-depth notes

This is the technical companion to the [portfolio root README](../README.md), which has the papers,
the headline results table, and the figures for this project. This file covers the statistics each
notebook leans on in more depth: what each method is for, the math behind it, and exactly where in
the notebooks it gets applied.

## The statistics behind it

### Rank-weighted treatment-effect evaluation

#### Elaboration

##### Intuitive description
A scoring rule for "who to call" is only useful if the people it ranks highly are the people a call
actually changes, not just the people who were always going to book. This method turns that question
into a curve: rank everyone by the score, sweep the fraction contacted from 0% to 100%, and at each
fraction plot the average treatment effect among the top-ranked group so far. A good uplift score
front-loads the curve toward the high-uplift leads; a plain "likely to book" score front-loads toward
leads that would have booked with or without a call, and can even show a *negative* effect in its own
top slice.

##### Mathematical formalism
For a scoring rule S(x) and outcome/treatment pair (Y, W) with known propensity, define the targeting
operating characteristic TOC(u; S) as the average treatment effect among the top-u fraction ranked by
S, minus the overall average treatment effect. Integrating TOC against a weight function α(u) gives a
single rank-weighted average treatment effect (RATE); α ≡ 1 gives an AUC-style summary, α(u) = u gives
a Qini-style summary that upweights the very top of the ranking more heavily. Each unit's contribution
is estimated with an augmented inverse-propensity-weighted (AIPW) score so the estimator stays
efficient under a known randomization design, and a half-sample bootstrap gives a standard error and
confidence interval for the resulting RATE.

#### Practical use cases in this Project
Notebook 1 §§1-3 implements TOC/RATE from these definitions, cross-checks the AUTOC/Qini closed forms
against each other, and validates bootstrap coverage and test size against known-truth simulations
before ever touching real data. Notebook 3 §3 applies it to compare the heuristic rule, the LLM fit
score, and a learned uplift model against the oracle uplift on the pilot data -- which is how the
"the heuristic rule is actively harmful" finding above gets its confidence interval instead of being
a single suspicious number.

---

### Calibrated LLM-based outcome pretesting

#### Elaboration

##### Intuitive description
Running a real test of two email variants means waiting for real leads to reply, which is slow and
burns real leads on variants that might lose. The tempting shortcut is to have an LLM read the email
and a synthetic persona and guess the reaction instead. The problem is that an LLM's guess is on a
different scale and has a different bias than a real human's binary yes/no, so a raw LLM-vs-LLM
comparison can be wildly miscalibrated. The fix is to spend a *little* real human data calibrating a
correction function between "what the LLM says" and "what a human actually does," then apply that
correction to the cheap, LLM-only comparison.

##### Mathematical formalism
Let Y be the real human outcome, Y* the LLM-generated surrogate, and X the covariates. Under a
surrogacy assumption (the treatment affects Y only through Y*, given X) and a comparability assumption
(the Y*-to-Y relationship is the same in the calibration sample and the target sample), a calibration
function μ(x, y*) = E[Y | X=x, Y*=y*] estimated by cross-fitting on a small human-labeled sample, and
plugged into the LLM-only sample, identifies the true treatment effect. A falsification test checks
comparability directly: it should not be possible to predict *treatment status* from (X, Y*) alone on
held-out human data; a nonzero, non-noise signal there is the earliest sign the shortcut isn't safe to
trust. Repeated LLM draws per unit (K draws) reduce attenuation from a noisy correction function.

#### Practical use cases in this Project
Notebook 1 §§4-5 builds this identification result and the falsification test from scratch and
verifies both the calibration bias-correction and the test's own size/power in simulation. Notebook 3
§2 runs the real thing: a DeepSeek-based persona simulates each lead's reaction to two email variants,
the correction is cross-fit on a small real-outcome sample, the falsification test is checked before
the calibrated number is trusted, and the section is explicit that at this pilot size the corrected
estimate's confidence interval still includes zero -- "no evidence of an effect," reported honestly
rather than as "no effect."

---

### Sample-split personalization testing

#### Elaboration

##### Intuitive description
Suppose you have three email templates and want to know if picking the *best template for each lead*
individually beats just sending everyone the single best-performing template. The naive way to check
is to fit a per-lead policy and then measure how much better it does *on the same data used to fit
it* -- but that's exactly what lets you "discover" a personalization benefit that isn't real, because
picking a winner and then grading that same winner on the same draw is a biased comparison (the
"optimizer's curse"). The fix is to never let the same slice of data both choose the policy and grade
it: split the data, fit on one part, grade on a disjoint part, and repeat that split many times to
average out which particular split you got unlucky or lucky with.

##### Mathematical formalism
Define the personalization effect ψ as the gap between the value of the best per-lead policy (from a
constrained policy class) and the value of the single best fixed action, both measured under a known
or estimated propensity via a doubly-robust score. A single split estimates ψ̂ by fitting the policy on
a policy-fold, fitting outcome/propensity nuisances on separate nuisance-folds, and evaluating the
doubly-robust score only on a held-out evaluation fold that touched neither fitting step. Repeating
this over S independent splits and combining the per-split estimates and their per-observation score
variances gives a final estimate ψ̃, a standard error, and a z-statistic whose rejection rate is
asymptotically controlled at the nominal level under the null of no personalization benefit.

#### Practical use cases in this Project
Notebook 1 §6 first demonstrates the naive-comparison bias directly (a >50% false-positive rate under
a constructed no-effect world) and then implements the split-sample test and its power curve in
simulation. Notebook 3 §4 runs both the naive and the split-sample version side by side on the real
pilot and on a constructed no-effect world, which is where the "60% false-positive rate vs ~1%"
comparison above comes from, plus the pilot-size power sweep that quantifies how much more data this
particular question would need.
