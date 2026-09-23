### The configuration grid and a per-episode loss cache

Every configuration is a point `(score threshold, stride, max gap, temporal pad, margin)` of Notebook 2's pipeline. The grid **Lambda** here extends NB2's toward more permissive settings (lower thresholds, more bridging, more padding) plus a few strided settings, because stride is the only knob that lowers cost. Grid design is a choice made before any calibration data are seen; P1 itself lists the discretisation of the parameter space as a design decision `[P1 limitations, facered-q3]`.

**Why per-episode counts.** For each configuration and each episode the cache keeps four integers: face instances, missed instances (strict rule: at least one visible frame not covered), blur box-frames, and false blur box-frames. Episodes are independent, so counts add across episodes and both risks used later (section 3) can be recomputed for any subset of episodes without re-running the pipeline. This is **derived here** engineering, not paper content.

**CHECK 1.4 (a-priori budget).** The estimated CPU time to evaluate the whole grid on TUNE, CAL and TRUTH is at most 1,500 CPU-seconds. It is an engineering budget, not a statistical claim.
