**Two target sets, fixed before any CAL or AUDIT label is touched.** P3's Assumption 1 prohibits "use of certification labels to construct, tune, order, or select" what is being certified `[P3 Assumption 1, facered-q7]`; this cell applies the same discipline to the targets, by writing them to `data/nb3/prereg_targets.json` and to the event log before CAL is evaluated.

- **T1** is the client's: alpha_miss = 0.01 (recall >= 99% per instance), alpha_fb = 0.05.
- **T2-v1** is a relaxed **demonstration** target (**derived here**), chosen by a written rule from TUNE point estimates only: the smallest alpha_miss on a ladder for which at least 8 configurations have TUNE miss at least 0.02 below it, and alpha_fb set 0.05 above the best TUNE false-blur risk among those configurations, rounded up to a multiple of 0.05.

T2 exists because T1 is expected to fail (it already fails on TUNE) and the reader should see what a certified result looks like, and what it costs, without pretending the client's targets were met.

**CHECK 3.1 (a-priori).** The event log shows the targets were recorded before the CAL grid was evaluated, and T2-v1 is strictly looser than T1 on both risks. Note this check verifies ordering in the log and looseness; it does not verify that the rule was sensible.
