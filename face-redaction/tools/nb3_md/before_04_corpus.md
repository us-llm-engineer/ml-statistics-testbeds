## 1. A larger corpus, split honestly

Notebook 2's 222 face instances cannot support a 99% certificate (section 2 shows why). Here the **same generator** (`SimConfig`) draws more episodes with its own seeds, and the episodes are split into three disjoint sets:

| Split | Job |
|---|---|
| **TUNE** | Learns the order in which configurations will be tested and fixes the targets. No guarantee is claimed from it. |
| **CAL** | The LTT test set (P1 App. D calls it the multiple testing set). |
| **AUDIT** | Sealed. Only opened in section 5, after the configuration is fixed (P3 Assumption 1). |

A second, fresh calibration split **CAL2** is generated now but its labels stay untouched until section 3(c); a separate large **TRUTH** population approximates "true" risks for the simulation in 3(d). Splits are made at the **episode** level: every track of an episode goes to one split.

**Unit of analysis (derived here).** The client's metric is per face instance. None of the three papers covers video. P1 assumes an "independent and identically distributed (i.i.d.) set of variables" `[P1, facered-q11]`, and P2 notes that its object-level guarantee "implicitly assumes independence between objects of a same image" `[P2 App. C, facered-q11]`. In this generator the tracks within an episode are drawn independently, so treating face instances as exchangeable is defensible **here**. On real footage, within-episode dependence (the same room, the same lighting, the same person returning) must be checked before any Clopper-Pearson or binomial bound is trusted.

**CHECKs (a-priori).**
- 1.1: at least 1,500 face instances across TUNE+CAL+AUDIT.
- 1.2: the three splits are pairwise disjoint at the episode level and jointly cover the corpus.
- 1.3: the larger corpus keeps NB2's distribution: zero-face-episode fraction in [0.45, 0.60] and every face-kind share within 3 binomial standard errors of NB2's *design weights*. The cell comment records that an earlier draft compared against NB2's realised shares within +-0.05 and failed, because NB2's own 222-instance sample is noisy; the comparison was changed to the design weights and the reason is printed.
