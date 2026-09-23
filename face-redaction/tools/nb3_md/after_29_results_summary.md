## Limitations and open engineering choices

**What the sources leave open:**

The notebook can compute certificates; it cannot make these choices. Each one changes what can honestly be promised.

1. **The targets.** The client's T1 (miss <= 1%, false blur <= 5%) is **not certified** on this synthetic corpus (empty certified set on CAL); the closest configuration misses 5.85% of instances with an episode-averaged false-blur of 0.308, and the lowest miss rate anywhere (0.0079) comes with a false-blur of 0.612. Options: relax the targets (T2 shows one certified point, at 5% miss and 45% false blur), accept a human-review step for what the pipeline flags, or improve the detector. Also decide what "precision 95%" means in the contract: LTT controls an episode-*averaged* false-blur fraction, the client's number is *pooled* over blur box-frames; the notebook reports both and they differ (**derived here**).
2. **The real labelled staged set and who labels it.** Every certificate here rests on labelled instances: certifying 1% needs at least 299 audited instances with zero misses, 473 with one, 628 with two (section 2's table, derived from Clopper-Pearson; P3's Sec. 4.6 table has the same arithmetic for the excluded pool [facered-q8]). Choose the vendor, the guidelines and the budget; the audit set must stay sealed.
3. **The real detector and its licence.** The detector here is a simulator plus a YuNet cost model. A different detector means a new grid, new calibration and a new audit.
4. **Blur method** (derived here; the papers are silent): destructive pixelation or solid fill is irreversible, a light Gaussian blur may not be. This is a policy choice, not a statistical one.
5. **Encoder and CRF 16** versus a smaller file, and **the GPU instance and real prices**. All dollar figures above are measured throughput times assumed prices.
6. **Screens and photos in the scene** (a face on a monitor, a photo on a wall): whether they count as faces to redact is a **contract clause**, not something any paper decides.
7. **Which certificate to put in the contract.** Only Prop. 26 over audited face instances supports a recall claim; the missed-mass bound of 5(b) and the stress bounds of 5(d) do not.

**Engineering choices left open:**

- **`redact.py` command-line tool**: resumable, JSON log of every decision, non-zero exit on failure.
- **A real labelled staged capture and a real-detector run**; nothing here uses real footage.
- **Per-stratum reporting** (lighting, distance, camera, scene): all guarantees here are *marginal* over the audit distribution: "all the guarantees in this paper are marginal" `[P1 Sec. 5, facered-q3]`, and Prop. 26 is likewise an average over audited positives. P3 Sec. 9 offers stratified audits with a union bound `[P3 Sec. 9, facered-q11]`; none is implemented here, so a per-stratum report (children, mirrors, small faces) would be a derived-here diagnostic, not a guarantee.
- **The SeqCRC localisation and classification step** (only the confidence step was implemented in section 4), P3's incremental-coverage diagnostic (Thm 24) and its noisy-label variants (Sec. 9).
- **Cost minimisation among certified configurations with a pre-registered cheaper path**: the certified set here contains only stride 1.

## What the three sources do not settle

- **Temporal dependence.** No paper covers video, tracking or frame-to-frame dependence [facered-q11, facered-q12(d)]. The exchangeable unit (instance or episode) is an assumption of this notebook (derived here), and the segment coverage study in 5(b) is a simulation, not a proof.
- **Per-instance versus per-frame units and temporal padding.** None of the three papers defines them [facered-q12(d)].
- **Precision.** SeqCRC does not guarantee it ("only empirically limits false positives" [P2 Sec. V-D, facered-q5]); LTT can control a false-blur risk jointly with recall [P1, facered-q12(a)], but no source certifies the client's *pooled* precision on an audit set; the bootstrap here is not a guarantee.
- **Cost.** Choosing the cheapest configuration among certified ones, and every dollar figure, are outside all three papers [facered-q12(d)].
- **Noisy labels.** P3 needs noiseless audit labels (Assumption 2) and weakens the missed-mass bound under a known sensitivity floor; its two-pool and positive-conditional certificates, including Prop. 26, break under unmodelled noise [P3 Assumption 2, Sec. 9, facered-q11].
- **Distribution shift.** No source gives coverage between calibration/audit data and deployment beyond a declared perturbation generator [facered-q11]; that is why section 7 demands a fresh audit set after any change.
- **Rare faces.** An absolute missed-mass bound is not recall when faces are rare [P3 Remark 9, facered-q11]; only Prop. 26 on audited instances speaks to the client's per-instance 99%.
