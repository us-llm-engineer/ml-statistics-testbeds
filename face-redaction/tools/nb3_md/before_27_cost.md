## 6. Cost per option

**Claim.** The client wants the cheapest configuration that meets the targets, but none of the three papers prices anything. P3 minimises a *review burden* B(g) = P(g(X) = 1) subject to a certified missed-mass bound [P3 Sec. 3, Eq. 1, facered-q9], and computational rendering, GPU latency or encoding cost is simply not addressed [facered-q12(d)]. So every dollar figure here is **derived here** or **assumed**, and cost minimisation among certified configurations is a design step of this notebook, not a result from the sources.

**What is measured and what is assumed.**
- **Measured**: throughput of the reused Notebook 2 pipeline on this shared WSL2 VM, from `data/smoke/timings.json` (a 15-frame 4K clip); section 3's throughput cell (`16_throughput`) builds the dictionary. NB2 recorded several-fold run-to-run variation on this machine.
- **Assumed placeholders, not provider quotes**: the CPU price, the GPU price and the GPU speed-up over the CPU path. A dollar per video-hour is *measured throughput x assumed price*.

**Options priced.** The T2-certified configuration(s); strided variants (only stride changes the cost; they are **uncertified**, either failing on CAL2 or not reached by the fixed-sequence walk, whose p-values are not certificates); the best-effort recall-first configuration (uncertified by LTT; the audit is its only evidence); and the GPU column under the assumed numbers. The cell prints the **break-even**: the GPU is cheaper only if speed-up > GPU price / CPU price, whatever the throughput.

**A-priori checks.** **6.1** cost is strictly decreasing in stride (1 > 2 > 4) for both columns (the lever exists). **6.2** at speed-up = GPU price / CPU price the two columns are equal (an identity of the cost function). **6.3** run-to-run variance in the measured timings moves the CPU $/video-hour by at least 2x, so that no single dollar figure should be quoted to the client (the threshold is stated as 2.0x; watch how close it is).
