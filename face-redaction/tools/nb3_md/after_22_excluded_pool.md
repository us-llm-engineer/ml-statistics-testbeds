**What the output says (synthetic AUDIT, T2 configuration).**

- 4,096 segments; 3,244 (79.2%) are excluded, and **none of them contains a face** (M0 = 0). The 300-segment audit sample finds K0 = 0.
- Certified missed mass: Thm 8 gives r <= 37/4,096 = **0.0090**, Thm 7 gives r <= p0 * U = **0.0097**, both under 1%.
- But only 13.0% of segments hold a face. The prevalence sample (42 of 300) gives a = 0.103, so the implied **segment-level recall is >= 0.912**, not 99%. True segment recall here is 1.000; the bound is loose, not the pipeline.
- Why K0 = 0: a face track almost always draws some box in every 45-s segment it lives in (bridging and padding), so its misses are uncovered *frames inside included segments*. Those belong to the included pool, where labels give no missed-mass bound (P3 Prop. 3), and only Prop. 26 in 5(a) speaks to them. This certificate is about a different, easier event than the client's.

**Coverage (weak configuration on TRUTH).** Thm 8: coverage 1.0000 against a threshold of 0.9664 (5.3 PASS). Thm 7 on dependent segments: 0.9983 against 0.956 (5.4 PASS); nominal 0.975. Read these with care: with a true excluded-pool rate of 0.0054, even K0 = 0 gives an upper bound above the truth, so coverage saturates near 1 and the test is not sharp. It shows the bounds were not broken by segment dependence *in this simulator*, not that dependence is harmless.
