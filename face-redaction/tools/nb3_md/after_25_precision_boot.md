**What the output says (sealed AUDIT, synthetic).**

| config | pooled precision | bootstrap 95% | episode-averaged false blur | its bootstrap 95% |
|---|---|---|---|---|
| T2 choice | 0.550 | [0.516, 0.585] | 0.371 | [0.342, 0.399] |
| best-effort | 0.237 | [0.214, 0.261] | 0.661 | [0.636, 0.688] |

**The client's 95% precision is not supported for either configuration**; the intervals sit far below 0.95 (5.8 PASS is only the sanity check). About 45% of the blur box-frames fall where the ground truth says nothing should be blurred (T2 choice), and about 76% for the recall-first configuration. This is a property of the synthetic detector and the simple blur rules here, not a claim about a real detector, but it shows the shape of the trade: recall bought by lowering the threshold is paid for in false blur. With the T1 targets (miss <= 1%, false-blur <= 5%) empty in section 3, this is the same finding seen from the audit side.

**If the audit had failed, or shown a miss:** the only allowed continuations are a fresh audit split or pre-registered error spending. Re-tuning on these labels burns them; the event log records that the AUDIT labels are now used up for design purposes.
