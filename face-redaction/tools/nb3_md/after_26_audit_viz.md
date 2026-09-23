#### How to read this chart

Three panels; bounds are one-sided upper bounds at delta = 0.05 (the stress bounds at delta/(2|G|)).

- **Left: Prop. 26 recall bound against audit size.** x = audited face instances Q; y = certified recall lower bound 1 - U_Q(C, 0.05); curves for C = 0, 1, 2 misses; red dashed line = the client's 99%. At Q = 515 the best-effort audit (C = 2) sits on its curve at 0.988, just under 99%; T2 (C = 10) is far below every curve at 0.967.
- **Middle: the Thm 19 walk.** x = nested score threshold (0.40 most inclusive); y is a symlog axis. The green instance miss rate rises from about 0.002 to 0.25; the purple Prop. 26 bound sits above it; the tan Thm 8 missed-mass bound stays near 0.006 to 0.008, under the client's 1% line everywhere. The star marks where the instance-level walk stops (thr 0.70). The gap between tan and green is the point: the segment certificate passes even where a quarter of faces are missed.
- **Right: all certificates (log x).** Only the missed-mass bar (0.009) is left of the 1% line; every recall-relevant bar (0.012 to 0.379) is right of it.

For the job: quote the left panel as the recall evidence; the tan line is a diagnostic.
