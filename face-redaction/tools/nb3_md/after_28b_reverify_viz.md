#### How to read this chart

Two panels; five cases on the x axis: (i) unchanged, (ii) model change, (iii-a) threshold 0.90, (iii-b) threshold 0.60, (iv) small 40-episode set.

- **Left: miss rate against the targets.** y (log) = instance miss rate; the dot is the point estimate, the bar the Clopper-Pearson interval at delta = 0.05. Green dashed line = LTT target 0.05; red dashed line = the client's 1%. Colour = LTT-target verdict. Case (i) is green, with an interval of roughly 0.007 to 0.033: under 0.05 but straddling 1%, so it passes the LTT target without settling the client's 99%. (ii) and (iii-a) are red, intervals above 0.16: the change is caught. (iii-b) is amber with a low miss rate; its INCONCLUSIVE comes from the false-blur leg, not shown here. (iv) is amber with a very wide bar straddling 0.05: too little data.
- **Right: pooled precision with a cluster-bootstrap 95% interval.** Every point is red ("client: FAIL", the combined 99%/95% verdict) and every interval is far below 0.95 (0.57, 0.60, 0.73, 0.45, 0.59).

Pattern: the left panel reacts to recall damage; the right does not react to anything, because the precision shortfall is structural in this synthetic setup. A re-verification is only as good as the size of the fresh audit set.
