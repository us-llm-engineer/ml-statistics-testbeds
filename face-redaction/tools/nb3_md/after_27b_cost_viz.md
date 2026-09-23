#### How to read this chart

Every dollar figure is measured throughput times an **assumed** price; the GPU numbers are assumed throughout.

- **Left: cost against stride.** x = detector stride (1, 2, 4); y = CPU $/video-hour; each dot is one of the 58 grid configurations, coloured by CAL2 miss rate. The dots form three horizontal bands (about 2.58, 1.62, 1.14) because cost depends only on stride. The stars (certified) and the diamond (recall-first, uncertified) all sit in the top band: certification chose among stride-1 options only.
- **Middle: GPU against CPU.** x = assumed GPU speed-up; y = stride-1 $/video-hour. Curves are the GPU at $0.30, $0.60 and $1.20 per hour; the red dashed line is the CPU cost ($2.58). A curve crossing the line marks break-even: about 6x, 12x and 24x. The verticals mark the 12x break-even (at $0.60 vs $0.05) and the assumed 8x, which lies to its left.
- **Right: measurement variance.** CPU cost at stride 1 (teal) and stride 4 (amber) under a slow, the measured and a fast smoke run: 4.63 / 2.58 / 2.29 and 3.19 / 1.14 / 0.85. The spread within one stride is as large as the gap between strides.

For the job: quote a range with the assumptions attached, and replace the placeholders with real prices.
