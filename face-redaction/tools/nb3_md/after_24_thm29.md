**What the output says.** Of 515 audited instances:

| config | misses in original / variants 1, 2, 3 | K_max | K_rand | rho_max bound | rho_avg bound |
|---|---|---|---|---|---|
| T2 choice | 10 / 31, 75, 169 | 170 | 66 | 0.3786 | 0.1649 |
| best-effort | 2 / 2, 7, 10 | 11 | 7 | 0.0405 | 0.0301 |

The identities hold (5.7 PASS). The T2 configuration degrades sharply under the harder detectors (10 misses become 169 under the hardest variant), so at least one variant of a face escapes it with probability up to 0.38. The recall-first configuration is much more robust (rho_max <= 0.041) but, as 5(e) shows, at a heavy price in false blur. These bounds hold **only** for the three declared detector shifts; they say nothing about lighting, blur or mirrors beyond them.
