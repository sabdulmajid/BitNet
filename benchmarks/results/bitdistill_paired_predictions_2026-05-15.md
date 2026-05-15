# BitDistill Paired Prediction Audit, 2026-05-15

Overall status: `pass`.

Rows complete: `44` / `44`. Pending: `0`. Failed: `0`.

This report is designed to become the main statistical comparison once long-warmup jobs write `eval_predictions.jsonl`.

Full-evaluation contract: `{'mnli': 9815, 'qnli': 5463, 'sst2': 872}` examples. Partial prediction traces cannot pass.

Primary baseline root: `checkpoints/bitdistill-glue-seqcls`.

Prediction-backfill baseline root: `checkpoints/bitdistill-glue-seqcls-predtrace`.

The baseline prediction-backfill reruns are not silently substituted for the primary headline metrics; their small accuracy deltas are reported below.

| task | run | primary acc | prediction acc | prediction-primary | primary n | prediction n |
| --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | 0.807641 | 0.808151 | 0.000509 | 9815 | 9815 |
| mnli | BitNet-SFT | 0.487621 | 0.489251 | 0.001630 | 9815 | 9815 |
| qnli | FP16-SFT | 0.898957 | 0.899506 | 0.000549 | 5463 | 5463 |
| qnli | BitNet-SFT | 0.596925 | 0.600037 | 0.003112 | 5463 | 5463 |
| sst2 | FP16-SFT | 0.925459 | 0.925459 | 0.000000 | 872 | 872 |
| sst2 | BitNet-SFT | 0.770642 | 0.777523 | 0.006881 | 872 | 872 |

Delta is candidate minus reference on the same eval indices; positive means the candidate is better.

| task | comparison | status | matched n | expected n | reference acc | candidate acc | delta | paired 95% CI | candidate wins | reference wins | McNemar p | pending/error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | BitNet-SFT minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.489251 | -0.318900 | [-0.330340, -0.307459] | 575 | 3705 | 0.000000 |  |
| mnli | gamma100 row minus tensor | pass | 9815 | 9815 | 0.641671 | 0.653591 | 0.011921 | [0.004958, 0.018883] | 667 | 550 | 0.000876 |  |
| mnli | paper-gamma row minus tensor | pass | 9815 | 9815 | 0.630260 | 0.617626 | -0.012634 | [-0.019604, -0.005663] | 548 | 672 | 0.000425 |  |
| mnli | gamma100 tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.641671 | -0.166480 | [-0.176513, -0.156447] | 581 | 2215 | 0.000000 |  |
| mnli | gamma100 row minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.653591 | -0.154559 | [-0.164393, -0.144726] | 571 | 2088 | 0.000000 |  |
| mnli | paper-gamma tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.630260 | -0.177891 | [-0.188045, -0.167737] | 575 | 2321 | 0.000000 |  |
| mnli | paper-gamma row minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.617626 | -0.190525 | [-0.200695, -0.180354] | 540 | 2410 | 0.000000 |  |
| mnli | paper-gamma lr1e-5 tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.604381 | -0.203770 | [-0.214170, -0.193369] | 560 | 2560 | 0.000000 |  |
| mnli | paper-gamma lr5e-5 tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.642384 | -0.165767 | [-0.175802, -0.155731] | 584 | 2211 | 0.000000 |  |
| mnli | paper-gamma headinit tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.627815 | -0.180336 | [-0.190501, -0.170171] | 570 | 2340 | 0.000000 |  |
| mnli | paper-gamma lr1e-5 minus default | pass | 9815 | 9815 | 0.630260 | 0.604381 | -0.025879 | [-0.032249, -0.019509] | 385 | 639 | 0.000000 |  |
| mnli | paper-gamma lr5e-5 minus default | pass | 9815 | 9815 | 0.630260 | 0.642384 | 0.012124 | [0.005520, 0.018728] | 607 | 488 | 0.000358 |  |
| mnli | paper-gamma headinit minus default | pass | 9815 | 9815 | 0.630260 | 0.627815 | -0.002445 | [-0.007744, 0.002853] | 340 | 364 | 0.386041 |  |
| mnli | gamma1k tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.647275 | -0.160876 | [-0.170907, -0.150845] | 599 | 2178 | 0.000000 |  |
| mnli | gamma10k tensor minus FP16-SFT | pass | 9815 | 9815 | 0.808151 | 0.635354 | -0.172797 | [-0.182997, -0.162596] | 603 | 2299 | 0.000000 |  |
| mnli | gamma1k tensor minus gamma100 tensor | pass | 9815 | 9815 | 0.641671 | 0.647275 | 0.005604 | [-0.001305, 0.012512] | 626 | 571 | 0.118535 |  |
| mnli | gamma10k tensor minus gamma100 tensor | pass | 9815 | 9815 | 0.641671 | 0.635354 | -0.006317 | [-0.013889, 0.001255] | 688 | 750 | 0.107670 |  |
| mnli | paper-gamma tensor minus gamma1k tensor | pass | 9815 | 9815 | 0.647275 | 0.630260 | -0.017015 | [-0.023663, -0.010367] | 472 | 639 | 0.000001 |  |
| qnli | BitNet-SFT minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.600037 | -0.299469 | [-0.314333, -0.284606] | 285 | 1921 | 0.000000 |  |
| qnli | gamma100 row minus tensor | pass | 5463 | 5463 | 0.787846 | 0.796998 | 0.009152 | [0.000093, 0.018212] | 344 | 294 | 0.052303 |  |
| qnli | paper-gamma row minus tensor | pass | 5463 | 5463 | 0.759656 | 0.760937 | 0.001281 | [-0.007915, 0.010478] | 332 | 325 | 0.814943 |  |
| qnli | gamma100 tensor minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.787846 | -0.111660 | [-0.123164, -0.100157] | 243 | 853 | 0.000000 |  |
| qnli | gamma100 row minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.796998 | -0.102508 | [-0.113755, -0.091261] | 240 | 800 | 0.000000 |  |
| qnli | paper-gamma tensor minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.759656 | -0.139850 | [-0.152055, -0.127644] | 250 | 1014 | 0.000000 |  |
| qnli | paper-gamma row minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.760937 | -0.138569 | [-0.150609, -0.126528] | 237 | 994 | 0.000000 |  |
| qnli | paper-gamma lr1e-5 tensor minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.757459 | -0.142046 | [-0.154444, -0.129649] | 264 | 1040 | 0.000000 |  |
| qnli | paper-gamma lr5e-5 tensor minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.790957 | -0.108548 | [-0.120090, -0.097007] | 253 | 846 | 0.000000 |  |
| qnli | paper-gamma headinit tensor minus FP16-SFT | pass | 5463 | 5463 | 0.899506 | 0.762951 | -0.136555 | [-0.148692, -0.124418] | 250 | 996 | 0.000000 |  |
| qnli | paper-gamma lr1e-5 minus default | pass | 5463 | 5463 | 0.759656 | 0.757459 | -0.002197 | [-0.009942, 0.005549] | 227 | 239 | 0.610404 |  |
| qnli | paper-gamma lr5e-5 minus default | pass | 5463 | 5463 | 0.759656 | 0.790957 | 0.031301 | [0.022905, 0.039698] | 362 | 191 | 0.000000 |  |
| qnli | paper-gamma headinit minus default | pass | 5463 | 5463 | 0.759656 | 0.762951 | 0.003295 | [-0.003474, 0.010064] | 187 | 169 | 0.367611 |  |
| sst2 | BitNet-SFT minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.777523 | -0.147936 | [-0.176904, -0.118967] | 28 | 157 | 0.000000 |  |
| sst2 | gamma100 row minus tensor | pass | 872 | 872 | 0.866972 | 0.854358 | -0.012615 | [-0.027678, 0.002448] | 17 | 28 | 0.135156 |  |
| sst2 | paper-gamma row minus tensor | pass | 872 | 872 | 0.841743 | 0.837156 | -0.004587 | [-0.021414, 0.012240] | 26 | 30 | 0.688880 |  |
| sst2 | gamma100 tensor minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.866972 | -0.058486 | [-0.080752, -0.036221] | 25 | 76 | 0.000000 |  |
| sst2 | gamma100 row minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.854358 | -0.071101 | [-0.093769, -0.048433] | 22 | 84 | 0.000000 |  |
| sst2 | paper-gamma tensor minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.841743 | -0.083716 | [-0.107398, -0.060033] | 22 | 95 | 0.000000 |  |
| sst2 | paper-gamma row minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.837156 | -0.088303 | [-0.113164, -0.063442] | 26 | 103 | 0.000000 |  |
| sst2 | paper-gamma lr1e-5 tensor minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.846330 | -0.079128 | [-0.102880, -0.055377] | 24 | 93 | 0.000000 |  |
| sst2 | paper-gamma lr5e-5 tensor minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.836009 | -0.089450 | [-0.113146, -0.065753] | 20 | 98 | 0.000000 |  |
| sst2 | paper-gamma headinit tensor minus FP16-SFT | pass | 872 | 872 | 0.925459 | 0.834862 | -0.090596 | [-0.115421, -0.065772] | 25 | 104 | 0.000000 |  |
| sst2 | paper-gamma lr1e-5 minus default | pass | 872 | 872 | 0.841743 | 0.846330 | 0.004587 | [-0.009985, 0.019159] | 23 | 19 | 0.643969 |  |
| sst2 | paper-gamma lr5e-5 minus default | pass | 872 | 872 | 0.841743 | 0.836009 | -0.005734 | [-0.020816, 0.009348] | 20 | 25 | 0.551484 |  |
| sst2 | paper-gamma headinit minus default | pass | 872 | 872 | 0.841743 | 0.834862 | -0.006881 | [-0.022455, 0.008694] | 21 | 27 | 0.470879 |  |
