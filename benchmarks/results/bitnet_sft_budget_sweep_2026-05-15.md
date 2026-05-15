# BitNet-SFT Budget Sweep Audit, 2026-05-15

Best completed sweep row is accuracy `0.628935` at steps=`10000`, lr=`2e-5`. The paper BitNet-SFT anchor remains `0.608000`, so the local row exceeds the anchor by `0.020935`.

Completed rows: `9/10`.

Default BitNet-SFT MNLI baseline: `0.487621`.

## Budget Trend

The best completed row at the largest completed step count is still improving over the previous completed step bucket by `0.054203`. This clears the paper BitNet-SFT anchor for this baseline, but it does not prove BitDistill or FP16-level recovery.

| steps | best lr | best accuracy | paper anchor - local | delta vs previous step bucket | MNLI epochs |
| --- | --- | --- | --- | --- | --- |
| 1000 | 5e-5 | 0.523892 | 0.084108 | - | 0.040743 |
| 3000 | 2e-5 | 0.574733 | 0.033267 | 0.050841 | 0.122230 |
| 10000 | 2e-5 | 0.628935 | -0.020935 | 0.054203 | 0.407434 |

## Runs

| steps | lr | metrics | job | accuracy | examples | full eval | delta vs default | paper anchor - local | train examples | MNLI epochs | last CE | A8 | SubLN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 5e-6 | true | 10058 | 0.457056 | 9.815e+03 | true | -0.030565 | 0.150944 | 16000 | 0.040743 | 0.734375 | true | 0 |
| 1000 | 1e-5 | true | 10059 | 0.485991 | 9.815e+03 | true | -0.001630 | 0.122009 | 16000 | 0.040743 | 0.656250 | true | 0 |
| 1000 | 2e-5 | true | 10060 | 0.487621 | 9.815e+03 | true | 0.000000 | 0.120379 | 16000 | 0.040743 | 0.644531 | true | 0 |
| 1000 | 5e-5 | true | 10061 | 0.523892 | 9.815e+03 | true | 0.036271 | 0.084108 | 16000 | 0.040743 | 0.585938 | true | 0 |
| 3000 | 5e-6 | true | 10062 | 0.542435 | 9.815e+03 | true | 0.054814 | 0.065565 | 48000 | 0.122230 | 1.164062 | true | 0 |
| 3000 | 1e-5 | true | 10063 | 0.564646 | 9.815e+03 | true | 0.077025 | 0.043354 | 48000 | 0.122230 | 1.218750 | true | 0 |
| 3000 | 2e-5 | true | 10064 | 0.574733 | 9.815e+03 | true | 0.087112 | 0.033267 | 48000 | 0.122230 | 1.234375 | true | 0 |
| 3000 | 5e-5 | true | 10065 | 0.567397 | 9.815e+03 | true | 0.079776 | 0.040603 | 48000 | 0.122230 | 1.132812 | true | 0 |
| 10000 | 2e-5 | true | 10066 | 0.628935 | 9.815e+03 | true | 0.141314 | -0.020935 | 160000 | 0.407434 | 0.566406 | true | 0 |
| 10000 | 5e-5 | false | 10067 | - | - | false | - | - | 160000 | 0.407434 | - | - | - |

## Job Tables

- `benchmark_results/bitnet_sft_budget_sweep_20260515_132906_2052239_12830.tsv`
- `benchmark_results/bitnet_sft_budget_sweep_long_20260515_135825.tsv`
