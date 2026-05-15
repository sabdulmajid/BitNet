# BitNet-SFT Budget Sweep Audit, 2026-05-15

Best completed sweep row is accuracy `0.523892` at steps=`1000`, lr=`5e-5`. The paper BitNet-SFT anchor remains `0.608000`, so the remaining gap is `0.084108`.

Completed rows: `4/10`.

Default BitNet-SFT MNLI baseline: `0.487621`.

## Runs

| steps | lr | metrics | job | accuracy | examples | full eval | delta vs default | paper anchor - local | train examples | MNLI epochs | last CE | A8 | SubLN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 5e-6 | true | 10058 | 0.457056 | 9.815e+03 | true | -0.030565 | 0.150944 | 16000 | 0.040743 | 0.734375 | true | 0 |
| 1000 | 1e-5 | true | 10059 | 0.485991 | 9.815e+03 | true | -0.001630 | 0.122009 | 16000 | 0.040743 | 0.656250 | true | 0 |
| 1000 | 2e-5 | true | 10060 | 0.487621 | 9.815e+03 | true | 0.000000 | 0.120379 | 16000 | 0.040743 | 0.644531 | true | 0 |
| 1000 | 5e-5 | true | 10061 | 0.523892 | 9.815e+03 | true | 0.036271 | 0.084108 | 16000 | 0.040743 | 0.585938 | true | 0 |
| 3000 | 5e-6 | false | 10062 | - | - | false | - | - | 48000 | 0.122230 | - | - | - |
| 3000 | 1e-5 | false | 10063 | - | - | false | - | - | 48000 | 0.122230 | - | - | - |
| 3000 | 2e-5 | false | 10064 | - | - | false | - | - | 48000 | 0.122230 | - | - | - |
| 3000 | 5e-5 | false | 10065 | - | - | false | - | - | 48000 | 0.122230 | - | - | - |
| 10000 | 2e-5 | false | 10066 | - | - | false | - | - | 160000 | 0.407434 | - | - | - |
| 10000 | 5e-5 | false | 10067 | - | - | false | - | - | 160000 | 0.407434 | - | - | - |

## Job Tables

- `benchmark_results/bitnet_sft_budget_sweep_20260515_132906_2052239_12830.tsv`
- `benchmark_results/bitnet_sft_budget_sweep_long_20260515_135825.tsv`
