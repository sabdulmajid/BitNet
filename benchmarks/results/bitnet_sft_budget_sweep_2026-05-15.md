# BitNet-SFT Budget Sweep Audit, 2026-05-15

Best completed sweep row is accuracy `0.485991` at steps=`1000`, lr=`1e-5`. The paper BitNet-SFT anchor remains `0.608000`, so the remaining gap is `0.122009`.

Completed rows: `2/8`.

Default BitNet-SFT MNLI baseline: `0.487621`.

## Runs

| steps | lr | metrics | job | accuracy | examples | full eval | delta vs default | paper anchor - local | last CE | A8 | SubLN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 5e-6 | true | 10058 | 0.457056 | 9.815e+03 | true | -0.030565 | 0.150944 | 0.734375 | true | 0 |
| 1000 | 1e-5 | true | 10059 | 0.485991 | 9.815e+03 | true | -0.001630 | 0.122009 | 0.656250 | true | 0 |
| 1000 | 2e-5 | false | 10060 | - | - | false | - | - | - | - | - |
| 1000 | 5e-5 | false | 10061 | - | - | false | - | - | - | - | - |
| 3000 | 5e-6 | false | 10062 | - | - | false | - | - | - | - | - |
| 3000 | 1e-5 | false | 10063 | - | - | false | - | - | - | - | - |
| 3000 | 2e-5 | false | 10064 | - | - | false | - | - | - | - | - |
| 3000 | 5e-5 | false | 10065 | - | - | false | - | - | - | - | - |

## Job Tables

- `benchmark_results/bitnet_sft_budget_sweep_20260515_132906_2052239_12830.tsv`
