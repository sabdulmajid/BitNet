# BitDistill Job Monitor, 2026-05-14

Job tables: `benchmark_results/bitdistill_longwarmup_downstream_20260514_163342.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_171512.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173303.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173304.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173758.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173828_1911478_14277.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173828_1911513_4036.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173828_1911554_24007.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_173828_1911577_12838.tsv, benchmark_results/bitdistill_longwarmup_downstream_20260514_181145_1915691_24602.tsv`.

## Stage-2 Warm-Up

| log | step | max steps | progress | latest CE | effective tokens | target tokens | save every | snapshots | latest snapshot | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 3900 | 20000 | 0.195000 | 4.475124 | 31948800 | 163840000 | 0 | 0 | - | 8.14h |

## Downstream Jobs

| job | task | format | scale | layer | steps | logit KD | attention KD | logit temp scale | excluded linears | state | elapsed | node/reason | metrics | accuracy | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9925 | mnli | - | tensor | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9926 | mnli | - | row | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9927 | qnli | - | tensor | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9928 | qnli | - | row | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9929 | sst2 | - | tensor | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9930 | sst2 | - | row | -8 | 1000 | 10.0 | 100.0 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9931 | mnli | - | tensor | -8 | 1000 | 10.0 | 100000 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9932 | qnli | - | tensor | -8 | 1000 | 10.0 | 100000 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9933 | sst2 | - | tensor | -8 | 1000 | 10.0 | 100000 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9934 | mnli | - | tensor | -8 | 1000 | 10.0 | 1000 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9935 | mnli | - | tensor | -8 | 1000 | 10.0 | 10000 | none | - | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9943 | mnli | causal_lm | tensor | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9944 | mnli | causal_lm | row | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9945 | qnli | causal_lm | tensor | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9946 | qnli | causal_lm | row | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9947 | sst2 | causal_lm | tensor | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9948 | sst2 | causal_lm | row | -8 | 1000 | 10.0 | 100.0 | none | lm_head | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
