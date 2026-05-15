# BitDistill Job Monitor, 2026-05-15

Job tables: `benchmark_results/bitdistill_rowwarmup_downstream_gamma100_20260515.tsv`.

## Stage-2 Warm-Up

| log | step | max steps | progress | latest CE | effective tokens | target tokens | save every | snapshots | latest snapshot | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 5710 | 20000 | 0.285500 | 4.034164 | 46776320 | 163840000 | 1000 | 5 | checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k/checkpoint-5000 | 7.35h |

## Operational Warnings

| warning |
| --- |
| none |

## Downstream Jobs

| job | task | format | scale | layer | steps | logit KD | attention KD | logit temp scale | excluded linears | state | elapsed | node/reason | metrics | accuracy | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10029 | mnli | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10030 | qnli | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10031 | sst2 | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
