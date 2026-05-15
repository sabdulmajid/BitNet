# BitDistill Job Monitor, 2026-05-15

Job tables: `benchmark_results/bitdistill_rowwarmup_downstream_gamma100_20260515.tsv, benchmark_results/bitdistill_rowwarmup_downstream_papergamma_20260515.tsv`.

## Stage-2 Warm-Up

| log | step | max steps | progress | latest CE | effective tokens | target tokens | save every | snapshots | latest snapshot | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 20000 | 20000 | 1.000000 | 3.255063 | 163840000 | 163840000 | 1000 | 20 | checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k/checkpoint-20000 | 0.00h |

## Operational Warnings

| warning |
| --- |
| none |

## Downstream Jobs

| job | task | format | scale | layer | steps | logit KD | attention KD | logit temp scale | excluded linears | state | elapsed | node/reason | metrics | accuracy | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10029 | mnli | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | not_in_squeue | - | - | true | 0.627713 | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10030 | qnli | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | not_in_squeue | - | - | true | 0.779791 | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10031 | sst2 | sequence_classification | row | -8 | 1000 | 10.0 | 100.0 | none | score\|classifier | not_in_squeue | - | - | true | 0.846330 | checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10032 | mnli | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | RUNNING | 0:53 | ece-nebula12 | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10033 | qnli | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | PENDING | 0:00 | (Resources) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10034 | sst2 | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | PENDING | 0:00 | (Priority) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
