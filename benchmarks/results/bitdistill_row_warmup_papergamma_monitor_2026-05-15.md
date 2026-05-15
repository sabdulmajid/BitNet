# BitDistill Job Monitor, 2026-05-15

Job tables: `benchmark_results/bitdistill_rowwarmup_downstream_papergamma_20260515.tsv`.

## Stage-2 Warm-Up

| log | step | max steps | progress | latest CE | effective tokens | target tokens | save every | snapshots | latest snapshot | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 860 | 20000 | 0.043000 | 5.259806 | 7045120 | 163840000 | 1000 | 0 | - | 9.65h |

## Operational Warnings

| warning |
| --- |
| none |

## Downstream Jobs

| job | task | format | scale | layer | steps | logit KD | attention KD | logit temp scale | excluded linears | state | elapsed | node/reason | metrics | accuracy | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10032 | mnli | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10033 | qnli | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 10034 | sst2 | sequence_classification | row | -8 | 1000 | 10.0 | 100000 | none | score\|classifier | PENDING | 0:00 | (Dependency) | false | - | checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
