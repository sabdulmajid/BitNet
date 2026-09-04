# BitDistill Gamma Balance Audit

Generated: `2026-09-04T03:45:19.306256+00:00`

Gamma-60 materially rebalances attention-KD updates under the local reductions.

This audit compares gradient/loss balance only. It is not a task-quality benchmark and does not update BitDistill reproduction status.

## Run State

| field | value |
| --- | --- |
| status | gamma60_rebalanced_attention_updates |
| quality claim | none |
| job id | 10257 |
| squeue state | not_in_squeue |
| squeue elapsed | - |
| gamma telemetry exists | true |
| gamma telemetry rows | 9 |
| gamma status report | complete |

## Balance Metrics

| metric | value |
| --- | --- |
| paper final grad attention/CE | 221.384986 |
| gamma60 final grad attention/CE | 0.346044 |
| attention grad reduction factor | 639.759089 |
| paper final loss attention/CE | 2.549e+03 |
| gamma60 final loss attention/CE | 1.270624 |
| attention loss reduction factor | 2.006e+03 |
| gamma60 max activation clipped | 0.000000 |
| gamma60 max activation edge | 0.000401 |
| gamma60 mean ternary flip fraction | 0.002206 |

## Paths

| artifact | path |
| --- | --- |
| paper dynamics | benchmarks/results/bitdistill_training_dynamics_2026-05-23.json |
| gamma status | benchmarks/results/gamma60_telemetry_status_2026-05-23.json |
| gamma telemetry | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl |
