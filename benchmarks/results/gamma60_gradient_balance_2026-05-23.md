# BitDistill Gamma Balance Audit

Generated: `2026-05-23T16:53:51.895992+00:00`

Gamma-60 telemetry is pending; no loss-normalization conclusion is available yet.

This audit compares gradient/loss balance only. It is not a task-quality benchmark and does not update BitDistill reproduction status.

## Run State

| field | value |
| --- | --- |
| status | pending_gamma60_telemetry |
| quality claim | none |
| job id | 10256 |
| squeue state | PENDING |
| squeue elapsed | 0:00 |
| gamma telemetry exists | false |
| gamma telemetry rows | 0 |
| gamma status report | - |

## Balance Metrics

| metric | value |
| --- | --- |
| paper final grad attention/CE | 221.384986 |
| gamma60 final grad attention/CE | - |
| attention grad reduction factor | - |
| paper final loss attention/CE | 2.549e+03 |
| gamma60 final loss attention/CE | - |
| attention loss reduction factor | - |
| gamma60 max activation clipped | - |
| gamma60 max activation edge | - |
| gamma60 mean ternary flip fraction | - |

## Paths

| artifact | path |
| --- | --- |
| paper dynamics | benchmarks/results/bitdistill_training_dynamics_2026-05-23.json |
| gamma status | benchmarks/results/gamma60_telemetry_status_2026-05-23.json |
| gamma telemetry | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl |
