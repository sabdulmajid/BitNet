# Active Stage-2 Extension Monitor

Status: **running**.

Quality claim: **none**. This report monitors job/artifact state only.

| job | id | slurm state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | RUNNING | 37:47 | ece-nebula12 |
| handoff | 10253 | PENDING | 0:00 | (Dependency) |
| gamma60 telemetry | 10254 | PENDING | 0:00 | (Dependency) |

| stage2 field | value |
| --- | --- |
| latest_step | 1210 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| progress | 0.030250 |
| latest_ce | 4.152130 |
| latest_lr | 0.000002 |
| log_elapsed_seconds | 2197.700000 |
| seconds_per_step | 1.816281 |
| steps_per_hour | 1982.072166 |
| eta_hours | 19.570428 |
| estimated_completion_utc | 2026-05-24T11:49:25.989430+00:00 |
| segment_token_presentations_per_second | 4510.315330 |
| latest_complete_snapshot_step | - |
| cumulative_token_presentations | 655360000 |

## Expected Snapshots

| step | dir exists | state | metrics | complete |
| --- | --- | --- | --- | --- |
| 10000 | False | False | False | False |
| 20000 | False | False | False | False |
| 30000 | False | False | False | False |
| 40000 | False | False | False | False |

## Artifacts

| artifact | exists | path |
| --- | --- | --- |
| stage2 root metrics | False | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/metrics.json |
| stage2 final state | False | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/custom_state_dict.pt |
| stage2 final snapshot metrics | False | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/metrics.json |
| handoff manifest | False | benchmarks/results/stage2_manifest_655m_2026-05-23.json |
| handoff report | False | benchmarks/results/stage2_655m_handoff_2026-05-23.json |
| telemetry artifact 1 | False | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl |
| telemetry artifact 2 | False | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/metrics.json |

## Caveat

This is a cumulative continuation from the verified 327.68M checkpoint with a fresh optimizer/scheduler segment. It is not an uninterrupted 80k-step Stage-2 run.
