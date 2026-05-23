# Active Stage-2 Extension Monitor

Status: **running**.

Quality claim: **none**. This report monitors job/artifact state only.

| job | id | slurm state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | RUNNING | 59:55 | ece-nebula12 |
| handoff | 10253 | PENDING | 0:00 | (Dependency) |
| gamma60 telemetry | 10254 | PENDING | 0:00 | (Dependency) |
| downstream MNLI | - | not_submitted |  |  |

| stage2 field | value |
| --- | --- |
| latest_step | 1940 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| progress | 0.048500 |
| latest_ce | 3.505811 |
| latest_lr | 0.000002 |
| log_elapsed_seconds | 3524.900000 |
| parsed_log_rows | 195 |
| recent_window_rows | 20 |
| recent_ce_mean | 3.897342 |
| recent_ce_min | 3.441311 |
| recent_ce_max | 4.278359 |
| seconds_per_step | 1.816959 |
| steps_per_hour | 1981.332804 |
| eta_hours | 19.209292 |
| estimated_completion_utc | 2026-05-24T11:49:53.805476+00:00 |
| segment_token_presentations_per_second | 4508.632869 |
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
| downstream metrics | False | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json |
| downstream predictions | False | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| telemetry artifact 1 | False | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl |
| telemetry artifact 2 | False | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/metrics.json |

## Downstream

| field | value |
| --- | --- |
| status | waiting_for_handoff |
| handoff_report_exists | False |
| handoff_report_status | - |
| output_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit |
| complete | False |
| caveat | This section tracks downstream artifact existence only; it does not compute or claim MNLI accuracy. |

## Caveat

This is a cumulative continuation from the verified 327.68M checkpoint with a fresh optimizer/scheduler segment. It is not an uninterrupted 80k-step Stage-2 run.
