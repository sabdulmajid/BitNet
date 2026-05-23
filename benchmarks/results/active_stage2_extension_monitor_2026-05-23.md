# Active Stage-2 Extension Monitor

Status: **running**.

Quality claim: **none**. This report monitors job/artifact state only.

| job | id | slurm state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | RUNNING | 3:20:20 | ece-nebula12 |
| handoff | 10255 | PENDING | 0:00 | (Dependency) |
| gamma60 telemetry | 10257 | PENDING | 0:00 | (Dependency) |
| downstream MNLI | - | not_submitted |  |  |
| postprocess | - | not_submitted |  |  |

| stage2 field | value |
| --- | --- |
| latest_step | 6580 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| snapshot_status | pre_first_snapshot |
| output_dir_exists | False |
| missing_output_dir_is_expected | True |
| first_snapshot_step | 10000 |
| next_snapshot_step | 10000 |
| steps_to_next_snapshot | 3420 |
| next_snapshot_eta_hours | 1.726979 |
| progress | 0.164500 |
| latest_ce | 3.750599 |
| latest_lr | 0.000002 |
| log_freshness_status | fresh_running_log |
| log_health_status | healthy |
| producer_config_status | matched |
| log_age_seconds | 1.662649 |
| time_limit_status | within_time_limit |
| time_limit_margin_seconds | 13626.706383 |
| log_elapsed_seconds | 11961.600000 |
| parsed_log_rows | 659 |
| recent_window_rows | 20 |
| recent_ce_mean | 3.766374 |
| recent_ce_min | 3.244417 |
| recent_ce_max | 4.217243 |
| seconds_per_step | 1.817872 |
| steps_per_hour | 1980.337079 |
| eta_hours | 16.875915 |
| estimated_completion_utc | 2026-05-24T11:50:18.323947+00:00 |
| segment_token_presentations_per_second | 4506.367041 |
| latest_complete_snapshot_step | - |
| cumulative_token_presentations | 655360000 |

## Time Limit Gate

| field | value |
| --- | --- |
| status | within_time_limit |
| slurm_state | RUNNING |
| elapsed | 3:20:20 |
| time_limit | 1-00:00:00 |
| elapsed_seconds | 12020 |
| time_limit_seconds | 86400 |
| eta_seconds | 60753.293617 |
| remaining_seconds | 74380 |
| margin_seconds | 13626.706383 |
| tight_margin_threshold_seconds | 1800 |
| caveat | Compares current ETA with Slurm time remaining; it is a runtime-risk signal, not quality evidence. |

## Log Freshness

| field | value |
| --- | --- |
| status | fresh_running_log |
| path | logs/bd-s2-655m-10250.out |
| exists | True |
| checked_utc | 2026-05-23T18:57:45.030897+00:00 |
| mtime_utc | 2026-05-23T18:57:43.368248+00:00 |
| age_seconds | 1.662649 |
| stale_after_seconds | 900 |
| slurm_state | RUNNING |
| caveat | Fresh logs are required while the Stage-2 producer is running. |

## Producer Log Health

| field | value |
| --- | --- |
| status | healthy |
| path | logs/bd-s2-655m-10250.out |
| parsed_step_rows | 659 |
| first_step | 1 |
| latest_step | 6580 |
| latest_ce | 3.750599 |
| latest_lr | 0.000002 |
| latest_elapsed_seconds | 11961.600000 |
| recent_window_rows | 20 |
| recent_ce_mean | 3.766374 |
| recent_ce_min | 3.244417 |
| recent_ce_max | 4.217243 |
| issue_count | 0 |
| fatal_match_count | 0 |
| caveat | This checks producer log structure and fatal patterns; it is not quality evidence. |

| check | value |
| --- | --- |
| has_step_rows | True |
| steps_monotonic | True |
| elapsed_monotonic | True |
| finite_numeric_values | True |
| constant_lr_matches_expected | True |
| latest_step_within_max_steps | True |

## Producer Config Gate

| field | value |
| --- | --- |
| status | matched |
| log_path | logs/bd-s2-655m-10250.out |
| header_exists | True |
| header_line_count | 18 |
| mismatch_count | 0 |
| caveat | This validates the producer log header against the submitted Stage-2 configuration. |

| key | expected | actual | mode | matched |
| --- | --- | --- | --- | --- |
| SLURM_JOB_ID | 10250 | 10250 | string | True |
| MODEL | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B | string | True |
| STAGE | continued_pretrain | continued_pretrain | string | True |
| METHOD | bitdistill | bitdistill | string | True |
| INIT_STATE_MANIFEST | benchmarks/results/stage2_manifest_2026-05-20.json | benchmarks/results/stage2_manifest_2026-05-20.json | string | True |
| INIT_STATE_DICT | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k/checkpoint-40000/custom_state_dict.pt | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k/checkpoint-40000/custom_state_dict.pt | string | True |
| SCALE_MODE | tensor | tensor | string | True |
| ACTIVATION_QUANTIZATION | 1 | 1 | string | True |
| USE_SUBLN | 1 | 1 | string | True |
| MAX_SEQ_LEN | 512 | 512 | string | True |
| MAX_STEPS | 40000 | 40000 | string | True |
| PER_DEVICE_BATCH_SIZE | 4 | 4 | string | True |
| GRAD_ACCUM_STEPS | 4 | 4 | string | True |
| LR | 0.000002 | 2e-6 | float | True |
| LR_SCHEDULER | constant | constant | string | True |
| SAVE_EVERY_STEPS | 10000 | 10000 | string | True |
| SAVE_MODEL_ARTIFACTS | 0 | 0 | string | True |
| OUTPUT_DIR | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m | string | True |

## Snapshot Gate

| field | value |
| --- | --- |
| status | pre_first_snapshot |
| output_dir | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m |
| output_dir_exists | False |
| first_snapshot_step | 10000 |
| next_snapshot_step | 10000 |
| steps_to_next_snapshot | 3420 |
| next_snapshot_eta_hours | 1.726979 |
| estimated_next_snapshot_utc | 2026-05-23T20:41:22.154225+00:00 |
| latest_complete_snapshot_step | - |
| missing_output_dir_is_expected | True |
| caveat | A missing output directory is expected before the first snapshot when save_every_steps has not been reached. |

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
| postprocess report | False | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |
| next decision report | True | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
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

## Postprocess

| field | value |
| --- | --- |
| job_id |  |
| slurm_state | not_submitted |
| expected_json | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |
| expected_json_exists | False |
| expected_md | benchmarks/results/stage2_655m_postprocess_2026-05-23.md |
| expected_md_exists | False |
| expected_next_decision_json | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| expected_next_decision_json_exists | True |
| expected_next_decision_md | benchmarks/results/bitdistill_next_decision_2026-05-23.md |
| expected_next_decision_md_exists | True |
| caveat | This section tracks report-regeneration job state only; it is not quality evidence. |

## Caveat

This is a cumulative continuation from the verified 327.68M checkpoint with a fresh optimizer/scheduler segment. It is not an uninterrupted 80k-step Stage-2 run.
