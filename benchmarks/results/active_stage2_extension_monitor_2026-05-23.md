# Active Stage-2 Extension Monitor

Status: **complete_artifacts_present**.

Quality claim: **none**. This report monitors job/artifact state only.

| job | id | slurm state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | not_in_squeue |  |  |
| handoff | 10259 | not_in_squeue |  |  |
| gamma60 telemetry | 10257 | not_in_squeue |  |  |
| downstream MNLI | 10260 | not_submitted |  |  |
| postprocess | 10261 | not_submitted |  |  |

| stage2 field | value |
| --- | --- |
| latest_step | 40000 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| snapshot_status | snapshots_present |
| output_dir_exists | True |
| missing_output_dir_is_expected | False |
| first_snapshot_step | 10000 |
| next_snapshot_step | - |
| steps_to_next_snapshot | - |
| next_snapshot_eta_hours | - |
| progress | 1.000000 |
| latest_ce | 3.426713 |
| latest_lr | 0.000002 |
| log_freshness_status | not_running |
| log_health_status | healthy |
| producer_config_status | matched |
| log_age_seconds | 8870465.347391 |
| time_limit_status | not_running |
| time_limit_margin_seconds | - |
| log_elapsed_seconds | 72805.900000 |
| parsed_log_rows | 4001 |
| recent_window_rows | 20 |
| recent_ce_mean | 3.463437 |
| recent_ce_min | 3.155323 |
| recent_ce_max | 3.803440 |
| seconds_per_step | 1.820147 |
| steps_per_hour | 1977.861684 |
| eta_hours | 0.000000 |
| estimated_completion_utc | 2026-09-04T03:53:18.643039+00:00 |
| segment_token_presentations_per_second | 4500.734144 |
| latest_complete_snapshot_step | 40000 |
| cumulative_token_presentations | 655360000 |

## Time Limit Gate

| field | value |
| --- | --- |
| status | not_running |
| slurm_state |  |
| elapsed | - |
| time_limit | - |
| elapsed_seconds | - |
| time_limit_seconds | - |
| eta_seconds | 0.000000 |
| remaining_seconds | - |
| margin_seconds | - |
| tight_margin_threshold_seconds | 1800 |
| caveat | Compares current ETA with Slurm time remaining; it is a runtime-risk signal, not quality evidence. |

## Log Freshness

| field | value |
| --- | --- |
| status | not_running |
| path | logs/bd-s2-655m-10250.out |
| exists | True |
| checked_utc | 2026-09-04T03:53:18.643350+00:00 |
| mtime_utc | 2026-05-24T11:52:13.295959+00:00 |
| age_seconds | 8870465.347391 |
| stale_after_seconds | 900 |
| slurm_state |  |
| caveat | Fresh logs are required while the Stage-2 producer is running. |

## Producer Log Health

| field | value |
| --- | --- |
| status | healthy |
| path | logs/bd-s2-655m-10250.out |
| parsed_step_rows | 4001 |
| first_step | 1 |
| latest_step | 40000 |
| latest_ce | 3.426713 |
| latest_lr | 0.000002 |
| latest_elapsed_seconds | 72805.900000 |
| recent_window_rows | 20 |
| recent_ce_mean | 3.463437 |
| recent_ce_min | 3.155323 |
| recent_ce_max | 3.803440 |
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
| status | snapshots_present |
| output_dir | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m |
| output_dir_exists | True |
| first_snapshot_step | 10000 |
| next_snapshot_step | - |
| steps_to_next_snapshot | - |
| next_snapshot_eta_hours | - |
| estimated_next_snapshot_utc | - |
| latest_complete_snapshot_step | 40000 |
| missing_output_dir_is_expected | False |
| caveat | A missing output directory is expected before the first snapshot when save_every_steps has not been reached. |

## Expected Snapshots

| step | dir exists | state | metrics | complete |
| --- | --- | --- | --- | --- |
| 10000 | True | True | True | True |
| 20000 | True | True | True | True |
| 30000 | True | True | True | True |
| 40000 | True | True | True | True |

## Artifacts

| artifact | exists | path |
| --- | --- | --- |
| stage2 root metrics | True | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/metrics.json |
| stage2 final state | True | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/custom_state_dict.pt |
| stage2 final snapshot metrics | True | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/metrics.json |
| handoff manifest | True | benchmarks/results/stage2_manifest_655m_2026-05-23.json |
| handoff report | True | benchmarks/results/stage2_655m_handoff_2026-05-23.json |
| downstream metrics | True | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json |
| downstream predictions | True | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| postprocess report | True | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |
| next decision report | True | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| telemetry artifact 1 | True | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/telemetry.jsonl |
| telemetry artifact 2 | True | checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200/metrics.json |

## Downstream

| field | value |
| --- | --- |
| status | complete_artifacts_present |
| handoff_report_exists | True |
| handoff_report_status | submitted_downstream |
| output_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit |
| complete | True |
| caveat | This section tracks downstream artifact existence only; it does not compute or claim MNLI accuracy. |

## Postprocess

| field | value |
| --- | --- |
| job_id | 10261 |
| slurm_state | not_submitted |
| expected_json | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |
| expected_json_exists | True |
| expected_md | benchmarks/results/stage2_655m_postprocess_2026-05-23.md |
| expected_md_exists | True |
| expected_next_decision_json | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| expected_next_decision_json_exists | True |
| expected_next_decision_md | benchmarks/results/bitdistill_next_decision_2026-05-23.md |
| expected_next_decision_md_exists | True |
| caveat | This section tracks report-regeneration job state only; it is not quality evidence. |

## Caveat

This is a cumulative continuation from the verified 327.68M checkpoint with a fresh optimizer/scheduler segment. It is not an uninterrupted 80k-step Stage-2 run.
