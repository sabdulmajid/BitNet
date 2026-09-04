# Active BitDistill Gate Watchdog

Generated: `2026-09-04T03:53:38.388215+00:00`

Status: **passed**.

Quality claim: **none**.

This watchdog refreshes status and validates artifacts; it does not create benchmark evidence.

## Summary

| field | value |
| --- | --- |
| monitor_status | complete_artifacts_present |
| ingestion_status | ingested_reports_rebuilt |
| snapshot_salvage_status | final_snapshot_available |
| snapshot_salvage_complete_count | 4 |
| handoff_preflight_status | ready_for_handoff |
| next_snapshot_step | - |
| steps_to_next_snapshot | - |
| next_snapshot_eta_hours | - |
| afterany_job_id | 10258 |
| afterany_status | dependency_pending |
| slurm_script_status | passed |
| traceability_status | in_progress |
| next_decision_status | run_gamma_balanced_downstream |
| next_blueprint_status | run_gamma_balanced_downstream |
| next_blueprint_action | run_matched_gamma60_mnli_downstream |
| stage2_job_id | 10250 |
| stage2_latest_step | 40000 |
| stage2_latest_ce | 3.426713 |
| stage2_progress | 1.000000 |
| stage2_log_freshness | not_running |
| stage2_log_health | healthy |
| stage2_producer_config | matched |
| stage2_time_limit_status | not_running |
| stage2_time_limit_margin_seconds | - |
| downstream_status | complete_artifacts_present |
| telemetry_state | not_in_squeue |

## Commands

| label | passed | returncode | elapsed seconds |
| --- | --- | --- | --- |
| monitor active Stage-2 extension | true | 0 | 0.137308 |
| audit 655M ingestion | true | 0 | 0.234090 |
| audit Stage-2 snapshot salvage | true | 0 | 0.080506 |
| audit 655M handoff preflight | true | 0 | 18.650691 |
| audit active Slurm batch scripts | true | 0 | 0.200844 |
| build next decision | true | 0 | 0.060654 |
| build next experiment blueprint | true | 0 | 0.057809 |
| build current goal status | true | 0 | 0.069999 |
| build deep research handoff | true | 0 | 0.058191 |
| build goal traceability | true | 0 | 0.091953 |
| build paper alignment audit | true | 0 | 0.055157 |
| build publication/product plan | true | 0 | 0.057287 |
| validate fail-closed reports | true | 0 | 0.055425 |
| compile Python sources | true | 0 | 0.056352 |
| check Slurm shell syntax | true | 0 | 0.006333 |

## Failures

none

## Source Artifacts

| artifact | path |
| --- | --- |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
| ingestion | benchmarks/results/stage2_655m_ingestion_2026-05-23.json |
| snapshot_salvage | benchmarks/results/stage2_snapshot_salvage_2026-05-23.json |
| handoff_preflight | benchmarks/results/stage2_655m_handoff_preflight_2026-05-23.json |
| afterany_submission | benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json |
| slurm_script_audit | benchmarks/results/active_slurm_batch_scripts_2026-05-23.json |
| traceability | benchmarks/results/bitdistill_goal_traceability_2026-05-23.json |
| paper_alignment | benchmarks/results/bitdistill_paper_alignment_2026-05-23.json |
| publication_product_plan | benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| next_experiment_blueprint | benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json |
