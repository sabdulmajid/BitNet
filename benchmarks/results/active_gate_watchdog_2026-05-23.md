# Active BitDistill Gate Watchdog

Generated: `2026-05-23T18:50:16.363388+00:00`

Status: **passed**.

Quality claim: **none**.

This watchdog refreshes status and validates artifacts; it does not create benchmark evidence.

## Summary

| field | value |
| --- | --- |
| monitor_status | running |
| ingestion_status | pending_handoff |
| snapshot_salvage_status | no_snapshot_expected_yet |
| snapshot_salvage_complete_count | 0 |
| next_snapshot_step | 10000 |
| steps_to_next_snapshot | 3670 |
| next_snapshot_eta_hours | 1.853215 |
| afterany_job_id | 10258 |
| afterany_status | dependency_pending |
| slurm_script_status | passed |
| traceability_status | in_progress |
| next_decision_status | pending_655m_downstream |
| next_blueprint_status | pending_655m_downstream |
| next_blueprint_action | wait_and_watch_655m_gate |
| stage2_job_id | 10250 |
| stage2_latest_step | 6330 |
| stage2_latest_ce | 3.851597 |
| stage2_progress | 0.158250 |
| stage2_log_freshness | fresh_running_log |
| stage2_log_health | healthy |
| stage2_producer_config | matched |
| stage2_time_limit_status | within_time_limit |
| stage2_time_limit_margin_seconds | 1.362e+04 |
| downstream_status | waiting_for_handoff |
| telemetry_state | PENDING |

## Commands

| label | passed | returncode | elapsed seconds |
| --- | --- | --- | --- |
| monitor active Stage-2 extension | true | 0 | 0.128164 |
| audit 655M ingestion | true | 0 | 0.097248 |
| audit Stage-2 snapshot salvage | true | 0 | 0.088523 |
| audit active Slurm batch scripts | true | 0 | 0.185004 |
| build next decision | true | 0 | 0.071149 |
| build next experiment blueprint | true | 0 | 0.067974 |
| build current goal status | true | 0 | 0.084755 |
| build deep research handoff | true | 0 | 0.073000 |
| build goal traceability | true | 0 | 0.097375 |
| build paper alignment audit | true | 0 | 0.067405 |
| build publication/product plan | true | 0 | 0.067796 |
| validate fail-closed reports | true | 0 | 0.073637 |
| compile Python sources | true | 0 | 0.073630 |
| check Slurm shell syntax | true | 0 | 0.003586 |

## Failures

none

## Source Artifacts

| artifact | path |
| --- | --- |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
| ingestion | benchmarks/results/stage2_655m_ingestion_2026-05-23.json |
| snapshot_salvage | benchmarks/results/stage2_snapshot_salvage_2026-05-23.json |
| afterany_submission | benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json |
| slurm_script_audit | benchmarks/results/active_slurm_batch_scripts_2026-05-23.json |
| traceability | benchmarks/results/bitdistill_goal_traceability_2026-05-23.json |
| paper_alignment | benchmarks/results/bitdistill_paper_alignment_2026-05-23.json |
| publication_product_plan | benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| next_experiment_blueprint | benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json |
