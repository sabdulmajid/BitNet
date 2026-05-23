# Active BitDistill Gate Watchdog

Generated: `2026-05-23T17:33:32.095150+00:00`

Status: **passed**.

Quality claim: **none**.

This watchdog refreshes status and validates artifacts; it does not create benchmark evidence.

## Summary

| field | value |
| --- | --- |
| monitor_status | running |
| ingestion_status | pending_handoff |
| slurm_script_status | passed |
| traceability_status | in_progress |
| next_decision_status | pending_655m_downstream |
| stage2_job_id | 10250 |
| stage2_latest_step | 3800 |
| stage2_latest_ce | 3.883600 |
| stage2_progress | 0.095000 |
| downstream_status | waiting_for_handoff |
| telemetry_state | PENDING |

## Commands

| label | passed | returncode | elapsed seconds |
| --- | --- | --- | --- |
| monitor active Stage-2 extension | true | 0 | 0.097388 |
| audit 655M ingestion | true | 0 | 0.097810 |
| audit active Slurm batch scripts | true | 0 | 0.112502 |
| build current goal status | true | 0 | 0.083586 |
| build deep research handoff | true | 0 | 0.071882 |
| build goal traceability | true | 0 | 0.099272 |
| build publication/product plan | true | 0 | 0.069453 |
| validate fail-closed reports | true | 0 | 0.068768 |
| compile Python sources | true | 0 | 0.079130 |
| check Slurm shell syntax | true | 0 | 0.005126 |

## Failures

none

## Source Artifacts

| artifact | path |
| --- | --- |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
| ingestion | benchmarks/results/stage2_655m_ingestion_2026-05-23.json |
| slurm_script_audit | benchmarks/results/active_slurm_batch_scripts_2026-05-23.json |
| traceability | benchmarks/results/bitdistill_goal_traceability_2026-05-23.json |
| publication_product_plan | benchmarks/results/bitdistill_publication_product_plan_2026-05-23.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
