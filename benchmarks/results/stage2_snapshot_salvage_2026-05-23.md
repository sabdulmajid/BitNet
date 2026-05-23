# Stage-2 Snapshot Salvage Audit

Generated: `2026-05-23T18:48:45.011146+00:00`

Status: **no_snapshot_expected_yet**.

Quality claim: **none**.

This inventories Stage-2 checkpoints for failover only. It does not run downstream evaluation or create quality evidence.

## Current State

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| slurm_state | RUNNING |
| slurm_time | 3:11:20 |
| latest_logged_step | 6280 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| next_snapshot_step | 10000 |
| steps_to_next_snapshot | 3720 |
| next_snapshot_eta_hours | 1.878478 |
| complete_snapshot_count | 0 |
| target_cumulative_token_presentations | 655360000 |
| recommendation | Keep watching; no checkpoint is expected before the first save interval. |

## Best Salvage Snapshot

| field | value |
| --- | --- |
| step | - |
| status | - |
| state | - |
| metrics | - |
| cumulative_token_presentations | - |
| last_ce | - |

## Snapshot Inventory

| step | status | state | metrics | cumulative tokens | last_ce | validation errors |
| --- | --- | --- | --- | --- | --- | --- |
| 10000 | missing | false | false | - | - | none |
| 20000 | missing | false | false | - | - | none |
| 30000 | missing | false | false | - | - | none |
| 40000 | missing | false | false | - | - | none |

## Salvage Manifest Command

No complete intermediate snapshot is available yet.

## Source Artifacts

| artifact | path |
| --- | --- |
| stage2_submission | benchmarks/results/stage2_655m_submission_2026-05-23.json |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
