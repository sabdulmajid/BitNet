# Stage-2 Snapshot Salvage Audit

Generated: `2026-09-04T03:53:18.957516+00:00`

Status: **final_snapshot_available**.

Quality claim: **none**.

This inventories Stage-2 checkpoints for failover only. It does not run downstream evaluation or create quality evidence.

## Current State

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| slurm_state | not_in_squeue |
| slurm_time | - |
| latest_logged_step | 40000 |
| max_steps | 40000 |
| save_every_steps | 10000 |
| next_snapshot_step | - |
| steps_to_next_snapshot | - |
| next_snapshot_eta_hours | - |
| complete_snapshot_count | 4 |
| target_cumulative_token_presentations | 655360000 |
| recommendation | Use the normal 655M handoff path; this report is only a fallback inventory. |

## Best Salvage Snapshot

| field | value |
| --- | --- |
| step | 40000 |
| status | complete |
| state | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/custom_state_dict.pt |
| metrics | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/metrics.json |
| cumulative_token_presentations | 655360000 |
| last_ce | 3.426713 |

## Snapshot Inventory

| step | status | state | metrics | cumulative tokens | last_ce | validation errors |
| --- | --- | --- | --- | --- | --- | --- |
| 10000 | complete | true | true | 409600000 | 3.866621 | none |
| 20000 | complete | true | true | 491520000 | 4.155915 | none |
| 30000 | complete | true | true | 573440000 | 3.667548 | none |
| 40000 | complete | true | true | 655360000 | 3.426713 | none |

## Salvage Manifest Command

`python benchmarks/build_stage2_manifest.py --output-dir checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m --snapshot-dir checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000 --allow-snapshot-metrics-root --parent-manifest benchmarks/results/stage2_manifest_2026-05-20.json --cumulative-token-presentations 655360000 --run-id qwen25-05b-bitdistill-tensor-stage2-655m-salvage-step40000-job10250 --job-id 10250 --model Qwen/Qwen2.5-0.5B --downstream-status salvage_pending_downstream --downstream-failed-job-id  --downstream-failure-mode salvage manifest from intermediate Stage-2 snapshot --output-json benchmarks/results/stage2_manifest_655m_salvage_step40000_2026-05-23.json --output-md benchmarks/results/stage2_manifest_655m_salvage_step40000_2026-05-23.md`

## Source Artifacts

| artifact | path |
| --- | --- |
| stage2_submission | benchmarks/results/stage2_655m_submission_2026-05-23.json |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
