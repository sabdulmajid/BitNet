# Stage-2 655M Handoff Preflight

Generated: `2026-05-23T18:57:53.005813+00:00`

Status: **pending_stage2_completion**.

Quality claim: **none**.

This validates the queued handoff path only. It does not run downstream evaluation or update quality claims.

## Current State

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| slurm_state | RUNNING |
| slurm_time | 3:20:20 |
| latest_step | 6580 |
| snapshot_status | pre_first_snapshot |
| next_snapshot_step | 10000 |
| steps_to_next_snapshot | 3420 |
| output_dir | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m |
| final_snapshot | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000 |

## Preflight Checks

| check | kind | path/command | passed | exists | returncode |
| --- | --- | --- | --- | --- | --- |
| parent manifest exists | file_exists | benchmarks/results/stage2_manifest_2026-05-20.json | true | true | - |
| parent manifest validates | command | python benchmarks/validate_stage2_manifest.py benchmarks/results/stage2_manifest_2026-05-20.json | true | - | 0 |
| build_stage2_manifest.py exists | file_exists | benchmarks/build_stage2_manifest.py | true | true | - |
| validate_stage2_manifest.py exists | file_exists | benchmarks/validate_stage2_manifest.py | true | true | - |
| handoff script exists | file_exists | slurm_stage2_655m_handoff.sh | true | true | - |
| postprocess script exists | file_exists | slurm_stage2_655m_postprocess.sh | true | true | - |
| handoff script syntax | command | bash -n slurm_stage2_655m_handoff.sh | true | - | 0 |
| postprocess script syntax | command | bash -n slurm_stage2_655m_postprocess.sh | true | - | 0 |
| downstream training script exists | file_exists | slurm_bitdistill_glue.sh | true | true | - |
| FP16 teacher directory exists | file_exists | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 | true | true | - |
| training save contract matches handoff assumptions | source_contract | train_bitdistill.py | true | true | - |

## Final Artifact Checks

| artifact | path | exists | size_bytes |
| --- | --- | --- | --- |
| final state dict | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/custom_state_dict.pt | false | - |
| final snapshot metrics | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/metrics.json | false | - |
| root metrics | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/metrics.json | false | - |

## Training Save Contract

| check | source pattern | passed |
| --- | --- | --- |
| root metrics are written regardless of save_model_artifacts | (output_dir / "metrics.json").write_text | true |
| root state dict is gated by save_model_artifacts | if args.save_model_artifacts: | true |
| snapshots write custom_state_dict.pt | snapshot_dir / "custom_state_dict.pt" | true |
| snapshots write metrics.json | snapshot_dir / "metrics.json" | true |
| active producer snapshot.complete flag is legacy false | "snapshot"] = {"step": step, "complete": False} | true |

The running 655M producer was submitted before any code change here. For this active run, snapshot usability is audited from actual state/metrics files, not from the legacy snapshot.complete flag.

## Manifest Command

`python benchmarks/build_stage2_manifest.py --output-dir checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m --parent-manifest benchmarks/results/stage2_manifest_2026-05-20.json --run-id qwen25-05b-bitdistill-tensor-stage2-655m-from327m-job10250 --job-id 10250 --downstream-status pending_submission --downstream-failed-job-id  --downstream-failure-mode  --output-json benchmarks/results/stage2_manifest_655m_2026-05-23.json --output-md benchmarks/results/stage2_manifest_655m_2026-05-23.md`

## Dry Run

| field | value |
| --- | --- |
| attempted | false |
| passed | - |
| build_returncode | - |
| validate_returncode | - |

## Source Artifacts

| artifact | path |
| --- | --- |
| stage2_submission | benchmarks/results/stage2_655m_submission_2026-05-23.json |
| handoff_submission | benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json |
