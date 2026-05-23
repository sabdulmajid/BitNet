# Stage-2 655.36M Submission, 2026-05-23

Status: **running**.

This run starts the next controlled Stage-2 token-budget point for the
BitDistill MNLI recovery curve.

| field | value |
| --- | --- |
| submitted_job_id | `10250` |
| cancelled_job_id | `10249` |
| partition | `midcard` |
| node_at_submission | `ece-nebula12` |
| model | `Qwen/Qwen2.5-0.5B` |
| stage | `continued_pretrain` |
| method | `bitdistill` |
| scale_mode | `tensor` |
| parent_manifest | `benchmarks/results/stage2_manifest_2026-05-20.json` |
| parent_tokens | `327,680,000` |
| segment_tokens | `327,680,000` |
| cumulative_tokens | `655,360,000` |
| paper_stage2_fraction | `6.5536%` |
| lr | `2e-6` |
| lr_scheduler | `constant` |
| warmup_steps | `0` |
| save_every_steps | `10000` |
| output_dir | `checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m` |

## Cancellation Note

Job `10249` was cancelled after 13 seconds because it was submitted before
`slurm_bitdistill_glue.sh` passed `LR_SCHEDULER` through to
`train_bitdistill.py`. Job `10250` was resubmitted after the wrapper patch, and
its log confirms `LR_SCHEDULER=constant`.

## Caveat

This is a cumulative continuation from the verified `327.68M` checkpoint with a
fresh optimizer/scheduler segment. It is not an uninterrupted 80k-step Stage-2
run.

## Next After Completion

1. Build a Stage-2 manifest with
   `--parent-manifest benchmarks/results/stage2_manifest_2026-05-20.json`.
2. Run matched MNLI downstream BitDistill with `INIT_STATE_MANIFEST` pointing at
   the new `655.36M` manifest.
3. Update the controlled Stage-2 curve only after `metrics.json` and
   `eval_predictions.jsonl` exist.
