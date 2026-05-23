# Stage-2 655.36M Handoff Submission, 2026-05-23

Status: **dependency pending**.

| field | value |
| --- | --- |
| stage2_job_id | `10250` |
| handoff_job_id | `10255` |
| cancelled_handoff_job_id | `10253` |
| dependency | `afterok:10250` |
| partition | `midcard` |
| script | `slurm_stage2_655m_handoff.sh` |
| postprocess_script | `slurm_stage2_655m_postprocess.sh` |
| expected_manifest_json | `benchmarks/results/stage2_manifest_655m_2026-05-23.json` |
| expected_downstream_output_dir | `checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit` |
| expected_postprocess_json | `benchmarks/results/stage2_655m_postprocess_2026-05-23.json` |

Job `10253` was cancelled while dependency-pending because Slurm had
snapshotted the pre-postprocess handoff script. Job `10255` was resubmitted with
the current handoff script that queues the downstream postprocess job.

The handoff job will only run if Stage-2 job `10250` exits successfully. It is
responsible for building and validating the `655.36M` Stage-2 manifest,
submitting the matched downstream MNLI BitDistill evaluation, and queuing a
postprocess job after that downstream run terminates.

Do not update quality claims until the downstream directory has both
`metrics.json` and `eval_predictions.jsonl`.
