# Stage-2 655.36M Handoff Submission, 2026-05-23

Status: **dependency pending**.

| field | value |
| --- | --- |
| stage2_job_id | `10250` |
| handoff_job_id | `10259` |
| cancelled_handoff_job_ids | `10253`, `10255` |
| dependency | `afterok:10250` |
| partition | `midcard` |
| script | `slurm_stage2_655m_handoff.sh` |
| postprocess_script | `slurm_stage2_655m_postprocess.sh` |
| producer_bitnet_commit | `10341701e5104c66d18fc9779ab9799bf2190c9a` |
| producer_llama_cpp_commit | `dc0bc5ee0423a2202d6284a4fc2d78d1e39905d7` |
| expected_manifest_json | `benchmarks/results/stage2_manifest_655m_2026-05-23.json` |
| expected_downstream_output_dir | `checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit` |
| expected_postprocess_json | `benchmarks/results/stage2_655m_postprocess_2026-05-23.json` |
| expected_next_decision_json | `benchmarks/results/bitdistill_next_decision_2026-05-23.json` |

Job `10253` was cancelled while dependency-pending because Slurm had
snapshotted the pre-postprocess handoff script. Job `10255` was cancelled while
dependency-pending because it lacked explicit producer commit metadata in the
manifest handoff path. Job `10259` was resubmitted with the provenance-pinned
handoff script that queues the downstream postprocess job.

The producer BitNet commit is inferred from the commit that captured the
`LR_SCHEDULER` wrapper patch used by Stage-2 job `10250`; the job log confirms
`LR_SCHEDULER=constant`.

The handoff job will only run if Stage-2 job `10250` exits successfully. It is
responsible for building and validating the `655.36M` Stage-2 manifest,
submitting the matched downstream MNLI BitDistill evaluation, and queuing a
postprocess job after that downstream run terminates.

Do not update quality claims until the downstream directory has both
`metrics.json` and `eval_predictions.jsonl`.
