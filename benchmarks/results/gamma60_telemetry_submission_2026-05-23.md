# Gamma-60 Gradient Telemetry Submission, 2026-05-23

Status: **dependency pending**.

| field | value |
| --- | --- |
| job_id | `10256` |
| cancelled_job_id | `10254` |
| dependency | `afterok:10250` |
| partition | `midcard` |
| script | `slurm_gamma60_telemetry.sh` |
| task | `MNLI` |
| method | `BitDistill` |
| attention_kd_weight | `60` |
| max_steps | `200` |
| telemetry_every_steps | `25` |
| component_grad_norms | `true` |
| output_dir | `checkpoints/bitdistill-glue-seqcls-telemetry-gamma60/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-gamma60-headinit-steps200` |

Job `10252` was cancelled while dependency-pending because its stored batch
script did not embed the diagnostic constants. Job `10254` was cancelled while
dependency-pending because its stored batch script did not include post-run
gamma-balance report generation. Job `10256` was resubmitted with
`slurm_gamma60_telemetry.sh`, which hardcodes `ATTENTION_KD_WEIGHT=60`, the
telemetry settings, and the post-run balance reporting in the Slurm batch
script.

This is a short gradient-balance diagnostic, not a quality benchmark. Its role
is to compare equalized-gamma component-gradient telemetry against the existing
paper-gamma telemetry.

Existing paper-gamma telemetry from
`benchmarks/results/bitdistill_training_dynamics_2026-05-23.json` measured:

| metric | value |
| --- | ---: |
| final grad attention/CE | `221.384986` |
| final loss attention/CE | `2549.206537` |

Do not use this run to update task-quality claims.

Expected post-run reports:

- `benchmarks/results/gamma60_telemetry_status_2026-05-23.json`
- `benchmarks/results/gamma60_telemetry_status_2026-05-23.md`
- `benchmarks/results/bitdistill_training_dynamics_2026-05-23.json`
- `benchmarks/results/bitdistill_training_dynamics_2026-05-23.md`
- `benchmarks/results/gamma60_gradient_balance_2026-05-23.json`
- `benchmarks/results/gamma60_gradient_balance_2026-05-23.md`
