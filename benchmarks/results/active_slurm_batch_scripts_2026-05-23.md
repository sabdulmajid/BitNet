# Active Slurm Batch Script Audit

Status: **passed**.

Quality claim: **none**. This validates queued script contents only.

| purpose | job | state | script available | passed |
| --- | --- | --- | --- | --- |
| 655M Stage-2 handoff | 10255 | PENDING | true | true |
| gamma-60 gradient telemetry | 10256 | PENDING | true | true |

## Required Snippets

| job | snippet | present |
| --- | --- | --- |
| 10255 | write_failure_report() | true |
| 10255 | trap 'status=$?; trap - ERR; write_failure_report | true |
| 10255 | --downstream-failed-job-id "" | true |
| 10255 | --downstream-failure-mode "" | true |
| 10255 | slurm_stage2_655m_postprocess.sh | true |
| 10255 | POSTPROCESS_JOB_ID | true |
| 10256 | write_status_report() | true |
| 10256 | export ATTENTION_KD_WEIGHT=60 | true |
| 10256 | export MAX_STEPS=200 | true |
| 10256 | export TELEMETRY_EVERY_STEPS=25 | true |
| 10256 | export TELEMETRY_COMPONENT_GRAD_NORMS=1 | true |
| 10256 | audit_bitdistill_gamma_balance.py | true |
| 10256 | validate_reports_fail_closed.py | true |
