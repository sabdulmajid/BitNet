# Active Slurm Batch Script Audit

Status: **passed**.

Quality claim: **none**. This validates queued script contents only.

| purpose | job | state | script available | passed |
| --- | --- | --- | --- | --- |
| 655M Stage-2 handoff | 10253 | PENDING | true | true |
| gamma-60 gradient telemetry | 10252 | PENDING | true | true |

## Required Snippets

| job | snippet | present |
| --- | --- | --- |
| 10253 | write_failure_report() | true |
| 10253 | trap 'status=$?; trap - ERR; write_failure_report | true |
| 10253 | --downstream-failed-job-id "" | true |
| 10253 | --downstream-failure-mode "" | true |
| 10252 | --telemetry-every-steps | true |
| 10252 | --telemetry-component-grad-norms | true |
| 10252 | --attention-kd-weight "$ATTENTION_KD_WEIGHT" | true |
