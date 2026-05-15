# BitDistill Controlled Postprocess Submission, 2026-05-15

This job refreshes the narrow controlled-matrix audits after the current
BitNet-SFT and BitDistill controlled jobs finish.

| field | value |
| --- | --- |
| job id | `10072` |
| dependency | `afterany:10067:10068:10069:10070:10071` |
| partition | `midcard` |
| script | `slurm_bitdistill_controlled_postprocess.sh` |
| stdout | `logs/bitdistill-ctrl-post-10072.out` |
| stderr | `logs/bitdistill-ctrl-post-10072.err` |
| current state at submission | `PENDING (Dependency)` |

The dependency is intentionally `afterany`, not `afterok`, so the postprocess
audits still run and record failures or missing artifacts if one of the
producer jobs fails.
