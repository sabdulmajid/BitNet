# BitDistill Warmup Finalizer Submission, 2026-05-14

Submitted this invocation: `true`.
Job ID: `10001`.
Dependency type: `afterany`.
Warmup producer jobs: `9894`.

## Existing Warmup Finalizers

| job | state | reason |
| --- | --- | --- |
| none | - | - |

## Command

```bash
sbatch --parsable --job-name bitdistill-warmup-finalize --dependency afterany:9894 slurm_bitdistill_warmup_finalizer.sh
```
