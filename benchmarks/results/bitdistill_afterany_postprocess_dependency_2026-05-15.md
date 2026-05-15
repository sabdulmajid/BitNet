# BitDistill Postprocess Dependency Audit, 2026-05-15

Overall status: `pass`.

Postprocess job: `10038`.

Expected producer jobs: `22`.

Warmup producer jobs: `-`.

Missing dependencies: `[]`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor JSON exists | pass | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_job_monitor_2026-05-15.json |  |
| postprocess job is discoverable | pass | name=bitdistill-postprocess-any, matches=1 |  |
| expected producer jobs are active | pass | warmup=0, downstream=20, extra=2, total=22 |  |
| postprocess depends on every active producer | pass | expected=22, dependency_ids=22, missing=[] |  |

## Extra Producer Jobs

| job | name | state | reason |
| --- | --- | --- | --- |
| 10025 | bitdistill-cpu-bench | PENDING | (Dependency) |
| 10037 | bitdistill-i2sr | PENDING | (Priority) |

## Dependency Text

`afterany:9972(unfulfilled),afterany:9973(unfulfilled),afterany:9974(unfulfilled),afterany:9975(unfulfilled),afterany:9976(unfulfilled),afterany:9978(unfulfilled),afterany:9979(unfulfilled),afterany:9980(unfulfilled),afterany:9981(unfulfilled),afterany:9982(unfulfilled),afterany:9983(unfulfilled),afterany:9987(unfulfilled),afterany:9988(unfulfilled),afterany:9989(unfulfilled),afterany:9990(unfulfilled),afterany:9991(unfulfilled),afterany:9992(unfulfilled),afterany:9993(unfulfilled),afterany:9994(unfulfilled),afterany:9995(unfulfilled),afterany:10025(unfulfilled),afterany:10037(unfulfilled)`
