# BitDistill Postprocess Dependency Audit, 2026-05-15

Overall status: `pass`.

Postprocess job: `10039`.

Expected producer jobs: `30`.

Warmup producer jobs: `-`.

Missing dependencies: `[]`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor JSON exists | pass | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_job_monitor_2026-05-15.json |  |
| postprocess job is discoverable | pass | name=bitdistill-postprocess, matches=1 |  |
| expected producer jobs are active | pass | warmup=0, downstream=28, extra=2, total=30 |  |
| postprocess depends on every active producer | pass | expected=30, dependency_ids=30, missing=[] |  |

## Extra Producer Jobs

| job | name | state | reason |
| --- | --- | --- | --- |
| 10025 | bitdistill-cpu-bench | PENDING | (Dependency) |
| 10037 | bitdistill-i2sr | PENDING | (Priority) |

## Dependency Text

`afterok:9960(unfulfilled),afterok:9961(unfulfilled),afterok:9962(unfulfilled),afterok:9963(unfulfilled),afterok:9964(unfulfilled),afterok:9965(unfulfilled),afterok:9966(unfulfilled),afterok:9971(unfulfilled),afterok:9972(unfulfilled),afterok:9973(unfulfilled),afterok:9974(unfulfilled),afterok:9975(unfulfilled),afterok:9976(unfulfilled),afterok:9978(unfulfilled),afterok:9979(unfulfilled),afterok:9980(unfulfilled),afterok:9981(unfulfilled),afterok:9982(unfulfilled),afterok:9983(unfulfilled),afterok:9987(unfulfilled),afterok:9988(unfulfilled),afterok:9989(unfulfilled),afterok:9990(unfulfilled),afterok:9991(unfulfilled),afterok:9992(unfulfilled),afterok:9993(unfulfilled),afterok:9994(unfulfilled),afterok:9995(unfulfilled),afterok:10025(unfulfilled),afterok:10037(unfulfilled)`
