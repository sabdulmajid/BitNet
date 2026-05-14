# BitDistill Postprocess Dependency Audit, 2026-05-14

Overall status: `pass`.

Postprocess job: `10005`.

Expected producer jobs: `42`.

Warmup producer jobs: `9894`.

Missing dependencies: `[]`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor JSON exists | pass | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_job_monitor_2026-05-14.json |  |
| postprocess job is discoverable | pass | name=bitdistill-postprocess-any, matches=1 |  |
| expected producer jobs are active | pass | warmup=1, downstream=38, extra=3, total=42 |  |
| postprocess depends on every active producer | pass | expected=42, dependency_ids=42, missing=[] |  |

## Extra Producer Jobs

| job | name | state | reason |
| --- | --- | --- | --- |
| 9949 | bitdistill-i2sr | PENDING | (Dependency) |
| 9967 | bitdistill-cpu-bench | PENDING | (Dependency) |
| 9997 | bitdistill-cpu-bench | PENDING | (Dependency) |

## Dependency Text

`afterany:9894(unfulfilled),afterany:9943(unfulfilled),afterany:9944(unfulfilled),afterany:9945(unfulfilled),afterany:9946(unfulfilled),afterany:9947(unfulfilled),afterany:9948(unfulfilled),afterany:9949(unfulfilled),afterany:9956(unfulfilled),afterany:9957(unfulfilled),afterany:9958(unfulfilled),afterany:9959(unfulfilled),afterany:9960(unfulfilled),afterany:9961(unfulfilled),afterany:9962(unfulfilled),afterany:9963(unfulfilled),afterany:9964(unfulfilled),afterany:9965(unfulfilled),afterany:9966(unfulfilled),afterany:9967(unfulfilled),afterany:9971(unfulfilled),afterany:9972(unfulfilled),afterany:9973(unfulfilled),afterany:9974(unfulfilled),afterany:9975(unfulfilled),afterany:9976(unfulfilled),afterany:9978(unfulfilled),afterany:9979(unfulfilled),afterany:9980(unfulfilled),afterany:9981(unfulfilled),afterany:9982(unfulfilled),afterany:9983(unfulfilled),afterany:9987(unfulfilled),afterany:9988(unfulfilled),afterany:9989(unfulfilled),afterany:9990(unfulfilled),afterany:9991(unfulfilled),afterany:9992(unfulfilled),afterany:9993(unfulfilled),afterany:9994(unfulfilled),afterany:9995(unfulfilled),afterany:9997(unfulfilled)`
