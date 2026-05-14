# BitDistill Postprocess Dependency Audit, 2026-05-14

Overall status: `pass`.

Postprocess job: `9977`.

Expected producer jobs: `34`.

Missing dependencies: `[]`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor JSON exists | pass | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_job_monitor_2026-05-14.json |  |
| postprocess job is discoverable | pass | name=bitdistill-postprocess, matches=1 |  |
| expected producer jobs are active | pass | downstream=29, extra=5, total=34 |  |
| postprocess depends on every active producer | pass | expected=34, dependency_ids=34, missing=[] |  |

## Extra Producer Jobs

| job | name | state | reason |
| --- | --- | --- | --- |
| 9949 | bitdistill-i2sr | PENDING | (Dependency) |
| 9967 | bitdistill-cpu-bench | PENDING | (Dependency) |
| 9985 | bitdistill-predtrace | PENDING | (Resources) |
| 9986 | bitdistill-predtrace | PENDING | (Priority) |
| 9984 | bitdistill-predtrace | RUNNING | ece-nebula10 |

## Dependency Text

`afterany:9956(unfulfilled),afterany:9957(unfulfilled),afterany:9958(unfulfilled),afterany:9959(unfulfilled),afterany:9960(unfulfilled),afterany:9961(unfulfilled),afterany:9962(unfulfilled),afterany:9963(unfulfilled),afterany:9964(unfulfilled),afterany:9965(unfulfilled),afterany:9966(unfulfilled),afterany:9967(unfulfilled),afterany:9943(unfulfilled),afterany:9944(unfulfilled),afterany:9945(unfulfilled),afterany:9946(unfulfilled),afterany:9947(unfulfilled),afterany:9948(unfulfilled),afterany:9949(unfulfilled),afterany:9971(unfulfilled),afterany:9972(unfulfilled),afterany:9973(unfulfilled),afterany:9974(unfulfilled),afterany:9975(unfulfilled),afterany:9976(unfulfilled),afterany:9978(unfulfilled),afterany:9979(unfulfilled),afterany:9980(unfulfilled),afterany:9981(unfulfilled),afterany:9982(unfulfilled),afterany:9983(unfulfilled),afterany:9984(unfulfilled),afterany:9985(unfulfilled),afterany:9986(unfulfilled)`
