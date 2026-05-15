# BitDistill Postprocess Dependency Audit, 2026-05-15

Overall status: `pass`.

Postprocess job: `10018`.

Expected producer jobs: `41`.

Warmup producer jobs: `9894`.

Missing dependencies: `[]`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor JSON exists | pass | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_job_monitor_2026-05-15.json |  |
| postprocess job is discoverable | pass | name=bitdistill-postprocess, matches=1 |  |
| expected producer jobs are active | pass | warmup=1, downstream=38, extra=2, total=41 |  |
| postprocess depends on every active producer | pass | expected=41, dependency_ids=41, missing=[] |  |

## Extra Producer Jobs

| job | name | state | reason |
| --- | --- | --- | --- |
| 9949 | bitdistill-i2sr | PENDING | (Dependency) |
| 10006 | bitdistill-cpu-bench | PENDING | (Dependency) |

## Dependency Text

`afterok:9894(unfulfilled),afterok:9943(unfulfilled),afterok:9944(unfulfilled),afterok:9945(unfulfilled),afterok:9946(unfulfilled),afterok:9947(unfulfilled),afterok:9948(unfulfilled),afterok:9949(unfulfilled),afterok:9956(unfulfilled),afterok:9957(unfulfilled),afterok:9958(unfulfilled),afterok:9959(unfulfilled),afterok:9960(unfulfilled),afterok:9961(unfulfilled),afterok:9962(unfulfilled),afterok:9963(unfulfilled),afterok:9964(unfulfilled),afterok:9965(unfulfilled),afterok:9966(unfulfilled),afterok:9971(unfulfilled),afterok:9972(unfulfilled),afterok:9973(unfulfilled),afterok:9974(unfulfilled),afterok:9975(unfulfilled),afterok:9976(unfulfilled),afterok:9978(unfulfilled),afterok:9979(unfulfilled),afterok:9980(unfulfilled),afterok:9981(unfulfilled),afterok:9982(unfulfilled),afterok:9983(unfulfilled),afterok:9987(unfulfilled),afterok:9988(unfulfilled),afterok:9989(unfulfilled),afterok:9990(unfulfilled),afterok:9991(unfulfilled),afterok:9992(unfulfilled),afterok:9993(unfulfilled),afterok:9994(unfulfilled),afterok:9995(unfulfilled),afterok:10006(unfulfilled)`
