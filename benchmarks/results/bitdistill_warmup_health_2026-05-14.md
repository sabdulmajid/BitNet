# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 7540 | 20000 | 0.377000 | 4.452291 | 4.369473 | -1.565512 | 1.820318 | 6.30h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=755, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=7540, observations=755 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.452291 |  |
| latest progress is within target | pass | latest=7540, max_steps=20000, progress=0.377 |  |
| log is fresh while job is active | pass | age_seconds=7.0 |  |
| ETA is finite | pass | seconds_per_step=1.820318302387268, eta_seconds=22681.16604774536 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
