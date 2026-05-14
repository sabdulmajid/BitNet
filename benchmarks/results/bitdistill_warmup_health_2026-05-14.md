# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 8590 | 20000 | 0.429500 | 3.955122 | 4.295503 | -1.639483 | 1.820338 | 5.77h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=860, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=8590, observations=860 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.955122 |  |
| latest progress is within target | pass | latest=8590, max_steps=20000, progress=0.4295 |  |
| log is fresh while job is active | pass | age_seconds=13.3 |  |
| ETA is finite | pass | seconds_per_step=1.8203376018626312, eta_seconds=20770.052037252623 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
