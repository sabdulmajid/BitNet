# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 12800 | 20000 | 0.640000 | 4.180564 | 4.150992 | -1.783994 | 1.820234 | 3.64h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1281, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=12800, observations=1281 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.180564 |  |
| latest progress is within target | pass | latest=12800, max_steps=20000, progress=0.64 |  |
| log is fresh while job is active | pass | age_seconds=18.3 |  |
| ETA is finite | pass | seconds_per_step=1.820234375, eta_seconds=13105.6875 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
