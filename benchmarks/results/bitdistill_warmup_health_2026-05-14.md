# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 7720 | 20000 | 0.386000 | 4.642119 | 4.313175 | -1.621811 | 1.820324 | 6.21h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=773, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=7720, observations=773 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.642119 |  |
| latest progress is within target | pass | latest=7720, max_steps=20000, progress=0.386 |  |
| log is fresh while job is active | pass | age_seconds=16.8 |  |
| ETA is finite | pass | seconds_per_step=1.8203238341968913, eta_seconds=22353.576683937823 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
