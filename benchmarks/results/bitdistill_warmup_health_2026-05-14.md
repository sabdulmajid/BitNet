# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 9970 | 20000 | 0.498500 | 4.059045 | 4.236360 | -1.698626 | 1.820271 | 5.07h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=998, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=9970, observations=998 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.059045 |  |
| latest progress is within target | pass | latest=9970, max_steps=20000, progress=0.4985 |  |
| log is fresh while job is active | pass | age_seconds=6.6 |  |
| ETA is finite | pass | seconds_per_step=1.8202708124373117, eta_seconds=18257.316248746236 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
