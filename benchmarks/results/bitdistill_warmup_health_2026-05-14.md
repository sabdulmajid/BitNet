# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 8340 | 20000 | 0.417000 | 4.494650 | 4.302079 | -1.632907 | 1.820348 | 5.90h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=835, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=8340, observations=835 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.49465 |  |
| latest progress is within target | pass | latest=8340, max_steps=20000, progress=0.417 |  |
| log is fresh while job is active | pass | age_seconds=10.4 |  |
| ETA is finite | pass | seconds_per_step=1.8203477218225421, eta_seconds=21225.25443645084 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
