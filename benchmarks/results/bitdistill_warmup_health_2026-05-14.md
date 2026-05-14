# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 12390 | 20000 | 0.619500 | 3.940150 | 4.172248 | -1.762738 | 1.820226 | 3.85h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1240, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=12390, observations=1240 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.94015 |  |
| latest progress is within target | pass | latest=12390, max_steps=20000, progress=0.6195 |  |
| log is fresh while job is active | pass | age_seconds=6.8 |  |
| ETA is finite | pass | seconds_per_step=1.820225988700565, eta_seconds=13851.9197740113 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
