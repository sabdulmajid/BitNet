# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 13280 | 20000 | 0.664000 | 4.100414 | 4.148844 | -1.786142 | 1.820256 | 3.40h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1329, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=13280, observations=1329 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.100414 |  |
| latest progress is within target | pass | latest=13280, max_steps=20000, progress=0.664 |  |
| log is fresh while job is active | pass | age_seconds=15.0 |  |
| ETA is finite | pass | seconds_per_step=1.8202560240963856, eta_seconds=12232.120481927712 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
