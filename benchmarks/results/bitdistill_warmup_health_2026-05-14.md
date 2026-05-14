# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 11530 | 20000 | 0.576500 | 4.471399 | 4.228564 | -1.706422 | 1.820173 | 4.28h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1154, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=11530, observations=1154 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.471399 |  |
| latest progress is within target | pass | latest=11530, max_steps=20000, progress=0.5765 |  |
| log is fresh while job is active | pass | age_seconds=1.4 |  |
| ETA is finite | pass | seconds_per_step=1.8201734605377275, eta_seconds=15416.869210754552 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
