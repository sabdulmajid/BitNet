# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 11220 | 20000 | 0.561000 | 4.211145 | 4.223673 | -1.711313 | 1.820187 | 4.44h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1123, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=11220, observations=1123 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.211145 |  |
| latest progress is within target | pass | latest=11220, max_steps=20000, progress=0.561 |  |
| log is fresh while job is active | pass | age_seconds=18.6 |  |
| ETA is finite | pass | seconds_per_step=1.820187165775401, eta_seconds=15981.24331550802 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
