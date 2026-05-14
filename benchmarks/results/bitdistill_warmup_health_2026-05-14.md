# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 10180 | 20000 | 0.509000 | 3.854085 | 4.219702 | -1.715284 | 1.820255 | 4.97h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1019, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=10180, observations=1019 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.854085 |  |
| latest progress is within target | pass | latest=10180, max_steps=20000, progress=0.509 |  |
| log is fresh while job is active | pass | age_seconds=0.5 |  |
| ETA is finite | pass | seconds_per_step=1.8202554027504911, eta_seconds=17874.908055009822 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
