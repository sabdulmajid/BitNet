# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 9140 | 20000 | 0.457000 | 4.467998 | 4.319034 | -1.615952 | 1.820317 | 5.49h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=915, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=9140, observations=915 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.467998 |  |
| latest progress is within target | pass | latest=9140, max_steps=20000, progress=0.457 |  |
| log is fresh while job is active | pass | age_seconds=2.2 |  |
| ETA is finite | pass | seconds_per_step=1.820317286652079, eta_seconds=19768.645733041576 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
