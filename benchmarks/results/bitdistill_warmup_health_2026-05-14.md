# BitDistill Warm-Up Health Audit, 2026-05-14

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 13080 | 20000 | 0.654000 | 4.680572 | 4.132581 | -1.802405 | 1.820260 | 3.50h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1309, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=13080, observations=1309 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.680572 |  |
| latest progress is within target | pass | latest=13080, max_steps=20000, progress=0.654 |  |
| log is fresh while job is active | pass | age_seconds=0.9 |  |
| ETA is finite | pass | seconds_per_step=1.8202599388379206, eta_seconds=12596.19877675841 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
