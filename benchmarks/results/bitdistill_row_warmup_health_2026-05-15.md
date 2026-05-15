# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 6670 | 20000 | 0.333500 | 4.051945 | 4.086693 | -1.832978 | 1.851859 | 6.86h | false | 6 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=668, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=6670, observations=668 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.051945 |  |
| latest progress is within target | pass | latest=6670, max_steps=20000, progress=0.3335 |  |
| log is fresh while job is active | pass | age_seconds=4.7 |  |
| ETA is finite | pass | seconds_per_step=1.8518590704647675, eta_seconds=24685.281409295352 |  |
| monitor identifies same warm-up job | pass | monitor_job=10028, parsed_job=10028 |  |

## Script Provenance

| field | value |
| --- | --- |
| current script | slurm_bitdistill_glue.sh |
| current sha256 | dd5ea8ef8474 |
| stored sha256 | dd5ea8ef8474 |
| stored script available | true |
| stored matches current | true |
| stored has snapshot guard | true |
| current has snapshot guard | true |
| stored script error | - |

## Warnings

| warning |
| --- |
| none |
