# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 8780 | 20000 | 0.439000 | 3.596343 | 3.939845 | -1.979826 | 1.852608 | 5.77h | false | 8 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=879, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=8780, observations=879 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.596343 |  |
| latest progress is within target | pass | latest=8780, max_steps=20000, progress=0.439 |  |
| log is fresh while job is active | pass | age_seconds=20.1 |  |
| ETA is finite | pass | seconds_per_step=1.8526082004555808, eta_seconds=20786.264009111615 |  |
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
