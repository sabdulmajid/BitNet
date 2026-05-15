# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 100 | 20000 | 0.005000 | 6.321851 | 7.774755 | 0.000000 | 1.803000 | 9.97h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=11, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=100, observations=11 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=6.321851 |  |
| latest progress is within target | pass | latest=100, max_steps=20000, progress=0.005 |  |
| log is fresh while job is active | pass | age_seconds=6.4 |  |
| ETA is finite | pass | seconds_per_step=1.8030000000000002, eta_seconds=35879.700000000004 |  |
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
