# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 13490 | 20000 | 0.674500 | 3.812809 | 3.526848 | -2.392822 | 1.854626 | 3.35h | false | 13 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=1350, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=13490, observations=1350 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.812809 |  |
| latest progress is within target | pass | latest=13490, max_steps=20000, progress=0.6745 |  |
| log is fresh while job is active | pass | age_seconds=1.2 |  |
| ETA is finite | pass | seconds_per_step=1.854625648628614, eta_seconds=12073.612972572277 |  |
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
