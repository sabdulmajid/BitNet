# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 3530 | 20000 | 0.176500 | 3.787263 | 4.516065 | -1.403605 | 1.846686 | 8.45h | false | 3 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=354, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=3530, observations=354 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.787263 |  |
| latest progress is within target | pass | latest=3530, max_steps=20000, progress=0.1765 |  |
| log is fresh while job is active | pass | age_seconds=14.5 |  |
| ETA is finite | pass | seconds_per_step=1.846685552407932, eta_seconds=30414.91104815864 |  |
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
