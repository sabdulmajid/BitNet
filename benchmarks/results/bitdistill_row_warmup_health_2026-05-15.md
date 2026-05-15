# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 8380 | 20000 | 0.419000 | 4.066986 | 3.933244 | -1.986426 | 1.853962 | 5.98h | false | 8 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=839, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=8380, observations=839 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.066986 |  |
| latest progress is within target | pass | latest=8380, max_steps=20000, progress=0.419 |  |
| log is fresh while job is active | pass | age_seconds=7.2 |  |
| ETA is finite | pass | seconds_per_step=1.8539618138424823, eta_seconds=21543.036276849645 |  |
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
