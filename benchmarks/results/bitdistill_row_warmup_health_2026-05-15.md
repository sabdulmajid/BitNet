# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 9940 | 20000 | 0.497000 | 3.919284 | 3.784343 | -2.135328 | 1.852495 | 5.18h | false | 9 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=995, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=9940, observations=995 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.919284 |  |
| latest progress is within target | pass | latest=9940, max_steps=20000, progress=0.497 |  |
| log is fresh while job is active | pass | age_seconds=11.0 |  |
| ETA is finite | pass | seconds_per_step=1.8524949698189135, eta_seconds=18636.09939637827 |  |
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
