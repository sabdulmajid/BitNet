# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 8890 | 20000 | 0.444500 | 3.501383 | 3.916950 | -2.002721 | 1.852238 | 5.72h | false | 8 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=890, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=8890, observations=890 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.501383 |  |
| latest progress is within target | pass | latest=8890, max_steps=20000, progress=0.4445 |  |
| log is fresh while job is active | pass | age_seconds=10.4 |  |
| ETA is finite | pass | seconds_per_step=1.8522384701912262, eta_seconds=20578.369403824523 |  |
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
