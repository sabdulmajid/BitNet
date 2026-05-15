# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 660 | 20000 | 0.033000 | 5.095449 | 6.186211 | 0.000000 | 1.815303 | 9.75h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=67, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=660, observations=67 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=5.095449 |  |
| latest progress is within target | pass | latest=660, max_steps=20000, progress=0.033 |  |
| log is fresh while job is active | pass | age_seconds=9.3 |  |
| ETA is finite | pass | seconds_per_step=1.8153030303030302, eta_seconds=35107.9606060606 |  |
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
