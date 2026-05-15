# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 5750 | 20000 | 0.287500 | 4.188304 | 4.186841 | -1.732829 | 1.851322 | 7.33h | false | 5 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=576, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=5750, observations=576 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.188304 |  |
| latest progress is within target | pass | latest=5750, max_steps=20000, progress=0.2875 |  |
| log is fresh while job is active | pass | age_seconds=22.7 |  |
| ETA is finite | pass | seconds_per_step=1.8513217391304349, eta_seconds=26381.334782608697 |  |
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
