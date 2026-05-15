# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10028.out | 10028 | RUNNING | 5090 | 20000 | 0.254500 | 4.410480 | 4.268116 | -1.651554 | 1.855422 | 7.68h | false | 5 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10028.out |  |
| warm-up has enough observations | pass | observations=510, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=5090, observations=510 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.41048 |  |
| latest progress is within target | pass | latest=5090, max_steps=20000, progress=0.2545 |  |
| log is fresh while job is active | pass | age_seconds=13.1 |  |
| ETA is finite | pass | seconds_per_step=1.8554223968565815, eta_seconds=27664.34793713163 |  |
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
