# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 17430 | 20000 | 0.871500 | 4.128016 | 4.077017 | -1.857968 | 1.820201 | 1.30h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1744, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=17430, observations=1744 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.128016 |  |
| latest progress is within target | pass | latest=17430, max_steps=20000, progress=0.8715 |  |
| log is fresh while job is active | pass | age_seconds=13.6 |  |
| ETA is finite | pass | seconds_per_step=1.8202008032128514, eta_seconds=4677.916064257028 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Script Provenance

| field | value |
| --- | --- |
| current script | slurm_bitdistill_glue.sh |
| current sha256 | dd5ea8ef8474 |
| stored sha256 | 57b131e197a5 |
| stored script available | true |
| stored matches current | false |
| stored has snapshot guard | false |
| current has snapshot guard | true |
| stored script error | - |

## Warnings

| warning |
| --- |
| SAVE_EVERY_STEPS is 0 and no final state exists yet; a job failure would lose current warm-up progress. |
| The running warm-up was submitted from an older batch script than the current checked-in launcher; current future launches have stricter snapshot guards, but this active job retains its submitted script. |
| The active warm-up stored script does not contain the current SAVE_EVERY_STEPS safety guard; this explains the no-snapshot live run and should not be treated as the current launcher policy. |
