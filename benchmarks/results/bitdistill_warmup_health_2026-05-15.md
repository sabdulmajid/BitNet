# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | not_in_squeue | 20000 | 20000 | 1.000000 | 3.738920 | 4.053051 | -1.881935 | 1.820090 | 0.00h | true | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=2001, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=20000, observations=2001 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.73892 |  |
| latest progress is within target | pass | latest=20000, max_steps=20000, progress=1.0 |  |
| log is fresh while job is active | pass | age_seconds=301.6 |  |
| ETA is finite | pass | seconds_per_step=1.8200900000000002, eta_seconds=0.0 |  |
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
| The running warm-up was submitted from an older batch script than the current checked-in launcher; current future launches have stricter snapshot guards, but this active job retains its submitted script. |
| The active warm-up stored script does not contain the current SAVE_EVERY_STEPS safety guard; this explains the no-snapshot live run and should not be treated as the current launcher policy. |
