# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | RUNNING | 16690 | 20000 | 0.834500 | 4.225062 | 4.096008 | -1.838978 | 1.820240 | 1.67h | false | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=1670, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=16690, observations=1670 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=4.225062 |  |
| latest progress is within target | pass | latest=16690, max_steps=20000, progress=0.8345 |  |
| log is fresh while job is active | pass | age_seconds=6.5 |  |
| ETA is finite | pass | seconds_per_step=1.8202396644697423, eta_seconds=6024.993289394847 |  |
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
