# BitDistill Warm-Up Health Audit, 2026-05-16

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-10070.out | 10070 | RUNNING | 29620 | 40000 | 0.740500 | 3.954908 | 3.932478 | -2.001633 | 1.822529 | 5.25h | false | 2 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-10070.out |  |
| warm-up has enough observations | pass | observations=2963, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=29620, observations=2963 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.954908 |  |
| latest progress is within target | pass | latest=29620, max_steps=40000, progress=0.7405 |  |
| log is fresh while job is active | pass | age_seconds=2.3 |  |
| ETA is finite | pass | seconds_per_step=1.8225286968264687, eta_seconds=18917.847873058745 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=10070, explicit_log_override=True |  |

## Script Provenance

| field | value |
| --- | --- |
| current script | slurm_bitdistill_glue.sh |
| current sha256 | 6e86e849d5fe |
| stored sha256 | 33dc357be7c4 |
| stored script available | true |
| stored matches current | false |
| stored has snapshot guard | true |
| current has snapshot guard | true |
| stored script error | - |

## Warnings

| warning |
| --- |
| The monitor JSON points at a different warm-up job, but an explicit log path was supplied; the explicit log is treated as authoritative for this live health audit. |
| The running warm-up was submitted from an older batch script than the current checked-in launcher; current future launches have stricter snapshot guards, but this active job retains its submitted script. |
