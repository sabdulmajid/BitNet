# BitDistill Warm-Up Health Audit, 2026-05-15

Overall status: `pass`.

## Overview

| log | job | state | step | max steps | progress | latest CE | last CE mean | last-first CE mean | sec/step | ETA | final state | snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logs/bitdistill-glue-9894.out | 9894 | squeue_error | 20000 | 20000 | 1.000000 | 3.738920 | 4.053051 | -1.881935 | 1.820090 | 0.00h | true | 0 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| warm-up log exists | pass | logs/bitdistill-glue-9894.out |  |
| warm-up has enough observations | pass | observations=2001, required=10 |  |
| step numbers are strictly increasing | pass | first=1, latest=20000, observations=2001 |  |
| CE values are finite | pass | nonfinite=0, latest_ce=3.73892 |  |
| latest progress is within target | pass | latest=20000, max_steps=20000, progress=1.0 |  |
| log is fresh while job is active | pass | age_seconds=22553.0 |  |
| ETA is finite | pass | seconds_per_step=1.8200900000000002, eta_seconds=0.0 |  |
| monitor identifies same warm-up job | pass | monitor_job=9894, parsed_job=9894 |  |

## Script Provenance

| field | value |
| --- | --- |
| current script | slurm_bitdistill_glue.sh |
| current sha256 | dd5ea8ef8474 |
| stored sha256 | - |
| stored script available | false |
| stored matches current | false |
| stored has snapshot guard | false |
| current has snapshot guard | true |
| stored script error | scontrol did not materialize a batch script |

## Warnings

| warning |
| --- |
| Could not recover the stored Slurm batch script for provenance: scontrol did not materialize a batch script. |
