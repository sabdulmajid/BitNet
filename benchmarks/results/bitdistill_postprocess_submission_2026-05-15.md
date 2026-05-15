# BitDistill Postprocess Finalizer, 2026-05-15

Submitted this invocation: `true`.
Job ID: `10039`.
Dependency type: `afterok`.
Producer jobs: `34`.

## Producer Breakdown

| source | count | job ids |
| --- | ---: | --- |
| Stage-2 warmup | 0 | `` |
| downstream GLUE/export rows | 32 | `9956, 9957, 9958, 9959, 9960, 9961, 9962, 9963, 9964, 9965, 9966, 9971, 9972, 9973, 9974, 9975, 9976, 9978, 9979, 9980, 9981, 9982, 9983, 9987, 9988, 9989, 9990, 9991, 9992, 9993, 9994, 9995` |
| extra producer jobs | 2 | `10025, 10037` |

## Existing Jobs

| job | state | reason |
| --- | --- | --- |
| none | - | - |

## Command

```bash
sbatch --parsable --job-name bitdistill-postprocess --dependency afterok:9956:9957:9958:9959:9960:9961:9962:9963:9964:9965:9966:9971:9972:9973:9974:9975:9976:9978:9979:9980:9981:9982:9983:9987:9988:9989:9990:9991:9992:9993:9994:9995:10025:10037 slurm_bitdistill_postprocess.sh
```
