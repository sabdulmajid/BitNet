# BitDistill Postprocess Finalizer, 2026-05-14

Submitted this invocation: `true`.
Job ID: `10014`.
Dependency type: `afterany`.
Producer jobs: `41`.

## Producer Breakdown

| source | count | job ids |
| --- | ---: | --- |
| Stage-2 warmup | 1 | `9894` |
| downstream GLUE/export rows | 38 | `9943, 9944, 9945, 9946, 9947, 9948, 9956, 9957, 9958, 9959, 9960, 9961, 9962, 9963, 9964, 9965, 9966, 9971, 9972, 9973, 9974, 9975, 9976, 9978, 9979, 9980, 9981, 9982, 9983, 9987, 9988, 9989, 9990, 9991, 9992, 9993, 9994, 9995` |
| extra producer jobs | 2 | `9949, 10006` |

## Existing Jobs

| job | state | reason |
| --- | --- | --- |
| 10010 | PENDING | (Dependency) |

## Command

```bash
sbatch --parsable --job-name bitdistill-postprocess-any --dependency afterany:9894:9943:9944:9945:9946:9947:9948:9949:9956:9957:9958:9959:9960:9961:9962:9963:9964:9965:9966:9971:9972:9973:9974:9975:9976:9978:9979:9980:9981:9982:9983:9987:9988:9989:9990:9991:9992:9993:9994:9995:10006 slurm_bitdistill_postprocess.sh
```
