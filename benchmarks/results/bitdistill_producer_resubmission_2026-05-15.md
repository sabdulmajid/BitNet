# BitDistill Producer Resubmission, 2026-05-15

Resubmitted stale pending producer jobs after hardening the Slurm scripts.

| role | old job | new job | dependency |
| --- | ---: | ---: | --- |
| causal I2_SR producer | 9949 | 10019 | `afterany:9943:9944:9945:9946:9947:9948` |
| CPU task benchmark producer | 10006 | 10020 | `afterany:9943:9944:9945:9946:9947:9948:9956:9957:9958:9959:9960:9961:9962:9963:9964:9965:9966:9971:9972:9973:9974:9975:9976:9978:9979:9980:9981:9982:9983:9987:9988:9989:9990:9991:9992:9993:9994:9995` |
| afterany diagnostic postprocess | 10017 | 10021 | refreshed from monitor |
| strict afterok postprocess | 10018 | 10022 | refreshed from monitor |

CPU producer env: `MAX_EVAL_SAMPLES=512 BATCH_SIZE=8 CHILD_TIMEOUT_SECONDS=900 BITNET_REPORT_DATE=2026-05-15`.

Warmup finalizer `10016` was not resubmitted because its stored script already contains `BITNET_REPORT_DATE` and it depends only on warmup job `9894`.
