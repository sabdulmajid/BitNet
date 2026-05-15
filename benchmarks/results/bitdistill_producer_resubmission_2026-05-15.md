# BitDistill Producer Resubmission, 2026-05-15

Resubmitted stale pending producer jobs after hardening the Slurm scripts.
Resubmitted the producer/finalizer jobs again after fixing the Slurm report
date handling so stored scripts respect `BITNET_REPORT_DATE` instead of
recomputing UTC date at runtime.

| role | old job | new job | dependency |
| --- | ---: | ---: | --- |
| warmup diagnostic finalizer | 10016 | 10023 | `afterany:9894` |
| causal I2_SR producer | 10019 | 10024 | `afterany:9943:9944:9945:9946:9947:9948` |
| CPU task benchmark producer | 10020 | 10025 | `afterany:9943:9944:9945:9946:9947:9948:9956:9957:9958:9959:9960:9961:9962:9963:9964:9965:9966:9971:9972:9973:9974:9975:9976:9978:9979:9980:9981:9982:9983:9987:9988:9989:9990:9991:9992:9993:9994:9995` |
| afterany diagnostic postprocess | 10021 | 10026 | refreshed from monitor |
| strict afterok postprocess | 10022 | 10027 | refreshed from monitor |

CPU producer env: `MAX_EVAL_SAMPLES=512 BATCH_SIZE=8 CHILD_TIMEOUT_SECONDS=900 BITNET_REPORT_DATE=2026-05-15`.
I2_SR producer env: `BITNET_REPORT_DATE=2026-05-15`.

Current stored-script audits pass for CPU producer `10025`, I2_SR producer
`10024`, diagnostic postprocess `10026`, and strict postprocess `10027`.
