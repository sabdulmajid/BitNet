# Sequence-Classification Native I2_SR Full CPU Submission, 2026-05-15

Submitted one full MNLI validation job for the native single-artifact
`bitnet-qwen` classifier path.

| field | value |
| --- | --- |
| job id | `10082` |
| partition / node | `midcard / ece-nebula12` |
| status at submission check | `RUNNING` |
| script | `slurm_seqcls_native_full_cpu.sh` |
| task | `mnli` |
| samples | full validation split (`9815`) |
| prompt input | `token_ids` |
| prompt batch size | `1` |
| reason for batch size | native batched classifier logits are not invariant; audited drift is position-dependent, not a simple row swap |
| output JSON | `benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_2026-05-15.json` |
| output report | `benchmarks/results/seqcls_native_i2sr_cpu_mnli_full_token_ids_2026-05-15.md` |

This job is meant to answer only one narrow product question: whether the same
packed GGUF classifier artifact can complete full MNLI validation in the safe
single-prompt mode. It is not a batched throughput claim.

Operational note: job `10081` was canceled after three minutes because the
evaluator's product-readiness flag did not explicitly depend on the batching
parity audit. The evaluator now makes batching parity a hard readiness gate,
and job `10082` was submitted with that fix.
