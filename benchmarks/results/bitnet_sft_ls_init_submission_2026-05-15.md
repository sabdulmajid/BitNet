# BitNet-SFT LS-Init Submission, 2026-05-15

Submitted one focused MNLI BitNet-SFT job to test whether least-squares ternary
fixed-point initialization improves the reproduction baseline. This is not a
new result yet; it is a controlled queued experiment.

| field | value |
| --- | --- |
| job id | `10079` |
| partition | `midcard` |
| model | `Qwen/Qwen2.5-0.5B` |
| task | `mnli` |
| method | `bitnet_sft` |
| task format | `sequence_classification` |
| scale mode | `tensor` |
| ternary init mode | `ls` |
| ternary init iterations | `8` |
| steps | `10000` |
| learning rate | `2e-5` |
| per-device batch size | `4` |
| gradient accumulation | `4` |
| changed axis | initialization only |
| output dir | `checkpoints/bitdistill-glue-seqcls-bitnet-sft-init/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-initls-steps10000-lr2em5` |
| job table | `benchmark_results/bitnet_sft_ls_init_submission_2026-05-15.tsv` |

Comparison target: existing absmean BitNet-SFT budget row
`checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5`.

Success gate: full MNLI validation accuracy and paired predictions improve over
the matched absmean run without changing task formulation, optimizer budget,
activation quantization, SubLN setting, or classifier-head treatment.
