# Qwen3 Paper-Alignment Postprocess Submission, 2026-05-15

This job refreshes the Qwen3-0.6B paper-alignment audit after the Stage-2
warmup, FP16 baselines, BitNet-SFT baselines, BitDistill rows, row-scale
variants, and MNLI attention-layer sweep jobs finish.

| field | value |
| --- | --- |
| job id | `10073` |
| dependency | `afterany:10040:10041:10042:10043:10044:10045:10046:10047:10048:10049:10050:10051:10052:10053:10054:10055` |
| partition | `dualcard` |
| script | `slurm_bitdistill_qwen3_postprocess.sh` |
| stdout | `logs/bitdistill-qwen3-post-10073.out` |
| stderr | `logs/bitdistill-qwen3-post-10073.err` |
| current state at submission | `PENDING (Dependency)` |

The dependency is `afterany`, so the audit will still produce a failure or
pending report if any producer job fails.
