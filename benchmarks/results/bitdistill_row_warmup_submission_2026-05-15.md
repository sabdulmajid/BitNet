# BitDistill Row-Scale Warm-Up Submission, 2026-05-15

Purpose: start a clean row-scale Stage-2 continued-pretraining branch so the row-scale BitDistill claim is not conflated with row-scale Stage-3 adaptation initialized from tensor-scale warm-up.

This is an exploratory/novelty branch. The existing 38-row strict paper-alignment matrix remains dependency-linked to tensor warm-up job `9894`.

## Submitted Job

| field | value |
| --- | --- |
| Slurm job | `10028` |
| partition | `midcard` |
| node | `ece-nebula12` |
| state at submission audit | `RUNNING` |
| output log | `logs/bitdistill-glue-10028.out` |
| error log | `logs/bitdistill-glue-10028.err` |
| output directory | `checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k` |

## Training Contract

| field | value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B` |
| stage | `continued_pretrain` |
| method | `bitdistill` |
| scale mode | `row` |
| max steps | `20000` |
| max sequence length | `512` |
| per-device batch size | `4` |
| gradient accumulation | `4` |
| effective token presentations | `163840000` |
| learning rate | `2e-5` |
| warmup steps | `100` |
| min LR ratio | `0.1` |
| dataset | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train` |
| training rows requested | `20000` |
| packed blocks requested | `20000` |
| checkpoint snapshots | every `1000` steps |

## Provenance Checks

| check | status | evidence |
| --- | --- | --- |
| stored Slurm script equals current launcher | pass | `dd5ea8ef8474` for both `slurm_bitdistill_glue.sh` and stored job script |
| current launcher has Stage-2 snapshot guard | pass | refuses `continued_pretrain` with `SAVE_EVERY_STEPS=0` unless explicitly overridden |
| job log confirms row-scale settings | pass | `SCALE_MODE=row`, `SAVE_EVERY_STEPS=1000`, `MAX_SEQ_LEN=512`, `MAX_STEPS=20000` |

## Interpretation

- This branch is required for a rigorous row-scale novelty claim.
- The currently queued row-scale downstream jobs are still useful, but they initialize from the tensor-scale Stage-2 warm-up. They test row-scale Stage-3/export adaptation, not a fully row-scale BitDistill pipeline.
- A publishable row-scale comparison should report both:
  - tensor Stage-2 -> row Stage-3, for compatibility/adaptation value;
  - row Stage-2 -> row Stage-3, for the clean row-scale method claim.

## Next Gates

| gate | required evidence |
| --- | --- |
| row warm-up health | strictly increasing finite CE, fresh log, snapshots every 1000 steps |
| row final checkpoint | `custom_state_dict.pt`, `ternary_state_dict.pt`, `metrics.json`, model/tokenizer artifacts |
| row downstream GLUE3 | MNLI/QNLI/SST2 sequence-classification metrics initialized from `bitdistill-row-20k` |
| row I2_SR export | valid row-scale GGUF export through the stable `I2_SR` path |
| Xeon benchmark | task quality plus CPU throughput, memory, and RSS against tensor BitDistill and FP16/BitNet baselines |
