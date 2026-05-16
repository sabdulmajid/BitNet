# BitDistill Reproduction Plan, 2026-05-16

Plan mode: `canonical_mnli_first`.

Model: `Qwen/Qwen2.5-0.5B`.

Primary task: `mnli`.

Included tasks: `mnli`.

Task format: `sequence_classification`.

Success criterion: BitDistill within 0.5-1.0 accuracy point (0.005-0.010 absolute accuracy) of FP16-SFT on the primary task before expanding axes.

Ordering constraint: Run FP16-SFT for each included task; those checkpoints become task teachers for BitDistill.

Warmup constraint: Run continued pretraining first and pass `checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor/custom_state_dict.pt` to every BitDistill task run.

Logits-KD constraint: Use paper-style logits KL with `LOGIT_KD_TEMPERATURE_SCALE=none`; the tau-squared convention is available only as an explicit diagnostic.

Expansion rule: Do not add QNLI/SST2, row-scale novelty, or attention-layer sweeps until the MNLI tensor-scale BitDistill gate is interpretable.

This matrix keeps paper reproduction separate from this fork's novelty claim. The default plan uses tensor-scale BitDistill on MNLI first. Secondary tasks, row-scale novelty, and attention-layer sweeps require explicit opt-in flags.

| # | phase | task | method | scale | layer | command |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | stage2_continued_pretraining | - | bitdistill | tensor | - | `MODEL=Qwen/Qwen2.5-0.5B STAGE=continued_pretrain METHOD=bitdistill SCALE_MODE=tensor MAX_STEPS=5000 SAVE_EVERY_STEPS=1000 OUTPUT_DIR=checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor sbatch slurm_bitdistill_glue.sh` |
| 2 | paper_baseline | mnli | fp16_sft | tensor | -1 | `MODEL=Qwen/Qwen2.5-0.5B TASK_FORMAT=sequence_classification LABEL_SCHEME=letters CANDIDATE_SCORE=mean TASK_NAME=mnli METHOD=fp16_sft SCALE_MODE=tensor DISTILL_LAYER=-1 MAX_STEPS=10000 sbatch slurm_bitdistill_glue.sh` |
| 3 | paper_baseline | mnli | bitnet_sft | tensor | -1 | `MODEL=Qwen/Qwen2.5-0.5B TASK_FORMAT=sequence_classification LABEL_SCHEME=letters CANDIDATE_SCORE=mean TASK_NAME=mnli METHOD=bitnet_sft SCALE_MODE=tensor DISTILL_LAYER=-1 MAX_STEPS=10000 sbatch slurm_bitdistill_glue.sh` |
| 4 | paper_baseline | mnli | bitdistill | tensor | -1 | `MODEL=Qwen/Qwen2.5-0.5B TASK_FORMAT=sequence_classification LABEL_SCHEME=letters CANDIDATE_SCORE=mean TASK_NAME=mnli METHOD=bitdistill SCALE_MODE=tensor DISTILL_LAYER=-1 MAX_STEPS=10000 TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 INIT_STATE_DICT=checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor/custom_state_dict.pt LOGIT_KD_TEMPERATURE_SCALE=none sbatch slurm_bitdistill_glue.sh` |
