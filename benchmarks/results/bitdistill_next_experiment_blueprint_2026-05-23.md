# BitDistill Next Experiment Blueprint

Generated: `2026-09-04T03:53:37.930806+00:00`

Status: **run_gamma_balanced_downstream**.

Quality claim: **experiment_blueprint_not_benchmark**.

## Current Recommendation

The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens.

## Current Action

| field | value |
| --- | --- |
| action | run_matched_gamma60_mnli_downstream |
| runnable now | true |
| why | The completed 655M row has weak marginal gain and gamma60 telemetry shows attention-KD updates are rebalanced, so the matched one-axis MNLI ablation is ready. |
| claim boundary | single MNLI ablation; do not broaden to QNLI/SST2 until paired MNLI result is ingested |

## Evidence Required

| required evidence |
| --- |
| next-decision status run_gamma_balanced_downstream |
| stage2_manifest_655m_2026-05-23.json exists and validates |
| gamma60_gradient_balance status indicates rebalanced updates |

## Commands

```bash
MODEL=Qwen/Qwen2.5-0.5B \
        STAGE=task_sft \
        METHOD=bitdistill \
        TASK_NAME=mnli \
        TASK_FORMAT=sequence_classification \
        LABEL_SCHEME=letters \
        CANDIDATE_SCORE=mean \
        TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
        INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_655m_2026-05-23.json \
        SCALE_MODE=tensor \
        EXCLUDE_LINEAR_REGEX='score|classifier' \
        DISTILL_LAYER=-1 \
        ATTENTION_SPLIT_HEADS=8 \
        ACTIVATION_QUANTIZATION=1 \
        USE_SUBLN=1 \
        LOGIT_KD_WEIGHT=10 \
        ATTENTION_KD_WEIGHT=60 \
        LOGIT_TEMPERATURE=5.0 \
        LOGIT_KD_TEMPERATURE_SCALE=none \
        ATTENTION_TEMPERATURE=1.0 \
        INIT_OUTPUT_HEAD_FROM_TEACHER=1 \
        MAX_SEQ_LEN=512 \
        MAX_STEPS=10000 \
        PER_DEVICE_BATCH_SIZE=4 \
        GRAD_ACCUM_STEPS=4 \
        LR=2e-5 \
        LR_SCHEDULER=cosine \
        SAVE_EVERY_STEPS=0 \
        SAVE_MODEL_ARTIFACTS=0 \
        OUTPUT_DIR=checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-gamma60-headinit \
        sbatch --partition=midcard --job-name=bd-mnli-655m-g60 slurm_bitdistill_glue.sh
```

## Action Catalog

| decision status | action | runnable now | claim boundary |
| --- | --- | --- | --- |
| pending_no_controlled_rows | materialize_controlled_row | false | status repair only; no quality claim |
| pending_655m_downstream | wait_and_watch_655m_gate | true | status only; quality_claim remains none until ingestion is ingested_reports_rebuilt |
| hold_for_gamma_balance | wait_for_gamma60_telemetry | true | diagnostic only; gamma60 telemetry is not a quality benchmark |
| run_gamma_balanced_downstream | run_matched_gamma60_mnli_downstream | true | single MNLI ablation; do not broaden to QNLI/SST2 until paired MNLI result is ingested |
| extend_stage2_curve | prepare_next_controlled_stage2_point | false | budget-curve extension only; keep recipe fixed and do not add new task axes |
| replicate_recovery_gate | replicate_passing_mnli_then_expand_glue | false | reproducibility gate; QNLI/SST2 remain gated behind replicated MNLI |
| pause_broad_stage2_audit_recipe | stop_broad_scaling_and_audit_recipe | true | root-cause audit only; do not submit larger Stage-2 runs before resolving recipe mismatch |
| ambiguous_recovery_continue_with_controls | choose_one_narrow_ablation | false | one-axis ablation only |

## Nonclaims

| nonclaim |
| --- |
| This report does not add benchmark evidence. |
| A runnable command is not permission to update quality claims. |
| Broad sweeps remain disallowed until the 655M gate is ingested. |

## Source Paths

| artifact | path |
| --- | --- |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| stage2_ingestion | benchmarks/results/stage2_655m_ingestion_2026-05-23.json |
| gamma_balance | benchmarks/results/gamma60_gradient_balance_2026-05-23.json |

This blueprint is decision support. It should be regenerated after the 655M downstream row and gamma telemetry complete.
