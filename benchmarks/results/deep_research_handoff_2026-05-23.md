# Deep Research Handoff

Generated: `2026-09-04T03:53:38.058573+00:00`

Status: **handoff_not_completion**.

## Thesis

- Original question: Can arbitrary pretrained FP16/BF16 models be post-hoc converted to BitNet-style W1.58A8 CPU inference?
- Current answer: No for the tested dense-Qwen setup; blind ternary PTQ collapses.
- Redirected question: Can task-specific ternary students be trained from pretrained teachers, and can CPU formats preserve the scale semantics those students learn?
- Core interpretation: Extreme ternary quantization is representation learning plus a runtime contract, not only compression or file conversion.

## Completed Findings

| finding | evidence | interpretation |
| --- | --- | --- |
| Blind ternary PTQ is a strong negative result in the tested dense-Qwen setup. | FP PPL 13.901 vs naive PTQ PPL 3813121.803; FP ten-task mean 0.644169 vs PTQ 0.348671. | The FP weight geometry is not preserved by a blind ternary projection. |
| QAT/distillation recovers signal but not FP quality. | Best row-scale QAT mean 0.499459; recovery over PTQ +0.150788; gap to FP -0.144710. | Training can move some function into the ternary family, but current runs do not close the gap. |
| BitDistill paper-level recovery remains governed by the latest completed Stage-2 row. | FP16-SFT MNLI 0.808151; latest 655.36M BitDistill 0.729903; delta -0.078248; status not_reproduced. | The 655.36M row is complete; its marginal gain over 327.68M is +0.009883, so loss balance is the next controlled variable. |
| The earlier weak BitNet-SFT baseline was mostly undertraining, not the main blocker. | default BitNet-SFT 0.487621; best budget row 0.628935; delta vs paper anchor +0.020935. | The remaining problem is BitDistill recovery/loss dynamics, not merely BitLinear replacement. |
| Row-scale semantics are material to the learned function. | TL2 one-scale output RMS error 1.904230; exact row-scale RMS error 0.000197. | A row-scale ternary student represents W approximately as s_row times T, so scales are model semantics. |
| I2_SR is a working row-scale packed CPU path but not a Q4 replacement. | I2_SR file 1211.3 MiB, PPL 38.8477, prompt 211.67 tok/s, decode 19.07 tok/s; Q4_K_M PPL 12.8112, file 940.4 MiB. | The systems path is real, but quality/storage tradeoffs remain unfavorable versus mature Q4. |

## Novelty Boundary

| classification | description |
| --- | --- |
| Not novel | BitDistill as a concept: SubLN, continued pretraining, logits KD, and attention-relation KD are Microsoft paper contributions. |
| Potentially novel | Independent reproduction-gap study with fail-closed artifacts and paired evidence for where local BitDistill diverges. |
| Potentially novel | Row-scale ternary retrofit variant and the measured requirement that runtime formats preserve row-scale semantics. |
| Potentially novel | I2_SR packed CPU runtime extension for compatible row-scale causal artifacts. |
| Potentially novel | Boundary study separating task quality, LM perplexity, file size, RSS, prompt speed, and decode speed. |

## Completed 655M Gate

| field | value |
| --- | --- |
| afterany_dependency | afterany:10250 |
| afterany_job_id | 10258 |
| afterany_status | historical_audit_failed_later_watchdog_passed |
| downstream_complete | True |
| downstream_status | complete_artifacts_present |
| eta_hours | 0.000000 |
| latest_ce | 3.426713 |
| latest_complete_snapshot_step | 40000 |
| latest_step | 40000 |
| log_health_status | healthy |
| max_steps | 40000 |
| next_snapshot_eta_hours | - |
| next_snapshot_step | - |
| producer_config_status | matched |
| progress | 1.000000 |
| snapshot_salvage_complete_count | 4 |
| snapshot_salvage_status | final_snapshot_available |
| stage2_job_id | 10250 |
| stage2_slurm_state | not_in_squeue |
| stage2_status | complete_artifacts_present |
| steps_to_next_snapshot | - |
| telemetry_job_id | 10257 |
| telemetry_slurm_state | not_in_squeue |
| time_limit_margin_seconds | - |
| time_limit_status | not_running |

## Next Action Policy

| field | value |
| --- | --- |
| decision_status | run_gamma_balanced_downstream |
| recommendation | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. |
| blueprint_action | run_matched_gamma60_mnli_downstream |
| runnable_now | True |
| claim_boundary | single MNLI ablation; do not broaden to QNLI/SST2 until paired MNLI result is ingested |
| required_evidence | next-decision status run_gamma_balanced_downstream, stage2_manifest_655m_2026-05-23.json exists and validates, gamma60_gradient_balance status indicates rebalanced updates |
| commands | MODEL=Qwen/Qwen2.5-0.5B \
        STAGE=task_sft \
        METHOD=bitdistill \
        TASK_NAME=mnli \
        TASK_FORMAT=sequence_classification \
        LABEL_SCHEME=letters \
        CANDIDATE_SCORE=mean \
        TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 \
        INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_655m_2026-05-23.json \
        SCALE_MODE=tensor \
        EXCLUDE_LINEAR_REGEX='score\|classifier' \
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
        sbatch --partition=midcard --job-name=bd-mnli-655m-g60 slurm_bitdistill_glue.sh |

## Open Research Questions

| question | evidence needed | current state |
| --- | --- | --- |
| Did doubling Stage-2 from 327.68M to 655.36M close the MNLI gap? | Completed 655M paired prediction trace against the fixed FP16 reference. | Answered: gain +0.009883; latest MNLI 0.729903; delta vs FP16 -0.078248. |
| Is the remaining BitDistill gap mostly compute budget or loss-normalization mismatch? | Matched 10k-step gamma-60 and paper-gamma MNLI runs from the same 655M checkpoint. | paper-gamma grad attention/CE 221.384986; gamma-60 0.346044; quality ablation not yet run. |
| Can the same artifact provide both quality and CPU runtime evidence? | Packed classifier or causal prompt-scoring artifact with task quality, RSS, file size, and throughput. | native classifier MNLI 0.652165, agreement 0.976668; not product-ready. |
| Do row-scale variants help generally or only in specific retrofit regimes? | Controlled tensor/row/group-scale comparisons across tasks/backbones with paired confidence intervals. | Row-scale runtime contract is strong; row-scale accuracy is not a universal guarantee. |
| Is MoE/Kimi feasible in this runtime path? | Real routed model mapping, expert layout, trained quality, and CPU expert-selection benchmarks. | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. |

## Nonclaims

- universal BitNet converter
- paper-level BitDistill reproduction
- Q4-quality I2_SR replacement
- Kimi/MoE runtime support

## Publishable Angles

- negative blind-ternary-PTQ result for tested dense Qwen models
- independent BitDistill reproduction-gap study
- row-scale ternary runtime-contract evidence
- I2_SR packed CPU row-scale extension for compatible causal artifacts
- boundary study separating task quality, LM perplexity, RSS, file size, and throughput

## Source Artifacts

| artifact | path | sha256 |
| --- | --- | --- |
| current_status | benchmarks/results/current_goal_status_2026-05-23.json | 0a53d5c46e041e4ad5b8876ddf72a51772793999784522f58309d99646396766 |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json | af9ec2e35931986c7caf63c178b7c482c3e93406f8d880774bbf8d114f27824c |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json | e93c4db2b5363d9999c0bab1a6e637526eca3e62be9541f1f99ad1291174cd10 |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json | 039f2ebb6d36d85073f7433fed5a21b35e966af60c646e4fc3d65cef60cbb071 |
| next_experiment_blueprint | benchmarks/results/bitdistill_next_experiment_blueprint_2026-05-23.json | e0ca9ab3f1a637ba0a89e6ca6cae6e1b5a7f3c0585c30643f8f5673642731955 |
