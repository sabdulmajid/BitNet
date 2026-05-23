# Active Slurm Batch Script Audit

Status: **passed**.

Quality claim: **none**. This validates queued script contents only.

| purpose | job | state | script available | passed |
| --- | --- | --- | --- | --- |
| 655M Stage-2 handoff | 10255 | PENDING | true | true |
| 655M Stage-2 postprocess script | local | local_file | true | true |
| gamma-60 gradient telemetry | 10257 | PENDING | true | true |
| 655M Stage-2 afterany audit | 10258 | PENDING | true | true |

## Required Snippets

| job | snippet | present |
| --- | --- | --- |
| 10255 | write_failure_report() | true |
| 10255 | trap 'status=$?; trap - ERR; write_failure_report | true |
| 10255 | --downstream-failed-job-id "" | true |
| 10255 | --downstream-failure-mode "" | true |
| 10255 | slurm_stage2_655m_postprocess.sh | true |
| 10255 | POSTPROCESS_JOB_ID | true |
| 10255 | INIT_STATE_MANIFEST="$MANIFEST_JSON" | true |
| 10255 | SCALE_MODE=tensor | true |
| 10255 | TASK_NAME=mnli | true |
| 10255 | TASK_FORMAT=sequence_classification | true |
| 10255 | LABEL_SCHEME=letters | true |
| 10255 | CANDIDATE_SCORE=mean | true |
| 10255 | TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1 | true |
| 10255 | ATTENTION_KD_WEIGHT=100000 | true |
| 10255 | LOGIT_KD_WEIGHT=10 | true |
| 10255 | LOGIT_TEMPERATURE=5.0 | true |
| 10255 | LOGIT_KD_TEMPERATURE_SCALE=none | true |
| 10255 | ATTENTION_TEMPERATURE=1.0 | true |
| 10255 | INIT_OUTPUT_HEAD_FROM_TEACHER=1 | true |
| 10255 | MAX_SEQ_LEN=512 | true |
| 10255 | MAX_STEPS=10000 | true |
| 10255 | PER_DEVICE_BATCH_SIZE=4 | true |
| 10255 | GRAD_ACCUM_STEPS=4 | true |
| 10255 | LR=2e-5 | true |
| 10255 | LR_SCHEDULER=cosine | true |
| 10255 | SAVE_MODEL_ARTIFACTS=0 | true |
| 10255 | OUTPUT_DIR="$DOWNSTREAM_OUTPUT_DIR" | true |
| 10255 | sbatch --parsable --partition=midcard --job-name=bd-mnli-655m slurm_bitdistill_glue.sh | true |
| local | build_bitdistill_next_decision.py | true |
| local | DECISION_JSON | true |
| local | DECISION_MD | true |
| local | INGESTION_JSON | true |
| local | audit_stage2_655m_ingestion.py | true |
| local | validate_reports_fail_closed.py | true |
| 10257 | write_status_report() | true |
| 10257 | export ATTENTION_KD_WEIGHT=60 | true |
| 10257 | export MAX_STEPS=200 | true |
| 10257 | export TELEMETRY_EVERY_STEPS=25 | true |
| 10257 | export TELEMETRY_COMPONENT_GRAD_NORMS=1 | true |
| 10257 | audit_bitdistill_gamma_balance.py | true |
| 10257 | build_bitdistill_next_decision.py | true |
| 10257 | validate_reports_fail_closed.py | true |
| 10258 | audit_stage2_snapshot_salvage.py | true |
| 10258 | audit_stage2_655m_ingestion.py | true |
| 10258 | run_active_gate_watchdog.py | true |
| 10258 | bitnet-stage2-afterany-audit-v1 | true |
| 10258 | quality_claim | true |
| 10258 | This afterany audit refreshes postmortem/salvage status only | true |
| 10258 | exit "$EXIT_CODE" | true |

## Dependency Graph

| purpose | job | state | expected dependency | actual dependency | dependency matched | expected command | command matched | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 655M Stage-2 handoff dependency | 10255 | PENDING | afterok:10250 | afterok:10250 | true | slurm_stage2_655m_handoff.sh | true | true |
| gamma-60 telemetry dependency | 10257 | PENDING | afterok:10250 | afterok:10250 | true | slurm_gamma60_telemetry.sh | true | true |
| 655M Stage-2 afterany dependency | 10258 | PENDING | afterany:10250 | afterany:10250 | true | slurm_stage2_655m_afterany_audit.sh | true | true |
