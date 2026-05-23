# Stage-2 655.36M Ingestion Audit

Generated: `2026-05-23T18:33:30.155934+00:00`

Status: **pending_handoff**.

Quality claim: **none_until_complete_downstream_trace**.

This report is an ingestion receipt. It does not create a quality claim; it verifies that quality artifacts are present before other reports may use them.

## Slurm State

| job | id | state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | RUNNING | 2:56:05 | ece-nebula12 |
| handoff | 10255 | PENDING | 0:00 | (Dependency) |

## Downstream Artifacts

| artifact | exists | path/value |
| --- | --- | --- |
| metrics | false | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json |
| predictions | false | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| metric_accuracy | false | - |
| metric_eval_examples | false | - |
| prediction_rows | false | 0 |
| paired_status | true | pending |
| paired_matched | true | 0 |

## Report Ingestion

| item | status/value |
| --- | --- |
| postprocess_status | - |
| controlled_target_row_exists | false |
| controlled_accuracy | - |
| controlled_delta_vs_fp16 | - |
| controlled_paired_status | - |
| gap_latest_stage2_tokens | - |
| gap_latest_mnli | - |
| next_decision_status | pending_655m_downstream |

## Consistency

| field | value |
| --- | --- |
| complete | false |
| consistency_errors | none |

## Source Artifacts

| artifact | path |
| --- | --- |
| stage2_submission | benchmarks/results/stage2_655m_submission_2026-05-23.json |
| handoff_submission | benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json |
| handoff_report | benchmarks/results/stage2_655m_handoff_2026-05-23.json |
| postprocess_report | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-23.json |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| reference_predictions | checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/eval_predictions.jsonl |
