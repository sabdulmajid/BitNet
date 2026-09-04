# Stage-2 655.36M Ingestion Audit

Generated: `2026-09-04T03:53:18.842782+00:00`

Status: **ingested_reports_rebuilt**.

Quality claim: **none_until_complete_downstream_trace**.

This report is an ingestion receipt. It does not create a quality claim; it verifies that quality artifacts are present before other reports may use them.

## Slurm State

| job | id | state | time | reason |
| --- | --- | --- | --- | --- |
| stage2 | 10250 | not_in_squeue |  |  |
| handoff | 10259 | not_in_squeue |  |  |

## Downstream Artifacts

| artifact | exists | path/value |
| --- | --- | --- |
| metrics | true | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/metrics.json |
| predictions | true | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| metric_accuracy | true | 0.729903 |
| metric_eval_examples | true | 9815.000000 |
| prediction_rows | true | 9815 |
| paired_status | true | pass |
| paired_matched | true | 9815 |

## Report Ingestion

| item | status/value |
| --- | --- |
| postprocess_status | reports_rebuilt |
| controlled_target_row_exists | true |
| controlled_accuracy | 0.729903 |
| controlled_delta_vs_fp16 | -0.078248 |
| controlled_paired_status | pass |
| gap_latest_stage2_tokens | 655360000 |
| gap_latest_mnli | 0.729903 |
| next_decision_status | run_gamma_balanced_downstream |

## Consistency

| field | value |
| --- | --- |
| complete | true |
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
