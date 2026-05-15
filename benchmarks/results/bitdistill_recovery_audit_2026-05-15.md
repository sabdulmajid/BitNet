# BitDistill Recovery Run Audit, 2026-05-15

Pending: metrics or prediction traces are not available yet.

## Run

| field | value |
| --- | --- |
| candidate_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit |
| metrics_exists | false |
| predictions_exists | false |
| steps | - |
| metric_accuracy | - |
| metric_eval_examples | - |
| success_delta_from_fp | -0.010000 |

## Paired FP16 Comparison

| metric | value |
| --- | --- |
| status | pending |
| matched | 0 |
| reference_accuracy | - |
| candidate_accuracy | - |
| delta_vs_reference | - |
| paired_ci95 | - |
| candidate_wins | 0 |
| reference_wins | 0 |
| mcnemar_exact_p | - |
| errors | missing checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |
| missing | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit/eval_predictions.jsonl |

## Loss Components

| component | value |
| --- | --- |
| ce | - |
| logit_kd | - |
| weighted_logit_kd | - |
| logit_to_ce | - |
| attention_kd | - |
| weighted_attention_kd | - |
| attention_to_ce | - |

## Interpretation

This audit uses the local FP16-SFT prediction trace as the reference. It does not judge paper reproduction until the candidate has full MNLI predictions, a paired confidence interval, and the configured delta gate is satisfied.

