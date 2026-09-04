# Stage-2 655.36M Handoff

Status: **submitted_downstream**.

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| handoff_job_id | 10259 |
| downstream_job_id | 10260 |
| postprocess_job_id | 10261 |
| manifest_json | benchmarks/results/stage2_manifest_655m_2026-05-23.json |
| downstream_output_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit |
| postprocess_json | benchmarks/results/stage2_655m_postprocess_2026-05-23.json |

The downstream directory now contains both metrics.json and
eval_predictions.jsonl; the ingestion audit is the authoritative completion
receipt.
