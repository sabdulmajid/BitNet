# Stage-2 Checkpoint Manifest: qwen25-05b-bitdistill-tensor-stage2-655m-from327m-job10250

This manifest pins the exact warm-up checkpoint consumed by downstream BitDistill jobs.

| field | value |
| --- | --- |
| job_id | 10250 |
| model | Qwen/Qwen2.5-0.5B |
| method | bitdistill |
| scale_mode | tensor |
| steps | 40000 |
| token_presentations | 655360000 |
| segment_token_presentations | 327680000 |
| parent_token_presentations | 327680000 |
| final_ce | 3.426712989807129 |
| state_dict_path | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-655m-from327m/checkpoint-40000/custom_state_dict.pt |
| root_metrics_source | root_metrics |
| parent_manifest_path | benchmarks/results/stage2_manifest_2026-05-20.json |
| bitnet_commit | 10341701e5104c66d18fc9779ab9799bf2190c9a |
| llama_cpp_commit | dc0bc5ee0423a2202d6284a4fc2d78d1e39905d7 |
| producer_bitnet_commit_note | inferred from the commit that captured the LR_SCHEDULER wrapper patch used by Stage-2 job 10250; the job log confirms LR_SCHEDULER=constant |
| producer_llama_cpp_commit_note | recorded in the Stage-2 submission report for job 10250 |
| downstream_status | submitted_downstream |
| downstream_rerun_job_id | 10260 |
| downstream_rerun_output_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit |

## Downstream Note

No downstream failure mode is recorded for this manifest. Downstream quality claims still require materialized metrics and prediction traces.
