# Stage-2 Checkpoint Manifest: qwen25-05b-bitdistill-tensor-stage2-40k-job10070

This manifest pins the exact warm-up checkpoint consumed by downstream BitDistill jobs.

| field | value |
| --- | --- |
| job_id | 10070 |
| model | Qwen/Qwen2.5-0.5B |
| method | bitdistill |
| scale_mode | tensor |
| steps | 40000 |
| token_presentations | 327680000 |
| final_ce | 3.7840569019317627 |
| state_dict_path | checkpoints/bitdistill-glue-stage2-curve/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-40k/checkpoint-40000/custom_state_dict.pt |
| bitnet_commit | 6353f7e3e770618f2c03b053b0179bf486ef5fb4 |
| llama_cpp_commit | dc0bc5ee0423a2202d6284a4fc2d78d1e39905d7 |
| downstream_status | rerun_submitted |
| downstream_rerun_job_id | 10169 |
| downstream_rerun_output_dir | checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-40kwarmup-steps10000-lr2em5-papergamma-headinit-rerun |

## Downstream Note

Job 10071 failed before training/evaluation because it looked for a root-level `custom_state_dict.pt`. The valid state dict is the snapshot path recorded above.
