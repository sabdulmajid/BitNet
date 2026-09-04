# Current Goal Status

Generated: `2026-09-04T03:53:37.994031+00:00`

Git HEAD: `5ac81035fa995589c42c760d3a91816368144dae`

Objective achieved: **False**.

Completion status: **in_progress**.

## Verdict

Blind ternary PTQ is rejected for the tested dense-Qwen setup. BitDistill-style recovery status is not_reproduced; the completed 655.36M row reaches 0.729903 MNLI, -0.078248 versus FP16, so the next controlled gate is loss balance.

## Headline Metrics

| metric | value |
| --- | --- |
| blind_ptq_fp_ppl | 13.901475 |
| blind_ptq_naive_ppl | 3813121.803327 |
| qat_row_scale_ten_task_mean | 0.499459 |
| qat_recovery_vs_ptq | 0.150788 |
| qat_gap_vs_fp | -0.144710 |
| fp16_sft_mnli | 0.808151 |
| bitdistill_327_68m_mnli | 0.720020 |
| bitdistill_327_68m_delta_vs_fp | -0.088130 |
| bitdistill_latest_stage2_tokens | 655360000 |
| bitdistill_latest_mnli | 0.729903 |
| bitdistill_latest_delta_vs_fp | -0.078248 |
| bitdistill_latest_gain_vs_327_68m | 0.009883 |
| bitdistill_655_36m_status | complete_artifacts_present |

## Requirement Audit

| requirement | status | evidence | remaining gap |
| --- | --- | --- | --- |
| Arbitrary FP/BF16 to ternary retrofit | rejected_for_tested_dense_qwen_setup | FP WikiText PPL 13.901; naive PTQ PPL 3813121.803; FP ten-task mean 0.644169; PTQ mean 0.348671 | Do not market as a universal converter. |
| BitDistill paper-level MNLI recovery | not_reproduced | FP16-SFT 0.808151; latest 655.36M BitDistill 0.729903; delta -0.078248 | The completed 655.36M row remains 7.825 accuracy points below FP16. |
| BitNet-SFT baseline sanity | locally_sanity_checked | default 0.487621; best budget row 0.628935; delta vs paper anchor +0.020935 | This does not reproduce BitDistill recovery. |
| Row-scale runtime contract | supported | one-scale TL2 RMS error 1.904230; exact row-scale RMS error 0.000197 | TL2 row-scale kernels are not implemented; I2_SR is the supported row-scale path. |
| Packed CPU I2_SR path | working_not_q4_quality_competitive | I2_SR file 1211.3 MiB; PPL 38.8477; prompt 211.67 tok/s; decode 19.07 tok/s | Not quality/storage competitive with Q4_K_M. |
| Native packed classifier product | research_demo_not_product_ready | MNLI accuracy 0.652165; PyTorch agreement 0.976668; RSS 960.15 MiB | Agreement and task quality remain below product gates. |
| MoE/Kimi support | not_supported | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. | Needs real routed model mapping, quality, and CPU runtime evidence. |
| 655M evidence-chain guardrails | completed_with_paired_trace | producer_config matched; four complete snapshots; 655M manifest verified; downstream metrics and paired predictions complete | The evidence chain is complete; the quality gate itself failed. |

## Completed 655M Gate

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| stage2_status | complete_artifacts_present |
| stage2_slurm_state | not_in_squeue |
| latest_step | 40000 |
| max_steps | 40000 |
| progress | 1.000000 |
| latest_ce | 3.426713 |
| eta_hours | 0.000000 |
| time_limit_status | not_running |
| time_limit_margin_seconds | - |
| producer_config_status | matched |
| log_health_status | healthy |
| snapshot_salvage_status | final_snapshot_available |
| snapshot_salvage_complete_count | 4 |
| next_snapshot_step | - |
| steps_to_next_snapshot | - |
| next_snapshot_eta_hours | - |
| afterany_job_id | 10258 |
| afterany_status | historical_audit_failed_later_watchdog_passed |
| afterany_dependency | afterany:10250 |
| latest_complete_snapshot_step | 40000 |
| downstream_status | complete_artifacts_present |
| downstream_complete | True |
| telemetry_job_id | 10257 |
| telemetry_slurm_state | not_in_squeue |

## Next Gates

| gate | minimum next point | why |
| --- | --- | --- |
| Gamma-balanced downstream MNLI | matched 10k-step MNLI from the 655.36M checkpoint with attention-KD gamma 60 | The 327.68M to 655.36M gain is modest while paper-gamma attention gradients dominate CE under the local reductions. |
| Loss-normalization/gradient-balance sweep | full-quality comparison of gamma 60 versus paper gamma with all other axes fixed | Paper gamma is only comparable if CE, logits KD, and attention KD reductions match. |
| Same-artifact runtime quality | packed classifier head or primary causal prompt-scoring evaluation | The strongest PyTorch classifier result and strongest packed causal runtime are still separate artifacts. |
| Backbone alignment | one Qwen3-0.6B or exact Qwen2.5-0.5B MNLI run with matched logging | Paper-scale claims need exact/closest public Qwen3/Qwen2.5 recipe alignment. |

## Publishable Scope

Not publishable as:
- universal BitNet converter
- paper-level BitDistill reproduction
- Q4-quality I2_SR replacement
- Kimi/MoE runtime support

Potentially publishable as:
- negative blind-ternary-PTQ result for tested dense Qwen models
- independent BitDistill reproduction-gap study
- row-scale ternary runtime-contract evidence
- I2_SR packed CPU row-scale extension for compatible causal artifacts
- boundary study separating task quality, LM perplexity, RSS, file size, and throughput

## Inputs

| artifact | path | sha256 |
| --- | --- | --- |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json | af9ec2e35931986c7caf63c178b7c482c3e93406f8d880774bbf8d114f27824c |
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json | e93c4db2b5363d9999c0bab1a6e637526eca3e62be9541f1f99ad1291174cd10 |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json | d77c22b665d1a0bef087476eb7d2518ea305276f01852c15939fe702105c0c30 |
| snapshot_salvage | benchmarks/results/stage2_snapshot_salvage_2026-05-23.json | 149e130bde1a2a2225c040f5b64d9f580b61a1fcd7a2b159ab5bf2657470d33a |
| afterany_submission | benchmarks/results/stage2_655m_afterany_submission_2026-05-23.json | 807495d10307b7d9a8bb5e69196a3b0cc7c12e69a415cce65a868df43156cd77 |
