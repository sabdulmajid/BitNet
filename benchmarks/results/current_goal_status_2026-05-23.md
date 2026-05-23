# Current Goal Status

Generated: `2026-05-23T17:33:31.687324+00:00`

Git HEAD: `39abad5b3da13d287bc43d408419a3d7b624aaa1`

Objective achieved: **False**.

Completion status: **in_progress**.

## Verdict

Blind ternary PTQ is rejected for the tested dense-Qwen setup. BitDistill-style recovery status is not_reproduced; the active 655.36M Stage-2 gate is testing whether recovery continues with more tokens.

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
| bitdistill_latest_stage2_tokens | 327680000 |
| bitdistill_latest_mnli | 0.720020 |
| bitdistill_latest_delta_vs_fp | -0.088130 |
| bitdistill_655_36m_status | waiting_for_handoff |

## Requirement Audit

| requirement | status | evidence | remaining gap |
| --- | --- | --- | --- |
| Arbitrary FP/BF16 to ternary retrofit | rejected_for_tested_dense_qwen_setup | FP WikiText PPL 13.901; naive PTQ PPL 3813121.803; FP ten-task mean 0.644169; PTQ mean 0.348671 | Do not market as a universal converter. |
| BitDistill paper-level MNLI recovery | not_reproduced | FP16-SFT 0.808151; latest 327.68M BitDistill 0.720020; delta -0.088130 | 655.36M downstream MNLI is pending behind the active Stage-2 producer. |
| BitNet-SFT baseline sanity | locally_sanity_checked | default 0.487621; best budget row 0.628935; delta vs paper anchor +0.020935 | This does not reproduce BitDistill recovery. |
| Row-scale runtime contract | supported | one-scale TL2 RMS error 1.904230; exact row-scale RMS error 0.000197 | TL2 row-scale kernels are not implemented; I2_SR is the supported row-scale path. |
| Packed CPU I2_SR path | working_not_q4_quality_competitive | I2_SR file 1211.3 MiB; PPL 38.8477; prompt 211.67 tok/s; decode 19.07 tok/s | Not quality/storage competitive with Q4_K_M. |
| Native packed classifier product | research_demo_not_product_ready | MNLI accuracy 0.652165; PyTorch agreement 0.976668; RSS 960.15 MiB | Agreement and task quality remain below product gates. |
| MoE/Kimi support | not_supported | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. | Needs real routed model mapping, quality, and CPU runtime evidence. |

## Active 655M Gate

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| stage2_status | running |
| stage2_slurm_state | RUNNING |
| latest_step | 3800 |
| max_steps | 40000 |
| progress | 0.095000 |
| latest_ce | 3.883600 |
| eta_hours | 18.277295 |
| latest_complete_snapshot_step | - |
| downstream_status | waiting_for_handoff |
| downstream_complete | False |
| telemetry_job_id | 10257 |
| telemetry_slurm_state | PENDING |

## Next Gates

| gate | minimum next point | why |
| --- | --- | --- |
| Stage-2 token-budget curve | 655.36M cumulative token presentations with the same downstream recipe | Determine whether MNLI continues improving toward FP or saturates far below it. |
| Loss-normalization/gradient-balance sweep | component-gradient telemetry for gamma near equalized and paper values | Paper gamma is only comparable if CE, logits KD, and attention KD reductions match. |
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
| reproduction_gap | benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json | b5a37266b33dc7318b55a23569673467d11fa7aa67ba6725baaa374210a42820 |
| active_monitor | benchmarks/results/active_stage2_extension_monitor_2026-05-23.json | a26c55cab237e73e62f2da6f9865fac090a903c0e4360d79fd97d0d868b2ad86 |
