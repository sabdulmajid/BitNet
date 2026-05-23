# Canonical Evidence Bundle

This bundle is manifest/artifact based. Missing artifacts are fatal while building it.

| claim | status | evidence | caveat |
| --- | --- | --- | --- |
| Blind PTQ | strong_negative_tested_setup | FP PPL 13.901; PTQ PPL 3,813,121.803; FP mean 0.644169; PTQ mean 0.348671 | Dense Qwen2.5-1.5B tested setup; not a theorem for every architecture. |
| QAT/distill | partial_recovery_not_fp | row-scale QAT mean 0.499459; recovery +0.150788; gap -0.144710 | Row-scale QAT is a retrofit variant, not standard BitDistill. |
| BitDistill | not_reproduced_327m_complete | MNLI 40.96M 0.616607; 163.84M 0.691187; 327.68M 0.720020, delta -0.088130 | The 327.68M row improves over 163.84M but remains below the FP16 recovery gate. |
| Row-scale runtime | strong_systems_result | TL2 one-scale RMS 1.904230; exact row-scale RMS 0.000197 | This supports I2_SR/row-scale contracts; TL2 row-scale support is not implemented. |
| I2_SR CPU | working_not_q4_quality_competitive | I2_SR PPL 38.8477, prompt 211.67, decode 19.07 | Does not beat Q4_K_M on quality or file size. |
| Native classifier | research_demo_not_product_ready | MNLI 0.652165; agreement 0.976668; 7.456204 ex/s | Agreement remains below the 0.99 product gate. |
| MoE/Kimi | not_supported | No trained Kimi/MoE quality or CPU runtime evidence. | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. |

## Artifact Inventory

| label | path | sha256 |
| --- | --- | --- |
| controlled_curve | benchmarks/results/bitdistill_controlled_curve_2026-05-20.json | d892817f387ff52f1105439e6b7a7c2417d7c0124b0d0dbf56adb3f7585f6356 |
| cpu_frontier | benchmark_results/cpu_tradeoff_frontier_2026-05-15.json | 792515abce6eff6c0521a6cddfb8243c7778d254fabfbcf793e9a3529f1045f2 |
| fp_lm_eval | benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json | e48cdc1bb44f1979512cef71f50b31b370692b11d08a4baa45dc7757ac71d6dd |
| fp_ppl | benchmark_results/quality-9735/qwen15b_fp_wikitext.json | acd79bab6f4020657f052232e19474fdd484b87885e09c810cac27c2a2392e58 |
| gamma60 | benchmark_results/bitdistill_gamma60_diagnostic_2026-05-15.json | 428f308e82e67593c9b48effb2db6e19893087df4d618d381d4063170efdd7f9 |
| native_seqcls | benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_sequence_isolated_2026-05-15.json | 1a78dc21dc0ccb0c3be0fca41ccb7c6228ccdd5117c5ef5f2dd82ea97d84d75a |
| ptq_lm_eval | benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json | 46e7097b707eb8b145579eed40eabd6a3707cd73dd6d92a5d5665788fe197804 |
| ptq_ppl | benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json | 8481ac1399014e94c23860ce8e9dbc2d9c50512634ac15d5f6886625141bab34 |
| row_qat_lm_eval | benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json | 50cda719e02af1824b041667ecdde0dbccd999a53d8aaa08cbe88e9af4553a5b |
| stage2_curve | benchmark_results/bitdistill_stage2_curve_2026-05-16.json | 9973aab406aaa32ae352b7a7da0d5972907f9ca7042f67c8620cb6af468faeb9 |
| stage2_manifest | benchmarks/results/stage2_manifest_2026-05-20.json | 4de6e5ac19a17bc53cac7eb33024964c069a13a24ba36afaec8632ba14f79bdf |
| tl2_contract | benchmark_results/tl2_row_scale_runtime_contract_2026-05-15.json | 4e631065cf2ef7df7eeca3dbb911516abed7360e678c75fb1a6c6062ed3506f5 |
