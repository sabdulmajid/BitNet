# BitDistill Benchmark Scoreboard

Generated: `2026-09-04T03:43:39.988793+00:00`

Quality claim: **scoreboard_from_existing_artifacts_not_new_benchmark**.

Status: **mixed_supported_and_blocked**.

## Publishability Assessment

| field | value |
| --- | --- |
| publishable | true |
| framing | Publishable as an independent negative/positive boundary study and systems-contract prototype, not as a universal BitNet converter and not yet as a paper-level BitDistill reproduction. |
| strongest_contribution | Blind PTQ failure plus row-scale runtime-contract evidence showing that trained ternary scale semantics must be preserved in CPU formats such as I2_SR. |
| main_blocker | The same artifact still does not jointly satisfy paper-level task quality, general-LM quality, mature Q4-level storage/quality tradeoffs, and product-ready packed runtime. |

## Coverage

| field | value |
| --- | --- |
| quality_benchmark_count | 12 |
| lm_eval_task_count | 10 |
| model_families | FP, QAT KL-only, QAT KL-only dense lm_head, QAT KL-only row dense lm_head, QAT hidden-MSE, naive PTQ |
| sample_counts | FP=22382, QAT KL-only=22382, QAT KL-only dense lm_head=22382, QAT KL-only row dense lm_head=22382, QAT hidden-MSE=22382, naive PTQ=22382 |
| coverage_gate_passed | true |
| coverage_check_count | 108 |
| coverage_failed | none |

Benchmarks covered: WikiText perplexity, FineWeb heldout perplexity, arc_challenge, arc_easy, hellaswag, piqa, winogrande, boolq, copa, openbookqa, sciq, truthfulqa_mc1.

## Headline Scoreboard

| area | status | evidence | impact | limitation |
| --- | --- | --- | --- | --- |
| Blind ternary PTQ | rejected_for_tested_dense_qwen | FP WikiText PPL 13.901475; naive PTQ PPL 3,813,121.803327; FP ten-task mean 0.644169; PTQ mean 0.348671 | The universal one-click arbitrary FP/BF16-to-ternary retrofit claim is not supported. | Dense Qwen2.5-1.5B tested setup; not a theorem for every architecture. |
| QAT/distillation recovery | partial_recovery_not_fp | best row-scale QAT mean 0.499459; recovery vs PTQ +0.150788; gap vs FP -0.144710 | Training under ternary constraints recovers real signal, but not FP quality. | Row-scale QAT is a retrofit variant, not standard BitDistill. |
| BitDistill reproduction | not_reproduced_327m_complete | MNLI 40.96M 0.616607; 163.84M 0.691187; 327.68M 0.720020; 655.36M 0.729903; latest delta vs FP -0.078248 | Paper-level BitDistill quality is not reproduced; the 655M marginal gain is modest. | The 327.68M row improves over 163.84M but remains below the FP16 recovery gate. |
| Loss normalization / gamma | local_loss_normalization_mismatch | gamma-60 MNLI 0.738462; delta vs FP -0.069689 | The attention-KD coefficient cannot be interpreted without matching loss reductions. | This is a local normalization diagnostic, not a claim that the paper coefficient is wrong. |
| Row-scale runtime contract | strong_systems_result | one-scale TL2 RMS error 1.904230; exact row-scale RMS error 0.000197 | Row scales are model semantics, not optional metadata. | This supports I2_SR/row-scale contracts; TL2 row-scale support is not implemented. |
| Packed CPU I2_SR | working_not_q4_quality_competitive | I2_SR file 1211.3 MiB, PPL 38.8477, prompt 211.67 tok/s, decode 19.07 tok/s; Q4_K_M file 940.4 MiB, PPL 12.8112 | Dense row-scale ternary CPU execution works; audited Q4 comparison has 1.191x decode and 2.299x prefill speedups for I2_SR. | I2_SR is a speed-oriented proof of row-scale ternary runtime semantics. It improves decode speed versus FP16 and is faster than Q4_K_M in the audited run, but it is larger than Q4_K_M and has much worse PPL. It should not be claimed as a quality/storage win over mature Q4 quantization. |
| Native sequence classification | research_demo_not_product_ready | MNLI 0.652165; PyTorch agreement 0.976668; sequence-isolated 7.456204 ex/s; token-id runner 2.724140 ex/s | Native packed classifier plumbing exists as a research demo. | Agreement remains below the 0.99 product gate. |
| MoE / Kimi | not_supported | local Kimi artifacts 0; MoE product gates passed 6/9 | Synthetic Qwen2MoE plumbing is useful, but Kimi is future work. | Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven. |
| Benchmark coverage | passed | 12 quality benchmarks; 108 coverage checks; failed checks 0 | The current negative and partial-recovery claims are backed by broad audited coverage. | The broad coverage predates 655M, but the paired 655M MNLI row is included in the controlled curve. |
| Product scope | research_mvp_only | supported claims 5; unsupported claims 4 | CPU-first dense-Qwen retrofit evaluator with stable I2_SR runtime support; keep BitDistill quality claims behind the full GLUE reproduction gate. | This is a research MVP, not a universal converter product. |
| Active next decision | run_gamma_balanced_downstream | latest controlled row 0.729903; latest tokens 655,360,000; gamma status gamma60_rebalanced_attention_updates | The Stage-2 marginal gain is weak and gamma-60 rebalances updates. Run a matched 10k-step downstream MNLI row with the balanced coefficient before spending more Stage-2 tokens. | The next result must be a matched quality run; the 200-step gamma-60 trace is diagnostic only. |

## Novelty

- A fail-closed evidence stack that separates blind PTQ, QAT/distillation, GLUE quality, general-LM perplexity, packed runtime, and MoE plumbing.
- Empirical evidence that row-scale ternary semantics materially affect output fidelity and require a matching CPU runtime contract.
- An I2_SR row-scale packed-runtime path for compatible dense causal artifacts.
- A bounded product direction: a CPU-first ternary retrofit evaluator, not a universal converter.

## Nonclaims

- No claim that arbitrary FP16/BF16 models can be converted losslessly to BitNet.
- No claim that paper-level BitDistill has been reproduced.
- No claim that I2_SR beats Q4_K_M on quality or file size.
- No claim that current causal exports are useful general-purpose LLMs.
- No claim that native packed sequence classification is product-ready.
- No claim that Kimi or real MoE CPU ternary quality is supported.

## Next Steps

- Run the matched 10k-step gamma-60 MNLI ablation from the verified 655.36M checkpoint.
- Compare the gamma-60 paired prediction trace directly against FP16 and the paper-gamma 655M row.
- Only continue Stage-2 scaling if the loss-balanced result fails and a pre-registered budget hypothesis remains justified.
- Do not expand to QNLI/SST2 until MNLI reaches and replicates the within-1-point recovery gate.
- Keep MoE/Kimi work outside the main claim path until dense quality/runtime evidence is resolved.

## Source Artifacts

| artifact | path |
| --- | --- |
| canonical_bundle | benchmarks/results/canonical_evidence_bundle_2026-05-20.json |
| benchmark_matrix | benchmark_results/benchmark_matrix_audit_2026-05-15.json |
| product_scope | benchmark_results/product_scope_gate_2026-05-15.json |
| coverage_gate | benchmark_results/benchmark_coverage_gate_2026-05-15.json |
| seqcls_gap | benchmark_results/seqcls_runtime_gap_2026-05-15.json |
| moe_support | benchmark_results/moe_support_audit_2026-05-15.json |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
