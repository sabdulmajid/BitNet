# Benchmark Coverage Gate, 2026-05-15

Overall status: **PASS**.

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| FP has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| FP has expected logged samples | pass | samples=22382 |  |
| naive PTQ has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| naive PTQ has expected logged samples | pass | samples=22382 |  |
| QAT hidden-MSE has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| QAT hidden-MSE has expected logged samples | pass | samples=22382 |  |
| QAT KL-only has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| QAT KL-only has expected logged samples | pass | samples=22382 |  |
| QAT KL-only dense lm_head has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| QAT KL-only dense lm_head has expected logged samples | pass | samples=22382 |  |
| QAT KL-only row dense lm_head has ten selected lm-eval tasks | pass | tasks=10, missing=[] |  |
| QAT KL-only row dense lm_head has expected logged samples | pass | samples=22382 |  |
| row minus FP has ten paired task rows | pass | rows=10 |  |
| row minus FP has expected paired examples | pass | matched=22382 |  |
| row minus FP has macro CI | pass | -0.144710 [-0.185756, -0.103664] |  |
| row minus naive PTQ has ten paired task rows | pass | rows=10 |  |
| row minus naive PTQ has expected paired examples | pass | matched=22382 |  |
| row minus naive PTQ has macro CI | pass | +0.150788 [+0.053427, +0.248149] |  |
| row minus tensor dense-head has ten paired task rows | pass | rows=10 |  |
| row minus tensor dense-head has expected paired examples | pass | matched=22382 |  |
| row minus tensor dense-head has macro CI | pass | +0.015081 [+0.009028, +0.021134] |  |
| row minus KL-only has ten paired task rows | pass | rows=10 |  |
| row minus KL-only has expected paired examples | pass | matched=22382 |  |
| row minus KL-only has macro CI | pass | +0.016021 [+0.006145, +0.025897] |  |
| BitDistill paired audit is complete | pass | complete=44/44, pending=0, failed=0 |  |
| BitDistill paired audit has paired statistics for every row | pass | stats_rows=44/44 |  |
| BitDistill paired audit has BitNet baseline rows | pass | rows=3, path=benchmark_results/bitdistill_paired_predictions_2026-05-15.json |  |
| BitNet baseline paired rows cover full GLUE validation | pass | full_rows=3, matched=16150 |  |
| BitNet baseline paired rows have paired statistics | pass | stats_rows=3 |  |
| BitNet-SFT budget paired audit has completed full-MNLI rows | pass | complete=10/10, best_matched=9815, path=benchmark_results/bitnet_sft_budget_paired_2026-05-15.json |  |
| BitNet-SFT best budget row has paired CI and McNemar test | pass | delta=-0.17921548650025465, ci=[-0.18958024909015786, -0.16885072391035155], mcnemar=3.4383886495134495e-240 |  |
| BitNet-SFT mechanics audit passes | pass | passed=True, verdict=basic_mechanics_pass_bitdistill_recovery_pending, path=benchmark_results/bitnet_sft_mechanics_audit_2026-05-15.json |  |
| BitNet-SFT mechanics audit has exact projection replacement counts | pass | ternary=168, families={'down_proj': 24, 'gate_proj': 24, 'k_proj': 24, 'o_proj': 24, 'q_proj': 24, 'up_proj': 24, 'v_proj': 24} |  |
| BitNet-SFT mechanics audit confirms dense non-projection tensors | pass | score_dense=True, score_ternary=False, forbidden=[] |  |
| BitNet-SFT mechanics audit confirms three-symbol ternary distribution | pass | fractions={'-1': 0.33324317512931406, '0': 0.33317633827964027, '1': 0.33358048659104567}, entropy=1.5849622976253435 |  |
| SubLN activation-variance audit has finite logit drift | pass | inserted=48, rel_rms=0.7680435180664062, cosine=0.6982523202896118, path=benchmark_results/subln_activation_variance_2026-05-15.json |  |
| SubLN audit confirms projection-input normalization | pass | subln_output_rms=[0.9996804222464561, 0.99930373330911] |  |
| BitDistill root-cause audit has required claim statuses | pass | claims=7, mismatches={} |  |
| BitDistill root-cause audit marks controlled recovery incomplete | pass | controlled=2/3, all=False |  |
| BitDistill root-cause audit carries Q4-vs-I2_SR boundary ratios | pass | q4_vs_i2sr={'decode_speedup': 1.19061682213809, 'file_ratio': 1.2881333039180543, 'ppl_ratio': 3.0323232796303237, 'prefill_speedup': 2.298817760610607, 'rss512_ratio': 1.2685742155747033} |  |
| BitDistill telemetry coverage audit measures current loss diagnostics | pass | status=controlled_observability, measured=10/10 |  |
| BitDistill telemetry coverage audit handles advanced telemetry materialization | pass | missing=[], controlled=1, measured_activation=True, measured_dynamics=True |  |
| BitDistill training-dynamics audit parses materialized telemetry | pass | status=controlled_materialized, traces=3, smoke=2, controlled=1 |  |
| BitDistill training-dynamics audit blocks controlled claims until real traces exist | pass | status=controlled_materialized, controlled=1 |  |
| BitDistill loss-contract static checks pass | pass | passed=True, checks=6, status=loss_normalization_risk |  |
| BitDistill loss-contract records paper-gamma dominance risk | pass | status=loss_normalization_risk, max_attn_ce=37819.64134227373 |  |
| Original benchmark objective audit maps all six requested deliverables | pass | completion=5/6, status=partial |  |
| Original benchmark objective audit keeps TL2 row-scale blocker explicit | pass | partial_rows=1, partial=5. Convert repaired checkpoints into GGUF/TL2/I2_S and run CPU inference Dense GGUF and row-scale I2_SR/I2_S CPU inference exist, but TL2 is not quality-preserving for learned row-scale checkpoints until row/group-scale metadata and kernels |  |
| TL2 negative-result audit has CPU probe evidence | pass | cpu_executed=True, finite_quality=False |  |
| TL2 negative-result audit proves row-scale mismatch | pass | supported=True, current=1.9042302114103853, row_fp16=0.00019744640689756221, failed_checks=9 |  |
| TL2 row-scale implementation plan keeps runtime blocked | pass | ready=False, failed_checks=9, blockers=9 |  |
| TL2 row-scale implementation plan has source-mapped patch sequence | pass | step_count=6 |  |
| TL2 row-scale implementation plan carries scale-error proof | pass | one_scale=1.9042302114103853, row_fp16=0.00019744640689756221, scale_mib=1.23046875 |  |
| TL2 group-scale viability audit has group sweep | pass | groups=10, best_group=0.0986922473661489, exact_row=0.00019744640689756221 |  |
| TL2 group-scale viability keeps strict row-scale blocker | pass | strict_viable=False, best_group=0.0986922473661489, exact_row=0.00019744640689756221, ratio=499.84321779707915 |  |
| Ternary flip-dynamics audit has nonzero saved-snapshot flips | pass | status=pass, pairs=2, min_flip=0.06454656755885871, max_flip=0.1659562863332169 |  |
| Ternary threshold-dynamics audit has measured boundary movement | pass | status=measured_increase, snapshots=5, delta=0.01523485567980197, final_band=0.07576561952248598, monotonic=True |  |
| Second-order ternary init audit improves synthetic reconstruction | pass | status=synthetic_promising_task_quality_rejected, deltas=[-0.0775993565718333, -0.094802921017011] |  |
| Second-order ternary init audit ingests diag-LS task-quality result | pass | status=synthetic_promising_task_quality_rejected, diag_status=complete, diag_delta=-0.27794192562404485 |  |
| Second-order ternary init audit blocks quality overclaim | pass | quality_proven=False, ls_integrated=True, diag_integrated=True |  |
| BitNet-SFT LS-init submission changes only initializer | pass | status=submitted, changed_axis=ternary_init_mode_only, init=ls, method=bitnet_sft, task=mnli |  |
| BitNet-SFT LS-init submission has matched absmean baseline | pass | baseline=checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5, steps=10000, lr=2e-05 |  |
| BitNet-SFT LS-init result audit is pending or complete | pass | status=complete, baseline=0.6289353031074886, candidate=0.36189505858380033, quality_proven=False |  |
| BitNet-SFT LS-init complete result has paired MNLI statistics | pass | matched=9815, ci=[-0.27990675734690423, -0.25417373170047225], delta=-0.26704024452368824 |  |
| BitNet-SFT diag-LS init submission changes only initializer | pass | status=submitted, changed_axis=ternary_init_mode_only, init=diag_ls, cal=8, method=bitnet_sft, task=mnli |  |
| BitNet-SFT diag-LS init submission has matched absmean baseline | pass | baseline=checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5, steps=10000, lr=2e-05 |  |
| BitNet-SFT diag-LS init result audit is pending or complete | pass | status=complete, baseline=0.6289353031074886, candidate=0.3509933774834437, quality_proven=False |  |
| BitNet-SFT diag-LS complete result has paired MNLI statistics | pass | matched=9815, ci=[-0.2908559087154853, -0.2650279425326044], delta=-0.27794192562404485 |  |
| Sequence-classification runtime gap is narrowed but not closed | pass | status=native_classifier_sample_available_full_validation_blocked, seqcls=15, seqcls_exportable=0, causal_exportable=6, exports=6 |  |
| Sequence-classification I2_SR sidecar smoke passes | pass | status=prototype_smoke_passed, returncode=0, head_shape=[3, 896], finite_logits=True |  |
| Sequence-classification sidecar CPU quality mismatch is recorded | pass | status=quality_mismatch, examples=128, agreement=0.9140625, accuracy=0.609375 |  |
| Sequence-classification hidden contract is near but not exact | pass | status=hidden_contract_mismatch, token_match=True, hidden_rel_rms=0.10866150519771632, hidden_cosine=0.9940905307837791, logit_rel_rms=0.09191836414090784 |  |
| Sequence-classification architecture contract is identified and repaired | pass | status=bitnet_qwen_contract_available, hidden_act=silu, bitnet25_activation=relu_sqr, bitnet_qwen={'available': True, 'dispatch_line': 16898, 'ffn_activation': 'silu', 'loader_has_qkv_bias': True, 'silu_branch_line': 15544}, projection_biases=72, checks={'activation_mismatch': True, 'bitnet25_has_bias_slots': True, 'bitnet_qwen_contract_available': True, 'plain_bitnet_bias_contract_gap': True, 'plain_bitnet_has_silu_graph': True} |  |
| Sequence-classification runtime implementation plan keeps product blocked | pass | same_artifact=False, ready=False, seqcls_exportable=0 |  |
| Sequence-classification runtime implementation plan has source-owned steps | pass | step_count=5 |  |
| Sequence-classification native I2_SR smoke passes | pass | status=pass, single_artifact=True, returncode=0, logits=3, prediction=2, sidecar_prediction=2, rel_rms=1.0342961243324147e-07 |  |
| Sequence-classification native I2_SR smoke keeps product blocked | pass | full_validation=False, ready=False, prompt_tok_s=265.9 |  |
| Sequence-classification native I2_SR CPU token-ID result records agreement | pass | status=sample_only, prompt_input=token_ids, examples=64, accuracy=0.59375, agreement=0.96875 |  |
| Sequence-classification native I2_SR CPU result keeps product blocked until batching parity passes | pass | full_validation=False, ready=False, batching=None, examples_per_second=0.7173345972905404, rss_mib=950.640625 |  |
| Sequence-classification native mismatch audit isolates runtime drift | pass | status=runtime_hidden_drift, token_ids=True, text_roundtrip=False, native_sidecar=True |  |
| Sequence-classification native batching audit blocks batched throughput | pass | status=batching_parity_mismatch, invariant=False, changed=4, ready=False |  |
| Sequence-classification native batching audit diagnoses non-swap drift | pass | diagnosis=position_dependent_drift_not_row_swap, nearest_self=True, drifted=4 |  |
| Sequence-classification native full CPU progress is resumable | pass | status=partial, completed=795, contiguous=True, product=False |  |
| Qwen3 paper-alignment audit tracks required GLUE rows | pass | jobs=16, complete=15, ready=False |  |
| FP F16 CPU row is finite | pass | ppl=12.2808, prefill=114.468162, decode=5.555998 |  |
| FP Q8_0 CPU row is finite | pass | ppl=12.3056, prefill=124.864246, decode=10.131914 |  |
| FP Q4_K_M CPU row is finite | pass | ppl=12.8112, prefill=92.077037, decode=16.013125 |  |
| row-scale TQ2_0 CPU row is finite | pass | ppl=38.8224, prefill=169.460897, decode=18.675323 |  |
| row-scale I2_S CPU row is finite | pass | ppl=38.8832, prefill=218.172685, decode=18.973629 |  |
| row-scale I2_SR CPU row is finite | pass | ppl=38.8477, prefill=211.668328, decode=19.065496 |  |
| CPU tradeoff frontier has headline rows | pass | labels=['FP F16', 'FP Q4_K_M', 'FP Q8_0', 'row I2_S', 'row I2_SR', 'row TQ2_0'] |  |
| CPU tradeoff frontier reports Q4-vs-I2_SR ratios | pass | q4_vs_i2sr={'decode_speedup': 1.19061682213809, 'file_ratio': 1.2881333039180543, 'ppl_ratio': 3.0323232796303237, 'prefill_speedup': 2.298817760610607, 'rss512_ratio': 1.2685742155747033} |  |
| CPU speed uncertainty audit has I2_SR-vs-Q4 intervals | pass | prefill_ci=[2.2578672451091255, 2.34051098794489], decode_ci=[1.1861079640891115, 1.1951428201115284] |  |
| I2_SR-vs-Q4 speedup intervals stay above 1 | pass | prefill_ci=[2.2578672451091255, 2.34051098794489], decode_ci=[1.1861079640891115, 1.1951428201115284] |  |
| Benchmark matrix audit has at least ten complete quality benchmarks | pass | passed=True, quality_benchmark_count=12 |  |
| Benchmark matrix audit has Xeon runtime and RSS evidence | pass | finite_rows=5, rss_contexts=[512, 2048, 8192, 32768] |  |
| Benchmark matrix audit keeps TL2 row-scale excluded | pass | tl2_ready=False, failed=9 |  |
| Research redirect claim gate passes | pass | status=claim_guardrail_passed, supported=7/7 |  |
| Research redirect gate covers required claims | pass | missing=[] |  |
| Research redirect gate blocks overclaims | pass | blocked_statuses=['blocked', 'not_proven', 'prototype_only'] |  |
| fixed I2_SR RSS has four context rows | pass | contexts=[512, 2048, 8192, 32768] |  |
| evidence manifest has no missing artifacts | pass | path=benchmarks/results/evidence_manifest_2026-05-15.json, artifacts=287, missing=0, missing_labels=[] |  |
| productization gate passes for stable I2_SR | pass | passed=True, failed=0, stable_quality=True, layout=True |  |
