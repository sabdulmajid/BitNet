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
| BitNet-SFT budget paired audit has completed full-MNLI rows | pass | complete=9/10, best_matched=9815, path=benchmark_results/bitnet_sft_budget_paired_2026-05-15.json |  |
| BitNet-SFT best budget row has paired CI and McNemar test | pass | delta=-0.17921548650025465, ci=[-0.18958024909015786, -0.16885072391035155], mcnemar=3.4383886495134495e-240 |  |
| BitNet-SFT mechanics audit passes | pass | passed=True, verdict=basic_mechanics_pass_bitdistill_recovery_pending, path=benchmark_results/bitnet_sft_mechanics_audit_2026-05-15.json |  |
| BitNet-SFT mechanics audit has exact projection replacement counts | pass | ternary=168, families={'down_proj': 24, 'gate_proj': 24, 'k_proj': 24, 'o_proj': 24, 'q_proj': 24, 'up_proj': 24, 'v_proj': 24} |  |
| BitNet-SFT mechanics audit confirms dense non-projection tensors | pass | score_dense=True, score_ternary=False, forbidden=[] |  |
| BitNet-SFT mechanics audit confirms three-symbol ternary distribution | pass | fractions={'-1': 0.33324317512931406, '0': 0.33317633827964027, '1': 0.33358048659104567}, entropy=1.5849622976253435 |  |
| SubLN activation-variance audit has finite logit drift | pass | inserted=48, rel_rms=0.7680435180664062, cosine=0.6982523202896118, path=benchmark_results/subln_activation_variance_2026-05-15.json |  |
| SubLN audit confirms projection-input normalization | pass | subln_output_rms=[0.9996804222464561, 0.99930373330911] |  |
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
| fixed I2_SR RSS has four context rows | pass | contexts=[512, 2048, 8192, 32768] |  |
| evidence manifest has no missing artifacts | pass | path=benchmarks/results/evidence_manifest_2026-05-15.json, artifacts=193, missing=0, missing_labels=[] |  |
| productization gate passes for stable I2_SR | pass | passed=True, failed=0, stable_quality=True, layout=True |  |
