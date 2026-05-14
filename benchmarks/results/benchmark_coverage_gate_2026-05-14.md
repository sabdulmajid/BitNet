# Benchmark Coverage Gate, 2026-05-14

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
| FP F16 CPU row is finite | pass | ppl=12.2808, prefill=114.468162, decode=5.555998 |  |
| FP Q8_0 CPU row is finite | pass | ppl=12.3056, prefill=124.864246, decode=10.131914 |  |
| FP Q4_K_M CPU row is finite | pass | ppl=12.8112, prefill=92.077037, decode=16.013125 |  |
| row-scale TQ2_0 CPU row is finite | pass | ppl=38.8224, prefill=169.460897, decode=18.675323 |  |
| row-scale I2_S CPU row is finite | pass | ppl=38.8832, prefill=218.172685, decode=18.973629 |  |
| row-scale I2_SR CPU row is finite | pass | ppl=38.8477, prefill=211.668328, decode=19.065496 |  |
| fixed I2_SR RSS has four context rows | pass | contexts=[512, 2048, 8192, 32768] |  |
| evidence manifest has no missing artifacts | pass | path=benchmarks/results/evidence_manifest_2026-05-14.json, artifacts=155, missing=0, missing_labels=[] |  |
| productization gate passes for stable I2_SR | pass | passed=True, failed=0, stable_quality=True, layout=True |  |
