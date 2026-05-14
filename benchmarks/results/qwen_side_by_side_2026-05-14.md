# Qwen2.5-1.5B Side-by-Side Artifact Summary

Generated from benchmark JSON artifacts. Missing rows are intentionally shown as missing. The Xeon headline isolates the Intel Xeon Silver 4116 runs; the longer GGUF table also preserves older Threadripper control runs and should not be used for cross-machine speed ratios.

## Headline Verdict

| claim area | best current artifact | artifact Wiki/CPU PPL | reference FP PPL | artifact FineWeb/PPL | reference FP FineWeb/PPL | artifact ten-task mean | FP ten-task mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best HF quality recovery | QAT KL-only row-scale dense lm_head | 38.580 | 13.901 | 21.333 | 10.269 | 0.499459 | 0.644169 | not FP-quality |
| packed CPU candidate | direct I2_SR fixed x86 ACT | 38.848 | 12.281 | - | - | - | - | production gate pass |

## Reviewer Gate Summary

| gate | status | evidence | reviewer implication |
| --- | --- | --- | --- |
| benchmark coverage | pass | full ten-task, paired deltas, CPU quality/speed/RSS, manifest | This confirms artifact coverage, not product completion. |
| objective completion | not_complete | 7/9 complete | Open items are TL2 row/group-scale quality support and MoE/Kimi evidence. |
| product scope | research_mvp_only | negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path | Do not claim arbitrary lossless retrofit or MoE/Kimi support. |
| I2_SR active submodule | ready | active=True; patch_applies=False; blockers=0 | Quality-valid CPU path is active when the fork branch is reachable and the superproject pointer is clean. |
| MoE/Kimi packing | ready | tl2_3d=True; i2sr_3d=True; 2d_control=True | Synthetic contract now separates direct I2_S/I2_SR packing from TL2; no Kimi artifact exists. |

## Perplexity

| run | WikiText PPL | FineWeb PPL | Wiki tokens | FineWeb tokens | status |
| --- | --- | --- | --- | --- | --- |
| FP | 13.901 | 10.269 | 32704 | 32736 | present |
| naive PTQ | 3813121.803 | 9582923.269 | 32704 | 32736 | present |
| QAT hidden-MSE | 86.414 | 40.398 | 32704 | 32736 | present |
| QAT KL-only | 50.595 | 26.599 | 32704 | 32736 | present |
| QAT KL-only dense lm_head | 43.372 | 22.759 | 32704 | 32736 | present |
| QAT KL-only row dense lm_head | 38.580 | 21.333 | 32704 | 32736 | present |

## Full Ten-Task lm-eval

| run | selected mean | tasks | samples | status |
| --- | --- | --- | --- | --- |
| FP | 0.644169 | 10 | 22382 | present |
| naive PTQ | 0.348671 | 10 | 22382 | present |
| QAT hidden-MSE | 0.464809 | 10 | 22382 | present |
| QAT KL-only | 0.483438 | 10 | 22382 | present |
| QAT KL-only dense lm_head | 0.484378 | 10 | 22382 | present |
| QAT KL-only row dense lm_head | 0.499459 | 10 | 22382 | present |

## Full Ten-Task Detail

| task | metric | FP | naive PTQ | QAT hidden-MSE | QAT KL-only | QAT KL-only dense lm_head | QAT KL-only row dense lm_head |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc_challenge | acc_norm | 0.450 | 0.262 | 0.264 | 0.271 | 0.264 | 0.272 |
| arc_easy | acc_norm | 0.720 | 0.244 | 0.478 | 0.483 | 0.501 | 0.518 |
| hellaswag | acc_norm | 0.678 | 0.264 | 0.362 | 0.378 | 0.391 | 0.412 |
| piqa | acc_norm | 0.758 | 0.508 | 0.622 | 0.637 | 0.647 | 0.650 |
| winogrande | acc | 0.638 | 0.498 | 0.523 | 0.521 | 0.523 | 0.537 |
| boolq | acc | 0.726 | 0.506 | 0.593 | 0.596 | 0.597 | 0.605 |
| copa | acc | 0.830 | 0.510 | 0.640 | 0.700 | 0.680 | 0.690 |
| openbookqa | acc_norm | 0.404 | 0.276 | 0.312 | 0.312 | 0.308 | 0.316 |
| sciq | acc_norm | 0.934 | 0.199 | 0.613 | 0.695 | 0.700 | 0.733 |
| truthfulqa_mc1 | acc | 0.305 | 0.220 | 0.241 | 0.241 | 0.233 | 0.261 |

## Paired Ten-Task Delta Checks

| comparison | macro mean delta with 95% CI | example-weighted delta | matched examples | status |
| --- | --- | --- | --- | --- |
| QAT row-scale minus FP | -0.144710 [-0.185756, -0.103664] | -0.195336 | 22382 | present |
| QAT row-scale minus naive PTQ | +0.150788 [+0.053427, +0.248149] | +0.151506 | 22382 | present |
| QAT row-scale minus tensor-scale dense lm_head | +0.015081 [+0.009028, +0.021134] | +0.016755 | 22382 | present |
| QAT row-scale minus KL-only tensor-scale | +0.016021 [+0.006145, +0.025897] | +0.024975 | 22382 | present |

## Xeon Packed Runtime Headline

| artifact | CPU | file MiB | PPL | prefill tok/s | decode tok/s | quality status |
| --- | --- | --- | --- | --- | --- | --- |
| FP F16 | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 2950.4 | 12.2808 | 114.47 | 5.56 | ok |
| FP Q8_0 | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 1570.3 | 12.3056 | 124.86 | 10.13 | ok |
| FP Q4_K_M | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 940.4 | 12.8112 | 92.08 | 16.01 | ok |
| row-scale ternary TQ2_0 | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 1218.6 | 38.8224 | 169.46 | 18.68 | ok |
| row-scale ternary I2_S prototype | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 1211.3 | 38.8832 | 218.17 | 18.97 | ok |
| row-scale ternary I2_SR fixed candidate | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | 1211.3 | 38.8477 | 211.67 | 19.07 | ok |

## Packed GGUF CPU

| suite | CPU | artifact | kind | file MiB | prefill tok/s | decode tok/s | PPL | quality status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_f16 | fp_reference | 2950.4 | 104.23 | 5.47 | 12.2806 | ok |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 134.48 | 10.03 | 12.3207 | ok |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 94.03 | 15.73 | 12.8452 | ok |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_tq2_0 | blind_ternary_tq2 | 773.5 | 160.66 | 18.34 | 18041439.0235 | catastrophic |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_i2_s | blind_bitnet_i2s | 766.1 | 204.57 | 18.34 | 1206122008073269478858135651032362604219831211261952.0000 | catastrophic |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_f16 | qat_klonly_static_ternary_materialized | 3395.5 | 105.34 | 5.50 | 55.0971 | ok |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_tq2_0 | qat_klonly_static_ternary_tq2 | 1218.6 | 160.93 | 18.43 | 55.1562 | ok |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_i2_s | qat_klonly_static_ternary_i2s_single_thread_quant | 1208.9 | 205.76 | 18.60 | 54.7366 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_f16 | fp_reference | 2950.4 | 218.46 | 12.46 | 12.2808 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 215.00 | 23.14 | 12.3056 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 172.14 | 36.50 | 12.8112 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_f16 | static_ternary_materialized | 3395.5 | 222.07 | 12.48 | 47.2994 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_tq2_0 | static_ternary_tq2 | 1218.6 | 348.88 | 44.03 | 47.2823 | ok |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_i2_s | static_ternary_i2s_single_thread_quant | 1208.9 | 464.19 | 45.50 | 47.3435 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_f16 | fp_reference | 2950.4 | 219.27 | 11.99 | 12.2808 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 214.69 | 22.51 | 12.3056 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 172.37 | 36.82 | 12.8112 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 3395.5 | 221.64 | 12.49 | 38.8651 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 1218.6 | 345.32 | 44.85 | 38.8224 | ok |
| KL-only row dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_row_notie_static_ternary_i2_s | static_ternary_i2s_single_thread_quant | 1208.9 | 465.34 | 46.13 | 1197135.5848 | catastrophic |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_f16 | fp_reference | 2950.4 | 114.47 | 5.56 | 12.2808 | ok |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 124.86 | 10.13 | 12.3056 | ok |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 92.08 | 16.01 | 12.8112 | ok |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 3395.5 | 114.75 | 5.49 | 38.8651 | ok |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 1218.6 | 169.46 | 18.68 | 38.8224 | ok |
| KL-only row dense lm_head I2_S row-scale prototype suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 1211.3 | 216.03 | 18.83 | 38.8832 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_f16 | fp_reference | 2950.4 | 102.46 | 5.49 | 12.2806 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 136.02 | 10.05 | 12.3207 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 92.46 | 15.67 | 12.8452 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 3395.5 | 104.72 | 5.49 | 38.8652 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 1218.6 | 161.67 | 18.14 | 38.8357 | ok |
| KL-only row dense lm_head I2_S row-scale prototype native suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 1211.3 | 207.35 | 18.37 | 38.8853 | ok |
| KL-only row dense lm_head I2_S heap-fix confirmation | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype_heap_tmp_fix | 1211.3 | 218.17 | 18.97 | 38.8832 | ok |
| KL-only row dense lm_head I2_SR fixed x86 ACT candidate | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act | static_ternary_i2sr_row_scale_candidate_x86_act_pack | 1211.3 | 211.67 | 19.07 | 38.8477 | ok |

## Packed GGUF RSS

| suite | artifact | kind | ctx | file MiB | max RSS GiB | return code |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_fp_f16 | fp_reference | - | 2950.4 | 2.948 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_fp_q8_0 | llama_q8 | - | 1570.3 | 1.601 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_fp_q4_k_m | llama_q4 | - | 940.4 | 0.985 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | - | 3395.5 | 3.383 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | - | 1218.6 | 1.257 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS probe | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | - | 1211.3 | 1.250 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_f16 | fp_reference | 512 | 2950.4 | 2.948 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q8_0 | llama_q8 | 512 | 1570.3 | 1.601 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q4_k_m | llama_q4 | 512 | 940.4 | 0.985 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 512 | 3395.5 | 3.383 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 512 | 1218.6 | 1.257 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 512 | 1211.3 | 1.250 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_f16 | fp_reference | 2048 | 2950.4 | 2.989 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q8_0 | llama_q8 | 2048 | 1570.3 | 1.642 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q4_k_m | llama_q4 | 2048 | 940.4 | 1.027 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 2048 | 3395.5 | 3.424 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 2048 | 1218.6 | 1.298 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 2048 | 1211.3 | 1.291 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_f16 | fp_reference | 8192 | 2950.4 | 3.153 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q8_0 | llama_q8 | 8192 | 1570.3 | 1.806 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q4_k_m | llama_q4 | 8192 | 940.4 | 1.191 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 8192 | 3395.5 | 3.588 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 8192 | 1218.6 | 1.462 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 8192 | 1211.3 | 1.455 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_f16 | fp_reference | 32768 | 2950.4 | 3.812 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q8_0 | llama_q8 | 32768 | 1570.3 | 2.465 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_fp_q4_k_m | llama_q4 | 32768 | 940.4 | 1.850 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_f16 | static_ternary_materialized | 32768 | 3395.5 | 4.247 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_tq2_0 | static_ternary_tq2 | 32768 | 1218.6 | 2.121 | 0 |
| Qwen2.5-1.5B row-scale I2_S RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale | static_ternary_i2s_row_scale_prototype | 32768 | 1211.3 | 2.114 | 0 |
| Qwen2.5-1.5B row-scale I2_SR fixed x86 ACT RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act | static_ternary_i2sr_row_scale_candidate_x86_act_pack | 512 | 1211.3 | 1.250 | 0 |
| Qwen2.5-1.5B row-scale I2_SR fixed x86 ACT RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act | static_ternary_i2sr_row_scale_candidate_x86_act_pack | 2048 | 1211.3 | 1.291 | 0 |
| Qwen2.5-1.5B row-scale I2_SR fixed x86 ACT RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act | static_ternary_i2sr_row_scale_candidate_x86_act_pack | 8192 | 1211.3 | 1.455 | 0 |
| Qwen2.5-1.5B row-scale I2_SR fixed x86 ACT RSS context scaling | qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act | static_ternary_i2sr_row_scale_candidate_x86_act_pack | 32768 | 1211.3 | 2.114 | 0 |
