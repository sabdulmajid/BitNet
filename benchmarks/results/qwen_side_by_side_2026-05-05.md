# Qwen2.5-1.5B Side-by-Side Artifact Summary

Generated from benchmark JSON artifacts. Missing rows are intentionally shown as missing.

## Perplexity

| run | WikiText PPL | FineWeb PPL | Wiki tokens | FineWeb tokens | status |
| --- | --- | --- | --- | --- | --- |
| FP | 13.901 | 10.269 | 32704 | 32736 | present |
| naive PTQ | 3813121.803 | 9582923.269 | 32704 | 32736 | present |
| QAT hidden-MSE | 86.414 | 40.398 | 32704 | 32736 | present |
| QAT KL-only | 50.595 | 26.599 | 32704 | 32736 | present |
| QAT KL-only dense lm_head | 43.372 | 22.759 | 32704 | 32736 | present |
| QAT KL-only row dense lm_head | - | - | - | - | missing |

## Full Ten-Task lm-eval

| run | selected mean | tasks | samples | status |
| --- | --- | --- | --- | --- |
| FP | 0.644169 | 10 | 22382 | present |
| naive PTQ | 0.348671 | 10 | 22382 | present |
| QAT hidden-MSE | 0.464809 | 10 | 22382 | present |
| QAT KL-only | 0.483438 | 10 | 22382 | present |
| QAT KL-only dense lm_head | 0.484378 | 10 | 22382 | present |
| QAT KL-only row dense lm_head | - | 0 | 0 | missing |

## Packed GGUF CPU

| suite | CPU | artifact | kind | file MiB | prefill tok/s | decode tok/s | PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_f16 | fp_reference | 2950.4 | 104.23 | 5.47 | 12.2806 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 134.48 | 10.03 | 12.3207 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 94.03 | 15.73 | 12.8452 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_tq2_0 | blind_ternary_tq2 | 773.5 | 160.66 | 18.34 | 18041439.0235 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_fp_i2_s | blind_bitnet_i2s | 766.1 | 204.57 | 18.34 | 1206122008073269478858135651032362604219831211261952.0000 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_f16 | qat_klonly_static_ternary_materialized | 3395.5 | 105.34 | 5.50 | 55.0971 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_tq2_0 | qat_klonly_static_ternary_tq2 | 1218.6 | 160.93 | 18.43 | 55.1562 |
| KL-only all-linear static ternary suite | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz | qwen15b_klonly_static_ternary_i2_s | qat_klonly_static_ternary_i2s_single_thread_quant | 1208.9 | 205.76 | 18.60 | 54.7366 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_f16 | fp_reference | 2950.4 | 218.46 | 12.46 | 12.2808 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q8_0 | llama_q8 | 1570.3 | 215.00 | 23.14 | 12.3056 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_fp_q4_k_m | llama_q4 | 940.4 | 172.14 | 36.50 | 12.8112 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_f16 | static_ternary_materialized | 3395.5 | 222.07 | 12.48 | 47.2994 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_tq2_0 | static_ternary_tq2 | 1218.6 | 348.88 | 44.03 | 47.2823 |
| KL-only dense lm_head static ternary suite | AMD Ryzen Threadripper PRO 5945WX 12-Cores | qwen15b_klonly_notie_static_ternary_i2_s | static_ternary_i2s_single_thread_quant | 1208.9 | 464.19 | 45.50 | 47.3435 |
| KL-only row dense lm_head static ternary suite | - | - | - | - | - | missing |
