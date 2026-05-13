# Evidence Manifest, 2026-05-13
Artifacts: `64`. Missing: `0`.
| label | kind | exists | size bytes | sha256 prefix | parsed summary |
| --- | --- | --- | ---: | --- | --- |
| README | tracked_report | yes | 48050 | af3109405c34 |  |
| side_by_side_report | tracked_report | yes | 12803 | 1729d20a5d2a |  |
| publishable_claims | tracked_report | yes | 9726 | 9a01e43a6856 |  |
| progress_audit | tracked_report | yes | 22913 | d5d5679e3fc3 |  |
| active_goal_audit | tracked_report | yes | 9724 | cbe862d8b210 |  |
| direct_static_ternary_gguf_report | tracked_report | yes | 2072 | ff830f6c1d72 |  |
| direct_packed_gguf_support_report | tracked_report | yes | 2053 | 86a566cf3fad |  |
| direct_i2s_scalar_gguf_report | tracked_report | yes | 5514 | 2c5a76508f7a |  |
| direct_row_i2s_qwen05b_report | tracked_report | yes | 3648 | 7986c373ad38 |  |
| tl2_shape_report | tracked_report | yes | 2817 | 8dddd65315e0 |  |
| tl2_probe_report | tracked_report | yes | 3608 | dd7e4fa29cc1 |  |
| tl2_scale_report | tracked_report | yes | 2321 | 30bbfbd22559 |  |
| i2s_row_scale_format_report | tracked_report | yes | 1759 | 256abf73581c |  |
| row_scale_qtype_productization_gate_report | tracked_report | yes | 2099 | 9562e3121111 |  |
| moe_report | tracked_report | yes | 1882 | 342ac496a28d |  |
| latest_nonrow_audit | evidence_audit_md | yes | 4396 | 6d8c31e958f1 |  |
| row_notie_5000_audit | evidence_audit_md | yes | 2569 | 4615a642257d |  |
| row_i2s_heapfix_audit | evidence_audit_md | yes | 964 | 7077cdc4e602 |  |
| row_i2s_thread_scaling_audit | evidence_audit_md | yes | 848 | 1321b4c78650 |  |
| context_scaling_rss_audit | evidence_audit_md | yes | 939 | ef8e7c7a8727 |  |
| qwen05b_tl2_probe_audit | evidence_audit_md | yes | 777 | ff37bffc1242 |  |
| fp_wikitext | perplexity_json | yes | 473 | acd79bab6f40 | ppl=13.9015, tokens=32704.0 |
| fp_fineweb | perplexity_json | yes | 488 | f493c13eb9cf | ppl=10.2686, tokens=32736.0 |
| ptq_wikitext | perplexity_json | yes | 499 | 8481ac139901 | ppl=3.81312e+06, tokens=32704.0 |
| ptq_fineweb | perplexity_json | yes | 517 | 3d8b77f56169 | ppl=9.58292e+06, tokens=32736.0 |
| hidden_mse_wikitext | perplexity_json | yes | 504 | 78bf0bea21b4 | ppl=86.414, tokens=32704.0 |
| hidden_mse_fineweb | perplexity_json | yes | 519 | 46c9b60cae1f | ppl=40.3981, tokens=32736.0 |
| kl_wikitext | perplexity_json | yes | 515 | 71d506623bda | ppl=50.5954, tokens=32704.0 |
| kl_fineweb | perplexity_json | yes | 534 | 77bc5e4d4425 | ppl=26.599, tokens=32736.0 |
| kl_dense_head_wikitext | perplexity_json | yes | 526 | 05f3adf09ed6 | ppl=43.3724, tokens=32704.0 |
| kl_dense_head_fineweb | perplexity_json | yes | 545 | 84c6bcda2904 | ppl=22.759, tokens=32736.0 |
| row_dense_head_wikitext | perplexity_json | yes | 532 | 3e0d76dd2573 | ppl=38.5801, tokens=32704.0 |
| row_dense_head_fineweb | perplexity_json | yes | 546 | f16a9f346821 | ppl=21.3332, tokens=32736.0 |
| lm_eval_fp | lm_eval_json | yes | 87341049 | e48cdc1bb44f | mean=0.644169, tasks=10, samples=22382 |
| lm_eval_ptq | lm_eval_json | yes | 87307748 | 46e7097b707e | mean=0.348671, tasks=10, samples=22382 |
| lm_eval_hidden_mse | lm_eval_json | yes | 87323009 | 34dd9886b133 | mean=0.464809, tasks=10, samples=22382 |
| lm_eval_kl | lm_eval_json | yes | 87321092 | 602e3c07c792 | mean=0.483438, tasks=10, samples=22382 |
| lm_eval_kl_dense_head | lm_eval_json | yes | 87322204 | 7ea31be6cee4 | mean=0.484378, tasks=10, samples=22382 |
| lm_eval_row_dense_head | lm_eval_json | yes | 87327355 | 50cda719e02a | mean=0.499459, tasks=10, samples=22382 |
| gguf_kl_suite | gguf_summary_json | yes | 9552 | 07a517cd5c67 | rows=8, failed=0, nan=0, catastrophic=2, max_ppl=1.20612e+51 |
| gguf_dense_head_suite | gguf_summary_json | yes | 7291 | bd4f50ffc310 | rows=6, failed=0, nan=0, catastrophic=0, max_ppl=47.3435 |
| gguf_row_dense_head_suite | gguf_summary_json | yes | 7297 | 228c6502be56 | rows=6, failed=0, nan=0, catastrophic=1, max_ppl=1.19714e+06 |
| gguf_row_i2s_heapfix | gguf_summary_json | yes | 1647 | 734fa7c103f1 | rows=1, failed=0, nan=0, catastrophic=0, max_ppl=38.8832 |
| gguf_row_i2s_thread_scaling | thread_scaling_json | yes | 2296 | 0396f5c951bd | rows=7, max_prefill=245.314, max_decode=19.4941 |
| gguf_context_rss | gguf_memory_json | yes | 9911 | fd135b202cfd | rows=24, contexts=[512, 2048, 8192, 32768] |
| direct_gguf_tiny | direct_gguf_json | yes | 1972 | a01c51bbb398 | arch=LlamaForCausalLM, outtype=f16, ternary=8, tensors=12, reader_rc=0, smoke_rc=None |
| direct_gguf_qwen05b | direct_gguf_json | yes | 2546 | fb82f94beb57 | arch=Qwen2ForCausalLM, outtype=f16, ternary=168, tensors=291, reader_rc=0, smoke_rc=0 |
| direct_i2s_tiny | direct_i2s_json | yes | 1087 | 496c4a2bac8d | arch=LlamaForCausalLM, packed=7, row_packed=0, out_f16=1, tensors=12, row_qtype=None, native_py_consts=False |
| direct_i2s_qwen05b_conversion | direct_i2s_json | yes | 1146 | deecbf0d1031 | arch=Qwen2ForCausalLM, packed=168, row_packed=0, out_f16=1, tensors=291, row_qtype=None, native_py_consts=False |
| direct_i2s_qwen05b_suite | gguf_summary_json | yes | 2559 | d6451c122755 | rows=2, failed=0, nan=1, catastrophic=0, max_ppl=18.0984 |
| direct_row_i2s_qwen05b_conversion | direct_i2s_json | yes | 1157 | a7ee2b556bc3 | arch=Qwen2ForCausalLM, packed=168, row_packed=168, out_f16=1, tensors=291, row_qtype=None, native_py_consts=False |
| i2sr_writer_smoke_qwen05b | direct_i2s_json | yes | 1421 | 42f8bed864e7 | arch=Qwen2ForCausalLM, packed=168, row_packed=168, out_f16=1, tensors=291, row_qtype=i2_sr, native_py_consts={'i2s': False, 'i2sr': False} |
| direct_row_i2s_qwen05b_suite | gguf_summary_json | yes | 2637 | ebdafab23882 | rows=2, failed=0, nan=0, catastrophic=1, max_ppl=59401.5 |
| row_f16_qwen05b_suite | gguf_summary_json | yes | 2575 | cc4583593042 | rows=2, failed=0, nan=0, catastrophic=0, max_ppl=578.483 |
| row_i2s_quantized_qwen05b_suite | gguf_summary_json | yes | 2645 | 23b71114cdae | rows=2, failed=0, nan=1, catastrophic=0, max_ppl=18.0986 |
| row_tq2_qwen05b_suite | gguf_summary_json | yes | 2667 | d3005c427d44 | rows=2, failed=0, nan=0, catastrophic=1, max_ppl=5.11853e+06 |
| direct_packed_gguf_support_json | direct_packed_support_json | yes | 2426 | 4636ac5e9c52 | dense=True, packed_i2s=True, row_safe=False |
| tl2_shape_json | tl2_shape_json | yes | 12054 | a99430f0f4b0 |  |
| tl2_scale_json | tl2_scale_json | yes | 14344 | bf2d36674b72 | qwen15b_tensor_scale err=0; qwen15b_row_scale err=1.90423 |
| i2s_row_scale_format_json | i2s_format_json | yes | 1528 | e17297539b9d | default_ratio=30836.2, prototype_ratio=1.00157, stable_format_required=True |
| row_scale_qtype_productization_gate_json | row_scale_qtype_gate_json | yes | 2654 | cc961131c059 | passed=False, gates=7, failed=4, stable_qtype=False, writer=True |
| tl2_generic_summary | gguf_summary_json | yes | 4494 | 306a682203b2 | rows=4, failed=2, nan=1, catastrophic=0, max_ppl=18.8044 |
| tl2_avx512_summary | gguf_summary_json | yes | 4573 | 6ec820472f37 | rows=4, failed=0, nan=2, catastrophic=0, max_ppl=18.8239 |
| ptq_math | math_json | yes | 14010 | 8cedd88658e0 | trials=10, rel_error=0.512542 |
