# Direct Packed GGUF Support Audit, 2026-05-13

This audit distinguishes direct dense GGUF export, scalar direct packed `I2_S` export, and product-safe row-scale packed export.

## Checks

| check | value |
| --- | --- |
| cxx_has_i2s_ggml_type | True |
| cxx_has_i2s_llama_ftype | True |
| cxx_quantize_maps_i2s_ftype | True |
| cxx_quantize_cli_exposes_i2s | True |
| py_gguf_has_i2s_quant_type | False |
| py_gguf_has_i2s_file_type | False |
| py_gguf_has_i2s_quant_size | False |
| py_quants_has_i2s_trait | False |
| py_writer_has_i2s_special_layout | False |
| direct_i2s_writer_has_1x4_layout | True |
| direct_i2s_writer_has_i2s_fallback | True |
| direct_i2s_writer_has_i2sr_mode | True |
| direct_converter_blocks_quantized_by_default | True |
| row_scale_patch_reuses_i2s_type | True |
| row_scale_patch_changes_i2s_nbytes | True |

## Verdict

| claim | value |
| --- | --- |
| direct_dense_gguf_supported | True |
| direct_packed_i2s_supported | True |
| direct_packed_i2s_supported_via_native_py_stack | False |
| candidate_i2sr_writer_supported | True |
| product_safe_row_scale_packed_supported | False |
| requires_python_gguf_i2s_support | True |
| requires_stable_row_scale_type_or_version | True |
| requires_special_row_scale_nbytes | True |

## Required Gates

1. Keep the scalar direct I2_S writer covered by load/run quality-failure evidence.
2. Promote the candidate I2_SR patch into the active runtime or carry it as an explicit downstream patch.
3. Run a full I2_SR PPL/throughput/RSS suite on the strong Qwen2.5-1.5B row-scale checkpoint.
4. Only then claim product-safe direct packed row-scale GGUF support.

## Interpretation

Scalar direct packed `I2_S` export is now mechanically supported by the self-contained writer and covered by a Qwen0.5B load/run quality-failure artifact. Row-scale remains not product-complete: the older quality-preserving prototype overloads `I2_S`, while the cleaner `I2_SR` path is currently an apply-check/build-checked candidate patch plus writer smoke, not a full runtime benchmark suite.
