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
| direct_i2s_writer_has_x86_act_layout | True |
| direct_i2s_writer_has_i2s_fallback | True |
| direct_i2s_writer_has_i2sr_mode | True |
| direct_converter_blocks_quantized_by_default | True |
| direct_i2sr_packing_byte_verified | True |
| row_scale_patch_reuses_i2s_type | True |
| row_scale_patch_changes_i2s_nbytes | True |

## Verdict

| claim | value |
| --- | --- |
| direct_dense_gguf_supported | True |
| direct_packed_i2s_supported | True |
| direct_packed_i2s_supported_via_native_py_stack | False |
| candidate_i2sr_writer_supported | True |
| candidate_i2sr_quality_valid | True |
| candidate_i2sr_layout_verified | True |
| product_safe_row_scale_packed_supported | False |
| requires_python_gguf_i2s_support | True |
| requires_stable_row_scale_type_or_version | True |
| requires_special_row_scale_nbytes | True |

## Required Gates

1. Keep scalar direct I2_S covered by load/run evidence after the x86 ACT packing fix.
2. Promote the candidate I2_SR patch into the active runtime or carry it as an explicit downstream patch.
3. Keep byte-layout regression coverage for direct I2_SR packing against the known-good quantizer layout.
4. Only then claim product-safe direct packed row-scale GGUF support.

## Interpretation

Scalar direct packed `I2_S` export is mechanically supported by the self-contained writer. Row-scale direct export is now quality-valid and byte-layout-verified through the fixed x86 ACT `I2_SR` candidate path on Qwen2.5-1.5B, but it remains not product-complete because the cleaner row-scale qtype is still a downstream patch rather than active/default runtime support.
