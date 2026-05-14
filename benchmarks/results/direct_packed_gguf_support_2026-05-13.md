# Direct Packed GGUF Support Audit, 2026-05-13

This audit distinguishes direct dense GGUF export, scalar direct packed `I2_S` export, and product-safe row-scale packed export.

## Checks

| check | value |
| --- | --- |
| cxx_has_i2s_ggml_type | True |
| cxx_has_i2s_llama_ftype | True |
| cxx_quantize_maps_i2s_ftype | True |
| cxx_quantize_cli_exposes_i2s | True |
| cxx_has_i2sr_ggml_type | True |
| cxx_has_i2sr_llama_ftype | True |
| cxx_quantize_maps_i2sr_ftype | True |
| cxx_quantize_cli_exposes_i2sr | True |
| py_gguf_has_i2s_quant_type | True |
| py_gguf_has_i2s_file_type | True |
| py_gguf_has_i2s_quant_size | True |
| py_gguf_has_i2sr_quant_type | True |
| py_gguf_has_i2sr_file_type | True |
| py_gguf_has_i2sr_quant_size | True |
| py_quants_has_i2s_trait | False |
| py_writer_has_i2s_special_layout | False |
| direct_i2s_writer_has_x86_act_layout | True |
| direct_i2s_writer_has_i2s_fallback | True |
| direct_i2s_writer_has_i2sr_mode | True |
| direct_converter_blocks_quantized_by_default | True |
| direct_i2sr_packing_byte_verified | True |
| stable_i2sr_productization_gate_passed | True |
| legacy_row_scale_patch_reuses_i2s_type | True |
| legacy_row_scale_patch_changes_i2s_nbytes | True |

## Verdict

| claim | value |
| --- | --- |
| direct_dense_gguf_supported | True |
| direct_packed_i2s_supported | True |
| direct_packed_i2s_supported_via_native_py_stack | False |
| candidate_i2sr_writer_supported | True |
| candidate_i2sr_quality_valid | True |
| candidate_i2sr_layout_verified | True |
| stable_i2sr_runtime_supported | True |
| product_safe_row_scale_packed_supported | True |
| requires_python_gguf_i2s_support | False |
| requires_stable_row_scale_type_or_version | False |
| requires_special_row_scale_nbytes | False |

## Required Gates

1. Keep scalar direct I2_S covered by load/run evidence after the x86 ACT packing fix.
2. Keep the stable I2_SR qtype/file-type route active in both llama.cpp and the Python GGUF constants.
3. Keep byte-layout regression coverage for direct I2_SR packing against the known-good quantizer layout.
4. Keep Qwen2.5-1.5B I2_SR quality and Xeon CPU evidence attached to release claims.

## Interpretation

Scalar direct packed `I2_S` export is mechanically supported by the self-contained writer. Row-scale direct export is product-safe for the audited dense-Qwen research path when the stable `I2_SR` qtype/file type is active, byte-layout verification passes, and the Qwen2.5-1.5B Xeon quality run remains valid. This does not make arbitrary FP/BF16 retrofits or MoE/Kimi support valid.
