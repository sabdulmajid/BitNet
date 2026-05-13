# Direct Packed GGUF Support Audit, 2026-05-13

This audit distinguishes direct dense GGUF export from direct packed CPU-native GGUF export. The former is now validated; the latter still needs writer and format work.

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
| direct_converter_blocks_quantized_by_default | True |
| row_scale_patch_reuses_i2s_type | True |
| row_scale_patch_changes_i2s_nbytes | True |

## Verdict

| claim | value |
| --- | --- |
| direct_dense_gguf_supported | True |
| direct_packed_i2s_supported | False |
| product_safe_row_scale_packed_supported | False |
| requires_python_gguf_i2s_support | True |
| requires_stable_row_scale_type_or_version | True |
| requires_special_row_scale_nbytes | True |

## Required Gates

1. Add Python GGUF constants for the packed ternary type being written.
2. Add file-type metadata for packed I2_S or a new row-scale ternary type.
3. Teach the Python writer/reader the special packed layout instead of assuming a fixed block type size is enough.
4. Define a compatibility-safe row-scale layout instead of overloading existing tensor-scale I2_S.
5. Write direct packed tensors from ternary codes plus scales, then load with llama-cli and run PPL/throughput/RSS audits.

## Interpretation

The C++ runtime and `llama-quantize` path know about `I2_S`, but the Python GGUF writer stack used for direct `ternary_state_dict.pt` export does not expose a compatible `I2_S` writer contract. More importantly, row-scale deployment needs a compatibility-safe row-scale layout or new qtype; the current prototype patch changes the existing `I2_S` payload. Therefore direct dense GGUF export is a real improvement, but direct packed row-scale GGUF export is not complete.
