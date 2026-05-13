# Direct Static-Ternary GGUF Bridge, 2026-05-13

This note records a narrower direct-GGUF improvement: `ternary_state_dict.pt`
can now be used as the source for dense GGUF export without first writing a
materialized Hugging Face `model.safetensors` directory.

The implementation is
`benchmarks/convert_static_ternary_to_gguf.py`. It reconstructs effective dense
weights in memory from `*.ternary_weight * *.weight_scale`, reuses the vendored
llama.cpp GGUF metadata/tokenizer/tensor-name writer, and writes dense F16/BF16
or F32 GGUF.

## Validation

| checkpoint | architecture | outtype | ternary tensors | copied tensors | GGUF tensors | GGUF size | validation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `checkpoints/eval-ternary-tiny` | LlamaForCausalLM | F16 | 8 | 4 | 12 | 25,357,088 bytes | `llama-gguf ... r n` read 25 metadata keys and 12 tensors |
| `checkpoints/qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-1000` | Qwen2ForCausalLM | F16 | 168 | 123 | 291 | 1,266,423,040 bytes | `llama-gguf ... r n` read 23 metadata keys and 291 tensors; `llama-cli` smoke returned 0 |

Qwen smoke output:

```text
the capital of the capital. The capital is the capital of the capital. The
```

The smoke text is low quality, matching the known weak 0.5B ternary checkpoint;
the validation target here is GGUF structural correctness and runtime load, not
new model quality.

## Negative Finding

Direct `TQ2_0` through the Python GGUF converter is not solved by this path.
When tried on the Qwen2.5-0.5B dense-head checkpoint, the converter emitted
shape warnings and fell back to F16 for the Qwen matrices. The direct converter
therefore blocks quantized outtypes by default unless
`--allow-converter-quantized-outtype` is passed for experiments.

## Remaining Gap

This closes the intermediate-HF-directory part of the bridge for dense GGUF
export. It does not close the final packed runtime gap: production deployment
still needs a direct row-scale-aware packed GGUF writer and stable GGUF type for
`I2_S` or an equivalent ternary CPU-native layout.
