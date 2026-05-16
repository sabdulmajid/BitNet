# Sequence-Classification Runtime Implementation Plan, 2026-05-15

Do not call the sequence-classification artifact deployed yet. The sidecar proves the path is plausible, but native GGUF score-head metadata, runtime pooling/head execution, full validation accuracy, RSS, and batching parity are still required.

## Current Evidence

| field | value |
| --- | --- |
| status | sidecar_qwen_contract_available_native_head_blocked |
| same artifact quality+CPU ready | false |
| seqcls configs | 15 |
| seqcls causal-export compatible | 0 |
| sidecar status | prototype_smoke_passed |
| sidecar sampled accuracy | 0.609375 |
| sidecar agreement with PyTorch predictions | 0.914062 |
| hidden relative RMS | 0.108662 |
| hidden cosine | 0.994091 |
| bitnet-qwen graph available | true |
| ready to productize | false |

## Source-Owned Plan

| step | files | work | exit gate |
| --- | --- | --- | --- |
| 1. Define classifier GGUF contract | benchmarks/convert_static_ternary_to_i2s_gguf.py, 3rdparty/llama.cpp/gguf-py/gguf/constants.py, 3rdparty/llama.cpp/src/llama.cpp | Persist score-head tensors and label metadata in GGUF instead of an NPZ sidecar. Record num_labels, label order, pooling policy, and problem type. | GGUF reader lists classifier head tensors and metadata without requiring sidecar files. |
| 2. Add native Qwen sequence-classification graph path | 3rdparty/llama.cpp/src/llama.cpp, 3rdparty/llama.cpp/src/llama-model.cpp, 3rdparty/llama.cpp/src/llama-graph.cpp | Reuse the audited `bitnet-qwen` decoder semantics, then pool the last non-padding token and apply the dense score head in the runtime. | Single-sample logits match PyTorch/sidecar logits within a strict relative RMS threshold. |
| 3. Implement a CPU GLUE classifier evaluator | benchmarks/benchmark_seqcls_i2sr_sidecar_cpu.py, benchmarks/benchmark_bitdistill_glue_cpu.py, 3rdparty/llama.cpp/examples | Replace the Python sidecar loop with native GGUF classifier inference and preserve the same prompt/tokenization contract used by PyTorch validation. | Full MNLI validation runs from one native GGUF artifact and reports accuracy, RSS, and examples/sec. |
| 4. Prove batching parity | benchmarks/audit_seqcls_i2sr_sidecar_batching.py, benchmarks/audit_seqcls_runtime_gap.py | Native batching must preserve predictions relative to batch size 1. Existing separator batching changed `3/64` sidecar predictions, so it is not a safe semantic reference. | Batch-size 1 and production batch size have identical predictions on a fixed validation subset. |
| 5. Promote quality/runtime gate | benchmarks/audit_seqcls_runtime_gap.py, benchmarks/audit_benchmark_coverage.py, README.md | Only after native logits, full-split accuracy, RSS, and throughput are present should the sequence-classification product status move from prototype to deployed artifact. | same_artifact_quality_cpu_ready=true and coverage gate passes with native classifier rows. |

## Completion Criteria

This blocker closes only when a single packed classifier artifact carries the decoder, score head, labels, and pooling contract; produces native CPU logits matching PyTorch; runs full GLUE validation; and reports accuracy, RSS, and throughput from the same GGUF.
