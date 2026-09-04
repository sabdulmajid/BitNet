# Sequence-Classification Native CPU Benchmark, 2026-09-04

This benchmark evaluates one native GGUF artifact that contains the model backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, runtime parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | sample_only |
| task | mnli |
| CPU | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| threads | 12 |
| CPU affinity | 0-11 |
| GGUF MiB | 230.903564 |
| GGUF SHA256 | 2ea4723c829f496fc6f7c636560140934050cb7a129a409e8b5bf366e1e14e41 |
| runtime build SHA256 | 3f7d85a8faa9bbdf05e292b790ee133cf55cf5e445984cc544af12164fcf54e3 |
| embedding binary SHA256 | 21330513dd101afc4e8ed2170dca61ec3cb96727f154cf4ab68f73efd45a0d1c |
| BitNet revision | bb05e23fba9aecf3249fc34005d89fa72816a08f |
| llama.cpp revision | 7fe586546fef1aff17cddabc2ca262d3da4fba15 |
| examples | 512 |
| expected examples | 9815 |
| full validation | false |
| accuracy | 0.667969 |
| accuracy CI95 | [0.626058, 0.707377] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.980469 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 512 |
| embedding sequential | true |
| batching parity ready | false |
| sequence-isolated parity ready | true |
| runtime parity ready | true |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 127.809574 |
| examples/sec | 4.005960 |
| tokens/sec | 147.526154 |
| child peak RSS MiB | 957.042969 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked by: status=sample_only, full_validation_incomplete, saved_pytorch_agreement=0.980469<0.99. Multi-prompt batched execution remains blocked by position-dependent drift in the I2_SR path; sequence-isolated mode is a separate mitigation.
