# Sequence-Classification Native CPU Benchmark, 2026-09-04

This benchmark evaluates one native GGUF artifact that contains the model backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, runtime parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | sample_only |
| task | mnli |
| CPU | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| threads | 12 |
| CPU affinity | 0-11 |
| GGUF MiB | 948.109589 |
| GGUF SHA256 | 293431e5689485dd0bb66deb331ed1b7c912925bc92544253b4666be3290ed9c |
| runtime build SHA256 | 3f7d85a8faa9bbdf05e292b790ee133cf55cf5e445984cc544af12164fcf54e3 |
| embedding binary SHA256 | 21330513dd101afc4e8ed2170dca61ec3cb96727f154cf4ab68f73efd45a0d1c |
| BitNet revision | bb05e23fba9aecf3249fc34005d89fa72816a08f |
| llama.cpp revision | 7fe586546fef1aff17cddabc2ca262d3da4fba15 |
| examples | 512 |
| expected examples | 9815 |
| full validation | false |
| accuracy | 0.789062 |
| accuracy CI95 | [0.751638, 0.822182] |
| stored PyTorch accuracy | 0.807641 |
| agreement with saved PyTorch predictions | - |
| label agreement with saved trace | - |
| prompt input | token_ids |
| prompt batch size | 512 |
| embedding sequential | true |
| batching parity ready | false |
| sequence-isolated parity ready | true |
| runtime parity ready | true |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 53.253263 |
| examples/sec | 9.614434 |
| tokens/sec | 357.468452 |
| child peak RSS MiB | 1016.878906 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked by: status=sample_only, full_validation_incomplete, missing_saved_pytorch_agreement. Multi-prompt batched execution remains blocked by position-dependent drift in the I2_SR path; sequence-isolated mode is a separate mitigation.
