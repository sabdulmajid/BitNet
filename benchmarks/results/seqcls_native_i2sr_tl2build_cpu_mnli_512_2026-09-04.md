# Sequence-Classification Native CPU Benchmark, 2026-09-04

This benchmark evaluates one native GGUF artifact that contains the model backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, runtime parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | sample_only |
| task | mnli |
| CPU | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| threads | 12 |
| CPU affinity | 0-11 |
| GGUF MiB | 352.617432 |
| GGUF SHA256 | b8d285a6008750ab0852dc691d6d10f704cf7df7cf85ef212abf6c77249e13d7 |
| runtime build SHA256 | 308fe5562cf4619c74f191bc3b5cef0b480417b4ea963d30617100160178d109 |
| embedding binary SHA256 | 04ea725d39e66fb59f3de83a2ae469e2686fe4e6c777d5cebdf4d27a47fbb73a |
| BitNet revision | 19542e7e983b9509ecce53c0fe63a46d3bca210f |
| llama.cpp revision | a9d436e17165e6f59c875fa46eea226185cb346b |
| examples | 512 |
| expected examples | 9815 |
| full validation | false |
| accuracy | 0.667969 |
| accuracy CI95 | [0.626058, 0.707377] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.972656 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 64 |
| embedding sequential | true |
| batching parity ready | false |
| sequence-isolated parity ready | false |
| runtime parity ready | false |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 83.578957 |
| examples/sec | 6.125944 |
| tokens/sec | 237.527552 |
| child peak RSS MiB | 968.945312 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked by: status=sample_only, full_validation_incomplete, runtime_parity_not_ready, saved_pytorch_agreement=0.972656<0.99. Multi-prompt batched execution remains blocked by position-dependent drift in the I2_SR path; sequence-isolated mode is a separate mitigation.
