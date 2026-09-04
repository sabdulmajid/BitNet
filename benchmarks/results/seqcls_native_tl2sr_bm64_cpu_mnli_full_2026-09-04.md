# Sequence-Classification Native CPU Benchmark, 2026-09-04

This benchmark evaluates one native GGUF artifact that contains the model backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, runtime parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | pass |
| task | mnli |
| CPU | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz |
| threads | 12 |
| CPU affinity | 0-11 |
| GGUF MiB | 341.494995 |
| GGUF SHA256 | d66dcf79a71d8855116e7987727195ab0922acf7421dbeb536fc8ef4aaa09f74 |
| runtime build SHA256 | 308fe5562cf4619c74f191bc3b5cef0b480417b4ea963d30617100160178d109 |
| embedding binary SHA256 | 04ea725d39e66fb59f3de83a2ae469e2686fe4e6c777d5cebdf4d27a47fbb73a |
| BitNet revision | 19542e7e983b9509ecce53c0fe63a46d3bca210f |
| llama.cpp revision | a9d436e17165e6f59c875fa46eea226185cb346b |
| examples | 9815 |
| expected examples | 9815 |
| full validation | true |
| accuracy | 0.652878 |
| accuracy CI95 | [0.643402, 0.662235] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.978808 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 64 |
| embedding sequential | true |
| batching parity ready | false |
| sequence-isolated parity ready | false |
| runtime parity ready | false |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 1741.600349 |
| examples/sec | 5.635621 |
| tokens/sec | 216.471400 |
| child peak RSS MiB | 974.757812 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked by: runtime_parity_not_ready, saved_pytorch_agreement=0.978808<0.99. Multi-prompt batched execution remains blocked by position-dependent drift in the I2_SR path; sequence-isolated mode is a separate mitigation.
