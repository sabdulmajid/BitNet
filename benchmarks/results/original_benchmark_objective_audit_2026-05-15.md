# Original Benchmark Objective Audit, 2026-05-15

This audit maps the original six requested benchmark deliverables to concrete artifacts. It does not include later MoE/Kimi scope.

| field | value |
| --- | --- |
| objective achieved | false |
| completion | 5/6 |
| coverage passed | true |
| coverage checks | 98 |

## Checklist

| item | status | evidence | remaining gap |
| --- | --- | --- | --- |
| 1. Fix FSDP ternary export and re-export 1.5B step-5000 | complete | Qwen2.5-1.5B repaired hidden-MSE step-5000: ternary=197/197, scales=197, dense=True; Qwen2.5-0.5B step-1000: ternary=169/169, scales=169, dense=True; Qwen2.5-1.5B KL row-scale dense-head step-5000: ternary=196/196, scales=196, dense=True |  |
| 2. Run eval/prompt suites on repaired 1.5B and complete 0.5B | complete | Qwen2.5-1.5B repaired step-5000: 5 prompts; Qwen2.5-0.5B step-1000: 5 prompts |  |
| 3. Add WikiText, FineWeb, HellaSwag/PIQA/ARC lm-eval | complete | 12/12 finite artifacts=True; FP WikiText=13.901475466625362; naive PTQ WikiText=3813121.80332679; best row-scale WikiText=38.58006540148406; models=6, tasks/model=10, samples/model=22382; FP mean=0.644169; PTQ mean=0.348671; row-scale mean=0.499459 |  |
| 4. Add FP/PTQ/Q4/Q8/QAT/row-vs-tensor baselines | complete | row-FP=-0.144710 [-0.185756, -0.103664]; row-PTQ=+0.150788 [+0.053427, +0.248149]; row-tensor=+0.015081 [+0.009028, +0.021134]; row-KL=+0.016021 [+0.006145, +0.025897] |  |
| 5. Convert repaired checkpoints into GGUF/TL2/I2_S and run CPU inference | partial | direct dense/scalar writers exist; I2_SR candidate gate=True; active default gate=True; TL2 row-scale one-scale error=1.9042302114103853; row-fp16 design error=0.00019744640689756221 at 1.23046875 MiB; TL2 ready=False; TL2 current row-scale error=1.9042302114103853; TL2 failed checks=9 | Dense GGUF and row-scale I2_SR/I2_S CPU inference exist, but TL2 is not quality-preserving for learned row-scale checkpoints until row/group-scale metadata and kernels are implemented. |
| 6. Measure Xeon speed, prompt throughput, RSS, model size, and quality loss | complete | I2_SR PPL=38.8477, file=1211.3 MiB, prompt=211.67 tok/s, decode=19.07 tok/s; Q4_K_M file=940.4 MiB; RSS contexts=[512, 2048, 8192, 32768]; I2_SR decode=19.065496 tok/s; Q4_K_M decode=16.013125 tok/s |  |

## TL2 Blockers

| blocker |
| --- |
| Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error. |
| `transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales. |
| The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor. |
| `GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype. |
| `ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group. |
| Generated TL2 qgemm multiplies by `Scales[0]`; row-scale support needs kernels that index the learned output-row or row-group scale. |
| The x86 TL2 dispatch passes the same `wt->scales` pointer into every generated qgemm call, matching the one-scale kernel contract. |
| The active BitNet/Qwen TL2 loader path does not create learned row-scale sidecar tensors; TL2 expects scale metadata inside the packed tensor/kernel path. |
| No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence. |

## Interpretation

The original benchmark objective is substantially satisfied for dense Qwen through the stable I2_SR path. It is not fully complete because the requested TL2 path is not quality-preserving for learned row-scale checkpoints; current TL2 collapses row scales to one tensor scale.
