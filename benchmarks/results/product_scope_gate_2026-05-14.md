# Product Scope Gate, 2026-05-14

This gate separates what can be published or productized from what remains unsupported.

Scope status: `research_mvp_only`.

Publishable angle: negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path.

| claim | status | evidence | scope | blocker |
| --- | --- | --- | --- | --- |
| One-click lossless arbitrary FP/BF16-to-ternary retrofit | unsupported | Qwen1.5B FP mean=0.644169; naive PTQ mean=0.348671; FP WikiText PPL=13.901475466625362; PTQ WikiText PPL=3813121.80332679 | Do not market as arbitrary lossless conversion. | Blind PTQ destroys quality in both math and model-level artifacts. |
| Dense Qwen negative result plus QAT/distillation recovery path | supported | row-scale mean=0.499459; row-PTQ paired delta=+0.150788 [+0.053427, +0.248149]; row-FP paired delta=-0.144710 [-0.185756, -0.103664]; row WikiText PPL=38.58006540148406 | Qwen2.5 dense 1.5B evidence in this fork. |  |
| CPU row-scale ternary inference for dense Qwen through stable I2_SR | supported | I2_SR PPL=38.8477; file=1211.3 MiB; prompt=211.67 tok/s; decode=19.07 tok/s; active gate=True; patch gate=True | Dense Qwen2.5-1.5B I2_SR evidence in this fork on Intel Xeon Silver 4116. |  |
| BitDistill task-specific packed I2_SR runtime support | unsupported | causal I2_SR gate passed=False; complete rows=0/6; gate=benchmark_results/bitdistill_i2sr_export_gate_2026-05-14.json | Only causal prompt-scoring BitDistill checkpoints can use this packed path; sequence-classification heads remain PyTorch-only unless runtime support is added. | missing export row; missing export summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/export_summary.json; missing memory summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/memory/summary.json; missing suite summary benchmark_results/bitdistill-causal-longwarmup-i2sr-2026-05-14/gguf_suite/summary.json |
| Default committed runtime supports stable row-scale I2_SR | supported | active productization gate passed=True; promotion_ready=True; failed_gates=0 | Stable I2_SR is available in the active source state when this gate is supported. |  |
| Direct packed row-scale GGUF export is product-safe by default | supported | direct packed verdict=True; candidate_i2sr_quality_valid=True | Direct writer can be used for the audited dense-Qwen I2_SR path when this gate is supported. |  |
| TL2 product support for the strong row-scale Qwen checkpoint | unsupported | Qwen0.5B TL2 PPL=NaN; Qwen1.5B row-scale one-scale error=1.9042302114103853; group2 fp16 design error=0.0986922473661489; group32 fp16 design error=0.1428438042225288; exact row-fp16 design error=0.00019744640689756221 at 1.23046875 MiB | Exclude TL2 from MVP claims. | Current TL2 scale semantics are incompatible with row-scale checkpoint quality; fixing it requires row/group-scale metadata and generated kernels that index those scales. |
| MoE/Kimi retrofit and CPU runtime support | unsupported | Kimi artifacts=0; Kimi source matches=0; failed MoE gates=3/6; TL2 MoE runtime ready=False; TL2 expert byte underreport=0 | Treat as separate research milestone. | No Kimi-specific mapping, real Qwen2MoE/Kimi conversion artifact, TL2 3D expert packing/runtime support, router distillation, quality benchmark, or expert-locality benchmark exists; direct I2_S/I2_SR expert packing is only synthetic so far. |

## Recommendation

Product: CPU-first dense-Qwen retrofit evaluator with stable I2_SR runtime support; keep claims limited to distilled dense Qwen until MoE evidence exists.

Paper: Scope as a negative PTQ result plus measured distillation/row-scale/runtime recovery path; do not claim arbitrary or MoE support.

Next engineering gate: Validate release packaging and keep MoE/Kimi as a separate milestone.
