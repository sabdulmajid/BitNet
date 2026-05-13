# Product Scope Gate, 2026-05-13

This gate separates what can be published or productized from what remains unsupported.

Scope status: `research_mvp_only`.

Publishable angle: negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path.

| claim | status | evidence | scope | blocker |
| --- | --- | --- | --- | --- |
| One-click lossless arbitrary FP/BF16-to-ternary retrofit | unsupported | Qwen1.5B FP mean=0.644169; naive PTQ mean=0.348671; FP WikiText PPL=13.901475466625362; PTQ WikiText PPL=3813121.80332679 | Do not market as arbitrary lossless conversion. | Blind PTQ destroys quality in both math and model-level artifacts. |
| Dense Qwen negative result plus QAT/distillation recovery path | supported | row-scale mean=0.499459; row-PTQ paired delta=+0.150788 [+0.053427, +0.248149]; row-FP paired delta=-0.144710 [-0.185756, -0.103664]; row WikiText PPL=38.58006540148406 | Qwen2.5 dense 1.5B evidence in this fork. |  |
| CPU row-scale ternary inference for dense Qwen through fixed I2_SR candidate | supported_with_patch | I2_SR PPL=38.8477; file=1211.3 MiB; prompt=211.67 tok/s; decode=19.07 tok/s; patch gate=True | Downstream `patches/llama-i2sr-row-scale-qtype.patch` applied to the audited source state. |  |
| Default committed runtime supports stable row-scale I2_SR | unsupported | active productization gate passed=False; failed_gates=3 | Do not claim default/submodule support yet. | The committed submodule lacks the separate qtype/file type/runtime routing without applying the downstream patch. |
| Direct packed row-scale GGUF export is product-safe by default | unsupported | direct packed verdict=False; candidate_i2sr_quality_valid=True | Direct writer can be used for controlled experiments; product claims need stable qtype support. | The safe row-scale path is still tied to the downstream I2_SR candidate patch. |
| TL2 product support for the strong row-scale Qwen checkpoint | unsupported | Qwen0.5B TL2 PPL=NaN; Qwen1.5B row-scale one-scale error=1.9042302114103853; group2 fp16 design error=0.0986922473661489; group32 fp16 design error=0.1428438042225288; exact row-fp16 design error=0.00019744640689756221 at 1.23046875 MiB | Exclude TL2 from MVP claims. | Current TL2 scale semantics are incompatible with row-scale checkpoint quality; fixing it requires row/group-scale metadata and generated kernels that index those scales. |
| MoE/Kimi retrofit and CPU runtime support | unsupported | Kimi artifacts=0; Kimi source matches=0; failed MoE gates=3/6; TL2 MoE runtime ready=False; TL2 expert byte underreport=69632 | Treat as separate research milestone. | No Kimi-specific mapping, real Qwen2MoE/Kimi conversion artifact, TL2 3D expert packing/runtime support, router distillation, quality benchmark, or expert-locality benchmark exists; direct I2_S/I2_SR expert packing is only synthetic so far. |

## Recommendation

Product: CPU-first dense-Qwen retrofit evaluator with explicit downstream I2_SR runtime patch requirement.

Paper: Scope as a negative PTQ result plus measured distillation/row-scale/runtime recovery path; do not claim arbitrary or MoE support.

Next engineering gate: Make I2_SR active/default or keep it clearly as a patch-distribution runtime.
