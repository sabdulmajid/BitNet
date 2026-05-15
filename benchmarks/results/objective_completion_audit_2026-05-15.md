# Objective Completion Audit, 2026-05-15

This audit maps the active user objective to concrete artifacts in this fork. It is stricter than the benchmark coverage gate and is not a success declaration.

## Success Criteria

1. Repaired ternary checkpoint export has the expected ternary key counts.

2. Fixed prompt sanity suites and quality benchmarks exist for the repaired dense-Qwen checkpoints.

3. WikiText, FineWeb-heldout, and ten-task EleutherAI lm-eval comparisons cover FP, naive PTQ, QAT, dense-head, and row-scale variants.

4. GGUF/TQ2_0/I2_S/I2_SR CPU paths measure quality, file size, throughput, and RSS on the Xeon.

5. Product claims are limited to what the active/default runtime supports, with TL2 and MoE/Kimi gaps called out.

## Verdict

Objective achieved: `False`.

Completion status: `not_complete`.

Complete rows: `7` / `9`.

The dense-Qwen negative result, row-scale recovery path, and stable I2_SR CPU route are well supported. The full objective is still incomplete because quality-preserving TL2 support for the strong row-scale checkpoint and trained/Kimi/ternary MoE evidence remain missing.

## Prompt-To-Artifact Checklist

| requirement | status | evidence | remaining gap |
| --- | --- | --- | --- |
| Fix FSDP ternary export bug and re-export Qwen2.5-1.5B step-5000; target 197 ternary linear keys, not 1 | complete | Qwen2.5-1.5B repaired hidden-MSE step-5000: ternary=197/197, scales=197, dense=True; Qwen2.5-0.5B step-1000: ternary=169/169, scales=169, dense=True; Qwen2.5-1.5B KL row-scale dense-head step-5000: ternary=196/196, scales=196, dense=True |  |
| Run fixed prompt suites for repaired 1.5B and complete 0.5B checkpoints | complete | Qwen2.5-1.5B repaired step-5000: 5 prompts; Qwen2.5-0.5B step-1000: 5 prompts |  |
| Add WikiText and FineWeb heldout perplexity for FP, naive PTQ, hidden-MSE QAT, KL QAT, dense-head, and row-scale variants | complete | 12/12 finite artifacts=True; FP WikiText=13.901475466625362; naive PTQ WikiText=3813121.80332679; best row-scale WikiText=38.58006540148406 |  |
| Add HellaSwag/PIQA/ARC and broader ten-task EleutherAI lm-eval comparisons with logged samples | complete | models=6, tasks/model=10, samples/model=22382; FP mean=0.644169; PTQ mean=0.348671; row-scale mean=0.499459 |  |
| Add baselines: original FP, naive PTQ, llama.cpp Q4_K_M/Q8_0, QAT with/without hidden MSE, row-scale versus tensor-scale | complete | row-FP=-0.144710 [-0.185756, -0.103664]; row-PTQ=+0.150788 [+0.053427, +0.248149]; row-tensor=+0.015081 [+0.009028, +0.021134]; row-KL=+0.016021 [+0.006145, +0.025897] |  |
| Measure Xeon model size, quality, prompt throughput, decode throughput, and RSS/context scaling | complete | I2_SR PPL=38.8477, file=1211.3 MiB, prompt=211.67 tok/s, decode=19.07 tok/s; Q4_K_M file=940.4 MiB; RSS contexts=[512, 2048, 8192, 32768] |  |
| Convert repaired checkpoints into GGUF/TL2/I2_S and run actual bitnet.cpp/llama.cpp CPU inference | partial | direct dense/scalar writers exist; I2_SR candidate gate=True; active default gate=True; TL2 row-scale one-scale error=1.9042302114103853; row-fp16 design error=0.00019744640689756221 at 1.23046875 MiB | Packed row-scale CPU inference is active through stable I2_SR for dense Qwen; TL2 quality-preserving row-scale Qwen1.5B still requires row/group-scale runtime and kernel support. |
| Evaluate MoE/Kimi feasibility including converter mapping, router/expert execution, quality, throughput, and locality | not_complete | generic MoE checks present=6; productization gates failed=3/9; Kimi artifacts=0; Kimi source matches=0; Kimi config supported=False; config gaps=4; tiny Qwen2MoE FP16 fixture passed=True; tiny Qwen2MoE ternary I2_SR fixture passed=True; synthetic expert scaling passed=True; scaling rows=4 | The Kimi config audit shows missing direct Kimi/DeepSeekV3 loading, MLA conversion metadata, shared-expert mapping, and block-FP8 import. Tiny random Qwen2MoE FP16 and ternary I2_SR GGUF fixtures validate generic converter/runtime plumbing, merged expert packing, and routed shape execution, but no validated Kimi-specific mapping, trained Qwen2MoE/Kimi quality artifact, TL2 MoE runtime support, router distillation, MoE quality run, trained throughput run, or trained expert-locality benchmark exists. |
| Produce side-by-side comparison, evidence manifest, prune plan, and honest novelty/product verdict | complete | manifest artifacts=225, missing=0; coverage=True checks=64; publishable ledger scopes negative result plus recovery path |  |

## Remaining Blockers

- Convert repaired checkpoints into GGUF/TL2/I2_S and run actual bitnet.cpp/llama.cpp CPU inference: Packed row-scale CPU inference is active through stable I2_SR for dense Qwen; TL2 quality-preserving row-scale Qwen1.5B still requires row/group-scale runtime and kernel support.

- Evaluate MoE/Kimi feasibility including converter mapping, router/expert execution, quality, throughput, and locality: The Kimi config audit shows missing direct Kimi/DeepSeekV3 loading, MLA conversion metadata, shared-expert mapping, and block-FP8 import. Tiny random Qwen2MoE FP16 and ternary I2_SR GGUF fixtures validate generic converter/runtime plumbing, merged expert packing, and routed shape execution, but no validated Kimi-specific mapping, trained Qwen2MoE/Kimi quality artifact, TL2 MoE runtime support, router distillation, MoE quality run, trained throughput run, or trained expert-locality benchmark exists.

## Practical Next Step

Keep the stable `I2_SR` route packaged and benchmarked for dense Qwen. Do not expand product claims to TL2 or MoE/Kimi until those paths have quality-valid CPU benchmark artifacts.
