# Static Ternary GGUF Bridge, 2026-05-05

This fork now has a reusable bridge runner:

```bash
python benchmarks/build_static_ternary_gguf_bridge.py \
  --checkpoint-dir checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000 \
  --expect-ternary-keys 196 \
  --run-label qwen15b_klonly_row_notie_static_ternary \
  --out-model-dir models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense \
  --results-dir benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite \
  --llama-bin-dir build-portable-avx2/bin \
  --threads 12 \
  --run-suite
```

The runner performs the current validated deployment bridge:

1. materialize `ternary_state_dict.pt` to dense HF tensors with
   `--validate-codes`,
2. convert the dense HF checkpoint to F16 GGUF with the vendored llama.cpp
   Qwen2 converter,
3. quantize the GGUF to `TQ2_0` and `I2_S` through
   `benchmarks/quantize_gguf_safe.py`,
4. write a GGUF manifest, and
5. optionally run `benchmarks/run_gguf_suite.py` plus `benchmarks/audit_evidence.py`.

`slurm_gguf_static_ternary_suite.sh` now delegates to this runner after ensuring
the portable AVX2 llama.cpp binaries exist.

## Verification

Dry-run command used during this update:

```bash
python benchmarks/build_static_ternary_gguf_bridge.py \
  --checkpoint-dir checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000 \
  --expect-ternary-keys 196 \
  --run-label dryrun_row_bridge \
  --out-model-dir /tmp/bitnet-dryrun-row-bridge \
  --dry-run \
  --run-suite
```

This verified command construction for materialization, F16 GGUF conversion,
`TQ2_0` quantization, `I2_S` quantization, suite execution, and evidence audit.

## Limitation

This is not a direct `ternary_state_dict.pt` GGUF writer. It still materializes
dense HF tensors before GGUF conversion. That limitation remains one of the
active gaps before a production-quality packed ternary exporter can be claimed.
