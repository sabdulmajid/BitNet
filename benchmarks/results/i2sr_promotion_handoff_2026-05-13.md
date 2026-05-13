# I2_SR Promotion Handoff, 2026-05-13

This report turns the remaining llama.cpp Git blocker into an executable handoff. The default script mode is non-mutating; it only prepares a worktree/commit or pushes when explicitly requested.

Fork URL: `https://github.com/sabdulmajid/llama.cpp.git`.

Branch: `i2sr-row-scale-runtime`.

## Preflight

| check | value |
| --- | --- |
| root clean | `true` |
| submodule clean | `true` |
| root patch applies | `true` |
| submodule patch applies | `true` |
| candidate fork reachable | `false` |
| ready for handoff | `false` |

## Blockers

| blocker |
| --- |
| Candidate llama.cpp fork URL is not reachable. |

## Prepare Submodule Branch

```bash
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet/3rdparty/llama.cpp && git worktree add -b i2sr-row-scale-runtime /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/i2sr-promotion-llama-worktree HEAD)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/i2sr-promotion-llama-worktree && git apply /mnt/slurm_nfs/a6abdulm/projects/BitNet/patches/llama-i2sr-row-scale-qtype.submodule.patch)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/i2sr-promotion-llama-worktree && git add ggml gguf-py include src)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/i2sr-promotion-llama-worktree && git commit -m 'Add I2_SR row-scale runtime')
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/i2sr-promotion-llama-worktree && git push https://github.com/sabdulmajid/llama.cpp.git HEAD:refs/heads/i2sr-row-scale-runtime)
```

## Update Superproject Pointer

```bash
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git config -f .gitmodules submodule.3rdparty/llama.cpp.url https://github.com/sabdulmajid/llama.cpp.git)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git submodule sync 3rdparty/llama.cpp)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git -C 3rdparty/llama.cpp fetch https://github.com/sabdulmajid/llama.cpp.git i2sr-row-scale-runtime)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git -C 3rdparty/llama.cpp checkout FETCH_HEAD)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git apply patches/bitnet-i2sr-root-runtime.patch)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git add .gitmodules 3rdparty/llama.cpp src/ggml-bitnet-mad.cpp)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && git commit -m 'Promote I2_SR row-scale runtime')
```

## Post-Promotion Gates

```bash
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && python benchmarks/run_i2sr_active_patch_gate.py)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && python benchmarks/audit_i2sr_submodule_promotion.py --check-remote-write --candidate-fork-url https://github.com/sabdulmajid/llama.cpp.git)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && python benchmarks/audit_product_scope.py)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && python benchmarks/audit_objective_completion.py)
(cd /mnt/slurm_nfs/a6abdulm/projects/BitNet && python benchmarks/build_evidence_manifest.py)
```

## Worktree Result

| field | value |
| --- | --- |
| prepared | `n/a` |
| pushed | `n/a` |
| commit | `` |
| worktree | `` |
