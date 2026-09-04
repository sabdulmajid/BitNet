# BitDistill method-parity pilot submission

- Source revision: `bb8e92b`
- Pilot array: Slurm `10298` (`0-5%2`, `midcard`)
- Fail-closed postprocessor: Slurm `10299`, dependent on the complete array
- Per-pilot resources: 1 GPU, 12 CPUs, 24 GiB RAM, 6-hour limit
- Shared starting point: verified 655M-token Stage-2 manifest
- Scope: six 120-step MNLI diagnostics over at most 8,192 train and 512 validation examples

The matrix isolates three previously conflated choices: published Algorithm 1 cosine relations versus Equation 12 scaled-dot relations, one versus eight relation heads, and fixed versus gradient-balanced attention loss. It also compares the local sequence-classification and causal answer-token formulations without asserting that either is paper-exact.

These runs may select a method for full validation. They cannot establish a task-quality result because the validation subset and training budget are intentionally bounded.

The first submission attempt requested 32 GiB and was rejected before creating a job because the target node has 30,000 MiB physical memory. Commit `bb8e92b` changed only that scheduler request to 24 GiB, matching the existing proven wrapper.
