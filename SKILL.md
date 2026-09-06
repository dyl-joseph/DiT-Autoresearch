---
name: dit-autoresearch-a5500
description: Autonomous Diffusion Transformer inference performance research on NVIDIA RTX A5500/Ampere with immutable benchmarks, handbook-backed hypothesis generation, and strict no-quality-regression gates.
---

# DiT AutoResearch A5500 Skill

Use this skill when optimizing Diffusion Transformer inference on RTX A5500/Ampere hardware.

## Non-negotiable rules

- Read `program.md` before acting.
- Treat `knowledge/handbook/00_foundations/04_target_hardware_and_quality_contract.md` as the project override for generic hardware/precision advice.
- Quality is a hard constraint, not a weighted objective.
- Quantization may be tested only if the fixed benchmark suite shows **no required accuracy/quality regression**. Never keep a quantized candidate with a lower required benchmark score.
- Do not modify the benchmark corpus, quality evaluator, performance harness, or other `fixed_paths` during a session.
- Only edit paths listed in `mutable_paths`.
- Establish the unmodified baseline first.
- Compare candidate quality with the original baseline, not the previous candidate.
- Compare performance with the current best candidate in the branch's objective lane.
- One causal hypothesis per commit.
- Prefer exact implementation optimizations before numerical or algorithmic approximations.
- Measure on A5500. Do not import H100/B200 speedup numbers as expected A5500 results.
- FlashAttention-2/SDPA are relevant Ampere attention paths; FA3/FA4 and Hopper/Blackwell-specific features are research references, not target-native paths.

## Mandatory startup

1. Read `program.md` completely.
2. Read the following handbook files:
   - `knowledge/handbook/00_foundations/04_target_hardware_and_quality_contract.md`
   - `knowledge/handbook/00_foundations/02_benchmarking_and_quality_gates.md`
   - `knowledge/handbook/03_single_gpu/08_a5500_optimization_ladder.md`
   - `knowledge/handbook/02_bottlenecks/05_nsight_a5500_profiling_workflow.md`
   - `knowledge/handbook/09_reference/09_a5500_benchmark_runbook.md`
   - `knowledge/handbook/09_reference/10_reproducibility_and_claims_standard.md`
3. Run `dit-ar --config config/autoresearch.json doctor`.
4. Ensure the research branch is `autoresearch/<tag>` and the working tree is clean.
5. Run `dit-ar ... init` if this is a new session.
6. Run `dit-ar ... baseline` before changing optimized code.

## Knowledge retrieval

Before a non-trivial experiment, search the handbook:

```bash
dit-ar --config config/autoresearch.json search "<mechanism + bottleneck + model/hardware terms>"
```

Read the top relevant Markdown files. Use the handbook as the primary local research memory. When it lacks support for a hypothesis, label the idea as an external/new research hypothesis instead of pretending the handbook recommends it.

## Experiment classification

Use one of:

- `exact`: implementation/layout/scheduling/memory changes intended to preserve the algorithm;
- `numerical`: arithmetic mode changes such as FP16↔BF16 or TF32 behavior;
- `approximate`: changed/skipped model computation such as fewer steps, approximate caches, guidance cutoff, sparse approximation;
- `quantization`: any low-bit weight/activation/attention/storage path.

Quantization receives a strict zero-regression quality gate automatically.

## Keep/discard loop

For each hypothesis:

1. Inspect current best metrics and recent failures.
2. Search/read relevant handbook material.
3. State the hypothesis and expected bottleneck change.
4. Edit only mutable paths.
5. Run a cheap syntax/smoke check if useful.
6. Commit the candidate.
7. Run:

```bash
dit-ar --config config/autoresearch.json experiment \
  --classification <class> \
  --description "<short causal hypothesis>"
```

8. If **KEEP**, stay on the commit and continue.
9. If **DISCARD**, preserve the `.autoresearch/` logs, then `git reset --hard HEAD^` to return to the previous best.
10. If **CRASH**, fix only if the implementation is trivially wrong; otherwise discard and move on.
11. Continue until manually stopped.

## Research priorities on A5500

Favor, roughly in this order:

- end-to-end profiling and phase attribution;
- attention backend selection for exact shapes;
- compile/regional compile and graph-break removal;
- fusion of norms/modulation/pointwise operations;
- CUDA Graphs for stable shape buckets;
- GEMM algorithm/layout tuning;
- transfer elimination and component lifecycle scheduling;
- 24 GB VRAM engineering with offload/eviction/tiling;
- topology-aware two-A5500 strategies if available;
- custom Triton/CUDA/CUTLASS only after profiler evidence;
- approximations and quantization only behind the full quality firewall.

Never optimize a metric that the target product does not care about just because it is easy to improve.
