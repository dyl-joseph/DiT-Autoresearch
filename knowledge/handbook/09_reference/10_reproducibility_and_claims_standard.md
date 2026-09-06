# Reproducibility and Performance-Claim Standard

This handbook adopts a strict evidence standard inspired by the source policy in `wafer-ai/gpu-perf-engineering-resources` and makes it specific to Diffusion Transformer inference on RTX A5500.

The goal is to prevent three common failures:

1. copying performance multipliers from different GPUs;
2. calling an approximation “free” without model-level quality validation;
3. publishing a speed number without enough context to reproduce it.

---

## 1. Required fields for every performance claim

A claim such as “FA2 is 18% faster” or “compile saves 900 ms” is incomplete unless it specifies:

### Hardware

- exact GPU model;
- GPU count;
- topology for multi-GPU;
- power/clock policy if changed;
- memory capacity;
- device capability;
- thermally stable or cold-run state.

### Software

- NVIDIA driver;
- CUDA runtime/toolkit where relevant;
- PyTorch;
- Diffusers/runtime engine;
- attention/kernel library version or commit;
- compiler flags;
- custom extension commit.

### Workload

- model/revision;
- pipeline type;
- height/width;
- frames;
- prompt/token length;
- batch/concurrency;
- denoising steps;
- scheduler/solver;
- guidance/CFG mode;
- adapters/control modules;
- cache state.

### Precision and algorithm

- FP16/BF16/FP32/TF32 state;
- weight-storage format;
- quantization modules, if any;
- attention backend;
- compile mode;
- CUDA Graph state;
- approximate cache/step/guidance changes.

### Baseline

A claim must identify what it beats. Use the strongest relevant exact baseline, not a deliberately weak implementation.

### Correctness / quality method

- operator-level tolerance tests where relevant;
- model-level required benchmark suite;
- seeds/sample count;
- human evaluation protocol if used;
- acceptance thresholds;
- paired output archive for approximate changes.

Without these fields, record the observation as a hypothesis, not a performance result.

---

## 2. Hardware portability rule

A performance mechanism can transfer across architectures; a multiplier usually does not.

Example:

```text
Mechanism: FlashAttention reduces attention IO by tiling/recomputation.
Likely portable idea: yes.
H100 measured speedup: not transferable as an A5500 number.
A5500 action: benchmark FA2 vs SDPA on the exact production shapes.
```

The same applies to:

- Hopper TMA;
- Blackwell tensor memory;
- FP8/FP4 claims;
- NVSwitch scaling;
- HBM bandwidth conclusions;
- datacenter power envelopes.

The A5500 result must be measured on A5500.

---

## 3. No-regression quantization rule

Quantization has a special policy in this project:

> **If any required benchmark accuracy/quality metric regresses beyond the predeclared measurement-noise rule, the quantized candidate is rejected.**

This applies even if:

- latency improves dramatically;
- the model now fits one GPU;
- VRAM halves;
- throughput doubles;
- average visual quality looks similar.

A capacity problem should first be attacked with exact methods:

- phase-local component residency;
- prompt-embedding caching;
- text-encoder eviction;
- VAE tiling/chunking;
- sharded loading;
- second-GPU placement;
- exact attention memory reductions;
- compile/fusion.

Only then is quantization worth testing, and it still must pass the gate.

---

## 4. Exact versus approximate labels

Every optimization in experiment records should be labeled:

### Exact implementation optimization

Intended model algorithm/math is unchanged, subject only to normal floating-point implementation differences.

Examples:

- FA2 exact attention;
- PyTorch SDPA backend change;
- kernel fusion;
- `torch.compile`;
- CUDA Graph replay;
- layout elimination;
- component offload;
- model sharding;
- CFG branch parallel execution;
- memory allocator tuning.

### Numerical-mode change

Same high-level model, different arithmetic behavior.

Examples:

- FP16 ↔ BF16;
- TF32 enabling for FP32 GEMMs;
- low-bit storage with dequantization;
- weight/activation quantization.

These require model benchmark validation.

### Algorithmic approximation

Model computation itself is skipped/changed.

Examples:

- fewer denoising steps;
- approximate timestep/block caching;
- guidance cutoff;
- sparse/token-pruned attention;
- stale-state parallel methods;
- distilled model substitution.

These require the strongest quality evaluation.

The labels prevent exact kernel tuning and model-changing shortcuts from being presented as equivalent.

---

## 5. Minimum sample discipline

### Kernel microbenchmark

Use enough repetitions after warmup to report median and dispersion. Sweep all production-relevant shapes.

### Full image generation

Use a prompt × seed matrix. One image is not evidence.

### Video

Use multiple prompts/seeds and evaluate temporal metrics plus human playback review where temporal artifacts matter.

### Serving

Use enough requests to report percentiles and realistic arrival/concurrency patterns.

Report sample counts with every table.

---

## 6. Warm versus cold results

Always label:

- cold process startup;
- first request;
- compile/warmup;
- steady-state warm request;
- sustained/thermal steady state.

Do not compare a warm candidate to a cold baseline.

---

## 7. Memory claims

“Uses 18 GB” must specify:

- allocated or reserved;
- peak or persistent;
- which phase;
- whole request or isolated component;
- one rank or maximum across ranks;
- host RAM if offload is used.

For A5500, persistent resident weight memory and transient peak both matter because 24 GB leaves limited workspace headroom.

---

## 8. Multi-GPU claims

Every result must include:

- `nvidia-smi topo -m` summary;
- whether NVLink bridge is present/used;
- parallelism method;
- per-rank peak memory;
- latency;
- speedup;
- scaling efficiency;
- GPU-seconds/output;
- communication share if profiled.

A 1.7× speedup on 2 GPUs may be excellent for latency but more expensive per output. State both.

---

## 9. Quality-gate result schema

Recommended table:

| Benchmark | Baseline | Candidate | Delta | Allowed delta | Pass? |
|---|---:|---:|---:|---:|---|
| prompt adherence | | | | 0 / noise rule | |
| OCR/text | | | | 0 / noise rule | |
| face/anatomy | | | | 0 / noise rule | |
| edit locality | | | | 0 / noise rule | |
| temporal consistency | | | | 0 / noise rule | |
| product composite | | | | 0 / noise rule | |

Predeclare which rows are required. Do not remove a failing benchmark after seeing the result.

---

## 10. What to do with uncertain results

If repeated runs overlap heavily or quality metrics are noisy:

- increase samples;
- use paired seeds/inputs;
- report confidence intervals/bootstrap intervals where appropriate;
- avoid a hard performance claim until the result resolves;
- prefer the simpler exact configuration if the measured gain is within noise.

The documentation should say “no reproducible win observed,” not force a winner.

---

## 11. Claim wording

Prefer:

> On RTX A5500 x1, CUDA X, PyTorch Y, model revision Z, 1024²/28 steps/FP16, FA2 reduced warm p50 end-to-end latency from A to B across N runs, while the required quality suite produced no regressions.

Avoid:

> FlashAttention makes DiTs 2× faster.

The first is falsifiable and reusable. The second is context-free.

---

## 12. Evidence hierarchy for this handbook

Prefer sources in this order:

1. official hardware/software documentation and specifications;
2. original research paper introducing the mechanism;
3. official implementation repository;
4. direct implementer report with code and reproducible measurements;
5. this handbook's own A5500 measurements;
6. secondary explanations only for intuition.

Performance numbers from sources 1-4 still do not replace local A5500 benchmarking when the source hardware differs.

---

## 13. Experiment directory standard

```text
experiments/
  2026-09-05_flux_fa2_compile/
    README.md
    environment.txt
    config.json
    timings.csv
    memory.csv
    quality.csv
    nsys/
    ncu/
    outputs/
      baseline/
      candidate/
    conclusion.md
```

This makes later comparisons possible when drivers, PyTorch, or Diffusers change.

---

## 14. Final acceptance rule

A configuration enters the recommended A5500 path only when:

```text
quality passes
AND performance/resource target improves materially
AND result reproduces
AND operational complexity is acceptable
```

For quantization, `quality passes` is a hard prerequisite, not a weighted tradeoff.
