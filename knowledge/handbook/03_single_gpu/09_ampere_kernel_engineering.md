# Ampere Kernel Engineering for DiT Inference on RTX A5500

This chapter turns general GPU performance-engineering material into an RTX A5500-specific workflow for Diffusion Transformer inference. It is intentionally lower-level than the rest of the single-GPU section.

The core principle is simple:

> **Optimize the measured bottleneck on the hardware you actually own, while preserving model outputs and benchmark quality.**

For this project, the hardware is RTX A5500, not H100/B200. The primary arithmetic target is FP16/BF16 Tensor Core execution. The primary attention candidates are PyTorch SDPA and FlashAttention-2. Kernel changes should be exact whenever possible; quantized/approximate kernels are optional experiments only after the exact path is strong.

---

## 1. What a DiT step looks like to the GPU

A visual diffusion transformer step usually contains repeated classes of operations:

- Q/K/V linear projections;
- attention score/value products and softmax;
- output projection;
- MLP/FFN GEMMs;
- normalization;
- modulation/scale/shift operations;
- residual additions;
- timestep/conditioning projections;
- layout transforms, casts, copies, and pointwise glue.

At the Python level this looks like a transformer block. At the GPU level it is a sequence of kernels with different bottlenecks.

A typical profile may include:

- very large GEMMs that are compute-bound;
- attention kernels with substantial memory traffic and synchronization;
- tiny normalization/modulation kernels that are launch/latency-bound;
- copies or reformat kernels introduced by layout mismatch;
- allocator and host synchronization gaps between GPU launches.

There is no single “DiT bottleneck.” There is a bottleneck distribution that changes with resolution, frame count, batch/CFG shape, prompt length, compiler state, and attention backend.

---

## 2. Start from the roofline, not from folklore

The Wafer performance-engineering resource map puts roofline/arithmetic-intensity reasoning near the beginning for a reason. Before rewriting a kernel, determine whether more ALU/Tensor Core work or fewer bytes moved is likely to matter.

A rough decision:

```text
low arithmetic intensity  -> memory/IO optimization first
high arithmetic intensity -> compute/tensor-core utilization first
very small kernel          -> launch/fusion/occupancy overhead first
```

For A5500, use its actual memory bandwidth and measured achievable compute rather than importing H100 roofline thresholds. Peak specifications are ceilings, not benchmark results.

Measure arithmetic intensity from the operation actually executed, including conversions, temporary writes, and extra passes. A supposedly “cheap” post-op can materially reduce effective intensity if it forces a large tensor to leave registers/shared memory and come back through global memory.

---

## 3. GEMM: first-class DiT work

Linear projections and MLPs map to GEMM. Before writing a custom matmul, make sure the library path is not being sabotaged by shape/layout choices.

Check:

- dtype is FP16 or BF16 as intended;
- dimensions are favorable for Tensor Core tiling;
- tensors are contiguous or in the layout expected by the selected kernel;
- no hidden FP32 upcast is occurring in the hot path;
- no repeated transpose/materialization is inserted around GEMM;
- batch/sequence flattening produces large enough matrices to amortize launch overhead;
- the same shapes recur often enough for autotuning/compilation to pay off.

cuBLAS/cuBLASLt and compiler-selected kernels should be the baseline. CUTLASS/Triton/custom CUDA is justified only when the production shape is poorly served by the library or when fusion can remove materialized intermediates.

### Why a custom GEMM can lose

A custom implementation can have attractive occupancy and still lose because:

- global loads are not coalesced;
- register tiling is poor;
- shared-memory bank conflicts serialize access;
- epilogue writes/readbacks add traffic;
- Tensor Core instructions are underfed;
- tile shape mismatches the exact M/N/K;
- register pressure causes spills;
- synchronization cost is too high.

Do not optimize occupancy as an isolated objective. The classic high-performance GEMM literature repeatedly shows that lower nominal occupancy can still win when each resident warp does more useful work and hides latency effectively.

---

## 4. Tensor Core discipline on Ampere

Ampere has third-generation Tensor Cores and strong FP16/BF16 matrix throughput. Use them deliberately.

Practical checks:

1. Verify the model is actually running the intended FP16/BF16 GEMMs.
2. Inspect Nsight Compute Tensor Core utilization metrics for hot GEMMs.
3. Keep matrix dimensions and alignments friendly to the selected kernels where model architecture permits.
4. Avoid gratuitous casts around every block.
5. Separate **storage dtype** from **compute dtype** in your reasoning.

A low-bit weight file does not imply low-bit Tensor Core execution. On A5500, many compressed formats require unpack/dequantize into FP16/BF16 before the multiply. That may save residency or bandwidth, but it can also add instructions and temporary buffers. Measure it and apply the benchmark quality gate.

### TF32

TF32 can accelerate FP32 matrix operations on Ampere by using Tensor Cores with reduced mantissa precision. It is not relevant to a pipeline already fully executing FP16/BF16 GEMMs, but may matter for accidental/intentional FP32 subgraphs.

Treat TF32 as an output-affecting numerical mode: compare the full benchmark suite if enabling it changes the arithmetic path.

---

## 5. Memory coalescing and layout

Many “mysterious” kernels are layout conversions or pointwise operations reading large tensors inefficiently.

For custom CUDA/Triton code:

- map adjacent lanes to adjacent addresses when possible;
- avoid strided per-thread global access across the fastest-changing dimension;
- vectorize loads/stores when alignment and dtype permit;
- use shared memory to reorganize data when the reuse justifies the extra copy;
- watch for bank conflicts after moving data into shared memory;
- avoid writing an intermediate to global memory if the next operation can consume it inside the same fused kernel.

For framework-level code:

- check `.is_contiguous()` and stride patterns at block boundaries;
- use profiler traces to identify `contiguous`, transpose-copy, cast-copy, or reformat kernels;
- prefer a consistent layout across repeated blocks;
- avoid forcing copies just to satisfy one minor operation if a backend supports strided input directly.

A layout fix is especially attractive because it can be exact and can remove work without changing model math.

---

## 6. Shared memory on compute capability 8.x

Ampere exposes a combined L1/shared-memory architecture and supports asynchronous copies from global to shared memory. Custom kernels can use shared memory to tile data and reduce external-memory traffic.

But shared memory is not automatically faster. The tile must provide reuse or a better access pattern.

Failure modes:

- allocating so much shared memory that occupancy collapses;
- bank conflicts;
- extra barriers that offset reuse;
- copying data into shared memory that is only used once;
- choosing a tile tuned for A100/Hopper rather than the workstation Ampere shape/resource limits.

Use the installed Ampere tuning guide and query the actual device properties. Do not transplant Hopper TMA schedules: A5500 does not provide Hopper's TMA programming model.

---

## 7. Registers: the hidden occupancy tax

Fused DiT kernels often become register-heavy. Register pressure matters because it can reduce active warps or spill to local memory.

When comparing kernel variants, record:

- registers/thread;
- achieved occupancy;
- spill loads/stores;
- local-memory traffic;
- shared-memory usage;
- issue/eligible warp metrics;
- Tensor Core or FP pipeline utilization.

A fusion that saves one global-memory round trip but causes severe spilling can lose. Conversely, a kernel with lower occupancy may still be faster if it performs far fewer memory operations.

Use Nsight Compute to decide; do not infer from source-code complexity.

---

## 8. Pointwise fusion in transformer blocks

DiTs contain many bandwidth/launch-heavy elementwise operations around the main GEMMs and attention:

- bias;
- residual add;
- scale/shift modulation;
- gates;
- activation functions;
- normalization post-processing;
- dtype conversion.

The best fusion opportunities often occur in GEMM epilogues or adjacent pointwise chains.

Example conceptual transformation:

```text
GEMM -> write Y
read Y -> add bias -> write Z
read Z -> activation -> write A
read A -> residual -> write B
```

becomes:

```text
GEMM + bias + activation + residual -> one final write
```

when dependencies and numerical semantics allow it.

Benefits:

- fewer kernel launches;
- fewer global reads/writes;
- less synchronization;
- better cache locality.

Risks:

- register pressure;
- compiler graph break elsewhere;
- different reduction order/numerics;
- harder debugging;
- shape-specialization explosion.

`torch.compile` should be tried before hand-writing these fusions. Inspect what it generates; hand-written Triton/CUDA is the next step only for persistent hot patterns that the compiler misses.

---

## 9. Normalization kernels

LayerNorm/RMSNorm-like kernels are small relative to GEMMs but repeated many times. Once GEMM and attention are fast, normalization can become visible.

Optimization options:

- fuse normalization with adjacent scale/shift/gating where semantics permit;
- use a known optimized implementation;
- keep reduction local to a block/warp appropriate for hidden size;
- avoid extra precision conversions;
- avoid separate mean/variance materialization.

The online-normalizer/softmax literature is useful because it illustrates a broader GPU principle: reformulate reductions so you do not materialize large intermediates unnecessarily.

---

## 10. Exact attention on A5500

Attention deserves its own backend chapter, but at kernel level the key A5500 rule is:

> **Benchmark exact PyTorch SDPA and FlashAttention-2 first.**

FlashAttention's important insight is IO-awareness: compute tiles of attention without materializing the full score/probability matrices in external memory.

For visual DiTs, sweep:

- noncausal attention;
- sequence lengths produced by each image/video bucket;
- head dimensions;
- FP16 versus BF16;
- joint text+image sequence lengths for MMDiT;
- batch 1 and CFG batch shapes;
- mask/no-mask cases.

FlashAttention-3 is optimized for Hopper and FlashAttention-4 for newer Hopper/Blackwell-class paths; those are not the target recommendation here.

If a custom attention candidate changes arithmetic (quantized attention, sparse approximation, cached/stale states), it moves from “kernel optimization” to “model-quality-sensitive optimization” and must pass the no-regression suite.

---

## 11. Triton versus CUDA versus CUTLASS

### Stay in PyTorch/compiler when

- the graph compiles cleanly;
- generated kernels are close to the hardware limit;
- maintainability matters more than the last few percent;
- shapes vary enough that hand-specialization would explode.

### Use Triton when

- the operation is naturally block/tile expressed;
- you need a custom fusion;
- you want fast iteration/autotuning;
- the kernel does not require an exotic low-level feature unavailable in Triton.

### Use CUTLASS/CuTe when

- GEMM/attention tiling is central;
- you need precise control over matrix layouts and Tensor Core atoms;
- a reusable templated kernel family is worth the engineering complexity.

### Use handwritten CUDA when

- you need exact low-level scheduling/control;
- the framework/compiler cannot express a critical optimization;
- the hot kernel is stable enough to justify maintenance;
- profiler evidence says the potential gain is material.

The tool is not the optimization. The memory traffic, work partition, synchronization, and instruction schedule are the optimization.

---

## 12. Compile for the real architecture

Custom extensions should include the actual target architecture rather than relying on generic PTX forever.

At runtime, verify:

```python
import torch
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
```

For an A5500-class workstation Ampere device, build the matching SM target returned by the machine. Rebuild if the deployment GPU changes.

Why this matters:

- the compiler can select architecture-specific instructions and scheduling;
- generic code may miss throughput improvements;
- kernels tuned around a different register/shared-memory/Tensor Core profile can perform poorly.

Keep the binary and source revision in benchmark metadata.

---

## 13. CUDA Graphs for repeated denoising

A DiT performs many structurally similar forward passes. CUDA Graph capture can reduce CPU launch overhead when shapes, addresses, and control flow are stable.

Good candidates:

- fixed resolution/frame bucket;
- fixed CFG batch mode;
- static transformer shapes;
- preallocated latents/conditioning buffers;
- no dynamic Python branch inside the captured region.

Poor candidates:

- arbitrary shape every request;
- changing adapter structure;
- host-device copies with dynamic allocation inside capture;
- attention backend that performs unsupported dynamic behavior;
- frequent recompilation.

Graph capture is exact when it replays the same operations. It should still pass output checks because capture/in-place buffer changes can expose bugs.

---

## 14. Kernel launch overhead and the “fast model” inversion

A slow denoiser is dominated by large kernels. As you optimize it, the percentage spent in tiny operations and Python dispatch rises.

The optimization sequence often looks like:

```text
baseline: GEMM/attention dominate
  -> exact attention backend improves
  -> compile fuses pointwise ops
  -> big kernels get faster
  -> scheduler/launch gaps become visible
  -> CUDA Graph / deeper fusion now matters
```

This is why profiling must be repeated after major changes. The original bottleneck can disappear.

---

## 15. Benchmark custom kernels correctly

### Warm first

JIT compilation, autotune, allocator setup, and cache population can distort the first launches.

### Synchronize

CUDA is asynchronous. Use CUDA events for GPU-kernel timing or synchronize around wall-clock regions.

### Sweep production shapes

Do not tune only one synthetic matrix size.

### Measure distributions

For end-to-end workloads, keep p50/p90/p95 and variance, not just minimum latency.

### Compare against a strong baseline

For a custom GEMM, compare against the exact library path the model actually uses, not a deliberately naive kernel.

### Include correctness

Use `torch.testing.assert_close`, stress shapes, and Compute Sanitizer where appropriate before accepting timing results.

### Then run model benchmarks

An operator can be numerically close while still changing a sensitive diffusion trajectory enough to alter the benchmark. Model-level validation remains mandatory.

---

## 16. Nsight Systems: find the time

Use Nsight Systems first when you do not yet know which class of cost matters.

Questions to answer:

- Is the GPU continuously busy?
- Are there CPU gaps between launches?
- Is every denoising step copying data from host?
- Is attention a few giant kernels or many fragmented kernels?
- Does VAE decode dominate the tail?
- Are two GPUs communicating through the intended link?
- Are synchronization calls serializing work?
- Is compilation happening unexpectedly during steady state?

Add NVTX ranges around:

```text
prompt_encode
step_000 ... step_N
transformer
attention
vae_decode
postprocess
```

A good timeline tells you **where** to look next.

---

## 17. Nsight Compute: explain the kernel

Once a hot kernel is identified, Nsight Compute can answer why it is slow.

Useful categories:

- achieved memory throughput;
- L1/L2 hit behavior;
- DRAM transactions;
- Tensor Core utilization;
- FP pipeline utilization;
- warp eligibility/stalls;
- branch divergence;
- registers/thread;
- shared-memory usage;
- occupancy;
- roofline position.

Do not collect every counter on the full application. Metric replay can perturb execution severely. Profile a narrow region or selected kernel launches.

---

## 18. Compute Sanitizer before celebrating

Custom CUDA can be “fast” because it is wrong. Run appropriate Compute Sanitizer tools for:

- invalid memory access;
- race conditions;
- uninitialized memory;
- synchronization errors.

Then run numerical tests. Then run model-level benchmark quality. Correctness is layered.

---

## 19. A5500-specific optimization checklist

Before writing a new kernel, confirm all of these:

- [ ] exact baseline is frozen;
- [ ] profiler proves this operator materially affects end-to-end latency;
- [ ] current backend is identified, not guessed;
- [ ] GPU capability is queried at runtime;
- [ ] FP16/BF16 Tensor Core path is confirmed where applicable;
- [ ] layout/copy overhead is measured;
- [ ] `torch.compile` has been tried;
- [ ] existing SDPA/FA2/CUTLASS/library kernels have been benchmarked;
- [ ] candidate will be tested over production shapes;
- [ ] register/shared-memory limits are considered;
- [ ] numerical correctness tests exist;
- [ ] Compute Sanitizer is used for custom CUDA;
- [ ] model benchmark quality gate exists;
- [ ] performance claim records software/hardware/precision/baseline/correctness.

If several boxes are missing, custom kernel work is probably premature.

---

## 20. Highest-return exact kernel work for DiTs

On A5500, prioritize these classes:

1. **Exact attention backend selection** — SDPA vs FA2 by real shapes.
2. **Compiler fusion of normalization/modulation/residual chains.**
3. **Elimination of layout/cast copies between block operations.**
4. **GEMM shape/layout fixes that restore Tensor Core-friendly execution.**
5. **CUDA Graph capture for stable repeated denoising shapes.**
6. **VAE kernels/tiling only after denoiser optimization makes VAE visible.**
7. **Specialized Triton/CUDA/CUTLASS kernels only for persistent profiler-proven gaps.**

This order has an important property: the first six can usually preserve the model algorithm exactly.

---

## 21. What not to copy from H100/B200 optimization writeups

Do not transplant these as A5500 recommendations:

- Hopper TMA schedules;
- FlashAttention-3 FP8 kernels;
- Blackwell tensor-memory/tcgen05 schedules;
- FP4/NVFP4/MXFP4 execution assumptions;
- B200-specific shared-memory/throughput tuning constants;
- NVSwitch scaling conclusions;
- headline speedups measured against HBM3/HBM3e bandwidth.

The **principles** may transfer: tile, overlap, fuse, reduce IO, specialize by architecture. The code path and multiplier often do not.

---

## 22. The engineering loop

A disciplined kernel loop is:

```text
measure end-to-end
-> locate phase
-> locate hot kernel
-> classify bottleneck
-> form one mechanism hypothesis
-> implement exact candidate
-> numerical correctness
-> profile microkernel
-> benchmark production shapes
-> benchmark full pipeline
-> run model quality suite
-> document hardware/workload/precision/baseline/correctness
-> keep or revert
```

That loop is more valuable than any single optimization trick. It is also the central lesson taken from the GPU performance-engineering resource map: performance engineering is a chain of measured mechanisms, not a collection of magic flags.

---

## Sources to keep open while doing this work

Primary/near-primary references emphasized by the Wafer resource map and adapted here:

- NVIDIA CUDA C++ Programming Guide — execution and memory model
- NVIDIA CUDA C++ Best Practices Guide — coalescing, shared memory, occupancy, workflow
- NVIDIA Ampere Tuning Guide — architecture-specific execution limits/features
- Roofline paper — arithmetic-intensity model
- Volkov, *Benchmarking GPUs to Tune Dense Linear Algebra* — measured hardware behavior over simplistic occupancy rules
- CUTLASS / CuTe GEMM documentation — tiled GEMM construction
- Triton programming guide/repository — blocked kernel programming
- FlashAttention and FlashAttention-2 papers/repository — exact IO-aware attention
- Nsight Systems and Nsight Compute documentation — timeline and kernel analysis
- Compute Sanitizer documentation — CUDA correctness

Use [../09_reference/08_wafer_gpu_perf_resource_map.md](../09_reference/08_wafer_gpu_perf_resource_map.md) for the dependency-ordered reading map and [../09_reference/09_a5500_benchmark_runbook.md](../09_reference/09_a5500_benchmark_runbook.md) for experiment execution.
