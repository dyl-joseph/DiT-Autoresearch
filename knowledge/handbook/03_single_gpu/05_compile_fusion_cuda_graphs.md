# `torch.compile`, Fusion, and CUDA Graphs on RTX A5500

Compiler/runtime optimization is one of the most attractive paths for this project because it can reduce latency **without intentionally approximating the model**. On RTX A5500, it also targets a weakness that becomes visible in repeated DiT loops: many small kernel launches and global-memory round trips around the large GEMM/attention kernels.

---

## 1. What the compiler can improve

A DiT transformer block contains many operations around the major matrix multiplies:

- normalization;
- timestep/conditioning modulation;
- scaling and shifting;
- activation;
- gating;
- residual add;
- reshape/transpose;
- rotary/position transforms;
- masking.

Eager execution can launch separate kernels and write intermediates to global memory. Compilation can:

- fuse pointwise chains;
- eliminate redundant intermediates;
- specialize static shapes;
- choose optimized GEMM implementations;
- reduce Python dispatcher overhead;
- use CUDA Graph replay in some modes/configurations.

Because the transformer repeats across denoising steps, a small per-block reduction compounds.

---

## 2. Compile the hot module before the whole pipeline

Start with the transformer/denoiser, not the entire Diffusers pipeline.

Why:

- tokenization and scheduler Python can create graph breaks;
- VAE/text encoder may use different shapes;
- the denoiser is the repeated hot loop;
- failures are easier to debug.

Conceptual pattern:

```python
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="reduce-overhead",
    fullgraph=False,
)
```

Exact API compatibility changes with PyTorch/Diffusers versions. Pin versions in the experiment record.

---

## 3. Regional compilation

Many DiTs consist of repeated transformer blocks. Regional compilation compiles the repeated region rather than tracing every wrapper and one-off component.

Benefits:

- shorter compile time;
- fewer graph-break surfaces;
- reusable compiled code for repeated blocks;
- easier shape specialization.

This is often the best first compile strategy for large DiTs.

---

## 4. Shape buckets

Compilation becomes much easier when requests use a small set of shapes.

Define production buckets such as:

```text
image:
  768x768
  1024x1024
  1344x768
  768x1344

video:
  480p x 49 frames
  720p x 49 frames
  720p x 81 frames
```

Then compile/prewarm each supported bucket.

Do not accept arbitrary resolution if the product does not need it. Generality is expensive.

---

## 5. `sm_86` targeting

NVIDIA’s Ampere tuning guide recommends compiling explicitly for compute capability 8.6 to exploit cc 8.6 behavior rather than relying on an 8.0 binary.

For custom extensions:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

For NVCC:

```bash
nvcc -O3 -gencode arch=compute_86,code=sm_86 kernel.cu
```

Verify the actual device first:

```python
print(torch.cuda.get_device_capability())
```

Use CUDA Binary Utilities (`cuobjdump`, `nvdisasm`) if you suspect a package is running PTX JIT or a non-native cubin.

---

## 6. Compiler modes

PyTorch compiler modes change by version, but commonly available choices include behavior similar to:

- `default` — balanced compile/runtime;
- `reduce-overhead` — useful for reducing Python/launch overhead and often CUDA Graph-oriented;
- `max-autotune` — more expensive compilation/autotuning for potentially better steady-state kernels.

Benchmark modes rather than assuming the most aggressive one is best.

On a 24 GB GPU, aggressive autotune/CUDA Graph paths can consume extra memory for workspaces and static pools. Measure peak **reserved** memory, not only allocated tensor memory.

---

## 7. Graph breaks

A graph break prevents fusion/specialization across the boundary.

Common causes in DiT pipelines:

- Python branching on tensor data;
- unsupported custom ops;
- `.item()`;
- device transfers;
- dynamic shape logic;
- iteration over tensor values;
- unsupported attention backend wrappers;
- callbacks inside the denoising loop.

Use PyTorch compiler logging appropriate to your pinned version. Common `TORCH_LOGS` categories include graph-break/recompile diagnostics.

Fix high-frequency breaks inside the denoising loop first.

---

## 8. Recompilation

Recompilation can destroy production latency.

Track recompiles by:

- width/height;
- frame count;
- batch size;
- prompt sequence length if it enters compiled shapes;
- CFG branch mode;
- control input count;
- dtype;
- device.

If many shape variants compile, route into buckets or pad to stable dimensions.

---

## 9. CUDA Graphs

CUDA Graphs capture a stable sequence of launches and replay it with much lower CPU launch overhead.

They are attractive for DiTs because denoising repeatedly executes the same graph shape.

### Requirements/constraints

- stable shapes;
- stable memory addresses or graph-managed static buffers;
- compatible control flow;
- no unsupported dynamic allocations inside capture;
- no unexpected host synchronization.

### Benefits

- lower CPU overhead;
- fewer launch gaps;
- more consistent step latency;
- better GPU occupancy over time when Python was starving the device.

### Costs

- static memory pools;
- capture complexity;
- shape specialization;
- harder debugging;
- interactions with offload/dynamic caches.

On A5500’s 24 GB VRAM, the extra reserved memory can be a deciding factor.

---

## 10. Fusion targets specific to DiTs

After compilation, profile for remaining small hot kernels.

Good custom-fusion candidates:

- RMSNorm/LayerNorm + scale/shift;
- AdaLN-style modulation;
- SiLU/GELU + gating;
- residual + gate + cast;
- rotary embedding transforms;
- QKV packing/reshape if the attention backend permits;
- timestep embedding transforms;
- small per-step scheduler tensor operations moved fully onto GPU.

Bad custom-kernel target:

- a 0.1% one-time preprocessing function.

---

## 11. Ampere async global→shared copy

The Ampere tuning guide documents hardware-accelerated asynchronous copies from global memory to shared memory and CUDA pipeline APIs. These features are useful in hand-tuned kernels that stage data into shared memory while overlapping computation.

You do not need to write raw `cp.async` immediately. Mature CUTLASS/Triton/Inductor kernels may exploit appropriate Ampere mechanisms. But if you write a custom hot kernel, the architecture supports software-pipelined global→shared staging.

This is the A5500-era analogue of the broader “overlap data movement and compute” principle; do not copy Hopper TMA examples directly.

---

## 12. Compile and attention interaction

Test:

- SDPA eager;
- SDPA compiled;
- FA2 eager;
- FA2 compiled if integration supports it.

Why:

- compiler may fuse QKV reshapes around SDPA;
- FA2 may be an opaque external kernel with less surrounding fusion;
- graph breaks can eliminate one backend’s theoretical advantage;
- layout conversions can dominate.

End-to-end block latency decides.

---

## 13. Compile and offload interaction

Offload can make a graph dynamic because modules/tensors move devices.

Safer patterns:

- compile modules that remain resident;
- offload whole cold components outside compiled hot regions;
- use stable transformer-group staging if the offload framework supports it;
- avoid compiling across unpredictable device movement.

Measure both memory and latency. A compiled resident transformer may be worth moving the text encoder fully off GPU to preserve compiler memory headroom.

---

## 14. Compile and quantization interaction

Because quantization is quality-gated and not native-FP8 on A5500, compare against the strongest full-precision compiled baseline.

Required matrix:

- FP16/BF16 eager;
- FP16/BF16 compiled;
- quantized eager;
- quantized compiled.

A quantized candidate that only beats eager full precision is not enough.

---

## 15. Benchmark protocol

For each compile configuration record:

```text
compile mode
shape bucket
compile time
first call latency
steady p50/p95
peak allocated VRAM
peak reserved VRAM
graph count
graph-break count
recompile count
attention backend
quality benchmark result
```

Warm up until compilation/autotuning is complete.

---

## 16. Profiling successful compilation

In Nsight Systems, successful optimization often looks like:

- fewer tiny kernels;
- shorter CPU gaps;
- denser CUDA stream timeline;
- longer fused kernels replacing pointwise chains;
- more consistent per-step pattern.

In Nsight Compute, a fused kernel may trade higher register pressure for much lower memory traffic. Do not reject it because occupancy fell if total latency improved.

---

## 17. Common failure patterns

### Compile is slower

Possible causes:

- graph breaks;
- dynamic shapes;
- poor generated kernel for an unusual shape;
- external optimized kernel replaced by slower generated code;
- excessive autotune overhead included in measurement.

### Compile OOMs but eager fits

Possible causes:

- CUDA Graph static pools;
- autotune workspaces;
- duplicate compiled buffers;
- more aggressive caching by Inductor.

Try:

- regional compilation;
- a less memory-intensive mode;
- disable graph capture for that bucket;
- free text encoder before compiling/running denoiser;
- lower concurrent worker count.

### Compiled first request is terrible

Expected if compilation occurs on demand. Precompile/prewarm and cache artifacts where supported.

### One shape is fast, another recompiles

Use shape buckets or dynamic-shape settings only if they preserve performance.

---

## 18. A5500 recommended sequence

1. establish eager FP16/BF16 baseline;
2. benchmark SDPA versus FA2;
3. compile transformer region;
4. inspect graph breaks/recompiles;
5. define shape buckets;
6. test `reduce-overhead`/appropriate mode;
7. test CUDA Graph behavior if memory headroom exists;
8. profile remaining small kernels;
9. write custom Triton/CUDA only for material hot spots;
10. rerun full quality suite and sustained thermal benchmark.

This path attacks implementation overhead while respecting the strict accuracy requirement.
