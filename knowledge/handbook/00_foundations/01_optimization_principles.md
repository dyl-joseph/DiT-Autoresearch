# Optimization Principles for DiTs on RTX A5500

This chapter defines the rules that should govern every optimization experiment in this repository. It is deliberately stricter than a generic “make the model faster” checklist because the target hardware and quality constraints are fixed:

- target GPU: **NVIDIA RTX A5500 / Ampere**;
- target memory budget: **24 GB per GPU**;
- quantization: **rejected if required benchmark accuracy decreases**;
- newer Hopper/Blackwell-only optimization paths: architectural context, not the implementation target.

The uploaded *Inference Engineering* book emphasizes that optimization is a trade among latency, throughput, quality, and cost rather than a single number. The Wafer GPU-performance resource list adds a useful discipline: performance claims require the exact hardware, software, workload, precision/algorithm, baseline, and correctness method. This chapter combines those ideas into an experimental method for DiTs.

---

## Principle 1 — optimize the bottleneck you measured, not the one you expected

A visual DiT is often described as compute-bound. That is a useful prior, not a diagnosis.

The actual runtime can be dominated by:

- transformer GEMMs;
- attention;
- normalization/modulation launch overhead;
- VAE decode;
- text encoder;
- CPU scheduler logic;
- Python synchronization;
- CPU↔GPU copies from offload;
- graph recompilation;
- memory allocator fragmentation;
- peer/NCCL traffic on multiple GPUs;
- thermal throttling during long runs.

The first optimization act is therefore measurement.

Use three levels:

1. **end-to-end timer** — what the user sees;
2. **timeline profiler** — which phases and transfers consume time;
3. **kernel profiler** — why the hottest kernels are slow.

Do not jump to Nsight Compute on the whole application. Use Nsight Systems or `torch.profiler` to identify a small set of hot kernels, then profile those kernels deeply.

---

## Principle 2 — exact optimizations first

On the target hardware, the default priority is implementation efficiency rather than arithmetic approximation.

### High-priority exact or effectively exact changes

- `torch.inference_mode()`;
- FP16/BF16 reference precision supported by the checkpoint;
- PyTorch SDPA or FlashAttention-2;
- `torch.compile`;
- regional compilation;
- CUDA Graph replay for stable shapes;
- operation fusion;
- better GEMM algorithms;
- static shape buckets;
- removing CPU/GPU synchronizations;
- keeping invariant tensors resident;
- caching prompt embeddings for identical encoder input/state;
- freeing/offloading the text encoder after use;
- VAE tiling/chunking when the exact pipeline defines the same semantics and benchmarks confirm parity;
- memory-mapped/lazy loading;
- CPU/group offload;
- pinned-memory prefetching;
- exact multi-GPU sharding or context parallelism.

Even these can create tiny floating-point differences because operation ordering changes. That is why they still pass the quality benchmark. The difference is that they do not intentionally throw away model information.

### Lower-priority quality-sensitive changes

- quantization;
- TF32 if replacing an FP32 baseline;
- step reduction;
- scheduler changes;
- distilled checkpoints;
- CFG/guidance cutoff;
- timestep/hidden-state caches;
- approximate attention such as low-precision attention variants;
- sparsity/pruning.

These are research candidates, not first-line tuning steps.

---

## Principle 3 — A5500 is not a small H100

Do not mechanically copy datacenter-GPU tuning advice.

A5500 has:

- Ampere Tensor Cores, not Hopper/Blackwell Tensor Cores;
- no native FP8 path;
- GDDR6 rather than HBM;
- 24 GB VRAM;
- PCIe 4.0 x16;
- optional two-card NVLink rather than an NVSwitch fabric;
- different shared-memory, warp, register, and scheduling limits from H100/B200.

That changes the optimization ranking.

Examples:

- **FA2** is relevant; FA3/FA4 are not target kernels.
- low-bit storage may save VRAM but can cost time because dequantization feeds FP16/BF16 compute rather than native FP8 compute.
- fine-grained tensor parallelism can lose to communication much sooner than on NVSwitch datacenter nodes.
- CPU offload over PCIe is a larger latency risk than on coherent or very high-bandwidth CPU/GPU systems.
- static-shape compilation and launch reduction can be disproportionately valuable because the raw GPU is slower than the newest accelerators, making overhead fractions workload-dependent.

---

## Principle 4 — benchmark the exact shape distribution

DiT cost changes sharply with:

- image resolution;
- latent downsampling factor;
- patch size;
- token sequence length;
- frame count;
- batch size;
- CFG branch count;
- head dimension;
- model width/depth;
- control/reference inputs;
- VAE tile size;
- offload group size.

A kernel can be best at 1024×1024 and poor at 768×1344. A compiled graph can be excellent for one video frame bucket and recompile constantly under arbitrary shapes.

Create shape buckets from real production traffic or the evaluation set. Every performance claim should name the bucket.

---

## Principle 5 — separate capacity optimizations from speed optimizations

A 24 GB A5500 creates strong pressure to “optimize memory.” But memory reduction can make inference slower.

Examples:

- CPU offload lowers VRAM but adds PCIe transfers;
- aggressive VAE tiling lowers peak activation memory but increases launch count;
- low-bit weights lower residency but add dequantization;
- sequence parallelism lowers per-GPU activation memory but adds collective communication;
- allocator tricks can reduce fragmentation without changing the true tensor footprint.

For each memory optimization, record both:

- bytes/GB saved;
- latency/GPU-seconds added or removed.

A good memory optimization is one that crosses a hard fit boundary or creates headroom for a faster configuration.

---

## Principle 6 — use a quality contract, not vibes

Image and video quality are hard to reduce to one metric. That makes discipline more important, not less.

Your benchmark suite should include the actual required quality dimensions:

- prompt adherence;
- text rendering if relevant;
- object count/composition;
- identity consistency;
- edit fidelity;
- spatial relations;
- temporal consistency;
- motion quality;
- artifact rate;
- task-specific pass/fail tests;
- human preference or accepted-output rate when necessary.

The project’s quantization policy is binary:

> If quantization reduces required benchmark accuracy, it is not accepted.

Do not hide a regression inside a composite average. A quantized model that improves aesthetic score but loses text rendering is a failure if text rendering is required.

---

## Principle 7 — use paired experiments

Performance and quality comparisons should use the same:

- prompts;
- negative prompts;
- seeds;
- initial latents when possible;
- image/video shapes;
- scheduler;
- number of denoising steps;
- model revision;
- conditioning inputs.

Paired comparisons reduce noise and make per-case regressions visible.

For performance, randomize A/B run order or alternate configurations after warmup so clock/temperature drift does not bias one side.

---

## Principle 8 — keep cold-start and warm-state measurements separate

Compilation, kernel autotuning, checkpoint loading, CUDA context creation, JIT extension building, and cache population can make the first request radically different from steady state.

Report at least:

- process startup to ready;
- first request latency;
- first request after compile cache is warm;
- steady-state p50/p95;
- compile/recompile count.

For a workstation batch workflow, cold start may be irrelevant. For autoscaled serving, it may be one of the largest costs.

---

## Principle 9 — make hidden synchronization visible

Common synchronization traps:

- calling `.item()` inside the denoising loop;
- printing CUDA tensors;
- measuring time with CPU clocks without CUDA synchronization;
- copying tensors to CPU each step;
- Python branches on CUDA-computed values;
- allocator/debug calls in the hot loop;
- eager logging of generated intermediates.

Use CUDA events for kernel-region timing and only synchronize at intentional benchmark boundaries.

---

## Principle 10 — static shapes are a performance feature

DiT inference is unusually friendly to shape specialization because each denoising request repeats the same transformer many times.

Exploit that:

- define resolution/frame buckets;
- pad or route into buckets if the product allows it;
- compile repeated blocks;
- cache compiled artifacts;
- use CUDA Graphs for stable allocations and control flow;
- keep batch size stable within a worker profile.

Every extra shape creates a compile/cache/allocator problem. Generality has a performance cost.

---

## Principle 11 — optimize repeated work, not one-off work

A DiT may execute the same transformer 20–50 times per request. A 2% improvement inside that loop can matter more than a 50% improvement to a one-time preprocessing function.

Rank work by:

`total request time contribution = per-call cost × number of calls`

This is why:

- attention;
- MLP GEMMs;
- norms/modulation;
- scheduler synchronization;

deserve disproportionate attention.

But once the denoiser becomes fast, re-profile. The VAE or text encoder can become the new bottleneck.

---

## Principle 12 — profile memory traffic when compute is not saturated

The uploaded book correctly identifies visual generation as usually compute-bound. But “compute-bound” can be defeated by poor implementation.

If Tensor Core utilization is low, check:

- tiny GEMMs caused by unusual batch/sequence shapes;
- layout conversion kernels;
- non-contiguous tensors;
- repeated materialization of attention intermediates;
- pointwise chains that spill to global memory;
- host/device copies;
- underfilled SMs from extreme register/shared-memory use;
- graph breaks that interrupt fusion.

On A5500, global memory bandwidth is 768 GB/s. A supposedly compute-bound transformer can spend a large fraction of time on memory traffic if kernels are not fused or tiled well.

---

## Principle 13 — occupancy is a constraint, not the goal

Wafer’s resource list highlights Volkov’s dense linear algebra work because “maximize occupancy” is too simplistic.

Higher occupancy can hide latency, but a kernel can win with lower occupancy if it:

- uses registers effectively;
- reduces global-memory transactions;
- improves data reuse;
- uses Tensor Cores more efficiently;
- avoids synchronization.

Use Nsight Compute to understand what limits occupancy and whether that limit actually reduces throughput.

---

## Principle 14 — compile for the real architecture

For custom CUDA/Triton extensions on the target system:

1. verify `torch.cuda.get_device_capability()`;
2. build for the actual architecture, normally `sm_86` for this class of Ampere workstation GPU;
3. inspect generated binaries if performance is suspicious;
4. do not benchmark a first-use JIT compile as inference latency.

Typical build environment:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

For raw NVCC:

```bash
nvcc -O3 -gencode arch=compute_86,code=sm_86 ...
```

If distributing binaries, you may also include PTX for forward compatibility, but local performance measurements should use a native cubin for the target architecture.

---

## Principle 15 — one optimization can move the bottleneck

Examples:

- FA2 makes attention faster; MLP GEMMs become dominant.
- compile/fusion shrinks launch overhead; VAE decode becomes visible.
- offload makes the model fit; PCIe becomes dominant.
- second-GPU context parallelism halves local attention work; NCCL becomes dominant.
- prompt embedding caching removes encoder cost; denoiser dominates even more.

Therefore every meaningful optimization ends with a new profile.

---

## Principle 16 — do not stack unvalidated changes

Bad experiment:

`new attention + compile + quantization + new scheduler + cache + offload`

If it is faster but quality changes, you do not know why.

Good sequence:

1. baseline;
2. change one mechanism;
3. performance benchmark;
4. quality gate;
5. save artifact/config;
6. re-profile;
7. move to the next mechanism.

Once individual effects are known, benchmark combinations because interactions are real.

---

## Principle 17 — report GPU-seconds, not only wall time

A two-GPU configuration that cuts latency from 10 s to 7 s consumes:

- one-GPU baseline: 10 GPU-s;
- two-GPU candidate: 14 GPU-s.

That may be correct for an interactive latency SLO, but it is worse for throughput/cost.

Always record:

`GPU-seconds/output = wall-clock GPU service time × number of active GPUs`

Use this alongside latency.

---

## Principle 18 — exact reuse is better than approximate reuse

Before trying hidden-state caching, look for truly invariant data:

- tokenization;
- prompt embeddings;
- negative-prompt embeddings;
- static control encodings;
- position/rotary tables;
- masks;
- model constants;
- VAE scale constants;
- compiled shape-specific kernels.

These caches can save work with no semantic approximation.

Only then test timestep/layer caches that reuse stale activations.

---

## Principle 19 — power and thermals are part of reproducibility

Short bursts can run at higher clocks than sustained video generation.

Record:

- GPU temperature;
- SM clock;
- memory clock;
- power draw;
- throttling reason if exposed.

Warm the card to a stable state before comparing small speed differences.

---

## Principle 20 — the benchmark decides whether quantization exists in the deployment

The theoretical argument for quantization is irrelevant if the project quality constraint forbids its observed effect.

On A5500, quantization is additionally less attractive because there is no native FP8 Tensor Core path. Therefore:

- do not assume “8-bit = 2× faster”;
- do not assume “4-bit = 4× less runtime memory” end-to-end;
- separate checkpoint storage, resident weight bytes, temporary dequant buffers, and compute dtype;
- compare against exact offload/sharding alternatives;
- reject the candidate if benchmark accuracy moves down.

That is the final principle because it prevents a large class of misleading “optimization” work.
