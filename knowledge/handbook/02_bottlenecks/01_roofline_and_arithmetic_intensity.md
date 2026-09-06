# Roofline and Arithmetic Intensity for DiTs on RTX A5500

The uploaded *Inference Engineering* book uses arithmetic intensity and roofline reasoning to distinguish compute-bound from memory-bound inference. The Wafer GPU-performance resource list puts the original Roofline paper near the beginning of its learning path for the same reason: before optimizing kernels, know which resource is limiting them.

For this repository, the roofline must be built around **RTX A5500**, not H100/B200.

---

## 1. The basic model

Arithmetic intensity:

`AI = operations / bytes moved from the relevant memory level`

A simple roofline says attainable performance is bounded by:

`performance <= min(compute_peak, memory_bandwidth * AI)`

The “ridge point” is roughly:

`AI_ridge = compute_peak / memory_bandwidth`

But there are several compute peaks depending on dtype/instruction mode, and the A5500 datasheet’s Tensor figure includes sparsity assumptions. Therefore do not build a precise application roofline from a marketing number alone.

Use Nsight Compute’s measured roofline/throughput data where possible.

---

## 2. A5500 bandwidth anchor

Official A5500 memory bandwidth: **768 GB/s**.

That is the outer GDDR6 roof. Actual achieved bandwidth depends on:

- access coalescing;
- cache hit rate;
- memory-controller behavior;
- request size;
- contention;
- kernel occupancy;
- read/write mix.

If a kernel moves 1 TB of data per forward pass, the absolute best-case lower bound from GDDR6 alone would already exceed one second at 768 GB/s, before considering compute. Use this style of sanity check to detect impossible expectations.

---

## 3. Why DiTs are often compute-bound

Large linear layers reuse weights/activations across many multiply-accumulate operations, creating high arithmetic intensity. Visual attention also performs large matrix products over latent tokens.

Therefore the denoiser often spends most time in:

- Q/K/V projections;
- attention matrix operations;
- output projection;
- MLP up/down/gated projections.

This is why Tensor Core utilization matters.

But the model can still fail to reach the compute roof due to implementation overhead.

---

## 4. Ways a nominally compute-bound DiT becomes memory/launch bound

### Unfused pointwise chains

Each operation writes an intermediate to GDDR6 and the next reads it back.

Example:

`RMSNorm -> scale -> shift -> SiLU -> gate -> residual`

Fusion can reduce traffic even though none of these operators are huge individually.

### Bad attention implementation

Materializing the score/probability matrices increases global-memory traffic and peak VRAM.

### Offload

Once weights cross PCIe every step, the relevant bandwidth roof may be the host link, not GDDR6.

### Tiny GEMMs

Small matrix shapes may underutilize Tensor Cores. Peak FLOPS become irrelevant.

### Recompiles and CPU gaps

No roofline model saves a GPU that is idle waiting for the CPU to launch the next kernel.

---

## 5. Per-operator diagnosis

### GEMM

If Tensor Core utilization is high and the kernel is near a compute roof:

- algorithmic work reduction is needed for major gains;
- or use a better GEMM schedule if not already near the roof.

If memory throughput is high but Tensor utilization is low:

- inspect layout/transposes;
- batch/sequence shape;
- kernel selection;
- weight reload behavior.

### Attention

If attention is bandwidth-heavy due to intermediate matrices:

- SDPA/FA2 can move it toward a better IO schedule.

If attention is already an efficient fused kernel and compute-heavy:

- parallelize sequence/context;
- reduce token count only if model/quality policy permits;
- optimize projections/softmax scheduling.

### Norm/modulation

Usually lower arithmetic intensity; fusion is the main lever.

### VAE convolution

Can be compute or memory sensitive depending on resolution/tile shape. Profile separately.

---

## 6. Measure at the right memory level

A kernel can reuse data in:

- registers;
- shared memory/L1;
- L2;
- GDDR6.

Arithmetic intensity versus GDDR6 traffic can look excellent because most reuse happens in cache. That is good. Use Nsight Compute counters/roofline views to understand the actual level where traffic is limiting.

---

## 7. Ampere tiling constraints

For cc 8.6, NVIDIA documents:

- 100 KB shared memory per SM;
- up to 99 KB per block with dynamic-memory opt-in;
- 64K 32-bit registers per SM;
- 48 maximum concurrent warps per SM.

A more aggressive tile can:

- improve reuse;
- use more registers/shared memory;
- reduce resident blocks/warps.

The optimum balances reuse and concurrency. That is why the Wafer reading list includes material arguing against “occupancy above all.”

---

## 8. Precision and roofline on A5500

Use FP16/BF16 as the main Tensor Core compute modes.

Do not put FP8/MXFP8 as the default roofline ceiling because A5500 does not have native Hopper FP8 Tensor Cores.

A low-bit weight-storage kernel may still have a different effective roof because it reads fewer weight bytes, but then dequantization/packing instructions become part of the compute cost. Measure the actual kernel.

---

## 9. Shape-dependent rooflines

The same model can move between regimes.

### Lower resolution / short sequence

- GEMMs are smaller;
- launch/CPU overhead fraction increases;
- CUDA Graph/compile benefits can be larger.

### Higher resolution

- attention and MLP GEMMs become larger;
- Tensor Core utilization can improve;
- activation memory pressure grows;
- attention may dominate.

### Video

- sequence length and activation footprint can explode;
- attention can dominate compute;
- single-GPU fit may fail;
- sequence/context parallelism becomes relevant.

Benchmark every shape bucket.

---

## 10. A simple empirical roofline workflow

1. capture end-to-end trace in Nsight Systems;
2. select the top 3–5 kernels by total GPU time;
3. profile them with Nsight Compute SpeedOfLight/Roofline sections;
4. classify each as compute-, memory-, latency-, or occupancy-limited;
5. choose an optimization matching the limit;
6. rerun the full pipeline.

Example mapping:

| Observation | Likely lever |
|---|---|
| Tensor Core GEMM near roof | work reduction/parallelism, not tiny kernel tweaks |
| Low Tensor use + many layout kernels | compile/fuse/layout |
| High DRAM traffic in attention | FA2/SDPA |
| CPU gaps between small kernels | compile/CUDA Graphs |
| PCIe copies dominate | residency/offload strategy |
| NCCL dominates on 2 GPUs | coarser sharding/fewer collectives |

---

## 11. Do not confuse memory capacity with memory bandwidth

24 GB capacity determines whether the model fits.

768 GB/s bandwidth determines how quickly the GPU can move data to/from GDDR6.

Quantization can help capacity without helping speed. Offload can solve capacity while destroying effective bandwidth. VAE tiling can solve activation capacity while adding launch overhead.

Record both fit and bandwidth/latency effects.

---

## 12. Compute-bound does not mean batching always helps

Image/video DiT inference often already saturates compute for a single request. Increasing batch can:

- improve GEMM efficiency for small shapes;
- or simply double work and latency without much throughput gain;
- or OOM due activation growth.

Benchmark throughput versus latency explicitly.

---

## 13. Roofline and multi-GPU

With two A5500s, add a communication roof.

Local kernel time may shrink while peer/NCCL traffic does not.

For a parallel strategy:

`T_total ≈ max/overlap(local_compute, communication) + synchronization_overhead`

If communication cannot be hidden, the interconnect becomes the bottleneck.

A5500 NVLink is useful but far below local GDDR6 bandwidth. Fine-grained communication should therefore be treated carefully.

---

## 14. Practical rule

Every proposed optimization should be able to answer:

> Which roof or overhead is this supposed to move?

If the answer is “it is a popular optimization,” do not run the experiment yet.
