# Target Hardware and Quality Contract: RTX A5500 / Ampere

This file defines the constraints that override generic advice elsewhere in the handbook.

## 1. Target hardware

The intended accelerator is the **NVIDIA RTX A5500**, a workstation GPU based on NVIDIA Ampere.

Official NVIDIA specifications list:

- 24 GB GDDR6 GPU memory with ECC
- 384-bit memory interface
- 768 GB/s memory bandwidth
- 10,240 CUDA cores
- 320 third-generation Tensor Cores
- 34.1 TFLOPS advertised single-precision peak
- PCIe 4.0 x16 host interface
- 230 W total board power
- optional low-profile NVLink bridge connecting two RTX A5500s
- 112.5 GB/s bidirectional NVLink bandwidth for the supported two-GPU bridge

Primary reference: [NVIDIA RTX A5500 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/proviz-nvidia-rtx-a5500-datasheet-2130578-r3-us-web.pdf).

Do not turn peak vendor figures into expected application performance. The Wafer performance-engineering resource list explicitly separates vendor peaks from measured performance and requires hardware, workload, precision, baseline, and correctness information for performance claims.

## 2. Verify the actual machine; do not trust the document blindly

At the beginning of each benchmark run, capture what the machine says:

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem --format=csv
nvidia-smi topo -m
```

And from PyTorch:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0))
```

For custom CUDA extensions, verify the reported capability before setting an architecture target. Ampere workstation/GeForce devices are generally the `sm_86` class; the runtime query above is the authority for your installed card.

## 3. Ampere-specific mechanics that matter

NVIDIA’s current Ampere tuning guide documents several characteristics of compute capability 8.6 that matter when you build or select kernels:

- 48 concurrent warps per SM at the architectural maximum
- up to 16 resident thread blocks per SM
- 64K 32-bit registers per SM and up to 255 registers per thread
- 100 KB shared memory capacity per SM, with up to 99 KB addressable by one block after explicit opt-in for large dynamic shared-memory allocations
- 128 KB combined L1/texture/shared-memory capacity for cc 8.6
- hardware-accelerated asynchronous global-memory-to-shared-memory copy, exposed through CUDA pipeline mechanisms
- third-generation Tensor Cores with FP16 and BF16 matrix operations, plus TF32 for FP32-style workloads
- explicit recommendation to compile for compute capability 8.6 rather than relying on an 8.0 binary when targeting 8.6 hardware

Primary reference: [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).

The practical interpretation for DiTs is not “maximize occupancy.” It is:

1. keep the Tensor Core math units fed,
2. reduce avoidable global-memory traffic,
3. avoid launch bubbles,
4. tile attention/GEMM work to match the architecture,
5. control register/shared-memory pressure so a faster-looking tile does not reduce useful concurrency too far.

The Wafer resource list points to Volkov’s dense-linear-algebra work precisely because peak occupancy is not a universal objective. Measure achieved performance instead of optimizing one counter in isolation.

## 4. Precision policy on A5500

### 4.1 Default execution precision

For most modern DiTs, begin with the model’s supported **FP16 or BF16** inference path. Benchmark both only if the checkpoint/framework claims both are valid. Ampere Tensor Cores support both.

Do not change dtype merely because it is available. The quality baseline is the reference implementation you actually care about.

### 4.2 FP8 / MXFP8 / FP4 are not primary A5500 paths

The RTX A5500 predates native FP8 Tensor Cores. Therefore:

- Hopper-targeted FP8 kernels are not a speed plan for this machine.
- Blackwell MXFP8/MXFP4/NVFP4 paths are not a speed plan for this machine.
- FlashAttention-3 and FlashAttention-4 are architectural references, not target kernels.
- storing weights in a low-bit format and dequantizing to FP16/BF16 at runtime may save residency, but it does **not** become native low-precision Tensor Core execution on A5500.

That distinction is essential. A low-bit checkpoint can consume less VRAM and still run slower because of dequantization, layout conversion, or unfused kernels.

### 4.3 Quantization acceptance rule

Quantization is disabled by default.

A quantized configuration may be promoted only if:

1. it completes the exact required benchmark suite;
2. aggregate required accuracy/quality metrics do not decrease beyond the predefined measurement-noise rule;
3. no critical per-case benchmark regression appears;
4. image/video artifact checks do not reveal new failure modes;
5. the performance or capacity benefit is large enough to justify the added complexity.

If the organization’s rule is literally no observed accuracy decrease, use the stronger version:

- every required aggregate score must be `candidate >= baseline`;
- every deterministic pass/fail benchmark must satisfy `baseline pass -> candidate pass`;
- a candidate with one new critical failure is rejected even if the average score rises.

See [02_benchmarking_and_quality_gates.md](02_benchmarking_and_quality_gates.md) and [../09_reference/09_a5500_benchmark_runbook.md](../09_reference/09_a5500_benchmark_runbook.md).

## 5. Exact-first optimization policy

The main optimization queue should be dominated by mechanisms that do not intentionally approximate the model:

### Tier A — exact / implementation-level

- PyTorch `inference_mode()`
- correct autocast/reference dtype
- SDPA or FlashAttention-2 exact attention implementation
- `torch.compile`
- regional compilation of repeated transformer blocks
- CUDA Graph capture for stable shapes when memory allows
- fused normalization/modulation/activation kernels
- better GEMM algorithm selection
- static shape buckets
- removal of graph breaks and Python synchronization
- prompt-embedding caching when prompt + encoder state are unchanged
- reuse of deterministic masks/rotary tables/position tensors
- VAE/text-encoder lifecycle management
- CPU/model/group offload
- pinned-memory staging and transfer/compute overlap
- memory-mapped/lazy checkpoint loading
- allocator and fragmentation control
- two-GPU sharding/data parallel/context parallelism that passes numerical equivalence checks

### Tier B — output-preserving in theory but numerically different

These usually preserve the mathematical operation but can change floating-point accumulation order:

- alternative exact attention kernels
- tensor/context parallel reductions
- fused kernels
- different GEMM algorithms
- compiler-generated code

They still need the benchmark suite because “mathematically exact” is not bitwise-identical floating point.

### Tier C — quality-sensitive / approximate

Do not adopt without explicit quality proof:

- quantization
- TF32 when replacing a stricter FP32 path
- fewer denoising steps
- scheduler changes that alter the trajectory
- guidance cutoff / reduced CFG work
- timestep or hidden-state caching
- low-precision attention such as SageAttention variants
- sparse/approximated attention
- distilled checkpoints

For this project, Tier C is a **research branch**, not the default deployment path.

## 6. Memory strategy for 24 GB before quantization

If the model does not fit, try these in order:

1. remove/cycle the text encoder after prompt embeddings are produced;
2. cache prompt embeddings for repeated prompts or offline prompt preprocessing;
3. decode the VAE after freeing denoiser-temporary buffers;
4. enable VAE tiling/slicing/chunking when quality passes;
5. use model/group CPU offload for cold transformer groups;
6. prefetch the next group with pinned host memory and a side stream if it actually overlaps;
7. use a second A5500 for model/sequence/context parallelism if available;
8. only then test low-bit storage, with the no-regression quantization gate.

A 24 GB limit is a systems problem. Treat quantization as one possible capacity tool, not the definition of memory optimization.

## 7. Multi-GPU contract

NVIDIA documents an optional A5500 NVLink bridge with 112.5 GB/s bidirectional bandwidth and support for connecting two cards. Practical rules:

- verify the physical bridge is present;
- verify `nvidia-smi topo -m` shows the expected link;
- verify peer access from CUDA/PyTorch;
- do not assume two 24 GB cards act like one transparent 48 GB allocation;
- use application/framework sharding or peer-aware kernels;
- benchmark communication percentage in Nsight Systems;
- report **GPU-seconds per output**, not just wall-clock latency.

If there is no NVLink bridge, PCIe 4.0 communication can erase the benefit of aggressive fine-grained parallelism. Prefer coarser sharding or independent request/data parallelism when that happens.

## 8. Thermal and sustained-performance contract

The A5500 is a 230 W active-cooled workstation card. Long image/video runs can expose thermal or power behavior that a short benchmark misses.

For long tests, log at least:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv -l 1
```

Reject a claimed speedup if it only appears because one run started cold or at a different boost clock. Compare sustained runs after the GPU reaches a stable operating temperature.

## 9. Source discipline

This handbook follows the Wafer repository’s hierarchy of evidence:

1. original paper introducing a mechanism;
2. official specification/reference;
3. implementation repository;
4. direct implementer report with code and reproducible measurements.

A performance claim should state:

- hardware;
- software versions;
- workload shape/distribution;
- precision and algorithm;
- baseline;
- correctness/quality method.

If your experiment notes omit one of these, the result is incomplete.
