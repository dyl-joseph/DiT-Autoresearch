# Wafer GPU Performance Engineering Resources — DiT / RTX A5500 Map

Source repository: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)

The repository is a curated learning path for GPU performance engineering and production inference. This file turns that general path into a concrete reading/implementation order for Diffusion Transformers on RTX A5500.

## 1. The source-selection rule to copy verbatim in spirit

Wafer’s contribution policy asks what mechanism a source teaches, why it is primary, where it belongs in the dependency order, and what it replaces. It considers a paper, official specification/reference, implementation repository, or reproducible direct implementer report to be primary.

For this handbook, that becomes:

- use NVIDIA docs for A5500/Ampere limits;
- use PyTorch/Diffusers docs for runtime APIs;
- use FlashAttention/CUTLASS/Triton repositories and papers for kernel behavior;
- use xDiT/Cache-DiT/ParaAttention repositories for DiT-specific systems;
- use benchmark claims only when the full measurement context is available.

Wafer also requires every performance number to include hardware/software versions, workload shape/distribution, precision/algorithm, baseline, and correctness method. For DiTs, add **prompt/seed set, denoising schedule, and quality benchmark**.

---

## 2. Start-here resources and how they translate to DiTs

### Roofline: *An Insightful Visual Performance Model*

Wafer link: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html

**Mechanism:** arithmetic intensity versus compute and memory-bandwidth ceilings.

**DiT use:** decide whether a hot operator needs fewer bytes moved, more Tensor Core utilization, or a different algorithm. Visual DiTs are often nominally compute-bound, but offload, unfused norms, and poor attention implementations can make actual execution memory/launch bound.

**A5500 priority:** very high. The card has 768 GB/s GDDR6 bandwidth, so avoid reasoning from H100/B200 rooflines.

### CUDA C++ basics / programming model

Wafer links:

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

**Mechanism:** grids, blocks, warps, synchronization, memory spaces.

**DiT use:** understand why a fused RMSNorm/modulation kernel, a tiled GEMM, or an attention kernel wins or loses on a particular shape.

**A5500 priority:** high if writing/reading Triton/CUDA kernels; medium if only tuning framework code.

### CUDA Best Practices

Wafer link: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

**Mechanism:** coalescing, data transfer minimization, launch configuration, occupancy, synchronization, memory reuse.

**DiT use:** applies directly to fused transformer pointwise kernels, rotary/position operations, VAE kernels, and custom preprocessing.

**A5500 addition:** pair it with the Ampere tuning guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/

---

## 3. Matrix multiplication resources

### Volkov: *Benchmarking GPUs to Tune Dense Linear Algebra*

Wafer link: https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf

**Mechanism:** tune from measured machine behavior, not a simplistic “maximum occupancy is always best” rule.

**DiT use:** DiT MLPs and Q/K/V/output projections are dominated by GEMMs. A kernel that uses more registers but performs fewer global transactions can beat a nominally higher-occupancy kernel.

**A5500 priority:** high when custom GEMM/Triton work begins.

### CUDA matmul optimization worklog

Wafer link: https://siboehm.com/articles/22/CUDA-MMM

**Mechanism:** progression from naive global-memory matmul to shared-memory/register tiling.

**DiT use:** a concrete mental model for why fused/tiled transformer kernels matter.

### CUTLASS / CuTe

Wafer links:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html

**Mechanism:** tile layouts, copy atoms, GEMM collectives.

**DiT use:** use when profiling shows default GEMM algorithm selection underperforms for repeated fixed DiT shapes.

**A5500 caution:** tutorials increasingly showcase newer architectures. Filter examples for Ampere-compatible MMA/copy mechanisms; do not import TMA/tcgen05 assumptions.

---

## 4. Attention resources

### FlashAttention

Wafer links:

- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2307.08691
- https://github.com/Dao-AILab/flash-attention

**Mechanism:** exact IO-aware attention that avoids materializing large intermediate matrices and improves work partitioning.

**A5500 relevance:** extremely high. The official repository lists FlashAttention-2 CUDA support for Ampere/Ada/Hopper and fp16/bf16. This is the main external attention kernel family to test on A5500.

**Not A5500 paths:** FlashAttention-3 is explicitly Hopper-targeted; FlashAttention-4 targets Hopper/Blackwell. Treat them as design reading, not runnable recommendations.

### Online softmax

Wafer link: https://arxiv.org/abs/1805.02867

**Mechanism:** stable softmax without materializing all intermediate normalization state.

**DiT use:** helps understand FlashAttention’s memory-traffic reduction and why an “exact attention” kernel can be much faster without changing the model algorithm.

---

## 5. Triton and custom-kernel resources

Wafer links:

- Triton paper: https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- programming guide: https://triton-lang.org/main/programming-guide/chapter-1/introduction.html
- repository: https://github.com/triton-lang/triton

**DiT use cases worth custom-kernel work on A5500:**

- fused AdaLN / modulation / residual operations;
- RMSNorm/LayerNorm variants;
- rotary or positional transform kernels;
- small repeated elementwise chains that create many launches;
- unusual GEMM-adjacent transforms not fused by Inductor.

Do **not** begin here. First confirm the operator is materially hot after `torch.compile` and attention backend optimization.

---

## 6. Profiling and correctness — mandatory reading

### Nsight Systems

Wafer link: https://docs.nvidia.com/nsight-systems/UserGuide/

Use it for:

- CPU launch gaps;
- repeated synchronizations;
- H2D/D2H transfers;
- stream overlap;
- NCCL/peer communication;
- identifying the true end-to-end hot phases.

### Nsight Compute

Wafer link: https://docs.nvidia.com/nsight-compute/ProfilingGuide/

Use it for hot kernels after Systems finds them:

- roofline position;
- achieved memory throughput;
- Tensor Core utilization;
- instruction mix;
- register pressure;
- occupancy limits;
- shared-memory use;
- stall reasons.

### Compute Sanitizer

Wafer link: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/

Use it after writing or changing CUDA/Triton extensions. A fast kernel with a race or out-of-bounds read is not an optimization.

### CUTLASS GEMM measurement methodology

Wafer link: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html

Adopt its mindset even when benchmarking a full DiT:

- warm up;
- isolate initialization;
- use enough iterations;
- synchronize correctly;
- report exact shapes and dtypes;
- distinguish one-time setup from steady state.

---

## 7. Distributed resources translated to two A5500s

### NCCL

Wafer link: https://github.com/NVIDIA/nccl

Use for collective behavior, but remember that A5500 topology is not an 8-GPU NVSwitch node. The official A5500 bridge connects two cards at 112.5 GB/s bidirectional. Algorithm choices that assume datacenter topology can underperform.

### Megatron-LM parallelism paper

Wafer link: https://arxiv.org/abs/1909.08053

Useful for tensor/pipeline parallel mental models, but DiTs have different hot loops and activation shapes. For visual DiTs, sequence/context parallelism is often more relevant than LLM decode-specific strategies.

### Ring Attention

Wafer link: https://arxiv.org/abs/2310.01889

Useful for understanding exact distributed attention. On only two A5500s, ring-style communication has a short ring but can still be communication-heavy; profile it rather than assuming multi-GPU is faster.

---

## 8. Serving benchmark resources: what survives modality translation

Wafer lists MLPerf Inference and Etalon among serving references. Their exact LLM metrics are not directly your image/video metrics, but the methodological lessons transfer:

- define an SLO, not just average latency;
- report goodput under a latency target;
- distinguish queue time from service time;
- use realistic request-shape distributions;
- include cold/warm behavior;
- enforce correctness/quality alongside throughput.

For DiTs, replace token metrics with:

- end-to-end generation latency;
- denoising-only latency;
- frames or images per second;
- GPU-seconds per accepted output;
- queue + service p95/p99;
- benchmark quality pass rate.

---

## 9. Wafer resources to de-prioritize on A5500

These can teach future architecture concepts but should not drive implementation on the target hardware:

- Hopper Tuning Guide — TMA and Hopper-specific execution features do not map directly to A5500.
- Blackwell Tuning Guide — Blackwell-specific tensor memory and low-precision formats are out of scope.
- DeepGEMM — its highlighted production path is FP8/Hopper-oriented.
- FlashAttention-3 — Hopper-specific.
- FlashAttention-4 — Hopper/Blackwell schedule.
- OCP FP8/MX specifications — useful if studying formats, not a native A5500 performance path.
- Blackwell `tcgen05` instructions — not available on Ampere.

This is not a statement that those resources are bad. It is hardware triage.

---

## 10. Suggested learning order for this project

1. [Target hardware contract](../00_foundations/04_target_hardware_and_quality_contract.md)
2. CUDA programming model + Ampere tuning guide
3. Roofline paper + [roofline chapter](../02_bottlenecks/01_roofline_and_arithmetic_intensity.md)
4. FlashAttention / FlashAttention-2
5. Nsight Systems and Nsight Compute
6. [A5500 optimization ladder](../03_single_gpu/08_a5500_optimization_ladder.md)
7. Triton/CUTLASS only for remaining hot kernels
8. NCCL/topology if adding a second A5500
9. serving benchmark methodology once kernel/runtime work is stable

That order mirrors Wafer’s “single request -> single GPU -> kernels -> engines -> distributed system” organization while removing the LLM-specific detours that are not directly useful to DiT inference.
