# Sources and Further Reading — A5500 / DiT Edition

The most useful sources are organized in dependency order rather than by novelty.

## 1. Primary uploaded source

Philip Kiely, *Inference Engineering* (Baseten Books, 2026).

Most relevant sections:

- performance goals and benchmarking;
- image generation architecture and few-step models;
- video generation and attention;
- arithmetic intensity / bottlenecks;
- CUDA kernels and fusion;
- quantization quality risk;
- model parallelism;
- image/video modality optimization;
- production profiling, cold start, observability.

The book’s Hopper/Blackwell details are retained as context, but this project targets Ampere.

---

## 2. Wafer AI GPU performance engineering resources

Repository: https://github.com/wafer-ai/gpu-perf-engineering-resources

Why it matters:

- dependency-ordered GPU performance learning path;
- preference for primary papers/specifications/repositories;
- explicit measurement/correctness requirements for performance claims;
- strong coverage of CUDA fundamentals, GEMM, attention, Triton/CUTLASS, profiling, collectives, and serving.

Project mapping: [08_wafer_gpu_perf_resource_map.md](08_wafer_gpu_perf_resource_map.md)

The repository’s contributing guide states that a performance number needs hardware/software versions, workload shapes/distribution, precision/algorithm, baseline, and correctness method. This handbook adds DiT prompt/seed/scheduler/quality data to that list.

---

## 3. RTX A5500 official references

### NVIDIA RTX A5500 datasheet

https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/proviz-nvidia-rtx-a5500-datasheet-2130578-r3-us-web.pdf

Use for:

- 24 GB GDDR6 ECC;
- 768 GB/s memory bandwidth;
- PCIe 4.0 x16;
- 230 W;
- 2-card NVLink support;
- 112.5 GB/s bidirectional NVLink.

### NVIDIA Ampere tuning guide

https://docs.nvidia.com/cuda/ampere-tuning-guide/

Use for:

- cc 8.6 occupancy/register/shared-memory constraints;
- asynchronous global→shared copies;
- FP16/BF16/TF32 Tensor Core modes;
- L1/shared-memory organization;
- explicit cc 8.6 compilation guidance.

### CUDA GPU compute capability

https://developer.nvidia.com/cuda/gpus

Always verify the installed device with `torch.cuda.get_device_capability()` or CUDA `deviceQuery` rather than relying only on static tables.

---

## 4. CUDA fundamentals from the Wafer path

### CUDA Programming Guide

https://docs.nvidia.com/cuda/cuda-programming-guide/

### CUDA Best Practices Guide

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

### PTX ISA

https://docs.nvidia.com/cuda/parallel-thread-execution/

### CUDA Binary Utilities

https://docs.nvidia.com/cuda/cuda-binary-utilities/

Use `cuobjdump`/`nvdisasm` to verify custom extensions contain native target code when architecture mismatch is suspected.

---

## 5. Roofline / GPU math

### Roofline paper

https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html

### Volkov dense-linear-algebra tuning paper

https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf

### CUDA matmul optimization worklog

https://siboehm.com/articles/22/CUDA-MMM

These build the mental model for compute versus bandwidth and why maximum occupancy is not automatically maximum performance.

---

## 6. Attention

### FlashAttention / FlashAttention-2

Paper/repo:

- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2307.08691
- https://github.com/Dao-AILab/flash-attention

The official repository identifies FA2 CUDA support for Ampere/Ada/Hopper in fp16/bf16. This is the relevant external FlashAttention generation for A5500.

FA3 is Hopper-specific and FA4 is Hopper/Blackwell-targeted; they are not A5500 deployment paths.

### Diffusers attention backends

https://huggingface.co/docs/diffusers/en/optimization/attention_backends

Use for current integration API and backend selection behavior.

### SageAttention

https://github.com/thu-ml/SageAttention

Research-only for this project because it uses low-precision/approximate attention. Deployment requires strict no-regression quality proof.

---

## 7. Compilation and custom kernels

### PyTorch `torch.compile`

https://docs.pytorch.org/docs/stable/generated/torch.compile

### Triton

- https://triton-lang.org/main/programming-guide/chapter-1/introduction.html
- https://github.com/triton-lang/triton

### CUTLASS / CuTe

- https://github.com/NVIDIA/cutlass
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html

Filter examples for Ampere compatibility. Hopper TMA and Blackwell `tcgen05` examples are not A5500 mechanisms.

---

## 8. Profiling and correctness

### Nsight Systems

https://docs.nvidia.com/nsight-systems/UserGuide/

### Nsight Compute

https://docs.nvidia.com/nsight-compute/ProfilingGuide/

### Compute Sanitizer

https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/

### CUTLASS GEMM measurement methodology

https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html

These are required reading before serious custom kernel work.

---

## 9. Diffusers memory/performance

### Memory optimization

https://huggingface.co/docs/diffusers/optimization/memory

### General acceleration

https://huggingface.co/docs/diffusers/optimization/fp16

### Caching

https://huggingface.co/docs/diffusers/main/optimization/cache

### Distributed inference

https://huggingface.co/docs/diffusers/main/training/distributed_inference

### Quantization overview

https://huggingface.co/docs/diffusers/main/quantization/overview

Quantization docs are implementation references, not permission to accept benchmark degradation.

---

## 10. DiT-specific distributed/caching systems

### xDiT

https://github.com/xdit-project/xDiT

### ParaAttention

https://github.com/chengzeyi/ParaAttention

### Cache-DiT

https://github.com/vipshop/cache-dit

### AdaCache

https://github.com/AdaCache-DiT/AdaCache

### SmoothCache

https://github.com/roblox/SmoothCache

All approximate caching techniques require project-specific no-regression evaluation.

---

## 11. Distributed communication

### NCCL

https://github.com/NVIDIA/nccl

### Ring Attention

https://arxiv.org/abs/2310.01889

### Megatron-LM tensor/pipeline parallelism

https://arxiv.org/abs/1909.08053

Translate datacenter parallelism results to the actual two-A5500 topology; do not import NVSwitch assumptions.

---

## 12. Serving methodology

### MLPerf Inference

https://www.cs.toronto.edu/ecosystem/papers/ISCA_20/MLPerf%20Inference.pdf

### Etalon

https://arxiv.org/html/2407.07000

Their exact metrics may be LLM-oriented, but the methodology—tail latency, realistic load, correctness, SLOs—transfers to DiT serving.

---

## 13. Source-use policy for this handbook

When adding a new optimization claim:

1. prefer a primary source;
2. state the mechanism;
3. state the hardware applicability;
4. measure it on RTX A5500;
5. record shape/dtype/software;
6. compare against the strongest exact baseline;
7. run the quality benchmark;
8. omit portable speedup claims if the context is incomplete.

That policy is more valuable than any single optimization trick.
