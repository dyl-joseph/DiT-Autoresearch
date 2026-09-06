# Source Map and Provenance

This handbook now has three explicit source layers.

## Layer 1 — uploaded book: *Inference Engineering*

The uploaded PDF by Philip Kiely is the conceptual backbone. It establishes that inference optimization is a systems problem spanning runtime, hardware, kernels, techniques, and production, and it gives the specific visual-model observations this guide builds on.

The most important PDF-derived ideas for DiTs are:

- image/video generation is usually **compute-bound**, unlike bandwidth-bound autoregressive decode;
- attention over visual/video latent tokens is a major compute cost;
- kernel selection and fusion are required to reach the useful hardware roof;
- image/video generation has explicit quality-versus-speed tradeoffs through denoising steps and guidance;
- video attention can dominate total runtime;
- caching and parallelism become especially important for video;
- quantization can improve compute throughput but carries quality risk;
- benchmarking must separate inference-only behavior from end-to-end latency and must look beyond means to tails;
- optimization is about balancing latency, throughput, quality, memory, and economics rather than maximizing one number.

The book’s visual-inference material was converted into the DiT-specific notes in [03_pdf_dit_extraction_notes.md](03_pdf_dit_extraction_notes.md). The rest of the folder paraphrases and extends the ideas rather than reproducing the text.

### Important adaptation for this project

The book spends meaningful attention on Hopper and Blackwell because it surveys contemporary datacenter inference. Those GPU-specific recommendations do **not** automatically transfer to RTX A5500.

For this project:

- the **principles** transfer;
- Hopper/Blackwell-only instructions, FP8/MX formats, and FA3/FA4-specific implementation advice do not;
- Ampere/A5500-specific kernel and memory choices override them.

---

## Layer 2 — Wafer AI GPU performance engineering resource map

Repository: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)

The repository is not a DiT optimization implementation. Its value is methodology and source selection. It orders learning from one request and one GPU through kernels, profiling, inference engines, and distributed systems, and it emphasizes original papers, official specifications/documentation, creator repositories, and direct implementation work.

This handbook adopts four important Wafer principles.

### 1. Learn mechanisms in dependency order

The relevant chain for DiTs is:

`GPU execution/memory -> roofline -> GEMM/attention -> profiling -> compiler/runtime -> multi-GPU topology -> serving`

Skipping directly to random “optimization flags” makes it hard to diagnose why a speedup did or did not happen.

### 2. Prefer primary sources

For a kernel claim, prefer the FlashAttention paper/repository over a benchmark tweet. For an Ampere limit, prefer the NVIDIA tuning guide over a secondary spec site. For `torch.compile`, prefer PyTorch documentation and source behavior.

### 3. A performance number is incomplete without context

Wafer’s contribution guide requires:

- hardware and software versions;
- workload shape/request distribution;
- precision and algorithm;
- baseline;
- correctness method.

This guide extends “correctness” to visual model quality/benchmark accuracy.

### 4. Peak hardware figures are not measurements

A5500 peak FLOPS are useful for rough roofline thinking, not for claiming model latency. Measured kernel throughput, achieved occupancy, memory traffic, and end-to-end timing determine the actual result.

A detailed DiT/A5500 mapping of the Wafer reading list is in [../09_reference/08_wafer_gpu_perf_resource_map.md](../09_reference/08_wafer_gpu_perf_resource_map.md).

---

## Layer 3 — target-hardware and implementation references

### NVIDIA RTX A5500 datasheet

https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/proviz-nvidia-rtx-a5500-datasheet-2130578-r3-us-web.pdf

Used for:

- 24 GB GDDR6 ECC
- 768 GB/s memory bandwidth
- PCIe 4.0 x16
- 230 W board power
- optional two-card NVLink
- 112.5 GB/s bidirectional NVLink figure

### NVIDIA Ampere tuning guide

https://docs.nvidia.com/cuda/ampere-tuning-guide/

Used for:

- compute capability 8.6 tuning characteristics
- shared memory and register limits
- asynchronous global→shared copy
- Tensor Core FP16/BF16/TF32 support
- compiling explicitly for cc 8.6
- Ampere memory-system behavior

### CUDA and kernel engineering references highlighted by Wafer

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities: https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CUTLASS: https://github.com/NVIDIA/cutlass
- Triton: https://github.com/triton-lang/triton
- FlashAttention: https://github.com/Dao-AILab/flash-attention

### Profiling/correctness references highlighted by Wafer

- Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Compute Sanitizer: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/
- CUTLASS GEMM measurement methodology: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html

### Current DiT/runtime implementation references

- Diffusers optimization docs: https://huggingface.co/docs/diffusers/optimization/memory
- Diffusers attention backends: https://huggingface.co/docs/diffusers/en/optimization/attention_backends
- Diffusers caching: https://huggingface.co/docs/diffusers/main/optimization/cache
- Diffusers distributed inference: https://huggingface.co/docs/diffusers/main/training/distributed_inference
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile
- xDiT: https://github.com/xdit-project/xDiT
- Cache-DiT: https://github.com/vipshop/cache-dit
- ParaAttention: https://github.com/chengzeyi/ParaAttention

---

## What is source-derived versus project policy

### Directly source-derived

- visual-model compute-bound framing from the PDF;
- attention/kernel/fusion/parallelism emphasis from the PDF;
- A5500 hardware specs from NVIDIA;
- cc 8.6 architectural tuning constraints from NVIDIA;
- Wafer’s performance-claim completeness standard and primary-source hierarchy;
- FlashAttention-2’s Ampere support and FA3/FA4’s newer-architecture targeting from its implementation repository.

### Project-specific policy

The following are deliberate constraints for this project rather than universal claims from the sources:

- **quantization is rejected if benchmark accuracy decreases**;
- exact/runtime optimizations are attempted before arithmetic approximation;
- Hopper/Blackwell-only paths are excluded from the primary optimization plan;
- A5500 and `sm_86` are the reference compilation/benchmark target;
- 24 GB VRAM management should prefer offload/sharding before low-bit conversion when accuracy cannot move.

Keeping this distinction matters: a source can say a technique is fast in general while the project can still reject it for hardware or quality reasons.
