# Diffusion Transformer Inference Engineering — RTX A5500 / Ampere Edition

A practical, architecture-aware handbook for making Diffusion Transformers (DiTs) faster, more memory-efficient, and easier to serve **on NVIDIA RTX A5500 GPUs** while preserving benchmark accuracy.

**Scope date:** 2026-09-05.  
**Primary source:** Philip Kiely, *Inference Engineering* (2026), especially the visual-model, GPU, CUDA, benchmarking, quantization, and production chapters.  
**GPU-performance expansion source:** [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources), used as a primary-source-oriented reading and methodology map.  
**Target hardware:** NVIDIA RTX A5500, Ampere, 24 GB GDDR6 ECC, 768 GB/s memory bandwidth, PCIe 4.0 x16, optional 2-GPU NVLink.  
**Quality contract:** **quantization is not accepted if it reduces required benchmark accuracy.** The default plan assumes no quantization at all; quantized paths are experiments that must prove benchmark parity before they enter production.

This folder is not a transcription of the book or the Wafer repository. It is a derived engineering manual. The book supplies the inference-engineering principles and image/video baseline; the Wafer repository supplies a disciplined performance-engineering dependency order and source policy; the rest translates both into RTX A5500-specific DiT practice.

---

## Read this first: the hardware changes the optimization strategy

A large fraction of public DiT speed work targets H100/H200/B200-class accelerators. That is **not** the target here.

The RTX A5500 is an Ampere workstation GPU. The practical consequences are:

- **No native FP8 Tensor Core path.** FP8/MXFP8/FP4 recommendations for Hopper/Blackwell are out of scope as primary speed paths.
- **FlashAttention-3 and FlashAttention-4 are not the target kernels.** On A5500, focus on PyTorch SDPA and FlashAttention-2-class kernels, then benchmark the exact model shapes.
- **24 GB VRAM is a first-class constraint.** Large DiTs often need exact offload, sharding, prompt-encoder lifecycle management, VAE tiling, or multi-GPU execution before they need arithmetic approximation.
- **GDDR6 bandwidth is much lower than datacenter HBM.** Avoid unnecessary weight reloads, CPU↔GPU transfer churn, unfused intermediates, and allocator fragmentation.
- **Compile specifically for Ampere / `sm_86` after verifying the device capability on the machine.** Generic PTX or a binary tuned for another architecture can leave performance on the table.
- **Two A5500s can be linked with NVLink**, but that does not magically pool VRAM for arbitrary PyTorch code. Parallelism must explicitly use peer access/sharding/collectives and still be benchmarked.

The fastest path on this hardware is usually **exact work reduction at the implementation level**—better kernels, fewer launches, more reuse of truly invariant data, static-shape compilation, and better memory placement—before any approximation is considered.

---

## Non-negotiable optimization contract

Every candidate optimization must pass all three gates:

1. **Correctness / quality** — required benchmark accuracy does not regress.
2. **Performance** — the target metric improves under a reproducible benchmark.
3. **Resource safety** — VRAM, host RAM, thermals, startup time, and tail latency remain acceptable.

For quantization, the quality gate is especially strict:

> **If a quantized configuration reduces accuracy on any required benchmark beyond established measurement noise, reject it.**

If your policy is literally “zero observed drop,” use a zero-drop acceptance rule: candidate aggregate metrics must be at least the baseline, and paired pass/fail cases may not turn baseline passes into candidate failures. The benchmark chapter explains how to handle stochastic image/video metrics without pretending measurement noise is a real model regression.

This same discipline is recommended for approximate caching, reduced step count, CFG shortcuts, sparse attention, TF32 changes, and other techniques that can change outputs.

---

## The A5500 optimization order

Do not begin with quantization. Use this order unless profiling shows a different bottleneck:

1. **Freeze a baseline and benchmark suite.** Fixed model revision, scheduler, prompt set, seeds, resolution/frame buckets, software versions, and quality metrics.
2. **Profile before changing anything.** Nsight Systems for the timeline; Nsight Compute for hot kernels; PyTorch profiler for operator attribution.
3. **Use an Ampere-appropriate exact attention backend.** Benchmark SDPA and FlashAttention-2. Confirm you are not silently falling back to the math implementation.
4. **Compile the hot transformer.** Start with `torch.compile`; use regional compilation and static shape buckets; then test CUDA Graph capture where stable.
5. **Remove launch and memory traffic overhead.** Fuse compatible pointwise operations, avoid Python synchronization, eliminate repeated device transfers, and keep invariant tensors resident.
6. **Manage 24 GB VRAM deliberately.** Evict cold text encoders, cache prompt embeddings, use group/model offload, tile/chunk VAE work, and keep a safety margin for allocator/compiler workspaces.
7. **Tune pipeline components separately.** The text encoder and VAE can dominate when the denoiser becomes fast. Profile them rather than assuming the transformer is still the bottleneck.
8. **Use exact reuse where possible.** Prompt-embedding caching and static preprocessing caches are exact. Timestep/hidden-state caches are approximate and require the same quality gate as quantization.
9. **Scale to a second A5500 only after one-GPU execution is healthy.** Prefer NVLink-connected placement when available and confirm traffic actually uses the fast path.
10. **Only then test quality-sensitive shortcuts.** Quantization, step reduction, approximate caches, guidance cutoff, sparse attention, etc. are rejected unless the required benchmark suite is preserved.

---

## What “performance” means in this handbook

Performance is a vector:

- warm end-to-end generation latency
- denoiser-only latency
- latency per denoising step
- images/second, frames/second, or videos/hour
- GPU-seconds per accepted output
- p50/p90/p95/p99 latency
- peak allocated and peak reserved VRAM
- persistent model residency
- host RAM and pinned-memory footprint
- CPU↔GPU bytes transferred per request
- GPU utilization, SM utilization, and Tensor Core utilization
- kernel launch count and graph-break count
- compile time and compiled-cache hit rate
- multi-GPU scaling efficiency
- quality/accuracy on the required benchmark suite
- thermal stability and sustained clocks

A faster microbenchmark that worsens end-to-end latency is not an optimization. A lower-VRAM configuration that spends most of its time copying weights over PCIe is not an optimization. A quantized configuration that is faster but loses benchmark accuracy is not an optimization.

---

## Folder map

### `00_foundations/` — methodology and constraints

- [00_source_map.md](00_foundations/00_source_map.md) — what comes from the PDF, Wafer resource map, and A5500-specific research
- [01_optimization_principles.md](00_foundations/01_optimization_principles.md) — governing rules for optimization
- [02_benchmarking_and_quality_gates.md](00_foundations/02_benchmarking_and_quality_gates.md) — reproducible performance + no-regression quality gates
- [03_pdf_dit_extraction_notes.md](00_foundations/03_pdf_dit_extraction_notes.md) — deconstruction of the uploaded book into DiT rules
- [04_target_hardware_and_quality_contract.md](00_foundations/04_target_hardware_and_quality_contract.md) — **start here for the RTX A5500-specific contract**

### `01_architecture/` — DiT families and scaling behavior

- [01_dit_inference_anatomy.md](01_architecture/01_dit_inference_anatomy.md)
- [02_architecture_taxonomy.md](01_architecture/02_architecture_taxonomy.md)
- [03_image_video_3d_scaling.md](01_architecture/03_image_video_3d_scaling.md)
- [04_conditioning_and_pipeline_components.md](01_architecture/04_conditioning_and_pipeline_components.md)

### `02_bottlenecks/` — compute, bandwidth, launch, transfer, and communication diagnosis

- [01_roofline_and_arithmetic_intensity.md](02_bottlenecks/01_roofline_and_arithmetic_intensity.md)
- [02_attention_cost.md](02_bottlenecks/02_attention_cost.md)
- [03_memory_accounting.md](02_bottlenecks/03_memory_accounting.md)
- [04_profiling_symptoms.md](02_bottlenecks/04_profiling_symptoms.md)
- [05_nsight_a5500_profiling_workflow.md](02_bottlenecks/05_nsight_a5500_profiling_workflow.md) — **step-by-step Nsight Systems/Compute workflow**

### `03_single_gpu/` — highest-return work on one A5500

- [01_steps_schedulers_distillation.md](03_single_gpu/01_steps_schedulers_distillation.md)
- [02_cfg_and_guidance.md](03_single_gpu/02_cfg_and_guidance.md)
- [03_attention_backends.md](03_single_gpu/03_attention_backends.md)
- [04_quantization_and_precision.md](03_single_gpu/04_quantization_and_precision.md)
- [05_compile_fusion_cuda_graphs.md](03_single_gpu/05_compile_fusion_cuda_graphs.md)
- [06_caching_across_steps.md](03_single_gpu/06_caching_across_steps.md)
- [07_text_encoder_vae_pipeline_overhead.md](03_single_gpu/07_text_encoder_vae_pipeline_overhead.md)
- [08_a5500_optimization_ladder.md](03_single_gpu/08_a5500_optimization_ladder.md) — **step-by-step implementation order**
- [09_ampere_kernel_engineering.md](03_single_gpu/09_ampere_kernel_engineering.md) — **low-level GEMM/attention/fusion/Triton/CUDA/CUTLASS guidance for Ampere**

### `04_memory/` — fitting models into 24 GB without giving away quality

- [01_weight_storage_and_layerwise_casting.md](04_memory/01_weight_storage_and_layerwise_casting.md)
- [02_offloading_and_device_maps.md](04_memory/02_offloading_and_device_maps.md)
- [03_activation_and_attention_memory.md](04_memory/03_activation_and_attention_memory.md)
- [04_vae_tiling_slicing_chunking.md](04_memory/04_vae_tiling_slicing_chunking.md)
- [05_fragmentation_shapes_allocator.md](04_memory/05_fragmentation_shapes_allocator.md)

### `05_multi_gpu/` — one or more A5500 pairs

- [01_context_and_sequence_parallelism.md](05_multi_gpu/01_context_and_sequence_parallelism.md)
- [02_cfg_data_tensor_pipeline_parallel.md](05_multi_gpu/02_cfg_data_tensor_pipeline_parallel.md)
- [03_xdit_pipefusion_hybrid.md](05_multi_gpu/03_xdit_pipefusion_hybrid.md)
- [04_interconnect_topology.md](05_multi_gpu/04_interconnect_topology.md)

### `06_model_playbooks/` — architecture-specific recipes

- [01_small_and_mid_size_image_dits.md](06_model_playbooks/01_small_and_mid_size_image_dits.md)
- [02_sd3_mmdit.md](06_model_playbooks/02_sd3_mmdit.md)
- [03_flux.md](06_model_playbooks/03_flux.md)
- [04_qwen_image_hunyuan_image.md](06_model_playbooks/04_qwen_image_hunyuan_image.md)
- [05_video_dits.md](06_model_playbooks/05_video_dits.md)
- [06_control_lora_editing.md](06_model_playbooks/06_control_lora_editing.md)
- [07_other_modalities_and_hybrids.md](06_model_playbooks/07_other_modalities_and_hybrids.md)

### `07_hardware/` — A5500/Ampere hardware mechanics

- [01_gpu_selection_by_bottleneck.md](07_hardware/01_gpu_selection_by_bottleneck.md)
- [02_architecture_generations.md](07_hardware/02_architecture_generations.md)
- [03_vram_tier_playbooks.md](07_hardware/03_vram_tier_playbooks.md)
- [04_rtx_a5500_ampere_playbook.md](07_hardware/04_rtx_a5500_ampere_playbook.md) — **deep A5500 tuning guide**

### `08_production/` — preserving wins in a service

- [01_runtime_selection.md](08_production/01_runtime_selection.md)
- [02_serving_concurrency_batching.md](08_production/02_serving_concurrency_batching.md)
- [03_cold_start_compile_cache_shape_buckets.md](08_production/03_cold_start_compile_cache_shape_buckets.md)
- [04_observability_cost_reliability.md](08_production/04_observability_cost_reliability.md)

### `09_reference/` — runbooks, matrices, and primary sources

- [01_optimization_decision_tree.md](09_reference/01_optimization_decision_tree.md)
- [02_compatibility_interaction_matrix.md](09_reference/02_compatibility_interaction_matrix.md)
- [03_benchmark_harness_templates.md](09_reference/03_benchmark_harness_templates.md)
- [04_checklists.md](09_reference/04_checklists.md)
- [05_research_frontier.md](09_reference/05_research_frontier.md)
- [06_sources_and_further_reading.md](09_reference/06_sources_and_further_reading.md)
- [07_glossary.md](09_reference/07_glossary.md)
- [08_wafer_gpu_perf_resource_map.md](09_reference/08_wafer_gpu_perf_resource_map.md) — Wafer repo mapped specifically to DiT/A5500 work
- [09_a5500_benchmark_runbook.md](09_reference/09_a5500_benchmark_runbook.md)
- [10_reproducibility_and_claims_standard.md](09_reference/10_reproducibility_and_claims_standard.md) — **evidence standard for performance claims and no-regression changes**

---

## Recommended reading paths

### “I have one A5500 and the model is too slow”

`00_foundations/04 -> 09_reference/09 -> 02_bottlenecks/05 -> 03_single_gpu/08 -> 03_single_gpu/03 -> 03_single_gpu/05 -> 03_single_gpu/09`

### “The model does not fit in 24 GB and I refuse quality loss”

`00_foundations/04 -> 02_bottlenecks/03 -> 04_memory/01 -> 04_memory/02 -> 04_memory/04 -> 05_multi_gpu/04`

The guiding principle is: **offload or shard before quantizing** when benchmark parity cannot be guaranteed.

### “I have two A5500s”

`07_hardware/04 -> 05_multi_gpu/04 -> 05_multi_gpu/01 -> 05_multi_gpu/02 -> 09_reference/09`

### “I need to learn GPU performance engineering, not only copy flags”

`09_reference/08 -> 02_bottlenecks/01 -> 02_bottlenecks/04 -> 03_single_gpu/05 -> 07_hardware/04`

### “I am testing quantization but accuracy may not move at all”

`00_foundations/02 -> 03_single_gpu/04 -> 09_reference/09`

---

## Provenance and evidence standard

This handbook adopts the strongest idea from the Wafer repository’s source policy: a performance number is only meaningful when you know the **hardware, software versions, workload shape/distribution, precision/algorithm, baseline, and correctness method**. If one is missing, treat the number as anecdotal.

The uploaded *Inference Engineering* book emphasizes that image/video inference is usually compute-bound and that attention, kernel selection, fusion, quantization, and parallelism are the central performance levers. This folder keeps those principles, but changes the implementation choices for Ampere. In particular, newer-generation low-precision instructions and kernels are treated as architectural context rather than runnable recommendations.

See [00_foundations/00_source_map.md](00_foundations/00_source_map.md) for a detailed provenance map.
