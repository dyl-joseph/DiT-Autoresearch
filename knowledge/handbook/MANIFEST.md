# Manifest — RTX A5500 / Ampere Edition

Generated handbook inventory after integrating the Wafer GPU performance-engineering resource map and applying the RTX A5500/no-regression constraints.

- Content Markdown files (excluding this manifest): **58**
- Approximate content words: **60,272**
- Approximate content lines: **14,149**
- Target hardware: **NVIDIA RTX A5500 (Ampere), 24 GB**
- Quantization policy: **reject any candidate that reduces required benchmark accuracy/quality**
- Scope date: **2026-09-05**

## Major refinements in this edition

- Reoriented all optimization advice around RTX A5500/Ampere rather than Hopper/Blackwell.
- Removed FP8/MXFP8/FP4 and FlashAttention-3/4 from the target execution path; they remain only as architecture/future context.
- Made FP16/BF16 + SDPA/FlashAttention-2 + compilation/fusion the default exact optimization path.
- Added strict quality gates for quantization, approximate caching, fewer steps, guidance shortcuts, and other output-changing techniques.
- Added A5500-specific roofline, memory, topology, thermal, Nsight, kernel-engineering, benchmark, and reproducibility guidance.
- Integrated the Wafer repository's primary-source and performance-claim discipline: hardware, software, workload, precision/algorithm, baseline, and correctness/quality must accompany measurements.

## Root

- [`README.md`](README.md) — Diffusion Transformer Inference Engineering — RTX A5500 / Ampere Edition (1,598 words, 216 lines)

## `00_foundations/`

- [`00_foundations/00_source_map.md`](00_foundations/00_source_map.md) — Source Map and Provenance (913 words, 155 lines)
- [`00_foundations/01_optimization_principles.md`](00_foundations/01_optimization_principles.md) — Optimization Principles for DiTs on RTX A5500 (1,928 words, 436 lines)
- [`00_foundations/02_benchmarking_and_quality_gates.md`](00_foundations/02_benchmarking_and_quality_gates.md) — Benchmarking and No-Regression Quality Gates (1,768 words, 617 lines)
- [`00_foundations/03_pdf_dit_extraction_notes.md`](00_foundations/03_pdf_dit_extraction_notes.md) — PDF Deconstruction Notes: DiT-Relevant Parts of *Inference Engineering* (1,714 words, 266 lines)
- [`00_foundations/04_target_hardware_and_quality_contract.md`](00_foundations/04_target_hardware_and_quality_contract.md) — Target Hardware and Quality Contract: RTX A5500 / Ampere (1,410 words, 226 lines)

## `01_architecture/`

- [`01_architecture/01_dit_inference_anatomy.md`](01_architecture/01_dit_inference_anatomy.md) — DiT Inference Anatomy (944 words, 181 lines)
- [`01_architecture/02_architecture_taxonomy.md`](01_architecture/02_architecture_taxonomy.md) — Architecture Taxonomy: Which Kind of Diffusion Transformer Are You Optimizing? (1,366 words, 268 lines)
- [`01_architecture/03_image_video_3d_scaling.md`](01_architecture/03_image_video_3d_scaling.md) — How Image, Video, and 3D/World DiTs Scale (969 words, 206 lines)
- [`01_architecture/04_conditioning_and_pipeline_components.md`](01_architecture/04_conditioning_and_pipeline_components.md) — Conditioning, Adapters, and Pipeline Components (958 words, 218 lines)

## `02_bottlenecks/`

- [`02_bottlenecks/01_roofline_and_arithmetic_intensity.md`](02_bottlenecks/01_roofline_and_arithmetic_intensity.md) — Roofline and Arithmetic Intensity for DiTs on RTX A5500 (1,126 words, 271 lines)
- [`02_bottlenecks/02_attention_cost.md`](02_bottlenecks/02_attention_cost.md) — Attention Cost in Image, Video, 3D, and Multimodal DiTs (1,385 words, 200 lines)
- [`02_bottlenecks/03_memory_accounting.md`](02_bottlenecks/03_memory_accounting.md) — DiT Memory Accounting: Weights, Activations, Workspaces, Caches, and Transfers (1,129 words, 187 lines)
- [`02_bottlenecks/04_profiling_symptoms.md`](02_bottlenecks/04_profiling_symptoms.md) — Profiling Symptoms and Diagnosis on RTX A5500 (1,221 words, 416 lines)
- [`02_bottlenecks/05_nsight_a5500_profiling_workflow.md`](02_bottlenecks/05_nsight_a5500_profiling_workflow.md) — Nsight Profiling Workflow for DiTs on RTX A5500 (1,681 words, 504 lines)

## `03_single_gpu/`

- [`03_single_gpu/01_steps_schedulers_distillation.md`](03_single_gpu/01_steps_schedulers_distillation.md) — Steps, Schedulers, Solvers, and Distillation: The Largest Speed Lever (1,055 words, 191 lines)
- [`03_single_gpu/02_cfg_and_guidance.md`](03_single_gpu/02_cfg_and_guidance.md) — Classifier-Free Guidance and Guidance Optimization (881 words, 165 lines)
- [`03_single_gpu/03_attention_backends.md`](03_single_gpu/03_attention_backends.md) — Attention Backends on RTX A5500: SDPA, FlashAttention-2, xFormers, and Quality-Sensitive Alternatives (1,382 words, 310 lines)
- [`03_single_gpu/04_quantization_and_precision.md`](03_single_gpu/04_quantization_and_precision.md) — Quantization and Precision on RTX A5500 — Strict No-Regression Policy (1,606 words, 385 lines)
- [`03_single_gpu/05_compile_fusion_cuda_graphs.md`](03_single_gpu/05_compile_fusion_cuda_graphs.md) — `torch.compile`, Fusion, and CUDA Graphs on RTX A5500 (1,333 words, 383 lines)
- [`03_single_gpu/06_caching_across_steps.md`](03_single_gpu/06_caching_across_steps.md) — Caching Across Denoising Steps and Transformer Blocks (984 words, 204 lines)
- [`03_single_gpu/07_text_encoder_vae_pipeline_overhead.md`](03_single_gpu/07_text_encoder_vae_pipeline_overhead.md) — Text Encoders, VAE, Scheduler, and Non-DiT Pipeline Overhead (903 words, 170 lines)
- [`03_single_gpu/08_a5500_optimization_ladder.md`](03_single_gpu/08_a5500_optimization_ladder.md) — RTX A5500 DiT Optimization Ladder (1,444 words, 478 lines)
- [`03_single_gpu/09_ampere_kernel_engineering.md`](03_single_gpu/09_ampere_kernel_engineering.md) — Ampere Kernel Engineering for DiT Inference on RTX A5500 (2,778 words, 558 lines)

## `04_memory/`

- [`04_memory/01_weight_storage_and_layerwise_casting.md`](04_memory/01_weight_storage_and_layerwise_casting.md) — Weight Residency and Storage on a 24 GB RTX A5500 (760 words, 195 lines)
- [`04_memory/02_offloading_and_device_maps.md`](04_memory/02_offloading_and_device_maps.md) — CPU/GPU Offloading and Device Placement on RTX A5500 (753 words, 223 lines)
- [`04_memory/03_activation_and_attention_memory.md`](04_memory/03_activation_and_attention_memory.md) — Activation and Attention Memory for Long Visual Sequences (763 words, 149 lines)
- [`04_memory/04_vae_tiling_slicing_chunking.md`](04_memory/04_vae_tiling_slicing_chunking.md) — VAE Tiling, Slicing, and Temporal Chunking (626 words, 139 lines)
- [`04_memory/05_fragmentation_shapes_allocator.md`](04_memory/05_fragmentation_shapes_allocator.md) — Memory Fragmentation, Shape Buckets, Allocator Behavior, and OOM Stability (633 words, 145 lines)

## `05_multi_gpu/`

- [`05_multi_gpu/01_context_and_sequence_parallelism.md`](05_multi_gpu/01_context_and_sequence_parallelism.md) — Context and Sequence Parallelism for Diffusion Transformers (956 words, 187 lines)
- [`05_multi_gpu/02_cfg_data_tensor_pipeline_parallel.md`](05_multi_gpu/02_cfg_data_tensor_pipeline_parallel.md) — CFG, Data, Tensor, and Pipeline Parallelism for DiTs (841 words, 166 lines)
- [`05_multi_gpu/03_xdit_pipefusion_hybrid.md`](05_multi_gpu/03_xdit_pipefusion_hybrid.md) — xDiT, PipeFusion, USP, and Hybrid Parallelism (676 words, 142 lines)
- [`05_multi_gpu/04_interconnect_topology.md`](05_multi_gpu/04_interconnect_topology.md) — Interconnect and Topology for Multiple RTX A5500 GPUs (700 words, 222 lines)

## `06_model_playbooks/`

- [`06_model_playbooks/01_small_and_mid_size_image_dits.md`](06_model_playbooks/01_small_and_mid_size_image_dits.md) — Playbook: Small and Mid-Size Image Diffusion Transformers (776 words, 163 lines)
- [`06_model_playbooks/02_sd3_mmdit.md`](06_model_playbooks/02_sd3_mmdit.md) — Playbook: Stable Diffusion 3 / MMDiT-Style Architectures (831 words, 160 lines)
- [`06_model_playbooks/03_flux.md`](06_model_playbooks/03_flux.md) — Playbook: FLUX-Style Flow Diffusion Transformers (818 words, 177 lines)
- [`06_model_playbooks/04_qwen_image_hunyuan_image.md`](06_model_playbooks/04_qwen_image_hunyuan_image.md) — Playbook: Qwen-Image, Hunyuan-Image, and LLM-Heavy Image Transformers (733 words, 145 lines)
- [`06_model_playbooks/05_video_dits.md`](06_model_playbooks/05_video_dits.md) — Playbook: Video Diffusion Transformers (929 words, 191 lines)
- [`06_model_playbooks/06_control_lora_editing.md`](06_model_playbooks/06_control_lora_editing.md) — Playbook: Control, LoRA, Image Editing, Inpainting, and Multi-Condition DiTs (733 words, 143 lines)
- [`06_model_playbooks/07_other_modalities_and_hybrids.md`](06_model_playbooks/07_other_modalities_and_hybrids.md) — Playbook: 3D, World Models, Audio DiTs, MoE DiTs, Discrete Diffusion, and Hybrids (938 words, 202 lines)

## `07_hardware/`

- [`07_hardware/01_gpu_selection_by_bottleneck.md`](07_hardware/01_gpu_selection_by_bottleneck.md) — Working Within Fixed Hardware: RTX A5500 Bottleneck Matching (331 words, 110 lines)
- [`07_hardware/02_architecture_generations.md`](07_hardware/02_architecture_generations.md) — GPU Architecture Generations — What Matters When You Only Have Ampere (289 words, 77 lines)
- [`07_hardware/03_vram_tier_playbooks.md`](07_hardware/03_vram_tier_playbooks.md) — 24 GB VRAM Playbook: RTX A5500 (348 words, 118 lines)
- [`07_hardware/04_rtx_a5500_ampere_playbook.md`](07_hardware/04_rtx_a5500_ampere_playbook.md) — NVIDIA RTX A5500 / Ampere Playbook for Diffusion Transformer Inference (2,046 words, 528 lines)

## `08_production/`

- [`08_production/01_runtime_selection.md`](08_production/01_runtime_selection.md) — Runtime Selection: Diffusers/PyTorch, SGLang Diffusion, TensorRT, and Custom Engines (606 words, 175 lines)
- [`08_production/02_serving_concurrency_batching.md`](08_production/02_serving_concurrency_batching.md) — Serving, Concurrency, Batching, Queueing, and Admission Control (658 words, 142 lines)
- [`08_production/03_cold_start_compile_cache_shape_buckets.md`](08_production/03_cold_start_compile_cache_shape_buckets.md) — Cold Starts, Compile Caches, Shape Buckets, and Engine Prewarming (604 words, 135 lines)
- [`08_production/04_observability_cost_reliability.md`](08_production/04_observability_cost_reliability.md) — Observability, Cost, Reliability, and Quality in Production DiT Serving (567 words, 167 lines)

## `09_reference/`

- [`09_reference/01_optimization_decision_tree.md`](09_reference/01_optimization_decision_tree.md) — RTX A5500 DiT Optimization Decision Tree (400 words, 148 lines)
- [`09_reference/02_compatibility_interaction_matrix.md`](09_reference/02_compatibility_interaction_matrix.md) — RTX A5500 Optimization Compatibility Matrix (365 words, 104 lines)
- [`09_reference/03_benchmark_harness_templates.md`](09_reference/03_benchmark_harness_templates.md) — Benchmark Harness Templates — RTX A5500 / No-Regression Edition (2,145 words, 620 lines)
- [`09_reference/04_checklists.md`](09_reference/04_checklists.md) — RTX A5500 Optimization Checklists (490 words, 132 lines)
- [`09_reference/05_research_frontier.md`](09_reference/05_research_frontier.md) — Research Frontier: Where DiT Inference Is Still Moving (860 words, 190 lines)
- [`09_reference/06_sources_and_further_reading.md`](09_reference/06_sources_and_further_reading.md) — Sources and Further Reading — A5500 / DiT Edition (889 words, 286 lines)
- [`09_reference/07_glossary.md`](09_reference/07_glossary.md) — Glossary of DiT Inference Engineering (1,219 words, 152 lines)
- [`09_reference/08_wafer_gpu_perf_resource_map.md`](09_reference/08_wafer_gpu_perf_resource_map.md) — Wafer GPU Performance Engineering Resources — DiT / RTX A5500 Map (1,367 words, 265 lines)
- [`09_reference/09_a5500_benchmark_runbook.md`](09_reference/09_a5500_benchmark_runbook.md) — RTX A5500 DiT Benchmark and Profiling Runbook (1,070 words, 490 lines)
- [`09_reference/10_reproducibility_and_claims_standard.md`](09_reference/10_reproducibility_and_claims_standard.md) — Reproducibility and Performance-Claim Standard (1,076 words, 350 lines)
