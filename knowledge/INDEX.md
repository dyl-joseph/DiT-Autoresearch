# Knowledge Index for the AutoResearch Agent

The authoritative local knowledge base is `knowledge/handbook/`. It contains the complete compiled Diffusion Transformer inference-engineering documentation from the earlier research pass.

## Mandatory every-session reading

1. `handbook/00_foundations/04_target_hardware_and_quality_contract.md`
2. `handbook/00_foundations/02_benchmarking_and_quality_gates.md`
3. `handbook/03_single_gpu/08_a5500_optimization_ladder.md`
4. `handbook/02_bottlenecks/05_nsight_a5500_profiling_workflow.md`
5. `handbook/09_reference/09_a5500_benchmark_runbook.md`
6. `handbook/09_reference/10_reproducibility_and_claims_standard.md`

## Route by symptom

| Symptom | Start with |
|---|---|
| Attention dominates | `02_bottlenecks/02_attention_cost.md`, `03_single_gpu/03_attention_backends.md`, `03_single_gpu/09_ampere_kernel_engineering.md` |
| GPU gaps / CPU launch bound | `02_bottlenecks/04_profiling_symptoms.md`, `03_single_gpu/05_compile_fusion_cuda_graphs.md` |
| OOM / near 24 GB ceiling | `02_bottlenecks/03_memory_accounting.md`, all of `04_memory/`, `07_hardware/04_rtx_a5500_ampere_playbook.md` |
| GEMMs dominate | `02_bottlenecks/01_roofline_and_arithmetic_intensity.md`, `03_single_gpu/09_ampere_kernel_engineering.md` |
| VAE/text encoder overhead | `03_single_gpu/07_text_encoder_vae_pipeline_overhead.md`, `04_memory/02_offloading_and_device_maps.md`, `04_memory/04_vae_tiling_slicing_chunking.md` |
| Compile is unstable/slow | `03_single_gpu/05_compile_fusion_cuda_graphs.md`, `08_production/03_cold_start_compile_cache_shape_buckets.md` |
| Need 2 GPUs | all of `05_multi_gpu/`, especially `04_interconnect_topology.md` |
| Video DiT | `01_architecture/03_image_video_3d_scaling.md`, `06_model_playbooks/05_video_dits.md`, `05_multi_gpu/01_context_and_sequence_parallelism.md` |
| FLUX | `06_model_playbooks/03_flux.md` |
| SD3/MMDiT | `06_model_playbooks/02_sd3_mmdit.md` |
| Qwen/Hunyuan image | `06_model_playbooks/04_qwen_image_hunyuan_image.md` |
| Control/LoRA/editing | `06_model_playbooks/06_control_lora_editing.md` |
| Considering quantization | `03_single_gpu/04_quantization_and_precision.md`, `00_foundations/04_target_hardware_and_quality_contract.md`, `09_reference/10_reproducibility_and_claims_standard.md` |
| Need benchmark design | `00_foundations/02_benchmarking_and_quality_gates.md`, `09_reference/03_benchmark_harness_templates.md`, `09_reference/09_a5500_benchmark_runbook.md` |

## Search rule

Use `dit-ar search` before guessing. Search combines model family, mechanism, bottleneck, and hardware. Example:

```bash
dit-ar search "FLUX A5500 attention sequence length SDPA FA2"
```

The knowledge base is advisory. The fixed benchmark/evaluator decides whether a candidate is actually good on the target system.
