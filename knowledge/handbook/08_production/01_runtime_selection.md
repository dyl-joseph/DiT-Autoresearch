# Runtime Selection: Diffusers/PyTorch, SGLang Diffusion, TensorRT, and Custom Engines

## The runtime is an optimization envelope

The best runtime depends on whether you value:

- time to first working deployment
- maximum single-model speed
- model/adapter flexibility
- multi-GPU scaling
- compilation/caching
- production server features

The uploaded PDF describes high-performance image inference as commonly living lower in the stack than LLM inference and points to three broad options: optimized diffusion engines, TensorRT, and carefully tuned PyTorch.

## PyTorch + Diffusers

### Best for

- research and new architectures
- custom attention/cache logic
- LoRA/control/edit pipelines
- rapid model support
- fine-grained profiling

### Performance stack

- SDPA / attention dispatcher
- `torch.compile`
- regional compilation
- torchao quantization
- offloading/layerwise casting
- context parallelism
- cache hooks

### Drawbacks

- more tuning burden
- version interactions
- less turnkey serving

Use as the reference implementation even if production moves to another engine; it is useful for correctness and ablation.

## SGLang Diffusion

Current SGLang Diffusion describes itself as a high-performance image/video inference framework with native pipelines, Diffusers backend support, serving interfaces, optimized kernels, scheduler improvements, and cache acceleration.

### Best for

- popular supported image/video models
- production server path
- wanting optimized kernels without building everything manually
- multi-platform deployment where support exists

### Validate

- exact model variant support
- adapter/control support
- quality parity
- shape/precision support
- multi-GPU features

The runtime changes quickly; pin a tested version/commit.

## TensorRT

### Best for

- stable model architecture
- NVIDIA deployment
- maximum performance via engine building and vendor kernels
- fixed/bucketed shapes

### Advantages

- aggressive graph/kernels/fusion
- architecture-specific autotuning
- engine serialization

### Drawbacks

- build time
- less flexibility for novel custom blocks
- dynamic adapters/control paths can complicate engines
- engine artifacts tied to GPU/software profiles

## Custom PyTorch/Triton/CUDA engine

Build custom kernels only when profiling proves a meaningful unsupported bottleneck. Most teams should select/tune kernels rather than write attention or GEMM from scratch.

Custom work is justified for:

- a stable high-volume model
- unique fused modulation/norm patterns
- unusual attention layout
- deployment at enough scale to amortize maintenance

## Runtime evaluation matrix

| Dimension | Diffusers/PyTorch | SGLang Diffusion | TensorRT | Custom |
|---|---|---|---|---|
| New model support | best | varies | slower | manual |
| Peak tuned speed | high | high | very high | potentially highest |
| Flexibility | best | medium-high | medium | exact |
| Compile/build time | medium | medium | high | high |
| Serving features | basic/modular | strong | integrate yourself | build yourself |
| Maintenance | medium | medium | medium-high | highest |

## Benchmark runtime, not marketing

For each candidate run the same:

- model/checkpoint
- prompt/seed set
- shapes
- steps/guidance
- precision
- quality gate

Measure:

- cold start
- warm p50/p95
- peak VRAM
- throughput under concurrency
- multi-GPU efficiency
- feature correctness

## Engine profile versioning

A production engine artifact should be keyed by:

- model revision
- runtime version
- GPU architecture
- precision/quantization
- shape bucket
- attention backend
- adapter/control configuration

Rebuild on driver/CUDA/runtime upgrades if required.

## Fallback path

Keep a flexible PyTorch path for:

- unsupported shapes
- rare adapters
- new model versions
- debugging quality regressions

Route hot/common traffic to the optimized engine.

## Current references

- SGLang Diffusion: https://github.com/sgl-project/sglang/blob/main/docs/docs/sglang-diffusion/index.mdx
- TensorRT Diffusion demo: https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion
- Diffusers optimization docs: https://huggingface.co/docs/diffusers/optimization/fp16

## Source basis

**PDF-derived:** image inference runtime options, PyTorch customization, TensorRT, kernel selection/compilation.  
**Expansion:** production runtime decision matrix and engine-profile lifecycle.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
