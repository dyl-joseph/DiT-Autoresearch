# Playbook: FLUX-Style Flow Diffusion Transformers

## Why FLUX-like models are a good systems benchmark

FLUX-class text-to-image models combine a large transformer denoiser, flow-matching-style inference, large text conditioning, and high-resolution generation. They are large enough that precision, compile, and memory planning matter, but still commonly run on a single high-memory GPU.

This playbook is architectural, not tied to one checkpoint license or exact parameter count.

## Optimization priorities

1. Keep transformer resident if possible.
2. Cache/reuse prompt embeddings for repeated prompts.
3. Use model-native recommended step schedule before tuning.
4. Compile repeated transformer blocks.
5. Benchmark native/Flash/Sage attention.
6. On RTX A5500, keep FP16/BF16; use exact offload/sharding for fit. Low-bit storage is optional only if the strict quality gate passes.
7. Apply FirstBlockCache/Cache-DiT only after baseline.
8. Use CP only at extreme resolutions or multi-GPU latency targets.

## Precision

FLUX-like models have large linear layers, which makes them common quantization demonstrations on newer GPUs. **For RTX A5500, this does not imply a native FP8 speed path.** First use FP16/BF16 + `torch.compile` + FA2/SDPA and exact component/offload strategies. Treat low-bit weight storage only as a final capacity experiment.

If any low-bit/quantized configuration fails the quality benchmark:

- preserve norms/modulation/input/output layers
- keep attention higher precision
- use weight-only/layerwise casting for capacity

## Compile

Repeated transformer blocks are ideal for regional compilation. Two profiles are useful:

### Long-lived worker

- `max-autotune`
- fixed shape buckets
- CUDA Graphs if compatible

### Autoscaled / cold-start-sensitive worker

- regional compilation
- prewarm common shape
- cache artifacts where runtime supports it

## Attention

At normal 1K image sizes, both projection/MLP and attention matter. At larger canvases attention share rises.

Benchmark:

- native SDPA
- FlashAttention family matching GPU
- do not use SageAttention unless the strict no-regression benchmark passes; the default A5500 path is exact attention

Run text-heavy prompts because attention quantization may weaken prompt fidelity before general aesthetics visibly degrade.

## FirstBlockCache

FLUX is a common target for FirstBlockCache/ParaAttention/Cache-DiT because adjacent denoising states can be redundant.

Tune threshold by:

- model variant
- step count
- image resolution
- guidance configuration

Do not use a universal threshold from another implementation.

## Prompt encoder handling

FLUX pipelines commonly use substantial text encoders. For repeated prompts:

- cache embeddings
- unload encoders during denoising

For one-off prompts on a high-VRAM GPU, keeping them resident can reduce latency and simplify serving, but costs concurrency headroom.

## Multi-GPU

### Throughput

Data parallel replicas almost always win if one request fits and latency is acceptable.

### Single-request latency / large canvas

Context parallelism/ParaAttention can split token context. Use NVLink where possible. On PCIe, only sufficiently large sequences justify CP.

### CFG parallelism

Depends on the exact FLUX variant/pipeline and whether classic two-branch CFG is performed. Do not assume it is applicable.

## Consumer cards

On 24 GB or less:

- keep transformer weights in FP16/BF16 when they fit; otherwise use exact sharding/offload first
- offload text encoders and cache prompt embeddings
- test layerwise casting/low-bit storage only as a separately quality-gated capacity experiment
- avoid CPU offload inside every transformer block if interactive latency matters
- consider distilled/schnell/low-step variants if product quality permits

## Low-step variants

A fast/distilled FLUX-like checkpoint changes the systems problem: launch/compile/VAE overhead become a larger fraction. Re-profile rather than carrying over optimizations from the full-step model.

## High-resolution behavior

As resolution rises:

- token count increases
- attention becomes more expensive
- activation memory rises
- shape-specific compilation matters

At extreme resolutions, the tuning order becomes:

`FA2/SDPA -> compile -> exact component residency/offload -> CP/sharding if needed -> quality-gated cache -> VAE tiling`

while preserving step/quality policy.

## Example engine profiles

### “Quality”

- full recommended steps
- FP16/BF16 linear path; optional low-bit weight storage only after parity proof
- high-precision attention
- cache disabled or conservative

### “Balanced”

- reduced steps
- compiled FP16/BF16 transformer on A5500
- Flash attention
- moderate FirstBlockCache

### “Preview”

- distilled/few-step checkpoint
- compile/CUDA Graph
- guidance-minimized
- aggressive latency target

Treat these as product tiers, not one engine trying to do everything dynamically.

## Current references

- Diffusers FLUX pipelines: https://huggingface.co/docs/diffusers/main/api/pipelines/flux
- Diffusers torchao: https://huggingface.co/docs/diffusers/quantization/torchao
- ParaAttention: https://github.com/chengzeyi/ParaAttention

## Source basis

**PDF-derived:** image DiTs are compute-bound; optimize attention/GEMM, compile, low precision, steps, guidance.  
**Expansion:** FLUX-specific component residency and current ecosystem composition.

## RTX A5500 recommended recipe

FLUX-class models are exactly where public optimization examples can mislead A5500 users because many examples showcase newer GPU low-precision paths.

Use this order:

1. FP16/BF16 reference transformer;
2. cache/precompute T5/CLIP embeddings when possible;
3. remove text encoders from GPU before denoising;
4. benchmark SDPA versus FA2 with the real FLUX head/sequence shapes;
5. regional `torch.compile` on repeated transformer blocks;
6. stable image-shape buckets;
7. VAE phase-local placement;
8. CPU group offload if still above 24 GB;
9. two-A5500 sharding/context parallelism if available;
10. low-bit weight storage only if strict benchmark parity is demonstrated.

If a low-bit FLUX build passes visual benchmarks but loses prompt adherence or typography on required tests, reject it.
