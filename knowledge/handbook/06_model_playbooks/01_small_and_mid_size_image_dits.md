# Playbook: Small and Mid-Size Image Diffusion Transformers

## Scope

This playbook covers image DiTs small enough to fit comfortably on one accelerator in BF16/FP16: classic patch-based DiT families, PixArt/AuraFlow/Sana-like image transformers, compact distilled models, and similar architectures where a single image latent is the main context.

The exact implementation differs by checkpoint, but the performance regime is often similar:

- one GPU is sufficient
- launch overhead and GEMM efficiency matter
- attention matters more at higher resolution
- weight offload is usually unnecessary
- fewer steps often dominates all kernel work

## Goal hierarchy

1. Minimize denoiser evaluations.
2. Keep the entire hot transformer resident.
3. Use FP16/BF16 on RTX A5500; quantization is not a default speed path and is rejected if benchmark accuracy drops.
4. Compile repeated blocks.
5. Select an attention backend by benchmark.
6. Cache only after the above baseline is stable.
7. Optimize text/VAE only when they become visible.

## Baseline configuration

A reasonable starting point:

- batch 1
- model-native scheduler and recommended steps
- BF16 if supported
- PyTorch native SDPA
- no CPU offload
- static 1K shape bucket
- `torch.inference_mode()`
- no cache

Record step latency and kernel breakdown.

## Most likely wins

### Step reduction / distilled checkpoint

If a 4-8 step distilled sibling passes your quality gate, it will usually beat a heavily optimized 30-50 step base model in latency.

### `torch.compile`

Small/mid-size models can be launch-sensitive. Regional compilation of repeated blocks often gives a good cold/warm tradeoff. Test `max-autotune` for long-lived services.

### Attention backend

At 512-768px, attention may not dominate. At 1K+ it becomes more important. Benchmark native SDPA vs Flash/Sage/xFormers candidates.

### Low-bit/quantization research branch on A5500

RTX A5500 has no native FP8 Tensor Core path. If memory pressure still justifies low-bit experimentation after exact offload/sharding, start with large linear-layer weight storage and require strict benchmark parity. Quantized attention is higher risk and should not be the first experiment.

## Avoid over-parallelizing

A small image DiT often becomes **slower** on two GPUs because communication exceeds saved compute. Prefer:

- one request per GPU (data parallel)
- CFG parallel only if single-image latency is the overriding objective

Do not use CP simply because multiple GPUs exist.

## Batch-size behavior

Because the model can be compute-bound, batching may increase throughput only modestly while increasing latency linearly-ish. Sweep batch 1/2/4/8 and plot:

- images/s
- latency/image
- GPU-seconds/image
- peak VRAM

Use offline batching only where throughput matters.

## Resolution buckets

Create separate engine profiles for:

- <=512/640
- 768
- ~1K
- >1K / high aspect ratio

The best attention kernel and compile specialization can differ.

## Cache policy

Cache is optional for small image DiTs. If generation is already 1-2 seconds, aggressive cache may save little absolute time while increasing approximation risk.

Use cache when:

- 20+ steps remain
- repeated blocks are expensive
- quality tests show redundancy

Tune FirstBlockCache-like thresholds per step count/resolution.

## Memory plan

If a 12-16 GB GPU is close to OOM:

1. offload text encoder after prompt encoding
2. exact component offload/sharding; low-bit/layerwise storage only after strict parity proof
3. use memory-efficient attention
4. tile VAE only if decode is the peak

Avoid per-block CPU offload unless no other option.

## Consumer GPU notes

On RTX-class GPUs:

- BF16/FP16 support and Tensor Core behavior vary by generation
- native FP8 benefits belong to newer architectures; A5500 should prioritize FP16/BF16 kernels, compile, and memory placement
- low-bit weight-only formats can help fit but may not speed up
- driver/PyTorch/kernel wheels matter enormously

Benchmark local stack; data-center numbers are not portable.

## Quality suite

Small image DiTs may fail first on:

- complex composition
- typography
- small details
- long prompts
- style specificity

Quantization, approximate caching, and step reduction can all hit the same weak points. **Test each change in isolation first** against the exact baseline; only test combinations after every constituent change passes its own quality gate.

## Recommended tuning sequence

```text
baseline
-> fewer steps/scheduler sweep
-> BF16/FP16 choice
-> compile repeated blocks
-> attention backend sweep
-> exact offload/sharding -> optional quality-gated low-bit weight storage
-> cache threshold sweep
-> VAE/text encoder tuning
```

Stop when the remaining hot kernels are near hardware limits or quality is the binding constraint.

## RTX A5500 recommended recipe

For small/mid image DiTs that fit in 24 GB:

1. FP16/BF16 reference path;
2. SDPA versus FlashAttention-2;
3. compile repeated blocks;
4. define fixed resolution buckets;
5. cache prompt embeddings when prompts repeat;
6. keep transformer resident and move encoder/VAE by phase only if needed;
7. use CUDA Graphs if the static pool still leaves safe VRAM headroom.

Do not quantize a model that already fits unless a benchmark-equal candidate proves a real end-to-end advantage. On A5500 the highest-return path is generally compiler/kernel efficiency, not low-bit compute.
