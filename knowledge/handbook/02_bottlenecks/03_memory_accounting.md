# DiT Memory Accounting: Weights, Activations, Workspaces, Caches, and Transfers

## The first rule: “model size” is not “VRAM required”

A Diffusion Transformer inference pipeline can consume GPU memory through at least seven independent buckets:

1. **resident weights** — transformer, text encoders, VAE, control/adapters
2. **activations** — hidden states and Q/K/V tensors inside the current forward
3. **attention temporaries/workspaces** — backend-dependent scratch and softmax state
4. **compiler/CUDA Graph workspaces** — captured buffers, autotuned kernels, persistent allocations
5. **cache state** — intermediate block/attention/timestep caches
6. **pipeline buffers** — latents, prompt embeddings, image/video tensors, VAE tiles
7. **allocator slack/fragmentation** — reserved but not currently allocated memory

OOM debugging requires identifying which bucket grows.

## Weight memory

For a rough lower bound:

`weight_bytes ~= parameter_count * bytes_per_stored_parameter`

Common storage costs before metadata/scales:

- FP32: 4 B/parameter
- BF16/FP16: 2 B/parameter
- FP8/INT8: 1 B/parameter
- 4-bit: 0.5 B/parameter

Quantized formats add scales/zero-points/packing overhead. Layerwise casting can store weights at low precision and upcast only for compute, changing residency without necessarily changing compute precision.

### Pipeline weights can dwarf the DiT

Text-to-image models may include one or more large text encoders. Stable Diffusion 3-style pipelines, for example, can include three text encoders, including a large T5. If prompt embeddings can be precomputed or encoders offloaded after use, much of this residency can disappear from the denoising hot loop.

For video, the VAE can also be substantial and may need tile/chunk support.

## Activation memory

Activation memory scales with:

- batch size
- CFG implementation (separate pass vs concatenated batch)
- latent token count
- hidden width
- number/shape of intermediates kept alive
- attention backend
- compile/fusion behavior
- control/adaptor branches

Unlike training, inference does not need to save activations for backward, so the working set is much smaller. But long video contexts can still make Q/K/V and attention state dominant.

### Rough transformer block accounting

For one block with sequence `N`, width `D`, and bytes `b`:

- hidden state: `N * D * b`
- Q/K/V: about `3 * N * D * b`
- MLP expansion: often `N * D_ff * b`
- additional modulation/norm/residual buffers depending on fusion

Efficient kernels free/reuse buffers aggressively. The peak is determined by *lifetime overlap*, not the sum of every theoretical intermediate.

## Attention memory

Naive attention's score matrix scales as `N^2`, which is catastrophic at large visual contexts. Flash/memory-efficient attention avoids materializing the complete score/probability matrix, dramatically reducing the temporary footprint.

If switching attention backends changes peak memory by many gigabytes, this is expected; the backend changes intermediate materialization.

## Compiler and CUDA Graph memory

`torch.compile(mode="reduce-overhead")` and `max-autotune` can use CUDA Graphs and cached workspaces. This may reduce CPU overhead but **increase persistent memory**. It is possible for an eager model to fit while the compiled model OOMs.

Always record both:

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

The gap helps reveal allocator/compiler workspace effects.

## Cache memory

Caching trades memory for compute. Depending on method, you may retain:

- attention outputs
- block residuals
- hidden states
- model output from prior timestep
- polynomial/Taylor history

A cache that makes the model 30% faster but reduces concurrency by 50% may be a throughput regression in production. Benchmark cache bytes per request.

## CFG memory

Classic classifier-free guidance may be implemented as:

- two sequential denoiser calls: lower peak activations, more launches/overhead
- one concatenated batch of conditional + unconditional samples: higher peak activations, potentially better utilization
- parallel branches on separate GPUs: lower wall-clock latency at hardware cost
- guidance-distilled / guidance-free model: removes the duplicate branch entirely

Memory and speed are coupled here. Do not treat “batch size two” as a fixed law; inspect the pipeline implementation.

## VAE memory

Decoding a large image or video can become the end-to-end memory peak even when the transformer fits. Common controls:

- slicing across batch
- tiling spatially
- temporal chunking for video
- lower precision, if decoder quality remains safe
- compile the VAE separately
- decode on a separate or lower-cost device in asynchronous pipelines

VAE tiling/chunking lowers peak memory at the cost of extra overlap/reassembly and possibly latency/seam risk.

## CPU RAM and pinned-memory pressure

Offloading does not make memory disappear; it moves it. If a 20+ GB transformer is repeatedly moved between CPU and GPU:

- host RAM must hold the weights
- pinned buffers may be needed for asynchronous DMA
- PCIe becomes part of every forward
- NUMA placement can matter on multi-socket systems

Offloading should be measured as a *bandwidth schedule*, not merely a VRAM checkbox.

## A peak-memory measurement protocol

```python
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# run exactly one warmed generation here

torch.cuda.synchronize()
print("allocated_GB", torch.cuda.max_memory_allocated() / 2**30)
print("reserved_GB", torch.cuda.max_memory_reserved() / 2**30)
```

Run after warmup because compiled kernels and lazy allocations can change the steady-state peak.

For multi-GPU jobs, print the peak on every rank; the worst rank determines fit.

## Memory diagnosis table

| Symptom | Likely bucket | Best first experiment |
|---|---|---|
| OOM during load | weights | shard/stream load, offload cold components, two-GPU placement; test compressed storage only after these exact approaches and only under the no-regression gate |
| OOM only at high resolution | activations/attention | memory-efficient attention, CP, smaller canvas |
| OOM only after compile | graph/workspace | disable CUDA graphs, regional compile, reserve margin |
| OOM during VAE decode | decoder activations | tiling/slicing/chunking |
| VRAM grows after enabling cache | cache state | inspect cache tensors, tighten interval/threshold |
| Large allocated/reserved gap | allocator/workspace | fixed shapes, allocator tuning, process restart |
| Fits first request, OOM with concurrency | per-request activations/cache | lower concurrency, memory scheduler |
| CPU RAM explodes with offload | host weights/buffers | component-specific offload, memory-mapped/sharded loading, reduce duplicate host copies, or use the second A5500; compressed host storage is optional only if benchmark parity is proven |

## Memory optimization hierarchy

1. Remove unneeded components from residency.
2. Store weights more compactly.
3. Use memory-efficient attention.
4. Reduce sequence size / frames / resolution if product permits.
5. Offload cold components before hot transformer layers.
6. Tile/chunk VAE.
7. Use context parallelism for activation-dominated long sequences.
8. Only then consider per-layer CPU offload of the hot DiT, because it can destroy latency.

## Source basis

**PDF-derived:** low precision reduces memory burden; image/video attention is large despite smaller parameter counts; kernel fusion and memory-efficient attention reduce accesses; video context parallelism replicates weights while splitting attention/latent work.  
**Expansion:** complete memory bucket accounting, compiled workspace effects, CFG/VAE/offload-specific diagnosis.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
