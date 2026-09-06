# Activation and Attention Memory for Long Visual Sequences

## When weights fit but the request OOMs

If a model loads successfully at 512px but OOMs at 2K or on a 49-frame video, the problem is usually **activation/workspace memory**, not weights.

Activation pressure grows with:

- token count
- hidden width
- batch/CFG
- control branches
- attention implementation
- cache state
- compiler workspaces

## Sequence length is the primary variable

For a patched 2D latent:

`N ≈ (H_lat / p_h) * (W_lat / p_w)`

For video:

`N ≈ (T_lat / p_t) * (H_lat / p_h) * (W_lat / p_w)`

This sequence drives Q/K/V and hidden-state sizes. Dense attention may also create quadratic workspaces if not using a memory-efficient kernel.

## Memory-efficient attention is mandatory at scale

FlashAttention/SDPA/xFormers-style algorithms avoid materializing the full attention matrix. If a high-resolution/video model is using a math attention fallback, fixing that should happen before almost any other memory technique.

Verify the actual backend rather than assuming the framework dispatched to flash.

## CFG activation multiplier

Batched CFG can approximately double the batch dimension for the DiT. If activation memory is the limiting factor:

- sequential CFG may fit where batched CFG OOMs
- guidance cutoffs reduce total work but not necessarily peak early-step memory
- guidance-free/distilled checkpoints can remove the branch

## Attention slicing

Older diffusion stacks use attention slicing to process heads/chunks sequentially. It can reduce memory but often slows inference. Modern Flash/memory-efficient attention is generally preferable when available.

## Chunking feed-forward layers

MLP activation memory can be reduced by processing sequence chunks through the feed-forward network. This trades extra launches and lower GEMM efficiency for capacity.

Use when:

- attention is already memory-efficient
- MLP intermediate is the remaining peak
- model framework exposes a chunking hook

## Context parallelism as activation sharding

Context/sequence parallelism splits tokens across GPUs. It is the direct answer when:

- one GPU can store the weights
- long-sequence activations do not fit
- multiple GPUs are available

Unlike weight sharding, each rank may keep a full weight copy. This is exactly why it is well-suited to video DiTs whose weights fit but latent context is enormous.

## Temporal/spatial chunking at architecture boundaries

Some video models or VAEs support:

- temporal tiling
- spatial tiling
- frame chunks

These can cap activation peaks but may alter global context or add boundary artifacts if applied inside the DiT rather than the decoder. Only use architecture-supported chunking.

## Cache state can erase activation savings

FirstBlock/PAB/Taylor-like cache methods keep additional intermediate tensors. At long contexts, these cached tensors can be gigabytes. Measure cache memory per request before enabling concurrency.

## Compile workspaces and activation headroom

Compiled/CUDA Graph paths may reserve fixed buffers sized for a shape bucket. A memory schedule should reserve:

`weights + max_activations + attention_workspace + graph_workspace + cache + safety_margin`

Do not fill the GPU to 99% with weights and hope activation allocations fit.

## Per-request concurrency planning

If weights are shared across requests but each request has activation/cache footprint `A`, a rough capacity bound is:

`max_concurrency <= floor((VRAM - weights - fixed_workspace) / A)`

Real schedulers need extra fragmentation margin. This formula explains why a 2 GB cache can be acceptable at concurrency 1 and disastrous at concurrency 8.

## Activation-memory profiling

Useful methods:

- peak CUDA allocated/reserved stats
- PyTorch memory snapshot
- layer/block hooks measuring active bytes
- Nsight memory traces

A binary-search technique is useful: insert synchronization + peak reset around groups of blocks to find the region that creates the peak.

## Resolution/frame bucketing by memory

Build a lookup table:

| Shape bucket | Steps | Peak FP16/BF16 | Peak experimental low-bit | Peak + cache | Recommended concurrency |
|---|---:|---:|---:|---:|---:|
| 1024² | | | | | |
| 1536² | | | | | |
| 720p x 49f | | | | | |

Use this table in admission control. Prevent OOM proactively instead of retrying after failure.

## Emergency memory levers in order of quality impact

1. memory-efficient attention
2. lower weight precision / layerwise casting
3. offload cold components
4. sequential instead of batched CFG
5. VAE tiling/chunking
6. context parallelism
7. MLP/sequence chunking
8. lower resolution/frame count
9. change model/checkpoint

Several of these affect speed before quality; use those first.

## Source basis

**PDF-derived:** visual attention is computationally heavy; FlashAttention reduces memory movement; video context parallelism splits the latent/attention while replicating weights.  
**Expansion:** concurrency formula, activation-specific controls, memory bucket table and admission-control strategy.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
