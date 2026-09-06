# Memory Fragmentation, Shape Buckets, Allocator Behavior, and OOM Stability

## Why “free VRAM” can still OOM

PyTorch's CUDA allocator reserves blocks and reuses them. Over time, a service with variable shapes and temporary workspaces can accumulate free fragments that are not large/contiguous enough for the next allocation pattern. You may see:

- `reserved >> allocated`
- OOM after many heterogeneous requests
- process restart temporarily fixes the issue

This is an allocator/workload-shape problem, not necessarily a true capacity shortage.

## Allocated vs reserved

- **allocated**: live tensors using memory
- **reserved**: memory held by the allocator for reuse

Track both. A large gap after warmup can be normal, especially with CUDA Graphs, but unexplained growth across requests needs investigation.

## Variable shapes create allocator churn

DiTs frequently accept arbitrary aspect ratios and video lengths. Different shapes request different:

- QKV buffers
- attention workspaces
- MLP intermediates
- VAE tensors
- compiled graph workspaces

Shape bucketing improves both compilation reuse and memory stability.

## Shape bucket design

Create a small set of latent-token buckets rather than an unbounded width/height grid. Group by **latent token count and aspect ratio**, because those determine compute/memory more directly than pixel dimensions.

For video, include frame/temporal length.

A bucket definition should specify:

- input range
- target/padded shape
- expected peak VRAM
- compiled graph key
- max concurrency

## CUDA Graph memory pools

CUDA Graph capture can hold stable memory pools for each captured shape. Capturing dozens of shapes can consume large persistent VRAM. Keep the graph cache bounded.

Eviction/restart strategy matters in long-lived serving.

## `empty_cache()` is not a hot-loop optimization

`torch.cuda.empty_cache()` returns unused cached blocks to the driver, but it:

- does not free live tensors
- can increase future allocation overhead
- may synchronize

Use only at controlled lifecycle boundaries or after unloading a large component, not every denoising step/request.

## Process-per-profile isolation

High-performance production often benefits from one worker process per GPU/profile:

- stable model/precision
- small shape-bucket set
- fixed attention backend
- fixed compile mode

This keeps allocator behavior deterministic. Route heterogeneous workloads to different worker pools rather than making one process serve every possible model/shape.

## OOM-safe admission control

Do not wait for CUDA OOM. Before enqueueing a request, estimate:

`needed = per_shape_activation + per_request_cache + temporary_margin`

and ensure:

`resident_weights + fixed_workspace + sum(active_requests) + needed < memory_budget`

Memory-aware concurrency is more reliable than a fixed request-count limit.

## Fragmentation diagnostic experiment

1. Start a fresh worker.
2. Run only one shape for 100 requests; record reserved memory.
3. Restart.
4. Alternate 10 very different shapes for 100 requests.
5. Compare reserved growth and OOM rate.

If heterogeneous traffic is much worse, use routing/buckets.

## Compiler cache limits

Dynamic shapes can create many compiled variants. Track:

- number of compiled graphs
- memory per graph/workspace
- code-cache size
- recompilation logs

A serving process that specializes on every new aspect ratio can eventually become both slow and memory-heavy.

## Reusing buffers

For custom runtimes, preallocate:

- latent buffers
- timestep arrays
- guidance merge buffers
- output staging buffers

Reuse by shape bucket. This avoids allocator churn and improves CUDA Graph compatibility.

## Memory safety margin

Never target 100% advertised VRAM. Leave margin for:

- driver/context
- NCCL
- lazy kernel modules
- compiler workspace
- output tensors
- monitoring/debugging hooks

The right margin is empirical and GPU/runtime specific.

## Source basis

**PDF-derived:** memory and caches are part of GPU performance; production inference requires reliable operating margins and benchmarking.  
**Expansion:** allocator fragmentation diagnosis, shape/graph-cache policies, memory-aware admission control.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
