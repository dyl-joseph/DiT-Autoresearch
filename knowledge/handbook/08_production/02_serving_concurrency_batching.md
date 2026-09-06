# Serving, Concurrency, Batching, Queueing, and Admission Control

## Compute-bound diffusion changes the batching story

For LLM decode, batching often increases arithmetic intensity and throughput dramatically. For a compute-bound image/video DiT, a single request may already saturate Tensor Cores. More batch can increase latency without proportional throughput gain.

The uploaded PDF notes that video generation commonly runs batch 1 even on a full multi-GPU node; the way to improve cost/throughput is often to make the model itself faster.

## Benchmark batch size rather than assuming

Sweep batch sizes and report:

- outputs/s
- latency per request
- GPU-seconds/output
- peak VRAM
- p95 under queue load

For image models, batch 2-4 may improve utilization if matrices are small. For video, batch 1 often remains optimal.

## Dynamic batching

Dynamic batching combines requests arriving near each other. Challenges for DiTs:

- different resolutions/frame counts
- different step counts
- different guidance modes
- different adapters/control inputs

Batch only compatible engine profiles. Otherwise padding/wasted steps can cost more than batching saves.

## Step-synchronous batching

A server can batch requests at the same denoising timestep/shape. This is easier if all requests use identical step schedules. Mixed scheduler/step requests fragment batches.

Production simplification: expose a small set of quality tiers with fixed schedules.

## Concurrency without batching

Multiple independent CUDA streams/requests can improve utilization only if one request leaves resources idle. On large compute-bound DiTs, concurrency can cause contention and *increase* total GPU-seconds.

Benchmark 1, 2, 4 concurrent requests at fixed shape.

## Memory-aware admission control

Every active request needs activations and perhaps cache. Define per-profile memory estimates and admit only if:

`fixed_weights + fixed_workspace + active_dynamic + new_request_dynamic < safe_budget`

This is more reliable than “max concurrency = 4.”

## Queueing policy

Different jobs have very different service times. Use separate queues or scheduling classes for:

- preview image
- high-quality image
- short video
- long/high-res video

Otherwise a long video can create head-of-line blocking for interactive images.

## Shortest-job / size-aware scheduling

If fairness requirements allow, scheduling by expected GPU-seconds can reduce mean latency. Estimate from:

- latent token count
- steps
- guidance evaluations
- model profile
- cache policy

Do not starve large jobs; use aging/fairness.

## Shape-aware routing

Route requests to workers already warmed/compiled for that shape bucket. Benefits:

- compile cache hits
- stable allocator memory
- predictable performance

A global load balancer should consider engine profile, not only GPU utilization.

## Multi-GPU group admission

For CP/TP video workers, reserve the entire group atomically. Do not let one rank serve an unrelated request. The request scheduler works in **GPU groups**, not individual cards.

## Data parallel replicas

If the model fits and latency is fine, replicate whole engines across GPUs. This usually scales throughput more cleanly than increasing per-request parallel degree.

Example 8 GPUs:

- 8 x single-GPU image replicas for throughput
- 2 x CP4 video replicas
- 1 x CP8 for premium low-latency/large video

Dynamic fleet partitioning can respond to workload mix.

## Cancellation

Generative requests are often canceled when a user changes prompt. Implement cooperative cancellation at denoising-step boundaries. Every canceled remaining step is pure saved GPU cost.

## Backpressure

When queue depth exceeds the SLO-safe limit:

- reject with retry hint
- degrade to preview tier
- spill to lower-priority/offline queue
- add replicas

Unbounded queues turn a fast model into a slow product.

## Metrics

- queue wait p50/p95/p99
- service time p50/p95
- GPU-seconds/output
- batch size distribution
- concurrency distribution
- memory admission rejections
- cancellation savings
- engine-profile cache hit

## Source basis

**PDF-derived:** latency-throughput tradeoff, batching/concurrency, video batch-1 compute-bound behavior, production queueing/routing/autoscaling.  
**Expansion:** DiT-compatible batching rules, memory-aware admission, size-aware scheduling and cancellation.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
