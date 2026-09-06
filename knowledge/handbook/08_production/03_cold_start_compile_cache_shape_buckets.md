# Cold Starts, Compile Caches, Shape Buckets, and Engine Prewarming

## Warm benchmarks are not enough

A `torch.compile(max-autotune)` DiT may be excellent after compilation and painful on a new worker. Production latency includes:

- container/image pull
- Python imports
- checkpoint download/read
- optional weight-format conversion/quantization **only for an already quality-approved deployment profile**
- GPU upload
- kernel JIT
- model compilation/autotuning
- CUDA Graph capture
- first request allocator setup

The uploaded PDF specifically notes that compilation can take minutes and that cached compiled engines are important for startup.

## Separate cold-start phases

Instrument:

1. process start
2. dependencies imported
3. weights available locally
4. weights loaded CPU
5. GPU residency complete
6. optional validated weight-format conversion complete (skip for the default exact profile)
7. compile started/completed
8. warmup complete
9. worker ready

This tells you where to invest.

## Weight locality

Avoid downloading multi-gigabyte checkpoints during autoscale. Options:

- bake into image/layer when practical
- local NVMe cache
- node-level shared cache
- prefetch before assigning traffic

Check model license/distribution requirements.

## Pre-converted weight formats at build time

The default A5500 profile is unquantized FP16/BF16. If—and only if—a quantized/compressed profile has already passed the full no-regression quality gate, serialize/preconvert that approved representation at build time when the backend supports it instead of repeating conversion during every worker startup. Store the exact quality-evidence/config ID alongside the artifact.

## Compile artifact cache

Cache keys must include every codegen-relevant variable:

- GPU architecture
- CUDA/driver compatibility
- PyTorch/Inductor/Triton version
- model code/revision
- dtype/quantization
- attention backend
- shape bucket
- compile flags

A stale cache can crash or silently select wrong behavior.

## Shape buckets

Each bucket can have its own warmed compiled graph. Keep bucket count small enough that:

- compilation time is bounded
- graph workspaces fit
- traffic per bucket amortizes setup

Use product analytics to choose high-frequency aspect ratios/frame lengths.

## Regional compilation for cold-start-sensitive services

Compile repeated transformer blocks rather than the full pipeline. Current Diffusers supports repeated-block regional compilation for many models. This can preserve runtime wins while reducing compilation wall time.

## Prewarming

Before marking a worker ready:

- run one or more requests in every high-priority shape bucket
- force attention kernel JIT/load
- capture graphs
- allocate cache buffers

Then reset metric counters so warmup is not mixed with service data.

## Lazy secondary profiles

Do not precompile every rare shape. Strategies:

- route rare shapes to flexible eager pool
- compile on first use only when queue is low
- evict least-used profiles with bounded cache

## Autoscaling economics

Compile time changes the minimum useful worker lifetime. If:

- startup = 180 s
- average service life = 300 s

then 37.5% of wall time is spent starting, before doing useful inference. Keep warm pools or use faster regional engines.

## Scale-to-zero tradeoff

Scale-to-zero saves idle cost but may be unacceptable for multi-minute cold starts. Consider:

- one warm replica
- lower-cost warm standby GPU
- predictive scaling around traffic patterns
- prewarm on queue-depth threshold

## Cold quality drift

The first compiled/captured request should be checked for numerical parity; lazy kernel selection and precision paths can differ from eager baseline.

## Source basis

**PDF-derived:** compilation/engine caching, cold starts, autoscaling and production reliability.  
**Expansion:** artifact cache keys, shape-bucket lifecycle, prewarm and autoscaling economics.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
