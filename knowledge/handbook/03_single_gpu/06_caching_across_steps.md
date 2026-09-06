# Caching Across Denoising Steps and Transformer Blocks

## Why caching works for diffusion

Successive denoising timesteps often produce similar intermediate representations. A full transformer forward recomputes many values even when the latent changes only slightly. Cache methods exploit this temporal redundancy by reusing or approximating intermediate outputs.

The uploaded PDF describes two broad video strategies:

- **timestep-based caching:** reuse outputs and skip entire model steps
- **transformer-based caching:** reuse hidden states and skip transformer layers

It reports that caching can materially accelerate video inference in practice, while warning that quality varies dramatically by algorithm/configuration.

## Cache taxonomy

### Full-step / denoiser-output reuse

Skip a denoiser evaluation and reuse/extrapolate a prior output.

**Maximum compute saved**, but approximation error directly changes the numerical trajectory.

### Block residual caching

Compute an early block or cheap change signal; if the state has not changed enough, reuse later block residuals.

Examples/concepts:

- FirstBlockCache
- DeepCache/DBCache-like residual gates
- Cache-DiT-style block caching

### Attention-output caching

Reuse attention results across nearby timesteps. PAB-style methods often use different reuse intervals for cross/spatial/temporal attention because their rate of change differs.

### Polynomial/Taylor prediction

Use recent history to approximate future intermediate outputs. TaylorSeer-style methods are an example.

### Learned caching/router policies

A learned controller decides which layers/timesteps can be skipped. These can outperform fixed heuristics but add training/calibration complexity.

## The cache decision signal

Common signals include:

- normalized L1/L2 difference in first-block input/output
- residual magnitude
- timestep distance
- attention output difference
- model-specific modulation/timestep embedding difference
- learned router score

A good signal is cheap. If the gate costs 10% of the block you hope to skip, the maximum speedup shrinks.

## Cache threshold tuning

Increasing threshold generally:

- increases hit rate
- lowers compute
- increases approximation error

Tune the **quality-speed Pareto curve**, not one threshold.

For every threshold record:

- skip ratio
- transformer time
- cache memory
- final quality
- worst-case prompts/videos

## Step-window policies

Caching is often unsafe at the very start and/or end of the trajectory. A robust policy can define:

- warmup: no cache for first `k` steps
- active region: cache allowed in middle steps
- cooldown: cache disabled for final refinement steps

This aligns with the broader principle that timesteps have unequal sensitivity.

## Video-specific caching

Video has extreme compute per timestep, so caching has enormous upside. Quality gates must include temporal metrics because cache error can manifest as:

- flicker
- frozen motion
- ghosting
- repeated texture
- identity drift
- motion discontinuity

Framewise CLIP/aesthetic scores may not detect these.

## PAB-style attention caching

Pyramid Attention Broadcast is based on the observation that attention outputs change slowly across some neighboring timesteps. Cross-attention may be reusable over longer intervals than spatial/temporal attention for some video architectures.

Not every video DiT has separate spatial, temporal, and cross-attention blocks. Inspect the model architecture before applying a generic PAB policy.

## FirstBlockCache

FirstBlockCache evaluates how much the early denoiser state changes. If the change is below a threshold, later expensive blocks can be skipped and prior residuals reused.

Why it is attractive:

- architecture-agnostic concept
- cheap gate relative to full transformer
- composable with attention kernels and some parallel methods

Why it is risky:

- threshold is model/scheduler/step-count dependent
- control/edit inputs can change sensitivity
- the same threshold may not generalize across resolutions

## Caching and quantization

Quantization changes the numerical differences used by many cache gates. A threshold tuned in BF16 may produce a different hit rate after any precision/quantization change. On A5500, treat low-bit cache interactions as experimental rather than assuming an FP8 runtime.

Retune in this order:

1. choose precision
2. choose scheduler/step count
3. choose attention backend
4. tune cache

Then evaluate the *combined* stack.

## Caching and compile

A dynamic cache branch can cause graph breaks. Prefer implementations that:

- keep cache logic in tensors/custom ops
- compile stable regions separately
- use predictable cache-state shapes

Sometimes compiling the full cache decision path is harder than compiling repeated blocks. Measure graph breaks.

## Caching and multi-GPU

Caches must be coherent across sequence/context-parallel ranks. A block residual is often sharded; cache reuse must preserve the same partition/layout. Some frameworks explicitly compose FirstBlockCache with context parallelism.

Cache can reduce compute without reducing communication proportionally, causing collectives to become the new bottleneck. Re-profile after enabling it.

## Cross-request caching

Most DiT cache methods are **intra-request**: they reuse states only within one denoising trajectory. Cross-request semantic caching is a newer direction and is harder because:

- prompts/seeds/shapes differ
- latent trajectories differ
- reuse risks quality/privacy issues
- cache lookup itself costs time/memory

Treat cross-request reuse as experimental unless the workload is highly repetitive and correctness is carefully defined.

## Cache memory budgeting

For each cache tensor compute:

`bytes = elements * dtype_bytes * number_of_cached_slots`

Then multiply by concurrency. A cache that is cheap for batch 1 can dominate fleet VRAM at high concurrency.

## A production cache tuner

For each model/shape/step bucket:

1. start from cache disabled
2. enable conservative threshold
3. collect skip ratio + latency + quality
4. increase threshold until quality gate is approached
5. back off with safety margin
6. canary on real traffic
7. monitor quality proxies and cache hit distributions

Never copy a threshold from a blog post without reproducing it on your checkpoint.

## Current references

- Diffusers caching: https://huggingface.co/docs/diffusers/main/optimization/cache
- Cache-DiT: https://github.com/vipshop/cache-dit
- ParaAttention: https://github.com/chengzeyi/ParaAttention
- AdaCache: https://github.com/AdaCache-DiT/AdaCache

## Source basis

**PDF-derived:** timestep- and transformer-based caching, large potential for video, and quality-risk warning.  
**Expansion:** cache taxonomy, threshold tuning, PAB/FirstBlockCache composition, compile/quantization/multi-GPU interactions.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
