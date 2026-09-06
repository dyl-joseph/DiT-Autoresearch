# RTX A5500 DiT Optimization Decision Tree

Use this after obtaining a baseline profile.

```text
START
  |
  |-- Does the reference configuration pass the quality benchmark?
  |       |-- no -> fix model/pipeline first; do not optimize
  |       `-- yes
  |
  |-- Does the full pipeline fit in 24 GB?
  |       |-- yes -> continue to speed diagnosis
  |       `-- no
  |            |-- Can text encoder be removed after embeddings?
  |            |       |-- yes -> do it, retest fit
  |            |       `-- no/insufficient
  |            |-- Can VAE be phase-local / tiled?
  |            |       |-- yes -> do it, retest
  |            |       `-- no/insufficient
  |            |-- Can exact CPU/group offload fit with acceptable PCIe time?
  |            |       |-- yes -> use it
  |            |       `-- no
  |            |-- Is second A5500 available?
  |            |       |-- yes -> exact sharding/CP/TP, prefer NVLink
  |            |       `-- no
  |            `-- Quantization experiment ONLY behind strict no-regression gate
  |
  |-- What dominates warm latency?
          |
          |-- CPU gaps / many tiny kernels
          |      -> torch.compile -> regional compile -> shape buckets -> CUDA Graphs
          |
          |-- Attention
          |      -> SDPA vs FlashAttention-2 -> profile layout/fallback
          |      -> if video/too-large: context/sequence parallelism
          |
          |-- GEMMs near compute roof
          |      -> compile/autotune -> multi-GPU for latency -> quality-safe work reduction
          |
          |-- Pointwise norm/modulation
          |      -> compile/fuse -> custom Triton only if still hot
          |
          |-- VAE
          |      -> compile/layout -> tiling if memory -> second GPU only if transfer wins
          |
          |-- Text encoder
          |      -> cache/precompute embeddings -> batch -> offload/free
          |
          |-- PCIe/offload stalls
          |      -> larger groups -> pinned prefetch -> more residency -> second GPU
          |
          |-- Multi-GPU communication
          |      -> verify topology/NVLink -> fewer collectives -> coarser split -> DP
          |
          `-- Queue/network/storage
                 -> move to production/infrastructure optimization
```

---

## Quality-sensitive branch

Only after the exact path is optimized:

```text
Need more speed/capacity?
  |
  |-- quantization
  |      -> if ANY required benchmark regression: REJECT
  |
  |-- timestep/hidden-state cache
  |      -> if benchmark regression: REJECT
  |
  |-- fewer steps / new scheduler / CFG cutoff
  |      -> if benchmark regression: REJECT
  |
  `-- sparse/approx attention
         -> if benchmark regression: REJECT
```

The order is intentional: do not trade quality to compensate for an unoptimized exact baseline.

---

## Attention sub-tree

```text
Attention hot?
  |
  |-- Is profiler showing PyTorch math fallback?
  |       -> fix backend compatibility/layout
  |
  |-- SDPA optimized kernel active?
  |       -> benchmark it
  |
  |-- FA2 supports exact shape/mask/head_dim?
  |       -> benchmark FA2
  |
  |-- xFormers installed/supported?
  |       -> benchmark if useful
  |
  `-- FA3/FA4?
          -> out of scope on A5500
```

---

## Compile sub-tree

```text
Many launches / CPU gaps?
  |
  |-- compile repeated transformer region
  |-- graph breaks?
  |      -> eliminate hot-loop breaks
  |-- recompiles?
  |      -> shape buckets
  |-- reserved VRAM too high?
  |      -> regional compile / no CUDA Graph for large bucket
  `-- still hot small kernels?
         -> custom fusion
```

---

## Two-GPU sub-tree

```text
Second A5500?
  |
  |-- topology shows NVLink pair?
  |       |-- yes -> verify P2P/NCCL bandwidth
  |       `-- no -> assume PCIe-level communication until measured
  |
  |-- goal = throughput and one model fits?
  |       -> data parallel independent requests
  |
  |-- goal = fit?
  |       -> coarse model/layer sharding first
  |
  |-- goal = high-res/video activation fit?
  |       -> context/sequence parallel
  |
  `-- goal = one-request latency?
          -> TP/CP/hybrid benchmark, report communication + GPU-seconds
```
