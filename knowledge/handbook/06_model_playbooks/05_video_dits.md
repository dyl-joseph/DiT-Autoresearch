# Playbook: Video Diffusion Transformers

## The hardest mainstream DiT workload

Video DiTs extend a 2D latent context with time. The uploaded PDF emphasizes the consequences:

- video latents encode far more information than image latents
- each frame interacts with other frames through attention or related global mechanisms
- models usually still perform dozens of denoising steps
- attention can account for roughly 70-80% of compute in representative models
- batch size is often one
- full-node multi-GPU inference is common

This makes video optimization primarily an **attention + algorithmic-reuse + parallelism** problem.

## Representative families

The same playbook applies, with model-specific details, to families such as:

- CogVideoX-like 3D transformers
- HunyuanVideo-like full-context video DiTs
- Wan-like video diffusion transformers
- Mochi-like video models
- LTX-Video-like latent video systems
- Open-Sora-style models
- image-to-video / text-to-video variants sharing a video DiT backbone

Always inspect whether attention is full spatiotemporal, factorized, windowed, causal, or hybrid.

## Optimization order

1. Use the best quality-acceptable step count/checkpoint.
2. Use BF16 baseline and a memory-efficient attention backend.
3. On A5500, benchmark FP16/BF16 SDPA/FA2 and compilation; FP8/MXFP8 native paths are not available.
4. Apply attention/block/timestep caching.
5. Compile stable blocks if compatible.
6. Use context/sequence parallelism.
7. Tune CP degree/topology.
8. Parallelize/chunk VAE decode.
9. Only then consider exotic hybrid parallelism.

## Attention first

Profile attention share. If it is >50%, spend engineering effort there:

- FlashAttention matching GPU generation
- low-precision/Sage attention only as strict no-regression research; exact FA2/SDPA first
- vendor/CuTe/CUTLASS kernels
- context parallelism
- PAB/attention caching

A faster MLP matters much less if attention is 80% of step time.

## Selective attention quantization

The PDF's high-value strategy:

- preserve early steps in higher precision
- if exact methods are insufficient, *experiment* with reduced precision only in later steps and reject it on any benchmark or temporal-quality regression
- preserve first/last layers
- if exact methods are insufficient, *experiment* with reduced precision only in candidate hidden layers and reject it on any benchmark or temporal-quality regression
- use blockwise/microscaling formats to protect outliers

For video, validate temporal coherence rather than only individual frames.

## Caching

Video has strong temporal *and denoising-step* redundancy. Candidates:

- PAB attention caching
- FirstBlockCache
- timestep output reuse
- transformer hidden-state cache
- adaptive/learned cache methods

The PDF cites 30-40% practical speed improvements from caching as a representative range, but quality can range from negligible impact to unusable output. Treat any speedup as model/config-specific.

## Context parallelism

When weights fit each GPU, replicate them and split tokens. Choose:

- Ulysses for moderate context / strong all-to-all
- ring for extreme sequence/memory pressure
- unified/hybrid for larger meshes

Video is the strongest case for CP because each request has enough compute to amortize communication.

## 8-GPU layouts

Possible profiles:

### CP8

Lowest latency / largest context. High communication.

### 2 x CP4

Two concurrent videos; often better fleet throughput.

### 2 guidance branches x CP4

If classic CFG branches are both required and latency matters.

### TP2 x CP4

If weights and context are both problematic; complex, only with mature engine.

## Frame count and resolution policy

Cost can grow sharply with both. Build explicit service tiers:

- 480p / short clip
- 720p / standard clip
- 1080p / premium
- long clip via segmentation/continuation rather than one gigantic latent when model supports it

Admission control should reserve an entire GPU group based on latent token count.

## Temporal VAE

Use model-native temporal tiling/chunking. The decoder may be only a few percent of baseline runtime, but after a 2x transformer speedup it can become 5-10%+ and worth parallelizing.

The PDF notes VAE decoding can be run across GPUs.

## Compile cautions

Video shapes are diverse: frame count, resolution, conditioning types. Full compile may generate many variants. Use:

- a small set of video buckets
- regional compilation
- avoid graph breaks from cache/CP code

Some current Diffusers CLI paths explicitly do not combine certain compile and context-parallel options; check version-specific support.

## Memory plan

If video OOMs:

1. memory-efficient attention
2. component/group offload or two-GPU context/model sharding; low-bit storage only if benchmark parity is proven
3. offload text/vision encoders
4. context parallelism
5. temporal VAE tiling
6. reduce frame count/resolution

Per-layer CPU offload of the hot video transformer can be extremely slow because every step moves huge weights.

## Quality gate

Must include motion:

- temporal flicker metric
- optical-flow consistency
- identity/subject consistency across frames
- camera-motion adherence
- action completion
- physics plausibility
- frame-level aesthetic quality
- human side-by-side video preference

Quantization/cache errors that are invisible in still frames can be obvious at playback speed.

## Current references

- Diffusers CogVideoX: https://huggingface.co/docs/diffusers/api/pipelines/cogvideox
- Diffusers HunyuanVideo: https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video
- Distributed inference: https://huggingface.co/docs/diffusers/main/training/distributed_inference
- Caching: https://huggingface.co/docs/diffusers/main/optimization/cache

## Source basis

**PDF-derived:** video compute regime, 70-80% attention example, caching, selective attention quantization, context parallelism, multi-GPU VAE.  
**Expansion:** family taxonomy, 8-GPU layouts, video quality gates, production shape policy.

## RTX A5500 recommended recipe

Video is the hardest workload for a 24 GB Ampere card because both compute and activation footprint grow with time.

Recommended order:

1. use the smallest required spatial/frame bucket;
2. FP16/BF16 exact attention via SDPA/FA2;
3. compile repeated video transformer blocks;
4. inspect activation peak and attention share;
5. if one GPU does not fit, use two-A5500 context/sequence parallelism, preferably over NVLink;
6. move text encoder/VAE out of the hot phase;
7. use exact temporal/spatial chunking only if model/runtime semantics support it;
8. only then test approximate timestep/hidden caches or low-bit paths, with temporal benchmark parity required.

A video candidate can pass single-frame image metrics and still fail due flicker or motion inconsistency. Temporal checks are mandatory.
