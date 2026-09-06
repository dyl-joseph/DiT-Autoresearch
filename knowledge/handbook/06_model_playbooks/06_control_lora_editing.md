# Playbook: Control, LoRA, Image Editing, Inpainting, and Multi-Condition DiTs

## Why conditioning changes performance

A base DiT benchmark can be misleading when production adds:

- ControlNet-like residual branches
- reference-image encoders
- IP-adapter/style embeddings
- LoRA adapters
- inpainting masks/extra channels
- edit instruction encoders
- multiple control images

These add weights, activations, extra forwards, or alter attention shapes.

## LoRA performance

LoRA adds low-rank projections to linear layers.

### Merge when possible

If one adapter is fixed for a worker:

- merge into base weights offline
- recompile the merged model for its stable shape/profile; if a separately validated quantized deployment exists, the merged checkpoint must be re-quantized and **re-run through the full no-regression gate** before use

Benefits:

- removes runtime low-rank matmuls
- simplifies compilation
- avoids adapter dispatch

Caveat: repeated merge/unmerge can be expensive and quantized base weights may require special handling.

### Dynamic LoRA

Necessary when every request selects a different adapter.

Performance considerations:

- extra GEMMs per adapted linear layer
- dynamic adapter selection may break fullgraph compilation
- many resident adapters consume VRAM

Serve popular adapters in dedicated compiled pools; use a flexible slower pool for the long tail.

## Control branches

Control modules may run once per step and therefore multiply with step count. Profile separately:

`step_time = base_DiT + control_branch + merge_overhead`

Optimizations:

- compile the control branch; test quantization only as a separate capacity experiment after exact placement is exhausted and reject it on any control/edit benchmark regression
- cache control features that are timestep-invariant
- pre-encode control image once
- reduce control evaluation frequency if algorithm supports it

Do not cache timestep-dependent residuals blindly.

## Image editing

Editing pipelines often encode an input image to latent first. Additional costs:

- VAE encode
- vision encoder
- reference attention tokens
- mask processing

These happen once and can be offloaded/cached. The denoiser may also see a larger input sequence/channels, so the base T2I engine profile may not apply.

## Inpainting

Masks/condition latents can change model input dimensions and compilation keys. Treat inpaint as a separate shape profile if the architecture differs.

## Multi-control memory

Multiple controls can create a large activation/residual footprint. Sum control branch memory and be careful with batched CFG. Peak memory may occur when control residuals and DiT activations overlap.

## Quantization strategy — optional, not the default

For control/editing workloads, exact FP16/BF16 execution is preferred because small numerical changes can show up as boundary drift, identity changes, or control-adherence failures. Solve residency with offload/caching/sharding first.

If quantization is still worth testing for capacity, isolate components rather than changing the whole pipeline at once: start with a large frozen one-time encoder, then (if needed) linear-heavy base/control weights, and leave attention as the last and highest-risk target. **Any required control/edit benchmark regression rejects the candidate.**

LoRA deltas are small; quantizing them usually yields little memory savings and can complicate execution.

## Cache strategy

Editing/control tasks can be more sensitive to approximation because the goal is often precise alignment. Use stricter cache thresholds and quality gates:

- edge/depth adherence
- mask boundary accuracy
- identity/reference similarity
- unchanged-region preservation

A generic aesthetic score is insufficient.

## CFG behavior

Control/edit pipelines may use guidance differently from base T2I. Count model forwards. Some controls are duplicated across guidance branches; others can be shared.

Potential optimization: compute conditioning features once and reuse across cond/uncond when mathematically identical.

## Compilation profiles

Dynamic adapter/control choices make compilation difficult. Strategies:

- compile the base transformer blocks and keep adapter dispatch outside
- dedicate workers to common adapter/control combinations
- use regional compilation to limit recompilation
- bucket by input shape and adapter count

## Production cache keys

If caching prompt/control embeddings or compiled engines, include:

- model revision
- adapter IDs + weights/scales
- control model revision
- input image hash
- preprocessing parameters
- prompt/negative prompt

Incorrect cache keys create silent semantic corruption.

## Source basis

**PDF-derived:** multi-model pipeline and general optimization principles.  
**Expansion:** adapter/control-specific compute, cache, compile, and quality-gate strategies.

## RTX A5500 recommended recipe

Editing/control pipelines add conditioning modules and can increase both residency and repeated compute.

First determine which control features are computed once and which run every denoising step. Keep repeated control features resident; phase-offload one-time encoders. If serving a fixed LoRA, evaluate fusing it into the base weights to remove adapter GEMM overhead, then rerun edit-fidelity benchmarks.

For dynamic LoRA/control workloads, shape/config diversity can cause compile-cache explosion. Bucket by adapter/control topology where practical.

Quantization is particularly risky when unchanged-region fidelity or identity is a hard benchmark requirement; reject any regression.
