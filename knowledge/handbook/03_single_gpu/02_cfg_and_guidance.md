# Classifier-Free Guidance and Guidance Optimization

## CFG can double the expensive part of inference

Classic classifier-free guidance (CFG) combines an unconditional and conditional prediction. Conceptually:

`prediction = uncond + scale * (cond - uncond)`

The expensive part is obtaining `uncond` and `cond`. Depending on implementation, that can mean two transformer evaluations or one evaluation at twice the effective batch.

The uploaded PDF highlights a simple but powerful fact for image generation: if guidance is no longer needed in later denoising steps, turning it off can remove one branch without reducing the number of steps.

## Four implementation modes

### 1. Sequential CFG

Run unconditional and conditional forwards separately.

**Pros**
- lower peak activation memory
- easiest to fit on small VRAM

**Cons**
- twice the launches/forward overhead
- poor latency if the GPU is already well utilized

### 2. Batched CFG

Concatenate unconditional and conditional inputs and run one larger forward.

**Pros**
- better GPU utilization on underfilled workloads
- fewer Python/launch boundaries

**Cons**
- roughly doubles activation batch
- may become slower if the single branch already saturates compute

### 3. CFG parallelism

Put conditional and unconditional branches on different GPUs, then combine predictions.

**Pros**
- near-ideal latency reduction when each branch is independent and communication is tiny

**Cons**
- duplicates weights
- spends two GPUs per request
- only useful when single-request latency is worth the hardware

### 4. Guidance-distilled / guidance-free models

Some models are trained so strong prompt adherence is available without the two-branch CFG process.

**Pros**
- eliminates duplicate denoiser work

**Cons**
- requires a compatible checkpoint/training method
- may change quality/style behavior

## Guidance scheduling

A single guidance scale across all steps is simple, not necessarily optimal. The broad structure is usually fixed early. Later steps may tolerate:

- lower guidance scale
- guidance every `k` steps
- guidance disabled after a cutoff
- model-specific adaptive guidance

Example evaluation grid for a 30-step model:

| Policy | Guided steps | Approx transformer work |
|---|---:|---:|
| Full classic CFG | 30 | 60 branch-evals |
| CFG first 20 | 20 | 50 branch-evals |
| CFG first 10 | 10 | 40 branch-evals |
| No CFG / distilled | 0 | 30 branch-evals |

This is only a work-count illustration; batched CFG and fused implementations alter wall time.

## What quality degrades first

When guidance is reduced too aggressively, watch:

- prompt adherence
- object count and relationships
- style specificity
- small rendered text
- edit instruction fidelity
- identity preservation
- scene composition

Aesthetic metrics alone can miss guidance failures.

## CFG and resolution

At higher resolution, the model spends more compute per branch, so eliminating a branch saves more absolute time. But higher-resolution requests may also have stricter detail/prompt requirements. Tune by resolution bucket.

## CFG and attention

Batched CFG doubles batch but does not necessarily double token sequence length. This matters for attention memory:

- memory grows roughly with batch for memory-efficient attention
- naive attention also carries the `N^2` term per batch item

A switch from sequential to batched CFG can expose an OOM at the same image size.

## CFG and caching

Some caching methods exploit similarity between successive timesteps. Conditional and unconditional branches can have different residual dynamics. If a cache implementation assumes a single stream, verify that its state is keyed/separated correctly for CFG branches.

The most aggressive stacks may combine:

- early-step CFG
- later-step guidance-off
- FirstBlockCache or PAB in the middle/late steps
- lower-precision compute in less sensitive regions only as a research candidate; RTX A5500 has no native FP8 path and the quality gate still applies

Every combination needs a joint quality gate.

## CFG and multi-GPU

CFG parallelism is often the cleanest multi-GPU strategy for image models if:

- weights fit on each GPU
- a request uses actual two-branch CFG
- you have spare GPUs
- single-request latency is more important than fleet throughput

It does not reduce weight memory because each GPU holds a complete model.

For video, context parallelism may be a better use of GPUs because one branch may not fit or may already be large enough to saturate all accelerators. Hybrids are possible: e.g., a CP group per guidance branch.

## Implementation checklist

1. Inspect whether the pipeline actually uses CFG at the requested guidance scale.
2. Count model forwards per nominal step.
3. Benchmark sequential vs concatenated CFG if both are supported.
4. Sweep guidance cutoffs on a fixed quality set.
5. Check VRAM impact.
6. Re-profile attention/GEMM share; changing batch changes kernel selection.
7. If multi-GPU, test CFG-parallel against data parallel throughput and context parallel latency.
8. Bake the selected policy into serving metadata so requests are reproducible.

## Common trap: guidance scale of zero/one semantics

Different pipelines define scales and “guidance disabled” conditions differently. Some skip unconditional computation below/at a threshold; others still build both branches. Never infer work from a UI number. Profile and inspect the pipeline.

## Source basis

**PDF-derived:** image denoising commonly evaluates guided/unguided predictions and later-step guidance can be skipped to reduce total passes while retaining quality in many cases.  
**Expansion:** sequential/batched/parallel CFG taxonomy, quality failure modes, interactions with attention/caching/parallelism.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
