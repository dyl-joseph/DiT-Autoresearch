# xDiT, PipeFusion, USP, and Hybrid Parallelism

## Why specialized DiT parallel engines exist

Conventional transformer parallelism was shaped by autoregressive LLMs. Diffusion has different structure:

- the same model runs repeatedly over a full latent
- no token-by-token KV-cache decode phase
- neighboring denoising steps are correlated
- image/video latents can be partitioned spatially/temporally
- CFG creates naturally independent branches

Systems such as xDiT exploit this structure with diffusion-specific hybrid parallelism.

## Unified Sequence Parallelism (USP)

USP-style approaches combine multiple sequence-parallel communication patterns to scale long visual sequences across GPUs. Conceptually they can mix:

- Ulysses/head-oriented all-to-all
- ring/sequence-oriented communication

Goal: choose a 2D decomposition that avoids the limitations of either alone.

This is especially useful for video or high-resolution images where attention dominates and sequence length is huge.

## PipeFusion

PipeFusion exploits the fact that neighboring diffusion timesteps are similar. Instead of treating each denoising forward as an indivisible whole, it partitions image/video patches and pipelines work across devices/stages, reusing/staging information between timesteps.

The systems insight:

> Diffusion's repeated trajectory creates a second dimension of pipeline opportunity that ordinary layer pipeline parallelism ignores.

Potential benefits:

- lower communication volume than naive tensor parallelism for some workloads
- overlap stage compute and transfers
- better device utilization than naive PP

Risks:

- stale/approximate dependencies depending method
- warmup/bubble overhead
- complexity with dynamic shapes/control branches
- sensitive tuning of patch/stage partition

## Sequence vs pipeline parallelism

### Sequence/CP

Every rank owns part of each attention operation and synchronizes within the block.

### PipeFusion

Ranks own stages/patch work and overlap successive work units/timesteps.

They can be combined because they partition different dimensions.

## Hybrid dimensions

A modern DiT engine may expose several axes:

- data parallel degree
- classifier-free-guidance parallel degree
- sequence/context parallel degree
- tensor parallel degree
- pipeline/PipeFusion degree

The product should satisfy:

`DP * CFGP * SP/CP * TP * PP = total GPUs`

Not every combination is supported or efficient. Search a constrained configuration space.

## How to tune a hybrid

### Step 1: define objective

- minimum latency
- maximum throughput
- minimum GPU-seconds/output
- fit a shape that otherwise OOMs

### Step 2: profile one GPU

Know attention vs MLP share and peak activation memory.

### Step 3: choose first partition

- activation/attention bound -> CP/USP
- weight bound -> TP
- classic CFG latency -> CFG parallel
- huge throughput -> DP

### Step 4: add a second dimension only if first saturates

For example, CP4 may scale well but CP8 may be communication-bound. On 8 GPUs, 2 x CP4 may produce better throughput than CP8 while retaining acceptable latency.

### Step 5: test PipeFusion only with a stable baseline

PipeFusion-style tuning introduces more variables. Compare against a conventional CP baseline at identical precision/steps/attention backend.

## Cache + hybrid parallelism

Cache can reduce local compute. If communication volume is unchanged, the job becomes more communication-bound. A configuration tuned without cache may use too many GPUs after cache is enabled.

Retune parallel degree after:

- FirstBlockCache/PAB
- low-precision attention only if the A5500-supported implementation passes the strict quality gate
- step reduction

All three reduce compute/communication ratio.

## Quantization + parallelism

Any local-kernel acceleration can make collectives dominate sooner. On A5500, do not assume FP8 acceleration; compare full-precision compiled kernels first. If a quality-passing quantized path ever reduces local compute, retune the parallel degree because communication may then dominate.

## Current ecosystem references

- xDiT: https://github.com/xdit-project/xDiT
- Diffusers distributed inference: https://huggingface.co/docs/diffusers/main/training/distributed_inference
- ParaAttention: https://github.com/chengzeyi/ParaAttention

APIs and supported models move quickly. Pin repository commits for reproducible production engines.

## Source basis

**PDF-derived:** context parallelism/ring attention for video, multi-GPU communication as the scaling limit, and topology-aware parallelism.  
**Expansion:** USP/PipeFusion concepts and systematic hybrid-tuning workflow.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
