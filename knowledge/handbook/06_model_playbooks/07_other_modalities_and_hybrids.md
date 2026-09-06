# Playbook: 3D, World Models, Audio DiTs, MoE DiTs, Discrete Diffusion, and Hybrids

## “Diffusion Transformer” is a systems pattern, not one architecture

The optimization framework in this folder generalizes beyond text-to-image/video. The key variables are:

- repeated iterative model evaluations
- token/context dimensionality
- attention pattern
- model width/parameter count
- conditioning pipeline
- output decoder
- quality sensitivity to approximation

This chapter maps less common DiT variants to those variables.

## 3D / 4D generation

3D latent representations may tokenize:

- voxel grids
- point clouds
- triplanes
- Gaussian parameters
- mesh tokens
- multi-view images

A dense XYZ grid can make token count explode even faster than 2D image latents. Optimization priorities:

1. sparse/hierarchical representation if architecture supports it
2. memory-efficient/sparse attention
3. RTX A5500: FP16/BF16 compiled kernels; low-bit only as a no-regression experiment
4. cache across denoising steps
5. context parallelism across spatial tokens
6. decoder/renderer disaggregation

If attention is local/sparse, kernel support—not dense FA—becomes the critical issue.

## World models / long spatiotemporal generation

World models can extend video with longer horizons, actions, state tokens, or autoregressive segments. They may mix:

- diffusion over local future chunks
- autoregressive progression between chunks
- memory/state recurrence

Performance must distinguish:

- within-chunk DiT latency
- between-chunk autoregressive/control loop
- state cache/memory

Long context makes CP and hierarchical attention central. Error accumulation makes aggressive cache/quantization quality testing more stringent.

## Audio DiTs

Audio diffusion transformers operate over spectrogram, codec, or latent sequences. Compared with image:

- sequence may be long in time
- attention can dominate
- decoder/vocoder can be substantial
- streaming requirements may matter

Optimization mapping:

- sequence attention -> Flash/CP
- repeated denoising -> step reduction/cache
- vocoder -> separate compile/disaggregation
- streaming -> chunked/causal architecture required; not solved by inference engineering alone

## Multi-modal “omni” DiTs

A single transformer may process combinations of:

- text
- image
- video
- audio

Joint attention can create very heterogeneous token types and masks. Custom kernels must support the exact masking/layout. Universal padding can waste huge compute; pack or bucket modalities when the architecture allows.

## Mixture-of-Experts DiTs

MoE replaces dense MLPs with routed experts.

Potential advantages:

- higher parameter capacity without activating all weights per token

Inference challenges:

- expert weights may dominate residency
- routing can create load imbalance
- expert parallelism introduces all-to-all communication
- visual token counts are large, so routing traffic can be huge

Optimization strategy:

- keep expert weights in FP16/BF16 by default; only test compressed expert storage when it is required for fit, and keep it only if the complete benchmark suite shows no regression
- keep active experts local where possible
- use expert parallelism for throughput/fit
- profile expert imbalance per timestep
- combine CP with EP only on fast topology

The standard dense-DiT rule “weights fit, activations dominate” may no longer hold.

## Discrete diffusion transformers

Discrete diffusion iteratively refines categorical tokens rather than continuous Gaussian/flow latents. Depending on algorithm, it may:

- update all tokens each step
- update only a subset/masked tokens
- vary active token count across trajectory

Optimization opportunities:

- skip converged tokens if supported
- bucket by active-token count
- dynamic sparse attention
- cache unchanged token representations

Dynamic sparsity can conflict with static CUDA Graphs/compilation; regional or masked fixed-shape execution may be faster despite extra nominal work.

## Autoregressive-diffusion hybrids

Some image/video systems combine global diffusion with autoregressive generation. Performance has two different phases:

- diffusion phase: compute-bound full-context repeated passes
- AR phase: memory-bandwidth/token-generation behavior closer to LLM decode

Do not apply one bottleneck model to both. The PDF notes research moving back toward autoregressive components in video to address full-context attention constraints.

## Cascade / multi-stage systems

Examples:

- low-res base DiT -> super-resolution diffusion
- keyframe generation -> interpolation model
- coarse 3D -> renderer/refiner

Optimize each stage separately and then allocate quality budget across stages. A 2x speedup in a stage that is 10% of latency is low value; reducing the number/resolution of refinement stages can be larger.

## Rectified flow / flow matching

Many modern “diffusion transformer” models use flow-matching/rectified-flow objectives rather than classic DDPM noise prediction. Inference engineering remains similar:

- repeated transformer evaluations
- scheduler/solver choice
- guidance
- attention/GEMM/precision

But step schedules and numerical sensitivity differ. Use model-native solver assumptions.

## Non-NVIDIA accelerators

The same hierarchy applies on AMD/Intel/Ascend/Apple/other accelerators, but kernels differ:

- use the platform-native low-precision type
- select the platform's memory-efficient attention
- compile with its backend
- benchmark interconnect collectives

Do not force CUDA-specific advice onto a platform where the fastest path is a different runtime.

## Universal decision tree for a new DiT

When encountering an unfamiliar architecture:

1. Count parameters per component.
2. Determine latent/token dimensions and attention pattern.
3. Count denoiser/model evaluations.
4. Profile attention vs MLP vs decoder.
5. Determine compute vs bandwidth vs launch vs communication bound.
6. Establish BF16 quality baseline.
7. Reduce steps/work.
8. Select precision and kernels.
9. Compile/fuse.
10. Test cache.
11. Scale with the parallelism dimension matching the bottleneck.

This is more reliable than searching for a one-off optimization recipe by model name.

## Source basis

**PDF-derived:** transformers span modalities; visual models are iterative denoisers, video/3D-like contexts are attention-heavy, autoregressive hybrids are an active direction, and MoE/parallelism principles generalize.  
**Expansion:** architecture-to-systems mapping for 3D/audio/world/discrete/MoE/cascade/hybrid DiTs.

## RTX A5500 recommended recipe

For 3D/world-model/hybrid DiTs, begin by identifying which axis creates token growth: spatial XY, temporal T, depth/view, or multimodal context. That determines whether the main pressure is compute, activation memory, or communication.

A5500 priorities remain constant:

- FP16/BF16 exact kernel path;
- FA2/SDPA where attention form is supported;
- compile/fusion;
- phase-local component residency;
- 2-GPU context/model sharding when needed;
- quality-sensitive compression only after task-specific benchmark parity.

Do not assume a text-to-image quality gate covers 3D consistency, physics, view consistency, or control fidelity.
