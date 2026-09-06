# Architecture Taxonomy: Which Kind of Diffusion Transformer Are You Optimizing?

“DiT” is now a family name, not one architecture. Optimization advice becomes much more useful when you first classify the model.

## Type A: Classic latent diffusion with a U-Net denoiser

Examples historically include Stable Diffusion 1.x/2.x/XL families. These are included here as a control group because many production techniques originated there.

Characteristics:

- convolution-heavy U-Net with attention blocks
- channels-last memory format can matter substantially
- VAE is separate
- attention sequence lengths are often smaller than in newer full-transformer models
- xFormers/SDPA/Flash attention can reduce memory
- compilation often targets U-Net + VAE

What transfers to DiTs:

- step reduction
- CFG optimization
- VAE tiling/slicing
- compilation
- lower precision
- model/component offload

What changes for DiTs:

- more runtime shifts into transformer attention/GEMMs
- context parallelism becomes more natural
- repeated transformer-block compilation is easier to exploit

## Type B: Vanilla DiT / spatial transformer denoiser

Conceptually closest to the original Diffusion Transformer idea: patchify latents, run transformer blocks, unpatchify.

Typical characteristics:

- one main 2D transformer
- timestep conditioning via adaptive normalization/modulation
- global self-attention
- prompt conditioning via cross-attention or additional embeddings
- relatively clean repeated block structure

Optimization priorities:

1. step count/distillation
2. BF16/FP16 -> keep as the A5500 default; low-bit storage only if strict benchmark parity is proven
3. attention backend
4. `torch.compile` / regional compile
5. cache across timesteps
6. context parallelism at high resolution

## Type C: MMDiT / multimodal joint-attention transformer

Stable Diffusion 3-style multimodal diffusion transformers jointly process text and image latent representations or use dual streams that interact.

Performance implications:

- sequence length includes more than image tokens
- text length can affect attention cost more directly than in simple cross-attention pipelines
- attention masking/padding behavior matters
- compiler graphs can depend on prompt-length shape
- variable prompt lengths may create recompilation or padding overhead

Optimization priorities:

- bucket/cap text lengths where product permits
- choose an attention backend that handles masks and joint sequence shapes efficiently
- compile repeated blocks
- keep linear-heavy transformer portions in FP16/BF16 by default; only test weight compression after exact fit strategies, and retain it only if the full benchmark suite shows no regression
- consider context parallelism if joint sequence length is large

## Type D: Flow-matching transformer (FLUX-like family)

Many modern DiTs use rectified-flow/flow-matching objectives. Runtime still resembles iterative denoising.

Distinctive operational properties can include:

- large transformer parameter count
- large text encoders
- fast distilled variants with very few steps
- “guidance” semantics that differ across variants
- transformer dominates steady-state runtime

Optimization priorities:

- select fast checkpoint first (`schnell`-like variants where quality permits)
- offload text encoder after prompt embedding if VRAM constrained
- exact offload/sharding for fit; weight quantization only as a final no-regression experiment
- compile transformer repeated blocks
- optimized attention
- first-block/timestep cache
- context parallelism for higher resolution or multi-GPU latency

## Type E: Large text-heavy image DiT

Modern image generators may use a full LLM/VLM as a text encoder. The uploaded PDF contrasts older CLIP-scale encoders with newer multi-billion-parameter encoders.

Performance consequences:

- checkpoint load time grows substantially
- CPU RAM/storage bandwidth become important
- prompt encoder can consume large VRAM even though it runs once
- separating encoder and denoiser residency can be highly effective

Best pattern on memory-constrained hardware:

1. encode prompt
2. keep embeddings
3. offload/free text encoder
4. run transformer hot loop entirely resident if possible
5. offload transformer only after denoising
6. decode VAE

This is usually preferable to repeatedly shuttling transformer blocks over PCIe.

## Type F: Video DiT with full spatiotemporal attention

The entire latent video is represented in a long sequence. The uploaded PDF notes that attention can consume the majority of compute for such models.

Characteristics:

- huge `N` due to width x height x time
- batch size often 1
- attention kernels matter more than almost anywhere else
- context/sequence parallelism is critical
- newer GPUs may gain from FP8 attention/activations, but RTX A5500 does not have native FP8 Tensor Core execution; use FA2/SDPA with FP16/BF16
- cache methods have large upside because adjacent denoising timesteps are redundant

Optimization priorities:

1. reduce frame/resolution/steps if product allows
2. best attention backend
3. cache acceleration
4. A5500: FP16/BF16 optimized compute; low-bit storage/compute only if benchmark parity and actual speedup are demonstrated
5. context parallelism
6. group offload only if needed for capacity
7. VAE chunking/tiling for memory

## Type G: Factorized video DiT

Some video architectures do not run one global attention operation over all space-time tokens. They may separate:

- spatial attention
- temporal attention
- cross attention

This changes optimization opportunities.

For example, Pyramid Attention Broadcast-style caching relies on the observation that different attention types can change at different rates across timesteps. A factorized architecture can therefore expose more selective caching than a monolithic joint-attention model.

Inspect the block graph before choosing a cache algorithm.

## Type H: Cascade / multi-stage DiT pipeline

A generation may include:

- low-resolution base generator
- latent upsampler
- refiner
- super-resolution stage
- separate face/detail stage

Optimization is now a scheduling problem across models.

Measure per-stage share. You can often:

- keep all stages FP16/BF16 by default; if exact placement cannot meet capacity, test compression on one isolated refinement stage and reject it on any required benchmark regression
- use fewer steps in the base but preserve high-quality refiner steps
- offload completed stages
- run different stages on different GPU classes
- cache or batch only the expensive stage

Do not assume one precision or one compiler strategy must apply to the whole cascade.

## Type I: Conditional / editing DiT

Inpainting, image-to-image, ControlNet-like conditioning, depth/pose/edge conditioning, reference-image conditioning, and multimodal editing add extra components or tokens.

Potential costs:

- reference-image encoder
- ControlNet / control transformer
- larger input channel count
- multiple conditioning sequences
- additional cross/joint attention

Optimization priorities:

- precompute static conditioning embeddings
- fuse/compile control branches if shapes are stable
- offload one-time encoders
- optimize auxiliary components separately from the base transformer; use exact offload/caching first, and test quantization only as an isolated no-regression experiment
- route requests by adapter/control set in production

## Type J: MoE or sparsely activated DiT

Mixture-of-experts diffusion transformers are less standardized than MoE LLMs but the performance logic is clear if present:

- total parameters predict memory
- active parameters predict compute
- routing can create communication pressure
- expert parallelism may become relevant

Do not apply dense-DiT assumptions blindly. Profile active expert count and all-to-all communication.

## Type K: Discrete/masked diffusion transformer

Diffusion-style iterative refinement can operate on discrete tokens rather than continuous visual latents. These models can resemble language transformers more closely.

Optimization may shift toward:

- token count
- mask schedule
- cacheability
- sparse token updates
- autoregressive-like serving techniques

This folder focuses primarily on continuous/latent visual DiTs, but the bottleneck and compilation methodology still transfers.

## Type L: Autoregressive-diffusion hybrid

The uploaded PDF identifies hybrid autoregressive image/video modeling as an emerging direction. These systems may divide generation into:

- autoregressive coarse structure
- diffusion refinement
- chunk-wise or frame-wise recurrent generation with global correction

The optimization plan must separately profile each phase. LLM-style KV caching or speculation may matter in the autoregressive phase, while DiT attention/caching/precision matters in the refinement phase.

## Classification table

| Architecture | Main scaling variable | Best first optimization |
|---|---|---|
| U-Net latent diffusion | steps + conv/attention | compile + steps |
| vanilla DiT image | spatial tokens | attention + compile |
| MMDiT/joint | image + text tokens | attention + shape control |
| large flow DiT | params + steps | fast variant + precision |
| full-attention video DiT | space-time tokens | attention + context parallel |
| factorized video DiT | per-axis tokens | selective cache + kernels |
| cascade | stage composition | per-stage profiling |
| editing/control | extra branches/tokens | precompute/offload auxiliary |
| hybrid AR+diffusion | phase-specific | separate phase optimization |

## Size classification

Parameter count is still useful for capacity planning:

- **tiny/small:** <2B transformer parameters
- **mid:** 2-6B
- **large:** 6-15B
- **very large:** 15-30B
- **extreme/multi-component:** >30B effective resident parameters or multiple large stages

These are engineering bands, not model-quality labels. See `07_hardware/03_vram_tier_playbooks.md` for memory implications.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
