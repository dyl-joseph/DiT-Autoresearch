# Conditioning, Adapters, and Pipeline Components

DiT inference is rarely “just the transformer.” Modern generation pipelines can contain multiple encoders, controls, adapters, and decoders. These components have different execution frequency, so they should be optimized differently.

## Classify components by reuse frequency

### Runs once per request

- text encoder
- vision/reference encoder
- input image VAE encoder
- edge/depth/pose preprocessors
- prompt tokenization

Best techniques:

- offload after use
- cache results when identical inputs repeat
- prefer offload/caching first; test quantization only as an isolated capacity experiment and keep it only under the project’s no-regression benchmark rule
- batch across requests if throughput matters

### Runs every denoising step

- main DiT transformer
- ControlNet-like control branch if evaluated each step
- adapter modules integrated into blocks

Best techniques:

- keep resident on GPU
- compile
- use Ampere-native FP16/BF16 Tensor Core paths; do not assume Hopper-style FP8 support
- fuse pointwise operations
- optimize attention
- cache across timesteps

### Runs once at the end

- VAE decoder
- super-resolution/refiner if separate
- safety/watermark/postprocess
- video encoder (H.264/AV1) after generation

Best techniques:

- optimize only after measuring share
- offload until needed
- tile/chunk for memory
- move non-neural encoding to CPU or dedicated hardware if it blocks GPU turnover

## Text encoder optimization

Large modern image models can use multi-billion-parameter text encoders.

### If VRAM is plentiful

Keep encoder resident to minimize per-request transfer and support concurrency.

### If VRAM is tight

Use component-level offload:

1. text encoder on GPU
2. produce prompt embeddings
3. move encoder to CPU / release device residency
4. bring transformer onto GPU
5. run all denoising steps

This is much better than layer-wise transformer offload if the transformer itself can fit once the encoder is removed.

### Prompt embedding cache

Useful when:

- repeated style templates
- fixed negative prompt
- batch generation from the same prompt
- iterative UI where only seed changes

Cache key must include:

- prompt text
- negative prompt
- tokenizer version
- encoder checkpoint
- max sequence length / truncation behavior
- dtype if exact reproducibility matters

## Negative prompt and CFG overhead

Classic CFG often requires an unconditional/negative branch every step. A fixed negative prompt can at least reuse its text embedding, but transformer work remains unless guidance is distilled, approximated, disabled, or parallelized.

Do not confuse **embedding caching** with **denoiser branch elimination**.

## LoRA

LoRA adds low-rank updates to linear projections.

Performance considerations:

- one small LoRA has little memory relative to a large base model
- many simultaneous adapters can add kernels and memory reads
- dynamic adapter switching can prevent optimal graph capture/compilation
- static production configurations can sometimes fuse/merge LoRA weights into base weights

Static deployment pattern:

- load base checkpoint
- apply approved adapter
- fuse/merge if the runtime supports it and rollback is easy
- compile that exact model variant
- cache compiled artifact per adapter variant

Dynamic marketplace pattern:

- keep base model stable
- route requests by LoRA
- maintain a bounded hot set of adapters in GPU/host memory
- avoid recompiling for arbitrary adapter combinations

## ControlNet-like conditioning

A control network may run every denoising step, making it much more expensive than a one-time encoder.

Measure:

- control branch time per step
- base transformer time per step
- extra activation memory
- whether control branch shares embeddings or latent features

Optimization options:

- lower precision for control branch separately
- compile control branch
- cache control features if architecture permits
- run control branch on another GPU only if communication is cheaper than local compute
- reduce control application to a subset of timesteps if model supports control guidance start/end parameters and quality is acceptable

The last technique is analogous to turning off CFG later: conditioning often matters most during structure formation.

## Image-to-image and inpainting

These add an input-image encoding stage and sometimes masks/extra channels.

Useful optimizations:

- pre-encode unchanged source images
- cache source latents for iterative editing
- cache mask transforms
- offload VAE encoder after source latent creation
- bucket source sizes to avoid compiler churn

## Reference-image / IP-Adapter-like conditioning

Reference-image encoders often run once. Keep their embeddings and discard/offload the encoder before the denoising loop.

If multiple references are used, prompt/reference token count can increase joint-attention cost. Long conditioning is not free.

## Multiple text encoders

Some pipelines use two or three text encoders. Optimize them independently.

Potential strategy:

- run encoders sequentially on one GPU
- keep only final embeddings on device
- free/offload encoder modules before loading/activating the transformer
- if exact placement still cannot fit, test compressed encoder storage as a separate last-resort candidate and retain it only if the required benchmark suite shows no regression

The fastest steady-state denoising path often uses more aggressive memory management for one-time components than for the repeated transformer.

## VAE / decoder considerations

The VAE can use substantial temporary memory because it expands latent resolution back to pixels.

Choose among:

- full decode: fastest if it fits
- slicing: lower batch/frame memory
- tiling: lower spatial memory
- temporal chunking: lower video memory

Do not leave tiling enabled automatically on a high-VRAM latency-critical deployment; it can add overhead.

## Safety filters and postprocessing

Production measurements should include:

- NSFW/safety classifier
- watermarking
- color conversion
- image encode
- video mux/encode
- upload/object storage

A 1.5-second GPU generation followed by 1 second of CPU video encoding is a 2.5-second product path.

## Component placement policy

A useful policy table:

| Component | Hot-loop frequency | Preferred placement |
|---|---:|---|
| text encoder | once | GPU then offload, or resident if VRAM allows |
| main DiT | every step | keep fully resident if at all possible |
| ControlNet | every step | resident/compiled or dedicated GPU |
| VAE encoder | once | on-demand |
| VAE decoder | once | on-demand; tile only if needed |
| safety model | once | CPU or separate GPU depending SLA |
| video encoder | once | CPU/NVENC/dedicated path |

## Rule of thumb

**Spend VRAM on the module that is executed repeatedly.**

If you must choose between keeping a 20B DiT resident and keeping a 7B text encoder resident, the transformer usually deserves the GPU memory because it executes tens of times per output.
