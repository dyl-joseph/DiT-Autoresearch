# Playbook: Stable Diffusion 3 / MMDiT-Style Architectures

## Architectural performance profile

MMDiT-style models jointly process image and text streams through transformer blocks rather than using only conventional cross-attention from image latents into frozen text context. This changes both memory and kernel shapes.

Stable Diffusion 3-style pipelines can also include **multiple text encoders**, including a large T5-class encoder. That means pipeline weight memory can be much larger than the denoiser alone.

## First optimization: separate prompt memory from denoiser memory

The text encoders are used before the denoising loop. If VRAM is constrained:

1. encode prompt
2. retain prompt embeddings
3. offload/unload text encoders
4. keep MMDiT resident for all steps
5. decode VAE afterward

This is usually much faster than offloading transformer blocks every step.

Current Diffusers SD3 documentation explicitly documents model CPU offload because of the large multi-encoder pipeline.

## Prompt-embedding cache

For workflows that generate many seeds/variations from one prompt:

- cache all encoder outputs
- key by tokenizer/text-encoder revision + prompt + negative prompt + clip-skip-like settings
- reuse across requests

This can eliminate a large one-time component and improve throughput.

## Attention strategy

Joint text-image attention may create slightly different sequence/head shapes than a pure image self-attention block. Benchmark:

- native SDPA
- FlashAttention compatible path
- SageAttention/quantized attention disabled by default; only test after strict prompt-adherence/typography benchmark proves no regression

Do not assume a kernel tuned for FLUX or video has the same win on MMDiT.

## Compile strategy

Compile the transformer first. Then optionally compile VAE decode. Use:

- regional compilation where repeated blocks support it
- fullgraph only after eliminating custom-processor graph breaks
- shape buckets for common image sizes

Current Diffusers SD3 docs include transformer/VAE compilation examples and report significant example speedups; treat those figures as environment-specific rather than promises.

## Precision plan

### Plenty of VRAM

- BF16/FP16 transformer
- full-precision sensitive norms/modulation
- offload text encoders if idle residency limits concurrency

### Moderate VRAM

- FP16/BF16 transformer; solve fit with encoder/VAE lifecycle, offload, or second-GPU sharding before low-bit experiments
- offload T5/text encoders after use or cache prompt embeddings; test encoder quantization only if exact placement still cannot fit and the full benchmark suite remains unchanged
- memory-efficient attention

### Low VRAM

- weight-only low-bit text encoders
- model CPU offload by component
- exact transformer sharding/offload and phase-local residency first; low-bit storage/layerwise casting only as a separately benchmarked no-regression capacity experiment
- VAE tiling if needed

Avoid fine-grained transformer offload until these are exhausted.

## CFG and step tuning

MMDiT checkpoints may use specific guidance and flow/scheduler conventions. Count actual transformer evaluations and test:

- recommended step range
- guidance scale range
- guidance cutoffs if mathematically compatible
- distilled/turbo variants if available

Quality gates should emphasize text-image semantic alignment because joint attention is one of the architecture's strengths.

## LoRA/adapters

Merged LoRA weights can be faster than applying many low-rank modules dynamically if:

- the adapter is fixed per worker
- merge does not degrade quantized format handling

For many per-request LoRAs, dynamic adapters preserve flexibility but add extra matmuls and can interfere with fullgraph compilation. Consider worker pools keyed by popular adapters.

## Memory hotspots

Potential peaks:

- simultaneous text encoders during prompt phase
- joint attention at high resolution
- compiled transformer workspace
- VAE decode after the transformer

Measure phase-specific peaks; a single `max_memory_allocated` number does not tell which component caused it.

## Production profile examples

### 24 GB class

- offloaded/cached text encoders; quantized encoders only when strict quality parity has been demonstrated
- FP16/BF16 transformer; quantization only as a no-regression capacity experiment
- compile transformer
- VAE decode resident only when needed

### 48 GB class

- keep transformer resident
- possibly keep VAE resident
- offload only largest text encoder
- do not expect native FP8 speed on A5500; only keep a low-bit candidate if it passes quality and beats the best exact baseline

### 80+ GB class

- keep full pipeline resident if it improves latency
- focus on steps, attention, compile, cache rather than memory

## Failure modes

- text encoder memory mistaken for DiT memory
- prompt embedding cache uses stale encoder revision
- low-bit T5 changes semantic embedding enough to hurt text fidelity
- compile graph breaks in joint-attention processor
- variable prompt lengths trigger recompiles or backend fallbacks

## Current reference

- Diffusers SD3: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3

## Source basis

**PDF-derived:** multi-model visual pipelines, attention/kernel/compile/quantization principles.  
**Expansion:** MMDiT joint-stream and SD3 multi-text-encoder-specific residency and serving strategy.

## RTX A5500 recommended recipe

MMDiT/SD3-style pipelines often have multiple text encoders plus a large transformer. On 24 GB, memory pressure may come from **component coexistence** rather than only the transformer.

Recommended order:

1. encode prompt(s), retain embeddings, offload/free text encoders;
2. run MMDiT in FP16/BF16 with SDPA/FA2 as supported;
3. compile repeated MMDiT blocks;
4. keep joint-attention tensor layouts stable to avoid transpose/copy kernels;
5. delay VAE residency until decode;
6. if transformer still does not fit, use group offload or two-A5500 sharding;
7. only then test low-bit storage behind typography/instruction no-regression benchmarks.

Joint text/image attention makes typography and prompt-binding regressions especially important. Do not accept an average aesthetic score as proof of parity.
