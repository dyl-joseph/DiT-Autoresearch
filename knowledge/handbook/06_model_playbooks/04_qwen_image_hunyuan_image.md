# Playbook: Qwen-Image, Hunyuan-Image, and LLM-Heavy Image Transformers

## Scope

Newer image generation families increasingly borrow from large language model architecture and scaling patterns: larger transformer widths, more sophisticated text understanding, multimodal conditioning, and sometimes text encoders that are themselves large LLMs. Examples include Qwen-Image- and Hunyuan-Image-like systems.

The performance consequences are:

- weight residency becomes more important than older image models
- prompt processing can be expensive
- compile and memory-placement work become high-return; low-bit paths remain optional capacity experiments under the no-regression rule
- attention can resemble LLM-style kernels but remains non-autoregressive full-context visual attention

## Qwen-Image-style optimization

Current Diffusers documentation uses Qwen-Image in examples for:

- quantization
- regional compilation
- attention backend switching
- FirstBlockCache

That makes it a useful reference pipeline for modern DiT tooling.

### Recommended order

1. BF16 baseline + native SDPA.
2. Compile repeated transformer blocks.
3. RTX A5500: FP16/BF16 + compile; low-bit torchao only as a strict no-regression capacity experiment.
4. Attention backend sweep.
5. FirstBlockCache threshold sweep.
6. Offload/cache the text encoder if needed; only test encoder quantization if exact placement is still insufficient and benchmark parity is proven.
7. CP only for large sequences/multi-GPU latency.

### Text rendering quality gate

Qwen-Image-class systems may be chosen for text rendering and instruction fidelity. These are precisely the qualities likely to reveal quantization/cache error. Include OCR-based evaluation and human legibility tests.

## Hunyuan-Image-like optimization

The PDF mentions newer image models increasingly resembling LLMs. For LLM-heavy visual transformers:

- large linear layers make kernel/compile optimization high-value; native FP8 attraction applies to newer GPUs, not A5500
- weight memory can become first-class
- kernel maturity follows how close the architecture is to standard transformer shapes

If a model has unusual multimodal blocks, custom processors may reduce compile/Flash compatibility. Profile graph breaks.

## Large text encoder/LLM conditioning

Strategies:

- encode once then offload
- precompute/cache embeddings for repeated prompts
- optimize/offload the conditioning model separately; only test its quantization after exact placement options and only under the no-regression gate
- serve conditioning model independently if fleet scale warrants it

Prompt encoder latency can be nontrivial for very large encoders. Measure it instead of assuming “text is free.”

## Regional compilation

Modern giant image transformers may take a long time to compile as a whole. Compile repeated block classes where supported. This can reduce cold start substantially while maintaining most steady-state speedup.

Current Diffusers quickstart documents `compile_repeated_blocks()` for Qwen-Image-style transformers.

## Attention backend

Large modern image transformers are good candidates for:

- native SDPA baseline
- RTX A5500: FlashAttention-2 or PyTorch SDPA; FA3/FA4 are newer-architecture paths
- SageAttention only as a quality-gated research path; exact attention first

Because typography/instruction following is sensitive, any quantized/approximate attention path must satisfy the project-wide zero-regression quality gate; on A5500 it is not a default speed path.

## Cache

FirstBlockCache is a natural fit. Tune threshold with:

- OCR accuracy
- prompt object relation score
- human preference

An image can remain “beautiful” while losing exact text or instruction details.

## VRAM profiles

### 24 GB

Likely require some combination of:

- exact offload/sharding first; low-bit weights only with benchmark parity
- text encoder offload
- layerwise casting
- VAE offload

### 48 GB

Transformer can often remain resident with compressed/offloaded conditioning. Optimize for speed rather than merely fit.

### 80+ GB

Keep more pipeline components resident; focus on compute reduction, compile, attention, cache.

## Multi-GPU

If weights fit per GPU but high-resolution activations do not, CP is appropriate. If weights themselves exceed one GPU after reasonable quantization, TP/device maps become more relevant.

## Model-variant policy

Families may have:

- base/high-quality models
- distilled/turbo variants
- editing variants
- control variants

Build separate engine profiles instead of forcing all variants through one compile/cache/quantization configuration.

## Current references

- Qwen-Image in Diffusers quickstart and optimization docs: https://huggingface.co/docs/diffusers/main/quicktour
- Attention backends: https://huggingface.co/docs/diffusers/en/optimization/attention_backends
- Caching: https://huggingface.co/docs/diffusers/main/optimization/cache

## Source basis

**PDF-derived:** recent image models can increasingly resemble LLM architecture; image inference remains iterative and compute-bound; kernel/precision optimization principles apply.  
**Expansion:** current Qwen-Image tooling patterns and quality gates for typography/instruction-following models.

## RTX A5500 recommended recipe

Instruction-following and text rendering make the quality contract especially important for Qwen/Hunyuan-style image models.

Priority:

- exact prompt-encoder lifecycle management;
- FP16/BF16 transformer;
- SDPA/FA2;
- compile/static shape buckets;
- exact VRAM offload/sharding;
- low-bit only with OCR/instruction benchmarks showing no regression.

Do not use a generic CLIP/aesthetic score as the only quantization gate. Include the model family’s differentiating capabilities in the acceptance suite.
