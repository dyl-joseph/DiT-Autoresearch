# Text Encoders, VAE, Scheduler, and Non-DiT Pipeline Overhead

## The transformer is not the whole service

A DiT can be 90% of runtime at baseline and only 60% after optimization. The remaining components then become worth engineering. Measure them separately.

Typical pipeline:

`input -> tokenizer/text encoder(s) -> prompt embeddings -> latent init -> DiT loop -> VAE decode -> image/video postprocess`

## Text encoder strategy

### Reuse prompt embeddings

If the same prompt or conditioning is reused across seeds/variants:

- encode once
- cache prompt embeddings
- pass embeddings directly to the pipeline if supported

This can remove one or more large text encoders from repeated requests.

### Precompute offline

For batch jobs with known prompts, precompute embeddings and store them with model/tokenizer revision metadata.

### Offload after encoding

Text encoders are cold during denoising. Component/model CPU offload is often ideal here because transfer occurs once, not every transformer layer/step.

### Encoder compression is a last-resort capacity experiment

Large T5/LLM-like encoders can consume significant VRAM. On the RTX A5500 target, start with **embedding reuse, phase-local residency, and offload after encoding**. Weight-only 8/4-bit storage may reduce residency, but it is not the default because the encoder can be quality-sensitive and low-bit compute is not a native A5500 speed path. Test it only if exact placement remains insufficient; reject it if any required benchmark regresses.

### Serve encoders separately at scale

For high traffic, a dedicated embedding service can:

- batch prompts
- share encoder replicas
- keep DiT workers focused on generation

Cost: network latency and cache consistency. Co-locate when possible.

## Tokenization and CPU overhead

Tokenization is usually small, but avoid accidental serialization under high concurrency. Pre-tokenize batch jobs. For extremely short low-step models, tokenizer + request handling can become a visible percentage.

## VAE decode

The VAE converts latent representation to pixels/frames. The PDF notes that video latent decoding may be a few percent of total time in heavy pipelines, but that percentage rises as the transformer gets faster.

### Speed optimizations

- use BF16/FP16 if stable
- compile VAE decode
- use optimized convolution kernels/layouts
- batch decodes for offline throughput
- overlap decode of request A with denoise of request B on separate stream/device only if resource contention is favorable

### Memory optimizations

- VAE slicing: split batch
- VAE tiling: split spatial regions
- temporal chunking: split frames/time

These lower peak memory but often add latency. Use only when required to fit or when enabling larger useful resolution/frame counts.

### Separate decoder GPU

With multiple A5500s, a second GPU can handle VAE decode or text encoding while the primary GPU denoises. This is a form of pipeline/component disaggregation. It only pays when transfer and queueing overhead are smaller than the accelerator time or VRAM headroom it frees.

## Preview decoding

Interactive products may decode a lower-resolution preview before the final image/video. This can improve **perceived latency** without reducing model time. Keep this distinct from actual compute speedup.

## Scheduler

Scheduler math is cheap, but per-step Python/device synchronization is not. After compilation:

- profile scheduler `step()`
- keep constant arrays on the appropriate device/host
- eliminate scalar syncs
- avoid creating fresh tensors per step

## Random latent initialization

Random number generation and latent allocation are typically tiny, but in multi-request services preallocating buffers can reduce allocator churn. Maintain correct seed isolation.

## Safety/watermark/postprocessing

Some pipelines include:

- safety checker
- watermark encoder
- image normalization
- PIL conversion
- video encode (H.264/AV1)

These are end-to-end latency. CPU video encoding can be seconds for long outputs; hardware encoders or asynchronous postprocessing may be more important than another 5% DiT speedup.

## Data movement

Watch transfers for:

- prompt embeddings CPU -> GPU
- latents between pipeline components on different devices
- decoded frames GPU -> CPU
- control images CPU -> GPU

Use pinned memory and nonblocking copies where the framework supports them. More importantly, minimize the number of crossings.

## Pipeline component residency pattern

A good single-GPU low-memory schedule often is:

1. load/execute text encoder
2. keep prompt embeddings on GPU
3. offload text encoder
4. bring DiT to GPU and run all steps resident
5. offload DiT if necessary
6. bring VAE to GPU and decode

This is far better for latency than moving DiT layers back and forth on every denoising step.

## Multi-image generation from one prompt

If generating `k` seeds for one prompt:

- encode prompt once
- reuse embeddings
- decide whether batch `k` or sequential requests better utilize GPU
- batch VAE decode if memory allows

Compute-bound DiTs may not scale throughput linearly with batch; benchmark.

## Production component metrics

Track:

- text-encode ms/request
- prompt-embedding cache hit rate
- denoiser ms/step and total
- VAE decode ms/megapixel or ms/frame
- host-device bytes/request
- postprocess/encode latency

This prevents a “fast DiT” from hiding a slow service.

## Current references

- Diffusers memory optimization: https://huggingface.co/docs/diffusers/optimization/memory
- Diffusers basic/accelerate inference: https://huggingface.co/docs/diffusers/optimization/fp16

## Source basis

**PDF-derived:** image pipelines consist of multiple models; VAE is a distinct decode stage; production performance must be evaluated end-to-end.  
**Expansion:** component residency schedule, embedding reuse/service, asynchronous decode/postprocess analysis.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
