# DiT Inference Anatomy

## The hot loop

A Diffusion Transformer repeatedly predicts how a noisy latent should change. The exact parameterization may be epsilon prediction, velocity prediction, flow matching/rectified flow, or a model-specific variant, but the performance structure is similar:

```text
conditioning = encode(prompt, images, masks, controls, ...)
latent = initialize_noise_or_encode_input()

for timestep in schedule:
    model_input = prepare(latent, timestep, conditioning)
    prediction = transformer(model_input)
    latent = scheduler_step(latent, prediction, timestep)

output = decoder(latent)
```

Every performance technique attacks one or more terms in that loop.

## Pipeline components

### 1. Text / multimodal encoder

Possible encoders include CLIP/T5-style text encoders, LLM/VLM encoders, or multiple encoders. They often have high parameter counts but execute only once per generation. This creates a distinctive optimization profile:

- large impact on model load and VRAM residency
- modest impact on steady-state latency when the denoiser has many steps
- excellent candidate for component offload after prompt embeddings are produced
- excellent candidate for prompt-embedding reuse in workflows with repeated prompts

A 7B text encoder can matter more to peak VRAM than to wall-clock time.

### 2. Patch / latent embedding

The latent is projected into transformer tokens. If latent spatial dimensions are `H_l x W_l` and patch size is `P x P`, the spatial sequence length is roughly:

```text
N_spatial = (H_l / P) * (W_l / P)
```

For video with temporal latent length `T_l` and temporal patch/compression factor `P_t`:

```text
N_video ~= (T_l / P_t) * (H_l / P) * (W_l / P)
```

This number is one of the most important performance quantities in the entire model because global attention cost grows approximately with `N^2`.

### 3. Repeated transformer blocks

Typical ingredients:

- self-attention or joint attention
- cross-attention or multimodal joint attention
- MLP/feed-forward layers
- RMSNorm/LayerNorm
- adaptive normalization / timestep modulation
- rotary or learned positional encoding
- residual connections

The repeated block structure is why regional compilation can work so well: compile one block shape and reuse the compiled form across many layers.

### 4. Scheduler / solver

The scheduler updates latents between denoiser evaluations. Its arithmetic is usually small compared with the transformer, but it can still hurt if it causes GPU-to-CPU synchronization or dynamic graph behavior.

Important distinction:

- **steps** are scheduler iterations
- **model evaluations** may exceed steps when CFG evaluates multiple branches

Latency should be modeled in denoiser evaluations, not just “steps.”

### 5. VAE / latent decoder

The decoder converts latent tensors back to pixels or video frames. Its share of runtime varies. The uploaded PDF places video VAE decode in the low-single-digit percentage range for a representative pipeline, which is useful as an optimization priority rule: profile before spending time here.

The VAE can still dominate **peak memory** at high resolution or frame count. Tiling and chunking are therefore often memory optimizations rather than speed optimizations.

## Classic classifier-free guidance changes the compute graph

Traditional CFG constructs conditioned and unconditioned predictions and combines them. Implementations often concatenate the two branches into a batch dimension, which means the transformer effectively sees batch size 2 for a single requested image.

Conceptually:

```text
pred_uncond = model(latent, empty_prompt)
pred_cond   = model(latent, prompt)
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```

Or equivalently one batched model call containing both branches.

Performance consequences:

- near-doubling of transformer math in the simplest case
- larger activations
- greater attention/GEMM work
- opportunity for 2-GPU CFG parallelism
- opportunity to disable guidance on later steps if quality remains acceptable

Not every modern DiT uses classic CFG this way. Guidance-distilled or guidance-embedded variants may behave differently. Always inspect the pipeline.

## Flow matching vs. diffusion: performance consequences

Modern image DiTs often use rectified flow or flow matching rather than the original DDPM formulation. From an inference-engineering perspective, the most important facts are:

- inference is still iterative
- the transformer is still the hot loop
- fewer high-quality solver evaluations are still a primary acceleration target
- attention/GEMM/compile/precision optimizations remain applicable

Do not over-index on the training objective when choosing runtime optimizations. Inspect the actual forward graph and scheduler.

## Why visual DiTs are usually compute-bound

The uploaded PDF compares visual denoising to LLM prefill rather than LLM decode. At each denoising evaluation, attention and MLPs process the whole latent sequence at once. That creates substantial matrix math per weight load.

But “compute-bound” does not mean memory work is irrelevant. Poor implementations may still waste bandwidth by materializing large intermediate attention matrices or repeatedly storing/loading norm/modulation outputs. High-performance kernels reduce this overhead so the workload can approach its theoretical compute roof.

## The four multipliers that dominate cost

A useful first-order model:

```text
cost ~ model_size
     * token_count_effect
     * denoiser_evaluations
     * precision_factor
```

Where `token_count_effect` is not linear if global attention dominates; it can behave closer to quadratic.

Then add:

- encoder/decoder overhead
- controls/adapters
- cache skip rate
- inter-GPU communication
- offload transfers
- compile/launch overhead

## A practical stage timing decomposition

Instrument these separately:

1. prompt preprocessing/tokenization
2. text/multimodal encode
3. latent setup / image encode
4. transformer denoising loop
5. scheduler arithmetic
6. VAE decode
7. image/video postprocess and transfer to CPU
8. serialization/encoding (PNG/JPEG/MP4)

For a server, separately record queue time and network time.

## Architecture inspection checklist

Before optimizing an unfamiliar pipeline, answer:

- How many transformer parameters?
- How many text encoder parameters?
- What dtype is the checkpoint designed for?
- What is latent downsampling ratio?
- What are spatial/temporal patch sizes?
- How many tokens at every supported output shape?
- How many transformer blocks?
- What head count and head dimension?
- Is attention global, windowed, factorized, or mixed?
- Are text and image/video tokens in one joint sequence?
- Is CFG used? If yes, how is it implemented?
- How many denoiser evaluations at default settings?
- Is there a fast/distilled checkpoint?
- Is VAE decode tiled/chunked?
- Does the model have repeated blocks recognized by Diffusers compilation helpers?
- Which modules contain unsupported/custom ops?

This architecture inventory determines almost every later optimization choice.
