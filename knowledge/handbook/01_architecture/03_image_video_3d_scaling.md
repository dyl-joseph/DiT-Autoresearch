# How Image, Video, and 3D/World DiTs Scale

## Start from latent token count

DiTs do not directly attend over every output pixel. A VAE or tokenizer compresses the output into latent space, then patching creates transformer tokens.

For an image:

```text
H_l = H_pixel / VAE_spatial_downsample
W_l = W_pixel / VAE_spatial_downsample
N = (H_l / patch_h) * (W_l / patch_w)
```

For a video:

```text
T_l = T_frames / VAE_temporal_downsample
N = (T_l / patch_t) * (H_l / patch_h) * (W_l / patch_w)
```

If the model uses global attention, attention score computation grows roughly with `N^2`.

## Resolution scaling is worse than it looks

Suppose you double both image width and height while every latent/patch factor stays constant.

- pixels: 4x
- latent tokens: 4x
- global attention pair count: ~16x

MLP and projection work rises more like 4x, so the fraction of runtime spent in attention grows with resolution.

This is why an attention backend that seems unimportant at 512x512 can become decisive at 1536x1536.

## Aspect ratio can matter even at equal pixel count

Equal-area images may not produce identical kernels:

- sequence length divisibility changes
- tiling behavior changes
- context-parallel splits may become uneven
- VAE tiling may use different tile counts
- compiler shape specialization may produce different kernels

Benchmark product aspect-ratio buckets rather than extrapolating from square images.

## Video multiplies spatial cost by time

Video adds a temporal axis. Even after temporal compression, latent token count can be one or two orders of magnitude larger than image generation.

Consequences:

- attention dominates
- batch size 1 is common
- multi-GPU is used for one request rather than for batching many requests
- cache methods are unusually valuable
- VAE decode needs chunking/tiling
- host-to-device offload can become catastrophic if repeated inside every layer

The uploaded PDF's mental model is useful: video uses roughly the same order of denoising steps as many image models, but each step processes vastly more latent data.

## Attention regime transitions

A rough progression:

### Small image

- MLP/GEMMs and attention both important
- launch overhead can be visible
- compilation yields substantial gains

### Large image

- attention share rises
- memory-efficient attention becomes mandatory
- context parallelism can make sense even when weights fit on one GPU

### Moderate video

- attention is often dominant
- efficient exact attention and caching can move the needle dramatically; on A5500, low-precision attention is experimental and quality-gated rather than a default FP8 path
- VAE memory becomes significant

### Long/high-resolution video

- one GPU often not viable
- context/sequence parallelism becomes architectural, not optional
- communication topology determines scaling

## Frame count scaling

For full spatiotemporal attention, doubling latent time approximately doubles `N`, which can quadruple the attention pair count.

This means “twice as long video” can cost much more than 2x if attention is global.

If temporal attention is factorized separately from spatial attention, scaling can be less severe, but you must inspect the actual architecture.

## 3D object generation

3D latent representations may use:

- voxels
- triplanes
- point tokens
- Gaussian splat parameters
- latent grids
- multi-view image sequences

The same performance question applies: **what is the transformer sequence and how does it scale with requested detail?**

If the representation has `X x Y x Z` tokens, global attention is even more explosive than image attention. Practical systems therefore tend to use compression, factorization, sparse attention, local windows, or staged generation.

Optimization priorities:

- reduce representation token count before kernel work
- exploit spatial sparsity/locality
- use sequence/context parallelism
- cache repeated denoising features
- keep decoder/renderer separate in profiles

## World models

World models may generate spatiotemporal latent states, often under stronger latency constraints than offline video generation.

They can combine:

- video-like DiT blocks
- recurrent state
- action conditioning
- autoregressive chunks
- diffusion refinement

For interactive world models, step reduction and streaming/chunked generation can matter more than absolute fidelity. A 50-step cinematic model may simply be the wrong model class for the latency budget.

## Audio DiTs

Audio diffusion transformers can operate on spectrogram or codec latent tokens. Sequence lengths may be long, but structure differs from 2D/3D visual attention.

Transferable optimizations:

- lower precision
- compile repeated blocks
- memory-efficient attention
- timestep caching
- chunked decoding
- context parallelism for long sequences

But quality gates must include temporal artifacts and phase/coherence issues specific to audio.

## A useful scaling spreadsheet

For each supported shape, compute:

| Variable | Meaning |
|---|---|
| H, W | output pixels |
| T | frames |
| d_s | VAE spatial downsample |
| d_t | temporal downsample |
| p_s | spatial patch size |
| p_t | temporal patch size |
| N | resulting transformer tokens |
| N² | naive attention pair count |
| steps | scheduler iterations |
| branches | CFG/model branches per step |
| evals | steps x branches minus cache skips |

Even approximate values help predict where a configuration will stop scaling.

## Why “lower resolution then upscale” can be a real systems optimization

If attention scales superlinearly with spatial tokens, generating at a smaller latent resolution and applying a separate fast upscaler can reduce total compute.

The tradeoff is architectural:

- base generator loses native high-frequency detail
- upscaler may hallucinate or alter text/faces
- pipeline complexity increases

But for latency-sensitive products, a two-stage system can be better than forcing one giant DiT to operate at the final resolution.

## Shape constraints improve multi-GPU efficiency

Context parallel algorithms often prefer sequence lengths divisible by the parallel degree. Recent “Anything” variants relax these requirements but can add padding/communication complexity.

Production strategy:

- choose a small set of output buckets that map cleanly to token partitions
- compile each bucket
- pre-benchmark the best Ring/Ulysses configuration for each bucket
- route arbitrary requests to the nearest supported bucket and crop/resize if acceptable

This converts shape flexibility into predictable performance.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
