# VAE Tiling, Slicing, and Temporal Chunking

## Why the VAE deserves its own memory plan

The VAE is outside the repeated DiT loop, but converting a large latent into full-resolution pixels or many video frames can create a large transient memory peak. Once the transformer is optimized or offloaded, VAE decode can become the step that determines whether a request fits.

## Three decomposition methods

### Batch slicing

Process items in a batch separately.

- reduces memory roughly with batch size
- preserves each image globally
- decreases throughput because batching is lost

Best for multi-image batches when one decode at a time fits.

### Spatial tiling

Split the latent/image plane into overlapping tiles, decode each, blend/reassemble.

- enables very large images
- lowers peak activation memory
- can introduce seam/blending issues
- repeats work in overlap regions
- adds Python/launch overhead

### Temporal chunking

Split video latent along time, often with overlap/context.

- reduces peak frame activation memory
- enables longer clips
- can create temporal boundary artifacts if decoder context is insufficient

Use model-native chunking when available because temporal VAEs often have architecture-specific receptive fields.

## Tiling is a capacity tool, not automatically a speed tool

If the full VAE decode fits, tiling often makes it slower due to:

- repeated launches
- overlap pixels
- reduced kernel sizes/occupancy
- assembly work

Enable it only for memory, or if a particular implementation shows an empirical speed win due to cache/algorithm behavior.

## Tile size tuning

Larger tile:

- more VRAM
- better GPU utilization
- less overlap overhead

Smaller tile:

- lower peak VRAM
- more overhead
- more seam risk

Choose the largest tile that fits with safety margin.

## VAE precision

Try BF16/FP16. Be more cautious with aggressive quantization because pixel reconstruction is direct and errors can be visually obvious. Some VAEs have numerically sensitive upsampling/norm paths.

## Compile the VAE separately

VAE decode often has convolution-heavy static graphs that compile well. Use shape buckets for common output sizes. A compiled decoder can be reused independently from the transformer engine.

## Channels-last

Convolution-heavy VAEs may benefit from channels-last memory format on NVIDIA GPUs. Benchmark; do not assume benefit for every architecture.

## Disaggregated decoding

At fleet scale, consider separate decode workers when:

- DiT GPU-seconds are much more expensive than decoder GPU-seconds
- latents are much smaller than final images/videos, so moving latents to a decoder service is cheap
- queueing does not hurt latency objectives

Architecture:

`expensive DiT GPU -> latent tensor -> decoder GPU -> encode/serialize`

This frees high-end GPUs earlier.

## Video encoding after VAE

The VAE outputs raw frames; a codec still needs to create MP4/WebM/etc. Hardware video encoders (NVENC or platform equivalent) can prevent CPU encode from dominating response latency. This is outside model inference but inside user-perceived latency.

## Preview strategy

For interactive systems:

- decode a smaller preview or selected frame first
- return it to user
- continue full-resolution decode/encode asynchronously

This improves perceived latency but should be reported separately from model speed.

## Memory test grid

For each resolution/frame bucket, benchmark:

- full decode
- slicing
- tiling tile sizes
- temporal chunk sizes
- compile on/off
- precision

Record:

- decoder latency
- peak VRAM
- seam/flicker quality checks
- total pipeline latency

## Source basis

**PDF-derived:** VAE decode is a separate visual pipeline stage and can be parallelized; video decode may be a minority but nonzero share of runtime.  
**Expansion:** tile/chunk decision rules, disaggregated decoder workers, hardware codec considerations.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
