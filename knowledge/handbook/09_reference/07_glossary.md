# Glossary of DiT Inference Engineering

**Activation** — Intermediate tensor produced during a forward pass. In inference it need not be saved for backward, but long visual sequences can still make activations a major VRAM cost.

**AdaCache** — Adaptive caching family for video DiTs that changes reuse decisions based on content/timestep signals.

**Admission control** — Deciding whether a new request can safely enter a worker based on memory, queue, and compute capacity.

**Arithmetic intensity** — Operations performed per byte moved. Used with a roofline model to diagnose compute vs bandwidth limits.

**Attention backend** — Specific implementation of scaled dot-product or related attention: native SDPA, FlashAttention, SageAttention, xFormers, etc.

**Attention slicing** — Processing attention in smaller chunks/heads to reduce memory, usually with a speed cost; less attractive than modern memory-efficient attention when the latter is supported.

**Batching** — Processing multiple samples/requests in one model forward. Helps utilization on underfilled workloads but is less universally beneficial for compute-bound visual models than LLM decode.

**BF16** — 16-bit floating-point format with FP32-like exponent range. Common safe inference baseline.

**Cache / caching** — Reuse of intermediate states across denoising timesteps or transformer layers to avoid recomputation.

**Classifier-Free Guidance (CFG)** — Combines conditional and unconditional model predictions to strengthen conditioning. Classic CFG may require two model branches per step.

**CFG parallelism** — Compute conditional and unconditional branches on separate GPUs.

**Cold start** — Time from a new worker/process/node to readiness, including load, quantization, compilation, graph capture, and warmup.

**Context Parallelism (CP)** — Split the input token/context sequence across GPUs while typically replicating weights; common for long video/high-res DiTs.

**CUDA Graph** — Captured fixed sequence of GPU operations replayed with low CPU launch overhead. Requires stable shapes/memory behavior.

**CUTLASS** — NVIDIA template library for high-performance matrix and tensor kernels.

**CuTe** — Tensor/tile abstractions used to implement high-performance CUDA kernels.

**Data parallelism (DP)** — Replicate model on multiple GPUs and serve independent samples; best for throughput when each request fits one GPU.

**Denoiser** — The neural network repeatedly called across diffusion/flow timesteps. In a DiT, this is a transformer.

**Device map** — Persistent placement of pipeline components/layers across CPU/GPUs.

**DiT (Diffusion Transformer)** — Transformer used as the denoising/vector-field model in an iterative diffusion/flow generative process.

**Dynamic batching** — Combining compatible requests that arrive over a short window into one batch.

**Engine profile** — A versioned optimized configuration keyed by model, GPU, shape, dtype, attention, compile/cache settings, etc.

**Expert Parallelism (EP)** — Place MoE experts across GPUs and route tokens between them.

**FA2/FA3/FA4** — Generations of FlashAttention kernels, with newer versions targeting newer hardware/techniques. Exact support is version-specific.

**FirstBlockCache** — Cache method that uses change in early block output/state to decide whether later expensive blocks can be reused/skipped.

**FlashAttention** — IO-aware exact attention algorithm that tiles/recomputes to reduce HBM traffic and avoid materializing the full attention matrix.

**Flow matching / rectified flow** — Modern generative training/inference formulation used by many systems still colloquially grouped with diffusion. Inference remains iterative and shares DiT systems characteristics.

**FP16** — IEEE half-precision floating point.

**FP8** — 8-bit floating-point compute/storage family supported efficiently on newer accelerators.

**FP4 / NVFP4** — Very-low-precision 4-bit floating formats, especially relevant on Blackwell-class hardware; higher quality risk.

**Fullgraph** — `torch.compile` mode/request that disallows graph breaks so the whole region is compiled as one graph.

**GEMM** — General matrix-matrix multiplication; core operation of linear layers.

**GPU-seconds/output** — Wall-clock generation time multiplied by GPU count. Useful cost-normalized metric for parallel configurations.

**Group offload** — Move groups of layers between CPU and GPU, potentially overlapping transfer and compute.

**HBM** — High Bandwidth Memory attached to datacenter GPUs.

**Head dimension** — Feature dimension per attention head; affects kernel compatibility and performance.

**Inference** — Running a trained model to generate output.

**Kernel fusion** — Combine multiple operators so intermediate values remain on-chip and kernel-launch/memory overhead is reduced.

**Layerwise casting** — Store layer weights in lower precision and upcast for computation on demand.

**Latent** — Compressed representation in which most image/video diffusion operates before VAE decoding.

**Microscaling / MXFP8 / MXFP4** — Formats with small-block scale factors to preserve dynamic range/outliers at low precision.

**MMDiT** — Multimodal Diffusion Transformer family that jointly processes text/image streams in transformer blocks.

**MoE** — Mixture of Experts; routes tokens to a subset of MLP experts.

**NVLink** — High-bandwidth GPU interconnect.

**NVSwitch** — Switch fabric connecting GPUs with high-bandwidth NVLink-style all-to-all topology.

**Offload** — Move weights/components between GPU and CPU/other devices to reduce VRAM.

**PAB (Pyramid Attention Broadcast)** — Cache/reuse technique based on similarity of attention outputs across denoising timesteps.

**PagedAttention** — KV-cache paging method important to autoregressive LLMs; less central to conventional full-context visual DiTs because they do not use an LLM-style persistent decode KV cache.

**Pipeline Parallelism (PP)** — Split layers/stages across GPUs. Naive batch-1 PP can underutilize devices.

**PipeFusion** — Diffusion-specific pipeline approach exploiting patch/timestep structure to overlap computation/communication across devices.

**Prompt embedding cache** — Reuse outputs of text encoders for repeated prompts/conditioning.

**Quantization** — Represent weights/activations at lower precision to reduce memory and/or use faster compute.

**Regional compilation** — Compile repeated submodules/blocks rather than the whole model to reduce compile time while retaining steady-state speedup.

**Ring attention** — Distribute attention by circulating K/V blocks around devices while each rank owns local queries.

**Roofline model** — Performance model comparing arithmetic intensity against compute and memory-bandwidth ceilings.

**SageAttention** — Low-precision attention kernel family that quantizes parts of QK/PV computations.

**Scheduler / solver** — Numerical procedure/timestep schedule used to move the latent through diffusion/flow inference.

**SDPA** — PyTorch Scaled Dot Product Attention API with backend dispatch.

**Sequence Parallelism (SP)** — Split sequence/tokens across GPUs; often used synonymously or closely with context parallelism in DiT systems.

**Shape bucket** — A controlled set of input sizes/frame lengths mapped to reusable compiled/graph engine profiles.

**Tensor Core** — Specialized matrix-multiply hardware on NVIDIA GPUs (similar accelerator units exist on other platforms).

**Tensor Parallelism (TP)** — Shard weight matrices/compute across GPUs, requiring frequent collectives.

**Timestep** — One iteration index/noise level in the denoising/flow trajectory.

**Ulysses attention** — Sequence-parallel attention pattern using all-to-all redistribution so ranks process subsets of heads over the global sequence.

**VAE** — Variational autoencoder (or related latent codec) mapping pixels/frames to/from latent space.

**VRAM** — GPU memory. For datacenter GPUs usually HBM; for consumer GPUs often GDDR.

**Weight-only quantization** — Store weights low-bit while computing activations/GEMMs at a higher precision or with mixed kernels.

## RTX A5500 / project-specific terms

**RTX A5500** — NVIDIA Ampere workstation GPU used as the reference hardware in this handbook: 24 GB GDDR6 ECC, 768 GB/s memory bandwidth, PCIe 4.0 x16, optional two-GPU NVLink.

**`sm_86`** — Native CUDA machine-code target associated with compute capability 8.6-class Ampere workstation/GeForce GPUs. Verify the actual device capability at runtime before building extensions.

**No-regression quality gate** — Project rule that an optimization, especially quantization, is rejected if required benchmark accuracy decreases under the predeclared evaluation rule.

**Exact-first optimization** — Policy of attempting implementation-level improvements—attention kernels, compilation, fusion, static shapes, offload, sharding—before approximating model arithmetic or denoising trajectory.

**Native low-precision compute** — Hardware execution mode that directly consumes a lower-precision format in specialized matrix instructions. A low-bit checkpoint that is unpacked/dequantized to FP16 before GEMM is low-bit storage, not native low-bit compute.

**Phase-local residency** — Keeping a pipeline component on GPU only during the phase when it is used; e.g., text encoder during prompt encoding and VAE during decode.

**GPU-seconds/output** — Wall-clock GPU service time multiplied by the number of GPUs used, useful for exposing the hardware cost of multi-GPU latency improvements.
