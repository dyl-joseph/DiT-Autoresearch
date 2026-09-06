# Attention Cost in Image, Video, 3D, and Multimodal DiTs

## Attention is where visual token count becomes expensive

Let `N` be the number of latent tokens, `H` the number of heads, and `D` the head dimension. Dense self-attention performs work proportional to roughly `O(N^2 * D)` for the score/value products, plus `O(N * D_model^2)` for Q/K/V/output projections. The exact crossover between projection-dominated and attention-dominated behavior depends on model width and token count.

For a 2D latent grid with patch size `p`, token count scales approximately as:

`N_image = (H_latent / p) * (W_latent / p)`

For a video latent with temporal patch size `p_t`:

`N_video = (T_latent / p_t) * (H_latent / p) * (W_latent / p)`

Doubling both spatial dimensions quadruples `N`; dense attention's score computation can then grow by roughly 16x. Adding frames multiplies the sequence again. This is why a model that looks comfortable at 768px can become a completely different systems problem at 2K or at dozens of frames.

## Self-attention vs cross-attention vs joint attention

### Self-attention

The expensive term is visual-token-to-visual-token attention. It is usually the dominant attention cost at large spatial/temporal contexts.

### Cross-attention

If the query is visual and K/V are prompt tokens, complexity is approximately `O(N_visual * N_text)`. Long prompts matter, but this usually scales better than visual self-attention because text sequence length is far smaller than visual sequence length.

### MMDiT / joint attention

Architectures that concatenate or jointly process text and image streams can create attention over a combined context. The cost depends on whether the implementation actually performs full joint attention, uses separate parameter streams, or factors interactions. Treat the model's processor implementation—not the marketing name—as the source of truth.

### Spatial/temporal factorization

Video architectures may split spatial and temporal attention rather than attending over all space-time tokens at once. This changes complexity and parallelization opportunities. Profiling must identify which attention family dominates.

## Memory behavior

A naive implementation may materialize an `N x N` score tensor for every head or batch element. At video-scale `N`, this is often impossible. Memory-efficient attention avoids storing the full matrix by tiling/recomputing softmax statistics.

This distinction is crucial:

- **mathematical complexity** can remain quadratic;
- **activation-memory complexity** can drop substantially;
- **HBM traffic** can drop enough to improve speed even on a compute-bound model.

The uploaded PDF emphasizes this IO-aware benefit of FlashAttention and notes that video attention can dominate total compute.

## Backend families

### PyTorch SDPA

PyTorch `scaled_dot_product_attention` dispatches among implementations based on shape/hardware. It is a strong baseline and is enabled by default in recent Diffusers pipelines.

Use it when:

- you want the least maintenance burden
- your shapes are conventional
- compile compatibility matters
- custom kernels do not show a measured win

### FlashAttention on the A5500 target

FlashAttention minimizes external-memory reads/writes through tiling and recomputation. For this handbook’s RTX A5500 target, **FlashAttention-2 is the relevant official generation**; FlashAttention-3 is Hopper-focused and FlashAttention-4 targets Hopper/Blackwell-class hardware. Benchmark FA2 against PyTorch SDPA for the exact head dimension, dtype, mask, layout, and sequence length rather than assuming either wins.

### SageAttention

SageAttention variants approximate/quantize portions of attention. They are **not a default A5500 optimization** in this project: the A5500 lacks the newer native FP8/microscaling paths that motivate many such results, and attention is quality-sensitive. Treat SageAttention only as a research candidate. It must beat the best exact SDPA/FA2 baseline **and** pass the full no-regression benchmark gate.

### xFormers / memory-efficient attention

Still useful for compatibility and some hardware/shape regimes. Treat it as a benchmark candidate rather than an unconditional upgrade over SDPA.

### Specialized/vendor kernels

CUTLASS, CuTe, cuDNN attention, Triton kernels, TensorRT plugins, SGLang kernels, and platform-specific backends may win for a specific architecture. Kernel selection is hardware-specific.

## How to benchmark attention correctly

A microbenchmark should sweep:

- sequence length `N`
- head count `H`
- head dimension `D`
- batch / CFG batch
- dtype
- causal vs noncausal (visual DiTs are commonly noncausal)
- attention masks
- contiguous/layout variants
- QKV packing form
- compile state
- GPU architecture

Record both:

- kernel latency
- peak temporary memory

Then run the backend end-to-end. A 20% faster attention kernel produces only a 4% overall speedup if attention is 20% of runtime.

## Attention share as an optimization threshold

A useful heuristic:

- `<20%` of denoiser GPU time: attention optimization is secondary
- `20-50%`: attention backend matters, but optimize MLP/GEMM/compile too
- `>50%`: attention is a primary optimization target
- `>70%`: common in large video contexts; exact attention-kernel selection and context/sequence parallelism become central. Approximate caching or attention quantization are optional research candidates only after quality-gated evaluation.

The PDF gives a representative video-model figure of 70-80% of compute time in attention. Treat it as a regime example, not a universal constant.

## Reducing attention work rather than only accelerating it

### Fewer steps

Every removed denoising step removes every attention call in that step. This is usually a much larger lever than a small kernel improvement.

### Cache attention or transformer states

PAB-like approaches exploit similarity of attention outputs across nearby timesteps. First-block and transformer cache methods may skip whole blocks after detecting small changes.

### Sparse/local/windowed attention

If the architecture supports it, structured sparsity changes the algorithmic cost. This generally requires training-time architectural support or a carefully validated approximation; it is not a drop-in kernel switch.

### Token reduction

Larger patches, latent downsampling, token merging, pruning, or hierarchical generation reduce sequence length. These are model- or algorithm-level changes and can affect quality significantly.

### Factorized video attention

Spatial and temporal factorization can reduce cost and make sharding easier, but it is an architectural design choice.

## Precision sensitivity

Attention is often more sensitive to quantization than large linear layers. The PDF recommends selective strategies for video:

- preserve early denoising steps at higher precision
- preserve first/last transformer layers
- if—and only if the exact baseline cannot meet the capacity/performance target—experiment with lower precision in less sensitive hidden layers/later steps, then reject it on any required benchmark regression
- on newer hardware, blockwise/microscaling formats can preserve outliers; this is architectural context, not an A5500-native recommendation

The same principle generalizes to image DiTs: build a **precision map** rather than assuming one dtype fits every operator.

## Multi-GPU implications

### Context / sequence parallelism

Split tokens across GPUs. This directly addresses activation memory and parallelizes attention. Common communication patterns include:

- ring attention: circulate K/V blocks
- Ulysses-style all-to-all: redistribute heads and sequence
- hybrid / unified sequence parallelism: combine dimensions

### Head divisibility and arbitrary shapes

Some algorithms require sequence/head divisibility. Modern implementations increasingly add “anything” variants that pad or redistribute arbitrary shapes. Shape compatibility must be part of the serving policy.

### Communication crossover

Parallel attention is only useful if the saved compute is larger than collective overhead. NVLink/NVSwitch helps; PCIe is much less forgiving. High-resolution/video workloads usually have enough compute per step to justify CP earlier than small image workloads.

## Failure modes

- Backend silently falls back to a slower implementation.
- A mask/layout forces the math path instead of FlashAttention.
- `torch.compile` introduces graph breaks around a custom processor.
- Quantized attention creates temporal flicker or prompt-adherence drift that average image metrics miss.
- CP reduces per-GPU memory but increases latency because the sequence is too short.
- A cache threshold tuned at 50 steps behaves differently at 20 steps.
- Variable shapes trigger recompilation and mask any kernel win.

## Decision rule

Do not ask “Which attention library is fastest?” Ask:

> For this exact GPU, model block, sequence length, head dimension, precision, compile state, and quality budget, which backend yields the lowest end-to-end denoiser time without increasing peak memory or quality error beyond the gate?

That is the only portable question.

## Current references

- Hugging Face Diffusers: Attention backends — https://huggingface.co/docs/diffusers/en/optimization/attention_backends
- FlashAttention — https://github.com/Dao-AILab/flash-attention
- SageAttention — https://github.com/thu-ml/SageAttention

## Source basis

**PDF-derived:** attention as a major cross-modality cost, FlashAttention's IO optimization, video attention dominance, selective attention quantization.  
**Expansion:** formulas for 2D/3D token scaling, backend decision process, microbenchmark matrix, context-parallel implications.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
