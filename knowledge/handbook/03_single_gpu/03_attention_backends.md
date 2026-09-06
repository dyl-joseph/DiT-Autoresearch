# Attention Backends on RTX A5500: SDPA, FlashAttention-2, xFormers, and Quality-Sensitive Alternatives

Attention is one of the highest-value kernel families to optimize in a DiT because visual and especially video latent sequences can be large. The uploaded book emphasizes attention as a central compute cost, and the Wafer resource list points to FlashAttention as the canonical example of a faster **exact** implementation rather than a different model algorithm.

The target GPU changes the backend ranking.

## Executive recommendation

For RTX A5500 / Ampere:

1. benchmark **PyTorch SDPA**;
2. benchmark **FlashAttention-2** if the model/backend integration supports the exact attention pattern;
3. benchmark **xFormers memory-efficient attention** if it is supported and competitive;
4. use model/runtime-specific fused attention if it is known to support Ampere;
5. treat SageAttention/other quantized attention as **experimental only** because the project rejects benchmark-accuracy loss;
6. do not plan around FA3 or FA4: the official FlashAttention repository identifies FA3 as Hopper-specific and FA4 as Hopper/Blackwell-targeted.

---

## 1. Why attention implementation matters

Naive attention materializes large score/probability matrices:

`QK^T -> softmax -> P V`

The mathematical operation may be compute-heavy, but intermediate reads/writes can waste memory bandwidth and VRAM. IO-aware attention kernels tile the operation so that more work happens in registers/shared memory and large intermediates do not round-trip through global memory.

That is particularly useful on A5500 because:

- GDDR6 bandwidth is 768 GB/s;
- visual sequence lengths can be much larger than text decode microbatches;
- repeated denoising multiplies any per-attention inefficiency by the step count.

---

## 2. PyTorch scaled-dot-product attention (SDPA)

Modern PyTorch exposes `torch.nn.functional.scaled_dot_product_attention` and selects among available implementations depending on dtype, shape, masks, device, and version.

Advantages:

- built into the framework;
- integrates well with `torch.compile`;
- can dispatch to efficient fused kernels;
- fewer external build/version problems than a separate package.

Risks:

- silent fallback to a slower math path;
- mask/layout/head-dimension combinations can change backend selection;
- backend behavior changes across PyTorch versions.

### Benchmark rule

Do not say “SDPA is enabled” based only on source code. Verify the actual profiler kernel names and latency for the target shape.

### Useful minimal test

```python
import torch
import torch.nn.functional as F

q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

with torch.inference_mode():
    y = F.scaled_dot_product_attention(q, k, v)
```

Profile the real model because standalone tensor layout may differ from the pipeline.

---

## 3. FlashAttention-2

The official FlashAttention repository currently states that FlashAttention-2 CUDA supports Ampere, Ada, and Hopper GPUs and fp16/bf16 dtypes. That makes FA2 the relevant FlashAttention generation for RTX A5500.

Primary source: https://github.com/Dao-AILab/flash-attention

### Why it is attractive

- exact attention algorithm;
- fewer global-memory reads/writes;
- better work partitioning than the original FlashAttention;
- strong fit for repeated DiT attention.

### A5500 build considerations

When building from source:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

Adjust `MAX_JOBS` to host RAM/CPU. Verify the GPU capability locally before hard-coding 8.6.

### What to benchmark

For every important shape:

- self-attention versus cross/joint attention;
- fp16 versus bf16 reference dtype;
- head dimension;
- sequence length;
- batch size;
- causal/noncausal mode;
- mask type;
- contiguous versus transposed layout;
- eager versus compiled integration.

A backend can be excellent at one head dimension and lose at another.

---

## 4. FlashAttention-3 and FlashAttention-4: out of scope for A5500

The FlashAttention implementation repository explicitly labels:

- FA3: optimized for Hopper GPUs such as H100;
- FA4: optimized for Hopper and Blackwell.

Therefore, do not spend A5500 engineering time trying to reproduce FA3/FA4 benchmark claims. Their design ideas are useful reading, but the target machine lacks the architecture features those schedules exploit.

This handbook may mention FA3/FA4 when explaining the evolution of attention kernels. Such mentions are not deployment recommendations.

---

## 5. xFormers memory-efficient attention

xFormers can provide an alternative fused/memory-efficient attention path and may already be integrated by a Diffusers pipeline.

Reasons to test it:

- installation may be simpler in some environments;
- certain shapes/masks can favor it;
- memory behavior can differ from PyTorch SDPA/FA2.

Reasons not to assume it wins:

- PyTorch’s native kernels have improved substantially;
- xFormers version and CUDA build matter;
- compile integration can change the end-to-end result.

Treat xFormers as a measured candidate, not a default.

---

## 6. SageAttention and other quantized attention

SageAttention-style kernels reduce precision in parts of the attention computation to access faster low-precision execution on supported hardware.

For this project there are two issues:

1. **quality:** attention quantization can change outputs, and the project does not accept benchmark-accuracy loss;
2. **hardware:** some low-precision advantages highlighted on Hopper/Blackwell do not transfer directly to A5500.

Therefore SageAttention is **disabled by default**.

Only test it if:

- the exact implementation supports Ampere;
- the required model shape is supported;
- the benchmark suite has enough power to detect prompt/typography/video regressions;
- the candidate passes the strict no-regression gate;
- it actually improves end-to-end latency on A5500.

A microbenchmark win is insufficient.

---

## 7. Joint attention / MMDiT considerations

MMDiT-style models may concatenate or jointly process image and text tokens. This changes attention dimensions and masking patterns.

Questions to answer:

- are text and image tokens concatenated into one sequence?
- is attention symmetric or are there modality-specific projections?
- are masks materialized?
- does the backend support the exact head dimension?
- does `torch.compile` fuse the QKV reshape/transposes around attention?

The fastest standalone attention kernel can lose end-to-end if integration requires extra permutes/copies.

Profile the whole transformer block.

---

## 8. Video attention

Video latent token count grows with spatial dimensions and time. Attention can become a dominant share of total inference.

On A5500, this makes three strategies especially important:

1. efficient exact attention kernel;
2. reducing unnecessary tensor materialization/layout changes;
3. multi-GPU context/sequence parallelism if one 24 GB card cannot hold the activation footprint.

Approximate attention or low-precision attention remains quality-gated.

---

## 9. Attention backend benchmark harness

Use the real model tensor shapes when possible. For a synthetic comparison:

```python
import statistics
import torch
import torch.nn.functional as F

@torch.inference_mode()
def bench(fn, q, k, v, warmup=20, iters=100):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    xs = []
    for _ in range(iters):
        start.record()
        fn(q, k, v)
        end.record()
        end.synchronize()
        xs.append(start.elapsed_time(end))
    return statistics.median(xs)

B, H, N, D = 1, 24, 4096, 128
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

print(bench(lambda q,k,v: F.scaled_dot_product_attention(q,k,v), q,k,v))
```

Then repeat inside the actual transformer block and pipeline.

---

## 10. Correctness comparison for exact backends

For alternative exact attention implementations:

- compare to a high-confidence reference backend;
- use the same Q/K/V tensors;
- report max absolute error and relative error;
- test multiple random and adversarial shapes;
- run the full model benchmark because small tensor differences can propagate across denoising steps.

Example:

```python
err = (candidate.float() - reference.float()).abs()
print("max_abs", err.max().item())
print("mean_abs", err.mean().item())
```

Tensor-level closeness is necessary, not sufficient; the full quality suite is still the acceptance gate.

---

## 11. Profiling symptoms

### Attention is hot and memory throughput is high

Try:

- SDPA/FA2;
- remove layout conversions;
- fuse surrounding QKV transforms;
- confirm no score/probability matrix materialization.

### Attention is hot but SM/Tensor Core use is low

Check:

- unsupported head dimension;
- fallback kernel;
- tiny batch/sequence shape;
- excessive register/shared-memory pressure;
- non-contiguous layout;
- masking path.

### Attention got faster but model latency barely changed

Re-profile. MLPs, VAE, CPU launch gaps, or offload may now dominate.

### FA2 uses more memory than expected

Check temporary layout conversions, compiled workspace, duplicated model components, and whether the integration actually bypassed the old attention path.

---

## 12. A5500 decision table

| Backend | A5500 relevance | Accuracy policy | Primary reason to test |
|---|---|---|---|
| PyTorch SDPA | very high | exact-ish numerical reorder | easiest integrated fast path |
| FlashAttention-2 | very high | exact algorithm | IO-efficient Ampere-supported attention |
| xFormers | medium-high | exact-ish numerical reorder | alternate fused path |
| model-specific Triton/CUDA | medium | depends | unusual shapes/fusion opportunities |
| SageAttention | experimental | **strict no-regression** | approximate low-precision attention |
| FlashAttention-3 | out of scope | n/a | Hopper-only target |
| FlashAttention-4 | out of scope | n/a | Hopper/Blackwell target |

The winning backend is the one that passes the benchmark suite and improves the real workload, not the one with the newest name.
