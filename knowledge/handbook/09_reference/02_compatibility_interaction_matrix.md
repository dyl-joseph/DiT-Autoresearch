# RTX A5500 Optimization Compatibility Matrix

Legend:

- ✅ usually compatible / quality-preserving in intent
- ⚠️ interaction requires benchmark/profile
- ❌ not target hardware path or fundamentally conflicting
- Q = quality-sensitive; must pass strict no-regression gate

| A \ B | FA2/SDPA | `torch.compile` | CUDA Graphs | CPU/group offload | Prompt cache | Approx DiT cache | 2-GPU CP/TP | Quantization |
|---|---|---|---|---|---|---|---|---|
| FA2/SDPA | — | ⚠️ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | Q |
| `torch.compile` | ⚠️ | — | ✅/⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️/Q |
| CUDA Graphs | ⚠️ | ✅/⚠️ | — | ⚠️ | ✅ | ⚠️ | ⚠️ | ⚠️/Q |
| CPU/group offload | ✅ | ⚠️ | ⚠️ | — | ✅ | ✅ | ⚠️ | ⚠️/Q |
| Prompt cache | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| Approx DiT cache | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | — | ⚠️ | Q + Q |
| 2-GPU CP/TP | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ | — | ⚠️/Q |
| Quantization | Q | ⚠️/Q | ⚠️/Q | ⚠️/Q | ✅ | Q + Q | ⚠️/Q | — |

---

## Hardware applicability matrix

| Technique | RTX A5500 status | Comment |
|---|---|---|
| FP16 Tensor Core | ✅ | primary path |
| BF16 Tensor Core | ✅ | primary path if model supports |
| TF32 | ⚠️ Q | only for FP32 path, quality check |
| Native FP8 Tensor Core | ❌ | Hopper-era feature, not A5500 |
| MXFP8/MXFP4/NVFP4 | ❌ | Blackwell-era, not A5500 |
| FlashAttention-2 | ✅ | official repo supports Ampere family |
| FlashAttention-3 | ❌ | Hopper-specific |
| FlashAttention-4 | ❌ | Hopper/Blackwell-specific |
| PyTorch SDPA | ✅ | benchmark actual backend |
| xFormers | ✅/⚠️ | version/shape dependent |
| SageAttention | Q | approximate/low-precision; strict quality gate |
| `torch.compile` | ✅ | high priority |
| CUDA Graphs | ✅/⚠️ | memory/static-shape constraints |
| Triton custom kernels | ✅ | compile for Ampere |
| CUTLASS Ampere kernels | ✅ | choose architecture-compatible path |
| TMA | ❌ | Hopper-specific concept/path |
| Ampere async global→shared copy | ✅ | architecture feature |
| NVLink 2-GPU | ✅ if bridge | 112.5 GB/s bidirectional official figure |
| NVSwitch assumptions | ❌ | not A5500 workstation topology |

---

## High-risk interaction notes

### Compile + offload

Device movement can create graph breaks. Keep compiled hot regions resident where possible.

### CUDA Graphs + offload

Static-address capture and dynamic residency fight each other. Prefer component offload outside the captured denoiser.

### Approx cache + quantization

Both change numerics/cache decisions. Never tune them together first. Establish each independently.

### Quantization + A5500

Low-bit storage is not native FP8 compute. Dequantization can erase speed gains.

### CP/TP + compile

Compiled local blocks can be excellent, but collectives become a larger fraction of step time after local kernels speed up. Re-profile communication.

### FA2 + compile

Benchmark whole block. External FA2 can win attention microbenchmarks while native SDPA + compile wins due surrounding fusion.

---

## Recommended safe stack

For most A5500 image DiTs:

```text
FP16/BF16
+ SDPA or FA2
+ torch.compile regional
+ shape buckets
+ prompt embedding cache
+ text encoder phase offload
+ VAE phase management
```

Optional:

```text
+ CUDA Graphs if memory allows
+ 2-GPU sharding/CP if needed
```

Research-only:

```text
+ quantization / approximate cache / step reduction
  ONLY if benchmark accuracy does not regress
```
