# GPU Architecture Generations — What Matters When You Only Have Ampere

This handbook mentions Hopper/Blackwell because the uploaded book and modern inference literature discuss them. They are **not** the target hardware.

## Target: Ampere RTX A5500

Relevant features:

- third-generation Tensor Cores;
- FP16/BF16 Tensor Core matrix math;
- TF32;
- cc 8.6-specific execution limits;
- asynchronous global→shared copy/pipeline support;
- optional two-GPU NVLink on A5500;
- no native FP8 Tensor Core execution.

Use:

- SDPA;
- FlashAttention-2;
- Ampere-compatible CUTLASS/Triton kernels;
- `sm_86` compilation;
- FP16/BF16.

---

## Hopper — architectural context only

Hopper adds mechanisms and native FP8 paths used by:

- FlashAttention-3;
- FP8 Transformer Engine workflows;
- TMA-heavy kernels.

These explain why H100 benchmark numbers can be much better, but they are not A5500 tuning recipes.

---

## Blackwell — architectural context only

Blackwell extends low-precision and data-movement capabilities, including newer microscaling formats and kernels such as FA4 schedules.

Do not copy:

- MXFP8/MXFP4/NVFP4 expectations;
- `tcgen05` code;
- B200/B300 roofline numbers;
- 8-GPU NVSwitch assumptions.

---

## Architecture translation table

| Generic recommendation | A5500 translation |
|---|---|
| “Use FP8” | use FP16/BF16; low-bit only as gated storage experiment |
| “Use FlashAttention-3/4” | benchmark FlashAttention-2 / SDPA |
| “Compile for H100/B200” | compile for verified `sm_86` |
| “Use NVSwitch TP” | two-GPU NVLink/PCIe topology-aware split |
| “Keep 70B-style model in huge HBM” | component lifecycle/offload/shard within 24 GB |
| “Use TMA” | rely on Ampere async copy/pipeline mechanisms in compatible kernels |

---

## Why this matters

A performance mechanism can be architecture-specific even when the model architecture is identical. FlashAttention is a perfect example: the mathematical attention operation is the same, but FA2/FA3/FA4 use different schedules and hardware features.

Always ask:

- what SM generation was the result measured on?
- what dtype/instructions were used?
- what memory system/interconnect?
- is the implementation available for Ampere?

If not, treat the result as research context.
