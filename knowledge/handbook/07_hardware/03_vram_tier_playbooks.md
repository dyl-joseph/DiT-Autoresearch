# 24 GB VRAM Playbook: RTX A5500

The target GPU has 24 GB. This chapter focuses on how to allocate that budget without sacrificing benchmark accuracy.

## 1. Reserve headroom

Do not target 24.0 GB resident tensors.

Budget for:

- CUDA context/runtime;
- allocator fragmentation;
- attention/GEMM workspaces;
- activations;
- compiler/CUDA Graph pools;
- VAE peak;
- control inputs;
- unexpected shape variation.

Measure safe headroom per bucket.

---

## 2. If model + components fit under budget

Best configuration:

- keep transformer resident;
- keep frequently used conditioning resident;
- compile;
- use FA2/SDPA;
- avoid offload.

Memory optimization should not be added merely because it exists.

---

## 3. If only full pipeline exceeds 24 GB

Likely solution:

- run text encoder, save embeddings, offload encoder;
- run denoiser with transformer resident;
- offload/free transformer temporaries;
- load/run VAE decode.

This preserves precision and minimizes PCIe traffic.

---

## 4. If transformer itself nearly fills 24 GB

Use:

- lower concurrency/batch;
- no persistent encoder/VAE;
- regional compile instead of memory-heavy graph mode;
- exact CPU group offload if necessary;
- second A5500 sharding.

Quantization is not first-line because quality cannot regress and A5500 lacks native FP8 acceleration.

---

## 5. If activation memory causes OOM

Typical high-resolution/video problem.

Use:

- memory-efficient attention;
- context/sequence parallelism;
- VAE tiling;
- chunked temporal/spatial operations if model/runtime supports exact semantics;
- reduce concurrent requests.

Weight quantization may not help if the peak is activation-dominated.

---

## 6. If compile/CUDA Graph causes OOM

Try:

- regional compile;
- no graph capture for large buckets;
- fewer shape variants;
- free text encoder before compiling hot region;
- lower autotune workspace mode.

Compiler memory is real memory.

---

## 7. If one GPU cannot fit even with exact offload

Use a second A5500:

- layer/model sharding;
- context parallelism;
- tensor parallelism if supported and measured.

NVLink is preferred when available.

Only after exact multi-GPU alternatives are understood should low-bit storage be evaluated.

---

## 8. Quantization rule

A quantized configuration that fits in 24 GB is accepted only if:

- it passes the strict no-regression quality gate;
- it improves the relevant objective versus exact offload/sharding;
- conversion/workspace memory is included.

If quality drops, the correct solution is more memory/system engineering, not accepting the regression.
