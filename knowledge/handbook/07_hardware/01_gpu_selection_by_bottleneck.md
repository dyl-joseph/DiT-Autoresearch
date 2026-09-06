# Working Within Fixed Hardware: RTX A5500 Bottleneck Matching

This project does not have a GPU selection problem: the available accelerator is RTX A5500. The engineering question is therefore **which part of the A5500 system should each optimization target?**

## 1. The fixed resources

- 24 GB GDDR6 capacity
- 768 GB/s local memory bandwidth
- Ampere Tensor Cores for FP16/BF16/TF32
- PCIe 4.0 x16 host link
- optional two-GPU NVLink at 112.5 GB/s bidirectional
- 230 W board power / workstation thermal envelope

---

## 2. If compute is the bottleneck

Use:

- FP16/BF16 Tensor Core path;
- FA2/SDPA for attention;
- `torch.compile` and GEMM autotuning;
- fusion;
- context/tensor parallelism for latency if a second GPU is available;
- work reduction only when quality benchmark remains intact.

Do **not** plan on FP8 native compute.

---

## 3. If GDDR6 bandwidth is the bottleneck

Use:

- fusion;
- IO-aware attention;
- remove layout conversions;
- increase data reuse;
- keep hot weights resident;
- avoid per-step casts;
- exact caching of invariant tensors.

Low-bit storage is optional only if quality passes and the unpack path is faster.

---

## 4. If VRAM capacity is the bottleneck

Use:

1. text encoder lifecycle/offload;
2. prompt-embedding cache;
3. VAE phase-local placement;
4. VAE tiling/chunking;
5. CPU/group offload;
6. second-GPU sharding;
7. quantization last, behind strict no-regression gate.

---

## 5. If PCIe is the bottleneck

Use:

- larger offload groups;
- pinned buffers;
- async prefetch/overlap;
- keep transformer resident longer;
- move cold components instead of hot blocks;
- second GPU/NVLink if available.

---

## 6. If CPU launch overhead is the bottleneck

Use:

- `torch.compile`;
- regional compile;
- CUDA Graphs;
- remove per-step Python callbacks;
- fuse pointwise kernels;
- static shape buckets.

---

## 7. If inter-GPU communication is the bottleneck

Use:

- data parallel rather than fine-grained parallelism if throughput is the goal;
- coarser sharding;
- fewer collectives;
- NVLink-connected pair;
- overlap communication;
- alternative partition dimension.

---

## 8. If thermals/power are the bottleneck

Use:

- better chassis airflow;
- avoid thermally choking adjacent cards;
- sustained rather than burst benchmarks;
- production power/clock telemetry.

Do not claim a software regression before ruling out throttling.
