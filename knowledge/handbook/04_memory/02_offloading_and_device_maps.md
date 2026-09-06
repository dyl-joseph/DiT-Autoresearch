# CPU/GPU Offloading and Device Placement on RTX A5500

Offload is a central strategy on a 24 GB A5500 because it can preserve the original model precision while solving residency limits. It is also dangerous: a bad offload plan replaces GPU compute with PCIe stalls.

---

## 1. Offload the coldest component first

Typical reuse frequencies:

| Component | Calls/request | Priority to keep on GPU |
|---|---:|---|
| transformer/denoiser | steps × branches | highest |
| control branch | model-dependent | high if repeated |
| text encoder | usually 1 | low after embeddings exist |
| VAE encode | 0–1 | phase-local |
| VAE decode | 1 | phase-local |

Therefore, exact first move:

- text encoder to CPU after prompt encoding;
- VAE to CPU until decode;
- transformer remains resident.

---

## 2. Component offload

Component offload moves whole modules between CPU and GPU at phase boundaries.

Best case:

- only a few transfers per request;
- large memory savings;
- no per-step transfer.

This is much better than layer-by-layer transformer offload when the denoiser repeats 30–50 times.

---

## 3. Model/group offload

If the transformer itself does not fit, group layers into chunks.

Let:

- `W` = total transformer weight bytes;
- `g` = number of resident groups at once;
- `S` = denoising steps.

If every group is transferred every step, total transferred weight volume is roughly proportional to:

`W × S`

Even a 20 GB transformer over 30 steps implies hundreds of GB of traffic. The goal is not just to fit; it is to hide or reduce this traffic.

---

## 4. Group size tradeoff

Small groups:

- lower peak VRAM;
- more transfers;
- more synchronization;
- more CPU launch overhead.

Large groups:

- higher VRAM;
- fewer transfers;
- better overlap potential;
- larger burst copy.

Sweep group size using real step timing.

---

## 5. Pinned memory

Use pinned host memory for asynchronous H2D transfers when beneficial.

Rules:

- preallocate a bounded pool;
- reuse it;
- avoid uncontrolled page-locking;
- monitor host RAM;
- ensure `non_blocking=True` is paired with truly pinned source memory.

Pinned memory is a performance resource, not free RAM.

---

## 6. Prefetch overlap

Ideal pipeline:

```text
GPU computes group k
copy stream transfers group k+1
when k finishes, k+1 is ready
```

Use CUDA streams/events to enforce dependencies without global synchronizations.

Nsight Systems should show copy rectangles overlapping compute. If not, the theoretical async design is not working.

---

## 7. PCIe behavior must be measured

A5500’s host interface is PCIe 4.0 x16. Real sustained transfer is lower than theoretical link rate.

Check:

- slot negotiated width/generation;
- NUMA placement of CPU memory;
- pinned/unpinned source;
- concurrent traffic;
- host memory bandwidth;
- CPU scheduling.

Large dual-GPU workstations can place GPUs/CPU memory behind different PCIe roots; topology matters.

---

## 8. Device maps

Automatic device maps are convenient but can place frequently communicating layers across slow boundaries.

For latency-sensitive DiTs, inspect the map manually.

Good placement tries to minimize transitions inside the repeated transformer loop.

Bad placement:

`GPU -> CPU -> GPU` multiple times per block/step.

A placement that fits is not necessarily a placement that performs.

---

## 9. Second GPU versus CPU offload

If two A5500s are available, a second GPU can act as a much faster weight/activation home than system RAM, especially with NVLink.

Compare:

- one-GPU + CPU offload;
- two-GPU sharding;
- one GPU denoiser + second GPU VAE/encoder;
- data parallel independent requests.

Use the benchmark suite and GPU-seconds metric.

---

## 10. Offload with compilation

Device movement can break compiled graphs.

Prefer:

- compile resident hot regions;
- move cold components outside those regions;
- keep group-offload schedule static if the compiler/runtime supports it.

A small eager copy manager around a compiled block can be better than trying to compile dynamic module migration.

---

## 11. Offload with CUDA Graphs

Graph capture wants stable addresses and allocations, while offload changes residency.

Usually easiest:

- graph-capture a resident denoiser bucket;
- use component offload outside the graph.

Group-offloaded transformer graphs are more complex and may not be worth the memory/capture complexity on 24 GB.

---

## 12. Exactness and quality

Offload does not intentionally change precision, so it is one of the preferred capacity strategies under the project’s quality constraint.

Still run the benchmark because:

- framework casts can sneak in;
- CPU fallback ops can use different kernels;
- race/order bugs in custom async prefetch can corrupt data.

---

## 13. Offload benchmark checklist

- [ ] persistent VRAM before request;
- [ ] peak allocated/reserved VRAM;
- [ ] host RAM peak;
- [ ] pinned-memory peak;
- [ ] H2D/D2H bytes;
- [ ] transfer time;
- [ ] overlap with compute;
- [ ] GPU idle due transfer;
- [ ] p50/p95 latency;
- [ ] quality PASS;
- [ ] no swap activity.

---

## 14. Recommended A5500 order

1. text encoder component offload;
2. VAE phase-local placement;
3. prompt embedding cache;
4. transformer group offload only if necessary;
5. tune pinned prefetch;
6. second-GPU sharding if available;
7. quantization only as last gated capacity experiment.
