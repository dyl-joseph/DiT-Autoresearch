# Profiling Symptoms and Diagnosis on RTX A5500

Use this file as a profiler-to-action dictionary. The goal is to prevent random tuning.

## 1. GPU utilization is low and there are visible CPU gaps

Likely causes:

- Python overhead;
- scheduler logic on CPU;
- `.item()`/sync calls;
- many tiny pointwise kernels;
- dynamic dispatch/graph breaks;
- dataload/preprocessing starvation.

Actions:

1. `torch.compile` repeated transformer region;
2. remove per-step logging/synchronization;
3. CUDA Graph replay for static buckets;
4. fuse repeated pointwise chains;
5. move scheduler tensor math to GPU where framework permits.

Use Nsight Systems, not Nsight Compute, first.

---

## 2. GPU utilization is high but model is still slow

High utilization alone says almost nothing.

Check hot kernels with Nsight Compute:

- Tensor Core use;
- SM throughput;
- DRAM bandwidth;
- register pressure;
- shared-memory pressure;
- stall reasons.

If the top GEMMs are near a compute roof, deeper kernel micro-optimization may have limited upside. Focus on reducing repeated work, parallelism, or other phases.

---

## 3. Attention takes a large fraction of GPU time

On A5500:

1. benchmark PyTorch SDPA;
2. benchmark FlashAttention-2;
3. test xFormers if relevant;
4. inspect layout/transpose kernels around attention;
5. if video/high resolution does not fit, test context/sequence parallelism.

Do not jump to FA3/FA4; they target newer GPU architectures.

Do not jump to quantized attention if quality cannot move.

---

## 4. Attention kernel is fast but QKV reshapes/transposes are hot

Symptoms:

- many `permute`, `contiguous`, copy, cast kernels;
- attention itself is a small share;
- memory throughput high around layout transforms.

Actions:

- compile/fuse surrounding transforms;
- maintain backend-preferred layout longer;
- avoid repeated contiguous conversions;
- benchmark whole block, not standalone attention.

---

## 5. GEMM kernels dominate

If they are near Tensor Core compute roof:

- algorithmic/work-count changes are needed for large speedups;
- multi-GPU may lower latency;
- custom GEMM work likely has diminishing returns.

If they are far below expected throughput:

- verify Tensor Core dtype;
- inspect matrix dimensions/alignment;
- compile/static shape specialization;
- inspect algorithm selected by cuBLASLt/Inductor;
- test custom Triton/CUTLASS only if repeated shape is material.

---

## 6. Norm/modulation kernels dominate more than expected

Symptoms:

- many short kernels;
- modest arithmetic intensity;
- large cumulative launch count across dozens of blocks/steps.

Actions:

- `torch.compile`;
- regional compile;
- custom fusion after compiler attempt;
- eliminate extra casts.

This is a classic GDDR6/launch-overhead opportunity.

---

## 7. VAE dominates after denoiser optimization

Actions:

- profile conv kernels/layouts;
- compile VAE if stable;
- test tiling/chunking only if memory requires it;
- free denoiser buffers before decode;
- consider second-GPU VAE placement only after transfer analysis.

Do not keep optimizing transformer attention when VAE is now 35% of request latency.

---

## 8. Text encoder dominates short/few-step model

Actions:

- cache prompt embeddings;
- batch encoder work;
- precompute embeddings;
- offload/free encoder after use;
- keep encoder on CPU or second GPU if latency still acceptable.

Few-step models often expose encoder/VAE overhead that was hidden by the denoiser.

---

## 9. Model fits at idle but OOMs during generation

Possible causes:

- attention workspace;
- MLP activations;
- VAE peak;
- compiler/CUDA Graph static pools;
- allocator fragmentation;
- duplicate model components;
- dynamic shape cache accumulation.

Actions:

- measure peak allocated + reserved;
- inspect phase of OOM;
- remove cold components before hot phase;
- VAE tile if decode peak;
- reduce graph variants;
- offload/shard before quantization if accuracy cannot change.

---

## 10. Quantized model fits but is slower

Expected possibility on A5500.

Likely:

- dequantization/unpack overhead;
- low-bit storage but FP16/BF16 compute;
- poor custom kernel shape;
- graph breaks;
- conversion copies.

Decision:

- compare to exact offload/sharding configuration;
- if no speed or essential capacity benefit, remove quantization;
- if quality drops, reject regardless of speed.

---

## 11. Offload path has sawtooth GPU utilization

Symptoms:

- GPU computes, then idle gap, then H2D copy;
- repeated every group/step.

Actions:

- pin host buffers;
- prefetch next group;
- increase group size;
- keep more groups resident;
- move text encoder/VAE out instead of hot transformer groups;
- use second GPU if available.

---

## 12. H2D copy stream overlaps poorly

Check:

- source memory is pinned;
- non-blocking transfer actually used;
- copy and compute use separate streams;
- dependency/event synchronization is correct;
- CPU can enqueue prefetch in time;
- PCIe link width/generation is correct.

Nsight Systems reveals whether overlap is real.

---

## 13. `torch.compile` makes one bucket fast but others slow

Likely:

- recompiles;
- dynamic shapes;
- graph breaks;
- cached graphs consuming memory;
- general kernel not optimal for all shapes.

Actions:

- explicit shape buckets;
- per-bucket compile cache;
- generic fallback for rare shapes;
- regional compile.

---

## 14. `torch.compile` OOMs

Likely:

- CUDA Graph static memory;
- autotune workspaces;
- duplicated compiled buffers;
- too many shape profiles.

Actions:

- compile only transformer blocks;
- use less aggressive mode;
- disable graph path for that bucket;
- free text encoder first;
- reduce worker concurrency.

---

## 15. CUDA Graph does not improve latency

Likely:

- GPU kernels themselves dominate with no CPU gaps;
- graph capture adds no benefit;
- large kernels already keep device busy.

Keep it only if it improves tail stability or CPU load without unacceptable memory cost.

---

## 16. CUDA Graph improves p50 but increases OOMs

Static pools reduce available dynamic memory. On 24 GB, that can be a bad trade.

Use graph only for smaller buckets or after freeing cold components.

---

## 17. Two GPUs are barely faster than one

Inspect:

- topology;
- NVLink presence;
- communication time;
- synchronization frequency;
- load balance;
- duplicated VAE/encoder work;
- CPU rank launch overhead.

If communication dominates, use coarser sharding or independent data parallelism.

---

## 18. Two GPUs fit a model but latency increases

That can be acceptable if fit is the goal. Report it honestly as a **capacity** optimization, not a speed optimization.

Try:

- sharding only the component that does not fit;
- keeping frequently communicating attention sequence chunks balanced;
- avoiding unnecessary all-gathers;
- using NVLink pair.

---

## 19. p50 improves but p95 gets worse

Likely:

- dynamic compilation;
- cache misses;
- shape-specific slow path;
- allocator pressure;
- thermal throttling;
- queue contention;
- intermittent offload stalls.

Trace slow samples specifically.

---

## 20. First request is much slower

Possible:

- model loading;
- CUDA context;
- JIT extension;
- compile;
- autotune;
- graph capture;
- lazy device movement.

Separate cold/first/warm metrics. Prewarm expected buckets in serving environments.

---

## 21. Sustained latency degrades over minutes

Check telemetry:

- temperature;
- power;
- clocks;
- other GPU processes;
- host memory pressure;
- CPU thermal limits;
- cache/allocator growth.

A short benchmark may be invalid for long video jobs.

---

## 22. Memory reserved keeps growing across shapes

Likely:

- compiler graph cache;
- CUDA Graph pools;
- allocator fragmentation;
- backend workspace caching.

Actions:

- bounded shape set;
- worker recycling policy if necessary;
- inspect memory snapshots;
- do not call `empty_cache()` every request as a substitute for architecture.

---

## 23. Low measured memory bandwidth but kernel is slow

Could be:

- dependency latency;
- low occupancy from registers/shared memory;
- instruction bottleneck;
- serial reduction;
- tensor-core underutilization;
- divergent control flow.

Use Nsight Compute stall/occupancy data rather than assuming memory bottleneck.

---

## 24. High occupancy but low throughput

Occupancy is not the goal. Check:

- instruction mix;
- memory coalescing;
- data reuse;
- Tensor Core use;
- synchronization.

This is one of the central lessons from the performance-engineering references in the Wafer list.

---

## 25. Profiler-to-action summary

| Symptom | First tool | First action |
|---|---|---|
| CPU gaps | Nsight Systems | compile/graphs/remove sync |
| attention hot | Systems + Compute | SDPA/FA2 |
| GEMM hot, low utilization | Nsight Compute | shape/algorithm/compile |
| pointwise launch storm | Systems | compile/fuse |
| PCIe stalls | Systems | residency/prefetch |
| OOM during VAE | memory stats | tile/free denoiser buffers |
| 2-GPU scaling poor | Systems + topology | reduce collectives/coarsen split |
| quantized slower | Systems/Compute | remove or use exact alternative |
| p95 spikes | request trace | compile/cache/thermal diagnosis |

The profiler is a decision system, not a screenshot generator.
