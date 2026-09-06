# Nsight Profiling Workflow for DiTs on RTX A5500

This is a practical profiling workflow for converting “the model feels slow” into an evidence-backed optimization plan on RTX A5500.

The intended order is:

```text
application timing
-> Nsight Systems timeline
-> phase attribution
-> hot kernel selection
-> Nsight Compute analysis
-> one optimization hypothesis
-> reprofile
```

Do not begin with Nsight Compute on every kernel. First find where the request spends time.

---

## 1. Establish a reproducible request

Choose one representative production bucket and freeze:

- model + revision;
- dtype;
- scheduler/solver;
- step count;
- guidance mode;
- prompt + negative prompt;
- seed;
- resolution/aspect ratio;
- frame count for video;
- batch/concurrency;
- attention backend;
- compile state;
- cache state;
- offload/device-map state.

Warm the model before steady-state profiling unless the explicit target is cold start.

Write the full configuration into the trace directory so an `.nsys-rep` file is never separated from the experiment definition.

---

## 2. Add NVTX ranges

Profiler traces are dramatically easier to read when pipeline phases have names.

```python
from torch.cuda import nvtx

nvtx.range_push("prompt_encode")
embeds = encode_prompt(...)
nvtx.range_pop()

for i, t in enumerate(timesteps):
    nvtx.range_push(f"denoise_step_{i:03d}")
    # transformer forward + scheduler update
    nvtx.range_pop()

nvtx.range_push("vae_decode")
image = decode(...)
nvtx.range_pop()
```

Add deeper ranges only when needed. Too many annotations can make the timeline noisy.

---

## 3. First Nsight Systems capture

A useful starting pattern:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=dit_a5500_baseline \
  python bench_one_request.py
```

Check installed `nsys --help` because flags can change by release.

Keep the capture short: one warmed request or a small number of steps is usually enough to identify major patterns.

---

## 4. Read the timeline from the outside inward

### A. Are there long CPU gaps between GPU kernels?

Possible causes:

- Python dispatch;
- scheduler work;
- dynamic shape logic;
- synchronization (`.item()`, printing CUDA tensors, explicit synchronize);
- compilation/recompilation;
- allocator work;
- data preparation.

Candidate fixes:

- `torch.compile`;
- regional compilation;
- static shape buckets;
- CUDA Graph capture;
- move scalar/control work out of inner loop;
- avoid accidental sync.

### B. Are there H2D/D2H copies inside every denoising step?

Likely cause: overly aggressive layer/model offload.

Candidate fixes:

- keep hot transformer resident;
- offload once-per-request components instead;
- use group offload only when the latency/fit tradeoff is favorable;
- use the second A5500 for exact sharding;
- pin/stage transfers only if they can overlap useful compute.

### C. Is the GPU continuously busy with a few large kernels?

Then launch overhead is not the first problem. Identify the largest kernel families and move to Nsight Compute.

### D. Are there hundreds/thousands of tiny kernels?

Likely pointwise/norm/modulation fragmentation. Try compiler fusion before manual CUDA.

### E. Is VAE decode a large tail after denoising?

Then transformer-only work has already improved enough that the decoder is a product-level bottleneck. Optimize/tile/compile VAE separately.

---

## 5. Generate a system summary

Nsight Systems can report aggregate statistics from a trace. Depending on installed version:

```bash
nsys stats dit_a5500_baseline.nsys-rep
```

Look at:

- CUDA GPU kernel summary;
- CUDA API calls;
- memory operations;
- NVTX ranges;
- OS runtime/synchronization.

Sort by total time, not merely invocation count.

A tiny kernel launched 10,000 times can matter. A giant kernel launched 20 times can also matter. Both need total-time context.

---

## 6. Classify the top kernels

Create a small table:

| Kernel family | Total GPU time | Calls | Avg | Phase | Hypothesis |
|---|---:|---:|---:|---|---|
| attention | | | | denoise | IO/compute |
| GEMM QKV | | | | denoise | compute |
| MLP GEMM | | | | denoise | compute |
| norm/modulation | | | | denoise | launch/bandwidth |
| memcpy | | | | all | transfer |
| VAE conv | | | | decode | compute |

Do not optimize the first scary kernel name you recognize. Rank by contribution to request time.

---

## 7. Move one kernel family into Nsight Compute

Nsight Compute metric collection can replay kernels and alter timing. Narrow the target.

Possible workflow:

1. use Nsight Systems to identify demangled kernel name;
2. create a minimal script that reaches the same shape;
3. warm several launches;
4. capture a small launch range.

Illustrative pattern:

```bash
ncu \
  --set roofline \
  --kernel-name-base demangled \
  --launch-skip 20 \
  --launch-count 5 \
  --output=ncu_attention_a5500 \
  python bench_attention_shape.py
```

Run `ncu --list-sets` and `ncu --list-sections` locally to match the installed version.

---

## 8. Interpret a hot GEMM

Questions:

- Is the intended FP16/BF16 Tensor Core path active?
- Is achieved compute throughput high relative to the A5500 ceiling?
- Is the kernel memory-bound instead?
- Are dimensions/strides forcing a less efficient algorithm?
- Is there a conversion/reformat kernel immediately before/after?
- Are registers or shared-memory limits suppressing useful concurrency?

If the library GEMM is already near a practical roof, focus on reducing how many GEMMs you execute (fewer steps/caching—quality-sensitive) or fusing surrounding work (exact) rather than rewriting the multiply.

---

## 9. Interpret a hot attention kernel

Questions:

- Which backend actually executed?
- How much request time is attention?
- What are sequence length and head dimension?
- Is full `N x N` intermediate materialization occurring?
- Is the kernel achieving strong memory throughput?
- Is softmax/reduction dominating?
- Are QKV layout conversions surrounding it?
- Does FA2 or SDPA provide a better exact path for this shape?

For A5500, do not jump directly to FA3/FA4 or FP8 attention. Those newer hardware paths are outside the target.

---

## 10. Interpret pointwise/norm kernels

If each kernel is tiny but aggregate time is meaningful:

- check launch gaps;
- inspect whether `torch.compile` fuses them;
- inspect graph breaks;
- test static shape buckets;
- evaluate a fused Triton kernel only if compiler output remains poor.

This is a common second-order bottleneck after attention/GEMM improve.

---

## 11. Memory bandwidth evidence

For a suspected bandwidth-bound kernel, look for:

- DRAM throughput near a large fraction of sustainable hardware bandwidth;
- low arithmetic intensity;
- low compute utilization because data delivery is limiting;
- large global read/write volume;
- poor cache hit rate where reuse should exist.

Fixes often include:

- fuse adjacent operations;
- improve coalescing/layout;
- avoid materialized intermediates;
- reuse data from shared memory/registers;
- eliminate redundant casts/copies.

Quantizing storage is not the first answer in this project because quality cannot regress and low-bit compute is not native on A5500. Exact IO reduction is preferred.

---

## 12. Compute-bound evidence

For a compute-bound GEMM/attention kernel:

- Tensor Core/pipeline utilization is high;
- memory throughput is not the ceiling;
- arithmetic intensity is high;
- throughput approaches the realistic precision-specific compute roof.

Potential actions:

- ensure Tensor Core-friendly dtype/layout;
- improve work partition/tile shape;
- use a better existing kernel implementation;
- reduce model evaluations (steps) only if the quality contract permits;
- parallelize across a second GPU only when communication is cheaper than the saved compute.

Do not waste time on PCIe transfer tuning if there are no transfers in the hot phase.

---

## 13. Occupancy: use it diagnostically

Low occupancy can indicate:

- high register pressure;
- high shared-memory allocation/block;
- too-large blocks;
- limited blocks/SM.

But “maximize occupancy” is not the objective. A compute-dense tiled kernel can outperform a high-occupancy kernel because it has better reuse and instruction efficiency.

Correlate occupancy with:

- stalls;
- issue rate;
- memory latency hiding;
- register spills;
- throughput.

---

## 14. Register spills

Spills create local-memory traffic backed by device memory/cache. In a fused kernel, they can erase the memory traffic saved by fusion.

If a candidate fusion regresses:

1. compare registers/thread;
2. check local-load/store metrics;
3. test a smaller tile or split fusion boundary;
4. reduce live intermediate count;
5. retune number of warps/stages where applicable.

Do not assume “more fusion = faster.”

---

## 15. Shared-memory pathologies

Inspect:

- bank conflicts;
- shared-memory throughput;
- barrier stalls;
- shared allocation per block;
- occupancy effect.

Shared memory should create reuse or reorganize accesses. If it only adds a copy/barrier, remove it.

---

## 16. Detect graph breaks/recompilation

When using `torch.compile`, a timeline may show compilation or shape-specialized behavior outside the expected warmup.

Keep a shape-bucket test:

```text
512x512 -> warm twice -> measure
768x768 -> warm twice -> measure
1024x1024 -> warm twice -> measure
768x1344 -> warm twice -> measure
repeat all buckets -> verify no surprise compilation
```

Log compiler diagnostics separately from the GPU trace.

---

## 17. Offload trace checklist

For every memory-saving offload configuration, answer:

- how many H2D bytes/request?
- how many D2H bytes/request?
- how many copy events/step?
- are copies on a separate stream?
- do copies overlap compute?
- does pinned memory help?
- is host RAM duplicating weights?
- does the hot transformer remain resident?

A configuration that saves 6 GiB but adds a large PCIe transfer before every block is a capacity workaround, not a speed optimization.

---

## 18. Two-GPU trace checklist

Before analyzing collective efficiency:

```bash
nvidia-smi topo -m
```

Record whether the pair is connected by the intended NVLink bridge or communicating through PCIe.

Then trace:

- peer copies;
- NCCL collectives;
- synchronization gaps;
- load imbalance between ranks;
- compute/communication overlap;
- per-rank memory peak.

An algorithm that scales well on an 8× H100 NVSwitch system may be communication-bound on a two-workstation-GPU topology.

---

## 19. Profiler perturbation

Profilers change execution:

- Nsight Compute may replay kernels;
- collecting many counters adds overhead;
- debug/sanitizer modes can drastically slow code;
- tracing every OS call can create huge files.

Use profiler timing for diagnosis, then re-run the normal benchmark harness without profiling to decide whether the candidate is actually faster.

---

## 20. Re-profile after every major optimization

A sensible sequence:

```text
P0 eager + SDPA
P1 exact attention winner
P2 compiled transformer
P3 fused/static bucket/CUDA Graph candidate
P4 memory placement optimized
P5 two-GPU candidate
```

After each stage, capture a short system trace. The ranking of bottlenecks may change completely.

---

## 21. Quality-sensitive methods require an extra branch

If the optimization changes model arithmetic or algorithm—quantization, timestep caching, step reduction, guidance cutoff, sparse approximation—profiling alone cannot accept it.

The workflow becomes:

```text
profile -> candidate faster?
              |
              yes
              v
       full benchmark quality gate
              |
         no regression?
          /       \
        no         yes
      reject   consider deploy
```

For quantization in this project, **any required benchmark regression rejects the candidate**.

---

## 22. Minimal profiler report template

```markdown
# Profile report: <candidate>

## Environment
- GPU: RTX A5500 x1/x2
- topology: <nvidia-smi topo -m summary>
- driver/CUDA/Torch: ...
- model/revision: ...
- dtype: FP16/BF16
- quantization: none / experimental

## Workload
- resolution/frames: ...
- steps/guidance: ...
- prompt length: ...
- batch/concurrency: ...

## Baseline
- p50/p95: ...
- peak VRAM: ...
- quality suite: PASS

## Nsight Systems finding
- dominant phases: ...
- dominant kernel families: ...
- CPU gaps/transfers: ...

## Nsight Compute finding
- kernel: ...
- suspected bottleneck: compute / bandwidth / latency
- evidence: ...

## Hypothesis
<one mechanism>

## Candidate change
<one change>

## Result
- p50/p95: ...
- peak VRAM: ...
- quality gate: PASS/FAIL
- decision: keep/revert
```

This report format makes optimization cumulative and auditable instead of anecdotal.
