# RTX A5500 DiT Optimization Ladder

This is the operational sequence for optimizing one Diffusion Transformer workload on an RTX A5500. It is designed to minimize wasted experiments and protect the no-regression quality requirement.

Each stage has four questions:

1. **What mechanism are we changing?**
2. **What profiler evidence justifies it?**
3. **What metric should move?**
4. **What quality/correctness gate applies?**

Do not advance by stacking unmeasured changes.

---

## Stage 0 — freeze the reference

Create an immutable baseline manifest.

Record:

- model repository/path + revision;
- all component revisions;
- scheduler;
- denoising steps;
- guidance/CFG configuration;
- dtype;
- prompt encoder settings;
- width/height/frame buckets;
- batch size;
- attention backend;
- offload settings;
- exact Python/PyTorch/CUDA/Diffusers versions;
- GPU UUID;
- benchmark corpus hash.

Set quantization to `none`.

Run the full quality benchmark and save the outputs/metrics. This is the reference every later configuration must beat or match.

---

## Stage 1 — remove obvious accidental overhead

Check:

```python
assert not torch.is_grad_enabled()
```

Prefer:

```python
with torch.inference_mode():
    ...
```

Also verify:

- model is on intended device;
- no tensors unexpectedly on CPU inside denoiser;
- no `.item()`/`.cpu()`/printing in hot loop;
- no debug callbacks each step;
- no repeated `torch.cuda.empty_cache()`;
- no safety checker or image serialization included in a denoiser-only microbenchmark;
- no repeated tokenizer/text-encoder work for identical prompt embeddings;
- `dtype` matches the intended FP16/BF16 reference.

Measure again. Many pipelines contain surprisingly large Python/transfer overhead.

---

## Stage 2 — capture an end-to-end timeline

Use Nsight Systems:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o baseline_a5500 \
  python bench_pipeline.py
```

Identify:

- text encoder region;
- each denoising step;
- attention kernels;
- GEMMs;
- pointwise/norm kernels;
- VAE decode;
- H2D/D2H copies;
- CPU launch gaps.

Write down the top five contributors by total request time.

---

## Stage 3 — select the attention backend

Test one by one:

1. PyTorch SDPA;
2. FlashAttention-2;
3. xFormers if supported;
4. model-specific Ampere-compatible backend.

For each:

- same model/dtype/shapes;
- warmup complete;
- quality benchmark run;
- profiler confirms actual backend;
- no hidden layout-copy penalty.

Reject Sage/quantized attention initially because it belongs to the approximate branch.

Promote the fastest quality-passing exact backend.

---

## Stage 4 — compile the repeated transformer region

Start regional:

- repeated transformer blocks;
- denoiser forward;
- stable tensor shapes.

Avoid compiling wrappers that create unnecessary dynamism.

Measure:

- compilation time;
- first-call time;
- warm p50/p95;
- graph breaks;
- recompiles;
- peak reserved VRAM.

If compile OOMs, remove cold components from GPU first or use a less memory-hungry mode.

---

## Stage 5 — establish shape buckets

Collect real/requested shapes.

Build a table:

| Bucket | Width | Height | Frames | Batch | Frequency | Compile profile |
|---|---:|---:|---:|---:|---:|---|
| square_1k | 1024 | 1024 | — | 1 | | |
| portrait | 768 | 1344 | — | 1 | | |
| landscape | 1344 | 768 | — | 1 | | |
| video_short | ... | ... | 49 | 1 | | |

Specialize common buckets. Route rare shapes to a generic/eager fallback if compile-cache explosion is worse than the benefit.

---

## Stage 6 — try CUDA Graph replay

Only after the compiled shape is stable.

Measure:

- CPU launch gap reduction;
- steady-state step latency;
- static memory pool size;
- capture/replay success rate;
- number of graph variants.

If VRAM becomes too tight, graph replay may not be worth it on 24 GB.

---

## Stage 7 — inspect remaining pointwise/kernel overhead

Re-run Nsight Systems.

If many small kernels remain around each block, inspect with PyTorch profiler/Inductor logs.

Potential custom fusion targets:

- norm + modulation;
- gate + residual;
- activation chains;
- timestep embedding transforms;
- rotary/positional transforms.

Only write Triton/CUDA when a repeated kernel contributes material total time.

For custom kernels:

- compile for `sm_86`;
- run reference tensor tests;
- use Compute Sanitizer for CUDA code;
- profile with Nsight Compute;
- rerun full model quality.

---

## Stage 8 — optimize text encoder lifecycle

If text encoder is more than a few percent of end-to-end latency or consumes valuable VRAM:

### Exact options

- cache prompt embeddings;
- batch prompts through encoder;
- precompute embeddings offline;
- move/free encoder before denoising;
- put encoder on second GPU if it does not create queue/transfer problems.

If prompts are unique and encoder time is small, do not over-engineer it.

---

## Stage 9 — optimize VAE lifecycle

Measure VAE separately.

If memory is the problem:

- free denoiser temporaries before decode;
- use tiling/slicing/chunking;
- move VAE in only after denoiser completes;
- evaluate second-GPU VAE placement.

If latency is the problem:

- compile VAE where effective;
- test channels-last/conv optimizations if supported;
- reduce unnecessary dtype/layout conversions;
- batch decode if throughput is the goal.

Quality benchmark must include tile seams/color/detail if tiling changes numerical path.

---

## Stage 10 — solve VRAM without quantization

If the model still does not fit:

1. remove text encoder from VRAM during denoising;
2. remove VAE from VRAM until decode;
3. use group/model CPU offload;
4. tune group size to amortize PCIe transfer;
5. pin/reuse host staging buffers;
6. overlap next-group H2D copy with current-group compute;
7. use a second A5500 for sharding/context parallelism.

The central rule: **do not move the same weights across PCIe every denoising step if you can avoid it.**

If group offload necessarily moves transformer groups each step, calculate total transfer volume:

`bytes_per_request ≈ transferred_weight_bytes_per_step × number_of_steps`

This can be enormous.

---

## Stage 11 — profile the offload path

Nsight Systems should show:

- H2D copies;
- copy/compute overlap;
- GPU idle gaps;
- whether pinned memory helps;
- whether CPU thread scheduling delays prefetch.

Tune:

- group size;
- number of prefetch groups;
- copy stream;
- pinned pool size;
- CPU affinity/NUMA if relevant.

If the GPU waits on transfers, make groups coarser or use the second GPU.

---

## Stage 12 — add second A5500 if available

First verify topology:

```bash
nvidia-smi topo -m
```

If NVLink bridge is present, verify peer access.

Choose the parallelism based on the bottleneck.

### If the problem is fit

- model/tensor sharding;
- sequence/context sharding;
- component placement.

### If the problem is attention activation memory

- context/sequence parallelism.

### If the problem is throughput

- data parallel independent requests often scales most cleanly.

### If the problem is one-request latency

- context/tensor/pipeline strategies may help, but communication must be profiled.

Report GPU-seconds/output.

---

## Stage 13 — optimize communication

For two GPUs:

- place ranks on the NVLink pair;
- avoid unnecessary all-gathers;
- overlap collectives with independent compute where possible;
- shard along a dimension that minimizes communication volume;
- fuse communication-adjacent transforms when framework supports it.

Fine-grained tensor parallelism can create many synchronization points. Context parallelism may have larger but fewer structured exchanges.

Benchmark both if the model supports them.

---

## Stage 14 — exact cache inventory

Before approximate DiT caches, cache things that are truly invariant:

- tokenization;
- prompt embeddings;
- negative prompt embeddings;
- model conditioning features that do not depend on timestep;
- masks;
- position tables;
- shape metadata;
- compiled graphs;
- loaded LoRA merged weights if the serving configuration is static.

Track hit rate and memory footprint.

---

## Stage 15 — only now consider approximate work reduction

Quality-sensitive candidates:

- fewer denoising steps;
- alternate scheduler;
- CFG cutoff;
- hidden-state/timestep cache;
- distilled checkpoint;
- sparse attention;
- quantization.

Change one mechanism at a time.

Because the user constraint specifically forbids quantization accuracy loss, quantization cannot move into the production branch until it proves no regression.

---

## Stage 16 — quantization experiment protocol

If a capacity problem remains:

1. choose one target class (e.g., MLP weights);
2. keep boundaries/norms/reference-sensitive layers unquantized;
3. record exact storage and compute dtype;
4. benchmark quantized eager;
5. benchmark quantized compiled;
6. compare against exact offload and exact 2-GPU alternatives;
7. run the full zero-regression benchmark;
8. reject immediately on any required accuracy drop.

Do not add quantized attention in the same experiment.

---

## Stage 17 — sustained benchmark

After the final configuration is selected, run long enough to reach thermal steady state.

Log:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv -l 1
```

Run the complete workload distribution, not only the easiest bucket.

---

## Stage 18 — production hardening

Pin:

- exact software versions;
- compiled cache keys;
- model revisions;
- attention backend;
- shape buckets;
- offload strategy;
- benchmark suite revision;
- quantization state (`none` unless proven).

Expose metrics:

- queue time;
- service time;
- per-phase latency;
- VRAM allocated/reserved;
- compilation/cache misses;
- OOMs;
- GPU clocks/power/temperature;
- accepted-output rate;
- quality regression alarms.

---

## Stage 19 — stop optimizing when the bottleneck moves outside the model

If the GPU is already near a stable compute roof and end-to-end latency is dominated by:

- request queue;
- upload/download;
- image encoding;
- network;
- storage;

then deeper kernel work will not improve user latency materially.

The inference-engineering principle from the uploaded book applies: once inference time is small, move attention to the infrastructure layer.

---

## Stage 20 — final scorecard

A production-ready A5500 configuration should have a one-page scorecard:

```text
GPU: RTX A5500 x1/x2
NVLink: yes/no
Model revision: ...
Reference dtype: fp16/bf16
Quantization: none / exact config + parity evidence
Attention: SDPA/FA2/...
Compile: mode/profile
CUDA Graphs: yes/no
Shape bucket: ...
Offload: ...

Warm p50: ...
Warm p95: ...
Denoiser: ...
VAE: ...
Encoder: ...
Peak allocated VRAM: ...
Peak reserved VRAM: ...
PCIe bytes/request: ...
GPU-seconds/output: ...
Quality benchmark: PASS
Critical regressions: 0
```

That scorecard is the endpoint of the optimization ladder.
