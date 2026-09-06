# Benchmarking and No-Regression Quality Gates

The benchmark suite is the optimization contract. Without it, “faster” can mean a model that silently produces worse images, less coherent video, more failed edits, or more dropped requests.

This chapter adopts the measurement discipline from both the uploaded *Inference Engineering* book and Wafer AI’s GPU performance resource list. Wafer’s rule is especially useful: a performance number is incomplete without **hardware/software versions, workload shape or distribution, precision/algorithm, baseline, and correctness method**. For DiTs, add prompt/seed data, denoising schedule, and output-quality evaluation.

The project-specific requirement is stricter still:

> **Quantization may not be used if it reduces required benchmark accuracy.**

---

## 1. Benchmark layers

Use four nested benchmark layers.

### Layer A — kernel microbenchmarks

Questions:

- Which attention backend is fastest for this exact `(batch, heads, sequence, head_dim, dtype)`?
- Which GEMM algorithm wins for the repeated projection shapes?
- Does a fused modulation kernel beat the unfused sequence?

Metrics:

- median kernel latency;
- p95 latency when launch noise matters;
- achieved bandwidth;
- achieved Tensor Core throughput;
- register/shared-memory pressure;
- occupancy;
- correctness versus reference tensor.

A kernel benchmark is not a model benchmark.

### Layer B — transformer/denoiser benchmark

Run the hot module with realistic latent/conditioning shapes.

Metrics:

- forward latency;
- peak VRAM;
- kernel launch count;
- graph breaks/recompiles;
- attention/MLP share;
- output numerical difference to reference.

### Layer C — full pipeline benchmark

Include:

- tokenization;
- text/vision encoders;
- latent setup;
- all denoising steps;
- scheduler work;
- VAE decode;
- safety/postprocessing if production includes them;
- CPU↔GPU transfers.

Metrics:

- warm end-to-end latency;
- denoising-only latency;
- VAE latency;
- encoder latency;
- peak allocated/reserved VRAM;
- host RAM;
- total bytes transferred if offloading;
- GPU utilization.

### Layer D — serving/workload benchmark

Include concurrency and queueing.

Metrics:

- p50/p90/p95/p99 end-to-end latency;
- service time versus queue time;
- throughput;
- GPU-seconds/output;
- memory OOM rate;
- benchmark-quality acceptance rate;
- cold-start latency;
- worker ready time;
- cache/compile hit rate.

---

## 2. Freeze the baseline before optimizing

Record the baseline as an immutable experiment manifest.

Minimum fields:

```yaml
model:
  repo_or_path: ...
  revision: ...
  transformer_revision: ...
  vae_revision: ...
  text_encoder_revision: ...

inference:
  dtype: fp16_or_bf16
  scheduler: ...
  steps: ...
  guidance_scale: ...
  height: ...
  width: ...
  frames: ...
  batch_size: ...
  attention_backend: ...
  compile: false
  cache: false
  quantization: none

hardware:
  gpu: NVIDIA RTX A5500
  count: 1
  nvlink: false
  driver: ...
  power_limit: ...

software:
  os: ...
  python: ...
  pytorch: ...
  cuda_runtime: ...
  diffusers: ...
  transformers: ...
  xformers: ...
  flash_attn: ...

benchmark:
  suite_revision: ...
  prompts_hash: ...
  seeds: [...]
  warmup_runs: ...
  measured_runs: ...
```

If the baseline changes, start a new experiment series.

---

## 3. Hardware-state capture

At the beginning and end of a run:

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv
nvidia-smi topo -m
```

Why:

- boost clocks can differ;
- thermal state can drift;
- power limits can be changed;
- a second GPU may use a different link topology;
- another process may consume VRAM.

Do not accept 2–3% performance claims from runs with uncontrolled clock/temperature differences.

---

## 4. Timing correctly with CUDA

CPU `time.time()` around an asynchronous CUDA call is wrong unless you synchronize.

For a hot region, use CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(warmup):
    run_once()
torch.cuda.synchronize()

times_ms = []
for _ in range(iters):
    start.record()
    run_once()
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))
```

For end-to-end user latency, use a wall clock and synchronize only at the actual completion boundary.

Do not place extra `torch.cuda.synchronize()` calls inside the production hot path just to make the benchmark easy; they can destroy overlap.

---

## 5. Warmup policy

Warmup needs to cover:

- CUDA context creation;
- memory allocator pool creation;
- attention-kernel autotuning;
- `torch.compile` compilation;
- CUDA Graph capture;
- lazy checkpoint movement;
- JIT extension load;
- cuBLAS/cuDNN algorithm selection.

Track compile/setup separately rather than hiding it.

Suggested reporting:

- `startup_to_ready_s`
- `first_request_s`
- `first_compiled_request_s`
- `steady_state_p50_s`
- `steady_state_p95_s`

---

## 6. Fixed paired quality corpus

For every optimization that can change arithmetic or outputs, use a fixed paired corpus.

Each test case should define:

```yaml
id: typography_0042
prompt: "..."
negative_prompt: "..."
seed: 123456
height: 1024
width: 1024
frames: null
steps: 30
guidance_scale: 3.5
condition_inputs: ...
required_checks:
  - text_accuracy
  - prompt_adherence
  - artifact_rate
```

The baseline and candidate consume the same case.

For video, include cases that specifically stress:

- temporal identity;
- camera motion;
- fast object motion;
- occlusion;
- repeated patterns;
- long-range frame consistency;
- small text/signage;
- hands/faces if relevant.

---

## 7. The strict quantization gate

A quantized candidate is **rejected** if it reduces required benchmark accuracy.

Use one of two policies depending on how literal the project requirement is.

### Policy Q0 — zero observed regression

Use when “no accuracy reduction” means exactly that.

Acceptance conditions:

1. every required aggregate accuracy/quality score is `candidate >= baseline`;
2. every deterministic pass/fail test that passes baseline also passes candidate;
3. no critical-case regression appears;
4. no new artifact class appears in human review;
5. performance/capacity benefit is independently demonstrated.

This is conservative and may reject harmless noise, but it matches a literal zero-drop requirement.

### Policy QN — zero regression outside measurement noise

Use only if the team accepts statistical treatment of noisy metrics.

Define a **measurement noise interval before testing candidates**, not after seeing results.

Example method:

- repeat baseline N times on identical case set;
- estimate metric standard error or paired bootstrap confidence interval;
- define the non-inferiority margin as zero model degradation, with only baseline measurement uncertainty tolerated;
- candidate passes only if its lower confidence bound does not indicate a decrease.

Do not set a convenient positive quality-loss margin after the fact.

---

## 8. Per-case monotonicity matters

Average metrics can hide failure mode swaps.

Suppose baseline passes 98/100 OCR prompts. Candidate also passes 98/100, but fails two cases baseline passed and fixes two different cases. Aggregate score is equal, but the behavior changed.

For critical benchmarks, record:

- baseline pass / candidate pass;
- baseline pass / candidate fail;
- baseline fail / candidate pass;
- baseline fail / candidate fail.

A strict deployment policy can require zero `baseline pass -> candidate fail` transitions on critical cases.

---

## 9. Quality metrics by task

The source material does not prescribe one universal image metric, and neither should this handbook. Use task-specific benchmarks.

### Text-to-image

Potential dimensions:

- prompt adherence;
- object count;
- attribute binding;
- spatial relation;
- typography/OCR;
- aesthetic preference;
- artifact rejection.

### Image editing / control

- source identity preservation;
- instruction compliance;
- unchanged-region preservation;
- geometry control accuracy;
- mask boundary quality.

### Video

- prompt adherence;
- temporal consistency;
- motion correctness;
- subject identity;
- flicker/artifact rate;
- frame-to-frame geometry consistency;
- camera instruction compliance.

### 3D/world models

- multiview consistency;
- geometry/pose error;
- temporal/world-state consistency;
- task-specific simulator metrics.

Use the benchmark that corresponds to the product requirement, not the easiest metric to improve.

---

## 10. Performance acceptance rule

Quality passing is necessary but not sufficient. The candidate should produce a meaningful performance benefit.

For each candidate report:

```text
metric                baseline      candidate      delta
warm p50 latency      ...           ...            ...
warm p95 latency      ...           ...            ...
denoiser latency      ...           ...            ...
VAE latency           ...           ...            ...
peak allocated VRAM   ...           ...            ...
peak reserved VRAM    ...           ...            ...
GPU-seconds/output    ...           ...            ...
quality score A       ...           ...            ...
quality score B       ...           ...            ...
critical regressions  0             0              pass/fail
```

State whether the win is:

- latency;
- throughput;
- memory capacity;
- cold start;
- multi-GPU scaling;
- some combination.

---

## 11. Benchmark quantization separately from memory fit

A common mistake is to compare:

- baseline: OOM;
- quantized: fits and runs.

That proves a capacity benefit, not a speedup or quality equivalence.

Use two comparisons:

### Capacity comparison

`unquantized single-GPU OOM` vs `candidate fits`.

### Fair performance comparison

Compare against another configuration that also fits without quality compromise, such as:

- unquantized CPU/group offload;
- two-GPU sharding;
- text-encoder offload + unquantized transformer;
- VAE tiling + unquantized transformer.

Then ask which fitting configuration has the best latency/GPU-seconds while preserving benchmark accuracy.

---

## 12. Profiling protocol

### Nsight Systems first

Capture a representative request:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o dit_a5500_timeline \
  python benchmark.py
```

Look for:

- CPU launch gaps;
- repeated CUDA synchronizations;
- long memory copies;
- gaps between denoising steps;
- VAE/text encoder overlap or serialization;
- NCCL/peer-copy duration;
- graph capture/replay behavior.

### Nsight Compute on selected kernels

Do not profile every kernel with every counter. Filter hot kernels first.

Typical starting point:

```bash
ncu \
  --set speedOfLight \
  --target-processes all \
  -o dit_a5500_ncu \
  python benchmark_hot_kernel.py
```

Then inspect:

- roofline position;
- SM throughput;
- memory throughput;
- Tensor Core instruction usage;
- occupancy limiters;
- registers/thread;
- shared memory/block;
- warp stall reasons.

---

## 13. Compiler benchmark rules

For `torch.compile` candidates record:

- compile time;
- first compiled call;
- steady-state latency;
- number of graphs;
- graph breaks;
- recompile count;
- peak reserved memory;
- shape bucket.

Useful debug environment variables vary by PyTorch release, but current builds commonly support `TORCH_LOGS` categories for graph breaks and recompiles. Pin the framework version in the result.

Do not compare eager first request to compiled steady state without showing both.

---

## 14. Offload benchmark rules

Offload changes the problem from GPU compute to a compute+transfer pipeline.

Record:

- H2D bytes/request;
- D2H bytes/request;
- transfer duration;
- whether transfers overlap compute;
- pinned host memory;
- host RAM peak;
- PCIe link state if available;
- p95 latency under repeated requests.

A configuration that fits but reloads all transformer weights every denoising step is usually pathological.

Group/layer offload must be designed around reuse across the repeated denoising loop.

---

## 15. Multi-GPU benchmark rules

For two A5500s:

- record whether an NVLink bridge is physically present;
- capture `nvidia-smi topo -m`;
- verify peer access;
- record per-rank memory use;
- record communication share of step time;
- report scaling efficiency;
- report GPU-seconds/output.

Scaling efficiency:

`efficiency_N = T1 / (N * TN)`

where `T1` is one-GPU latency and `TN` is N-GPU latency.

A two-GPU result can be useful even with low efficiency if it solves a hard latency/fit requirement, but the cost should be explicit.

---

## 16. Thermal stability benchmark

For long video or large image batches, log:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv -l 1
```

Compare steady-state windows rather than the first few seconds.

If a candidate increases power and causes lower sustained clocks, the microbenchmark may mislead.

---

## 17. Experimental result template

```markdown
# Experiment: <name>

## Hypothesis
<mechanism expected to improve>

## Environment
- GPU: RTX A5500 x1/x2
- NVLink: yes/no
- driver:
- CUDA:
- PyTorch:
- Diffusers:
- attention package:

## Workload
- model/revision:
- dtype:
- batch:
- width/height:
- frames:
- steps:
- guidance:
- prompt corpus revision:
- seeds:

## Baseline
<configuration>

## Candidate
<single mechanism changed>

## Correctness/quality method
<benchmarks and acceptance rule>

## Performance
<table>

## Quality
<table + per-case regressions>

## Profiler evidence
<what moved and why>

## Decision
ACCEPT / REJECT

## Reason
<one paragraph>
```

---

## 18. Final acceptance matrix

| Candidate class | Quality gate | Performance proof | Extra requirement |
|---|---|---|---|
| Exact attention backend | full benchmark | end-to-end + denoiser | no fallback |
| `torch.compile` | full benchmark | warm + compile cost | graph/recompile count |
| CUDA Graphs | full benchmark | steady-state + memory | static-shape safety |
| Offload | full benchmark | latency + transfer | host RAM/pinned memory |
| Multi-GPU | full benchmark | latency + GPU-s | topology/comm profile |
| Quantization | **strict no-regression** | fit/latency benefit | exact quant config |
| Approximate cache | strict no-regression | step/latency benefit | hit/skip behavior |
| Step/CFG reduction | strict no-regression | forward-count benefit | same prompt corpus |

No candidate skips the benchmark because it is “known to be safe.”
