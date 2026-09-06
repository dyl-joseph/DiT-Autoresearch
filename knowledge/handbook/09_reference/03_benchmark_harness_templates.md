# Benchmark Harness Templates — RTX A5500 / No-Regression Edition

This file contains reusable harness patterns for the handbook's target environment: one or two NVIDIA RTX A5500 GPUs, Ampere-native FP16/BF16 execution, and a strict rule that an optimization is deployable only when the required benchmark suite does not regress.

The objective is not to produce a single flattering latency number. The objective is to produce evidence that lets you answer four questions:

1. Is the candidate actually faster end-to-end?
2. Does it use less peak/resident memory or reduce transfer pressure?
3. Is the gain stable across the shapes and prompts that matter?
4. Does it preserve benchmark accuracy/quality?

A candidate that fails question 4 is rejected even when questions 1-3 look excellent.

---

## 1. Benchmark record schema

Every result should be self-describing. At minimum, save:

```json
{
  "run_id": "2026-09-05_flux_a5500_sdpa_compile_001",
  "model": "org/model",
  "revision": "git-or-hf-revision",
  "pipeline": "text-to-image",
  "gpu_name": "NVIDIA RTX A5500",
  "gpu_count": 1,
  "device_capability": "record-at-runtime",
  "driver": "...",
  "cuda_runtime": "...",
  "torch": "...",
  "diffusers": "...",
  "xformers": null,
  "flash_attn": null,
  "dtype": "float16",
  "weight_storage": "float16",
  "quantization": "none",
  "attention_backend": "sdpa",
  "compile_mode": "regional",
  "cuda_graph": false,
  "height": 1024,
  "width": 1024,
  "frames": null,
  "steps": 28,
  "guidance": 3.5,
  "batch": 1,
  "prompt_length_tokens": 77,
  "cache": "off",
  "warmup_runs": 3,
  "measured_runs": 20,
  "quality_suite": "dit-prod-v1",
  "quality_gate": "no-required-benchmark-regression"
}
```

Do not write `gpu: ampere` and stop there. Kernel behavior can depend on the precise GPU, CUDA, PyTorch, kernel package, shapes, and dtype.

---

## 2. Environment snapshot

Capture this once per benchmark session:

```bash
mkdir -p bench_artifacts
python -V > bench_artifacts/environment.txt
python -m pip freeze >> bench_artifacts/environment.txt
nvidia-smi -q >> bench_artifacts/environment.txt
nvidia-smi topo -m >> bench_artifacts/environment.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv >> bench_artifacts/environment.txt
```

And from Python:

```python
import json, torch

snapshot = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
}
print(json.dumps(snapshot, indent=2))
```

Do not hard-code `sm_86` without checking the machine. Build/install custom kernels for the capability actually returned.

---

## 3. Warm end-to-end latency harness

Use wall-clock time around the full pipeline, with explicit synchronization before and after each measured request.

```python
from __future__ import annotations
import gc
import statistics
import time
import torch


def percentile(xs, p):
    xs = sorted(xs)
    idx = round((len(xs) - 1) * p)
    return xs[idx]


def benchmark_pipeline(pipe, call_kwargs, warmup=3, runs=20):
    # Warm compilation, allocator, lazy kernel loading, and caches.
    for _ in range(warmup):
        _ = pipe(**call_kwargs)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(**call_kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.pstdev(times),
        "p50_s": percentile(times, 0.50),
        "p90_s": percentile(times, 0.90),
        "p95_s": percentile(times, 0.95),
        "p99_s": percentile(times, 0.99),
        "min_s": min(times),
        "max_s": max(times),
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
```

Use enough runs for stable percentiles. For long video generations, fewer runs may be practical; then report the sample count and avoid pretending p99 is meaningful from five observations.

---

## 4. Cold-start and first-request harness

A5500 deployments may compile, load large checkpoints, build CUDA kernels, or transfer weights at startup. Record each phase separately.

```python
import time

marks = {}
def mark(name):
    marks[name] = time.perf_counter()

mark("process_ready")
# import heavy modules
mark("imports_done")
# create/load pipeline on CPU
mark("weights_loaded_cpu")
# move target components to GPU
mark("gpu_residency_ready")
# optional torch.compile invocation
mark("compile_requested")
# first warm request
mark("first_request_done")
# second request after caches/compile settle
mark("steady_state_ready")

for a, b in zip(list(marks), list(marks)[1:]):
    print(f"{a}->{b}: {marks[b]-marks[a]:.3f}s")
```

Keep cold-start metrics separate from warm latency. Optimizations that move cost from every request to startup are often good, but only if the deployment lifecycle can tolerate that startup cost.

---

## 5. Denoiser-only timing with CUDA events

CUDA events are useful for a specific GPU-resident phase. They are not a substitute for end-to-end timing.

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()
# transformer / denoiser region
end.record()
end.synchronize()
print("denoiser_ms", start.elapsed_time(end))
```

When possible, measure:

- prompt encoding;
- each denoising step or block class;
- VAE decode;
- H2D/D2H transfer phases;
- total request.

This prevents a 15% denoiser win from being mislabeled a 15% product-latency win.

---

## 6. Per-step recorder

Denoising is repetitive, which makes per-step traces extremely useful.

Suggested CSV:

```text
request_id,step,timestep,transformer_ms,attention_ms,cache_hit,dtype,allocated_mib,reserved_mib
r0001,0,1000,112.4,46.2,false,fp16,18122,19370
r0001,1,964,107.9,43.7,false,fp16,18122,19370
...
```

For this A5500 handbook, do **not** put an FP8 late-step policy into the default harness. If you later test any reduced-precision policy, use a separate candidate name and route it through the quality gate.

---

## 7. Exact attention backend sweep

The first attention comparison on A5500 should stay exact:

```python
candidates = [
    "pytorch_sdpa",
    "flash_attention_2",
    "xformers_memory_efficient",
]
```

The concrete API names vary by Diffusers/PyTorch version, so query installed backend support rather than copying a stale string.

For each candidate, record:

- end-to-end latency;
- transformer-only latency;
- peak allocated/reserved VRAM;
- whether the intended kernel was actually selected;
- graph breaks/recompiles;
- output parity/correctness.

A backend is not accepted merely because its standalone attention microbenchmark is faster.

Approximate/quantized attention (for example SageAttention variants) belongs in a **separate experimental sweep** and only after the exact winner is known.

---

## 8. Shape matrix

DiT kernels can change winners by resolution, aspect ratio, frame count, and prompt length. Define the production matrix explicitly.

Example image matrix:

```python
image_shapes = [
    (512, 512),
    (768, 768),
    (1024, 1024),
    (768, 1344),
    (1344, 768),
]
```

Example video matrix:

```python
video_shapes = [
    # (height, width, frames)
    (480, 832, 49),
    (720, 1280, 49),
    (720, 1280, 81),
]
```

Do not benchmark only the most favorable square size if users generate portrait/landscape or longer clips.

---

## 9. Compilation experiment matrix

Do not combine all compiler changes into one opaque candidate. Sweep incrementally:

```text
E0 eager + SDPA
E1 eager + FA2
C1 torch.compile(transformer) + exact attention winner
C2 regional compile + exact attention winner
C3 static shape bucket + regional compile
C4 C3 + CUDA Graph capture (only if stable)
```

For each configuration, measure:

- first-call compile time;
- first request latency;
- warm p50/p95;
- graph breaks;
- recompilation count;
- peak VRAM;
- generated-output quality.

Compilation should be evaluated per shape bucket. A configuration that recompiles continuously under real aspect ratios can lose despite an excellent single-shape benchmark.

---

## 10. Peak-memory phase isolation

```python
import gc, torch


def peak_for(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out = fn()
    torch.cuda.synchronize()
    return {
        "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }, out

prompt_peak, embeds = peak_for(encode_prompt)
dit_peak, latents = peak_for(lambda: denoise(embeds))
vae_peak, image = peak_for(lambda: decode(latents))
```

Phase isolation is diagnostic. Confirm any memory fix in the actual whole pipeline because component overlap and persistent buffers change the real peak.

---

## 11. Transfer accounting for offload experiments

A 24 GB card often forces pipeline-placement work. Memory saved is not enough; track transfers.

At minimum, record:

```text
candidate,peak_vram_gib,host_ram_gib,h2d_gib_per_request,d2h_gib_per_request,e2e_p50_s,e2e_p95_s
resident_exact,...
group_offload_exact,...
component_offload_exact,...
two_gpu_exact,...
```

Use Nsight Systems to verify where H2D/D2H copies occur and whether they overlap useful GPU work. Repeatedly moving the hot DiT over PCIe is usually a poor latency strategy.

---

## 12. Two-A5500 scaling harness

Record both speedup and **GPU-seconds/output**.

| GPUs | Topology | Method | Latency s | Speedup | Efficiency | GPU-s/output | Peak/rank GiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | n/a | exact baseline | | 1.00 | 1.00 | | |
| 2 | NVLink or PCIe | CFG parallel | | | | | |
| 2 | NVLink or PCIe | component shard | | | | | |
| 2 | NVLink or PCIe | CP/TP candidate | | | | | |

Definitions:

```text
speedup(P)    = T_1 / T_P
efficiency(P) = speedup(P) / P
GPU_seconds   = P * T_P
```

Always save `nvidia-smi topo -m` with the result. “2× A5500” is not a topology description.

---

## 13. Strict quality suite manifest

Define the benchmark before the optimization.

```yaml
suite: dit-prod-v4
baseline:
  model_revision: "..."
  dtype: fp16
  quantization: none
  attention_backend: sdpa
  scheduler: "..."
  steps: 28
  guidance: 3.5
prompts:
  - id: typography_01
    prompt: "..."
    tags: [text, signage, spelling]
  - id: face_01
    prompt: "..."
    tags: [face, skin, eyes]
  - id: hands_01
    prompt: "..."
    tags: [hands, anatomy]
  - id: composition_01
    prompt: "..."
    tags: [counting, spatial_relation]
  - id: edit_boundary_01
    prompt: "..."
    tags: [editing, locality]
seeds: [17, 23, 41, 97, 131, 191]
acceptance:
  required_scalar_metrics:
    ocr_accuracy_drop_max: 0.0
    benchmark_score_drop_max: 0.0
  paired_cases:
    baseline_pass_to_candidate_fail_max: 0
```

For stochastic metrics with nonzero measurement noise, document the statistical rule. “Looks close” is not a gate.

---

## 14. Quantization candidate protocol

Quantization is **not** part of the baseline optimization sweep. If it is ever necessary for capacity or worth testing experimentally, use this sequence:

```text
Q0 exact FP16/BF16 strongest baseline
Q1 candidate compressed weights, eager
Q2 candidate compressed weights + best exact attention backend
Q3 candidate compressed weights + compile, if supported
```

For every Q candidate:

1. run the full required benchmark suite;
2. compare paired outputs/cases against Q0;
3. reject immediately if a required benchmark regresses;
4. only if quality passes, compare performance and memory;
5. retain it only if it provides a meaningful end-to-end or capacity advantage over exact alternatives.

Do not hide a quality drop inside a weighted aggregate. A required benchmark is required.

---

## 15. Paired-output archive

For image/video changes, save baseline and candidate outputs using identical seeds and inputs:

```text
outputs/
  typography_01/
    seed_017/
      baseline.png
      candidate.png
      metadata.json
  motion_01/
    seed_023/
      baseline.mp4
      candidate.mp4
      metadata.json
```

This makes regressions inspectable and allows later re-scoring when the benchmark improves.

---

## 16. Correctness test for custom kernels

Before performance testing a custom Triton/CUDA/CUTLASS kernel:

```python
ref = reference_op(inputs)
out = candidate_op(inputs)

torch.testing.assert_close(
    out,
    ref,
    rtol=your_rtol,
    atol=your_atol,
)
```

Then test adversarial shapes and values:

- smallest and largest production sequence lengths;
- odd/non-power-of-two dimensions where supported;
- masks/padding;
- large-magnitude values and outliers;
- contiguous and expected strided layouts;
- FP16 and BF16 separately if both are supported.

Use Compute Sanitizer for custom CUDA memory/race/synchronization issues. Numerical closeness of an operator is **not** a substitute for the model-level quality suite.

---

## 17. Nsight Systems capture recipe

Use a short representative request after warmup:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o nsys_dit_a5500 \
  python bench_one_request.py
```

Look for:

- CPU gaps between kernels;
- excessive tiny kernels;
- H2D/D2H copies inside every denoising step;
- synchronization points;
- repeated allocator work;
- multi-GPU collectives and copy engines;
- VAE/text-encoder phases that have become dominant.

Annotate phases with NVTX so the timeline is readable.

---

## 18. Nsight Compute hot-kernel recipe

Do not profile the entire application with every metric first. Identify hot kernels with Nsight Systems, then target a small kernel set.

Typical command pattern:

```bash
ncu \
  --set roofline \
  --kernel-name-base demangled \
  --launch-skip 20 \
  --launch-count 10 \
  -o ncu_hot_kernel \
  python bench_kernel_region.py
```

Depending on Nsight Compute version, section/set names may differ. Use `ncu --list-sets` and `ncu --list-sections` on the installed version.

Interpret the result against the A5500 rather than an H100/B200 roofline.

---

## 19. Thermal and clock stability

Workstation GPUs can benchmark differently when cold versus after sustained generation. During long runs, log:

```bash
nvidia-smi dmon -s pucvmet -d 1
```

Record at least:

- power;
- temperature;
- SM/memory clocks;
- utilization;
- memory usage.

If a candidate appears 5% slower only after several minutes, check thermal/power behavior before attributing the change to kernels.

---

## 20. Acceptance table

Use one final table per experiment:

| Candidate | Quality gate | Warm p50 | Warm p95 | Peak VRAM | GPU-s/output | Compile/startup | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| baseline exact | PASS | | | | | | keep reference |
| FA2 exact | PASS/FAIL | | | | | | |
| compile exact | PASS/FAIL | | | | | | |
| offload exact | PASS/FAIL | | | | | | |
| 2-GPU exact | PASS/FAIL | | | | | | |
| quant candidate | PASS/FAIL | | | | | | reject if any benchmark regresses |

The decision field should explain **why**, not just “faster.”

---

## 21. Benchmark anti-patterns

Reject results produced by these methods:

- timing asynchronous CUDA calls without synchronization;
- comparing a cold baseline to a warm candidate;
- changing scheduler/steps while claiming a kernel-only speedup;
- using one prompt/seed to approve an approximate method;
- reporting a microkernel win as an end-to-end win;
- comparing a 4-bit model that fits with an exact model that was deliberately configured to OOM instead of testing exact offload/sharding;
- omitting precision or GPU model;
- measuring only mean latency for a serving workload;
- ignoring compilation and cache warmup;
- assuming an H100/B200 result transfers numerically to A5500;
- declaring quantization “quality preserving” without running the required product benchmarks.

---

## 22. Minimum evidence package for a claimed optimization

For this project, do not merge an optimization claim into the main recommendation set without:

1. environment snapshot;
2. model revision and pipeline settings;
3. workload/shape matrix;
4. baseline definition;
5. candidate definition;
6. warm and (if relevant) cold latency;
7. peak VRAM;
8. profiler evidence supporting the proposed bottleneck mechanism;
9. required quality-suite results;
10. paired output archive for approximate/model-changing methods;
11. rerun variance or multiple repeated trials;
12. exact command/config needed to reproduce the result.

That standard is deliberately strict. It prevents the handbook from becoming a list of context-free “X is 2× faster” claims that collapse when moved to the A5500 target.
