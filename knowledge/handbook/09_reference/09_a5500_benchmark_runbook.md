# RTX A5500 DiT Benchmark and Profiling Runbook

Use this as the experiment procedure for every performance change.

## 1. Create a run directory

```bash
RUN="runs/$(date +%Y%m%d_%H%M%S)_a5500"
mkdir -p "$RUN"
```

Store:

- environment snapshot;
- model config hash;
- benchmark corpus hash;
- raw latency samples;
- memory samples;
- Nsight reports;
- quality metrics;
- generated outputs when policy permits;
- decision summary.

---

## 2. Environment capture

```bash
python -m torch.utils.collect_env > "$RUN/torch_env.txt"
nvidia-smi -q > "$RUN/nvidia_smi_q.txt"
nvidia-smi topo -m > "$RUN/topology.txt"
nvcc --version > "$RUN/nvcc.txt" 2>&1 || true
pip freeze > "$RUN/pip_freeze.txt"
```

Also capture git/model revisions.

---

## 3. Device verification

```python
import json, torch

props = torch.cuda.get_device_properties(0)
info = {
    "name": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "total_memory": props.total_memory,
    "multi_processor_count": props.multi_processor_count,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
}
print(json.dumps(info, indent=2))
```

If the capability is not what your custom build assumes, stop and fix the build.

---

## 4. Benchmark corpus

Create a versioned JSONL/YAML corpus.

Example:

```json
{"id":"square_detail_001","prompt":"...","seed":1,"width":1024,"height":1024,"steps":30,"guidance":3.5}
{"id":"portrait_text_002","prompt":"...","seed":2,"width":768,"height":1344,"steps":30,"guidance":3.5}
```

Video adds frame count/FPS/conditioning.

Never change the corpus during an A/B test.

---

## 5. Latency measurement harness

A robust structure:

```python
import statistics
import time
import torch

@torch.inference_mode()
def benchmark(run_once, warmup=5, iters=20):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    wall = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)

    wall_sorted = sorted(wall)
    def pct(p):
        i = min(len(wall_sorted)-1, round((len(wall_sorted)-1)*p))
        return wall_sorted[i]

    return {
        "mean_s": statistics.mean(wall),
        "median_s": statistics.median(wall),
        "p90_s": pct(0.90),
        "p95_s": pct(0.95),
        "min_s": min(wall),
        "max_s": max(wall),
        "samples_s": wall,
    }
```

For isolated GPU regions, CUDA events are preferable.

---

## 6. Peak memory

Before a measured request:

```python
torch.cuda.reset_peak_memory_stats()
run_once()
torch.cuda.synchronize()

print("allocated_peak", torch.cuda.max_memory_allocated())
print("reserved_peak", torch.cuda.max_memory_reserved())
```

Record both. Compilers/graphs can raise reserved memory even when active tensors do not.

Also record steady persistent residency after warmup.

---

## 7. Phase timing

Instrument:

- prompt encoding;
- latent setup;
- denoising loop total;
- one representative denoising step;
- VAE decode;
- postprocess.

Use NVTX ranges where possible:

```python
with torch.autograd.profiler.emit_nvtx():
    ...
```

Or framework-specific NVTX helpers.

These labels make Nsight Systems substantially easier to read.

---

## 8. Baseline Nsight Systems capture

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "$RUN/nsys_baseline" \
  python bench_one_case.py --case square_detail_001
```

Answer:

- what are the top GPU-time kernels?
- are there CPU launch gaps?
- are there H2D/D2H copies inside the loop?
- are denoising steps uniform?
- is attention dominant?
- is VAE significant?
- is there synchronization after every step?

Write the answers in `profile_summary.md`.

---

## 9. Nsight Compute capture

Choose a hot kernel; do not full-profile thousands of launches unless needed.

Starting command:

```bash
ncu \
  --set speedOfLight \
  --target-processes all \
  --force-overwrite \
  -o "$RUN/ncu_hot" \
  python bench_hot_region.py
```

Questions:

- compute or memory roof?
- SM throughput?
- memory throughput?
- Tensor Core instructions present?
- registers/thread?
- shared memory/block?
- occupancy limiting factor?
- dominant stall reason?

Then choose the next optimization from evidence.

---

## 10. Attention comparison sheet

For each shape bucket:

```text
backend: SDPA / FA2 / xFormers
model block: ...
B,H,N,D: ...
dtype: ...
compiled: yes/no
median attention/block: ...
full denoiser: ...
end-to-end: ...
peak VRAM: ...
quality: PASS/FAIL
fallback observed: yes/no
```

Do not accept a backend if the profiler shows a fallback math kernel.

---

## 11. Compile comparison sheet

```text
mode: ...
regional/full: ...
shape bucket: ...
compile time: ...
first call: ...
warm p50: ...
warm p95: ...
graph breaks: ...
recompiles: ...
peak reserved VRAM: ...
quality: PASS/FAIL
```

If a compiled configuration requires more memory than the worker can safely reserve, reject it even if p50 is lower.

---

## 12. Offload comparison sheet

```text
offload type: component/model/group
host memory: ...
pinned memory: ...
H2D bytes/request: ...
D2H bytes/request: ...
copy time: ...
overlap percentage: ...
GPU idle due transfers: ...
peak VRAM: ...
p50/p95: ...
quality: PASS/FAIL
```

Use realistic repeated denoising. An offload benchmark with one transformer call can dramatically understate request transfer cost.

---

## 13. Two-GPU comparison

Capture:

```bash
nvidia-smi topo -m > "$RUN/topology.txt"
```

Record:

- NVLink bridge present?;
- P2P available?;
- parallelism strategy;
- per-GPU VRAM;
- communication time/step;
- one-GPU latency;
- two-GPU latency;
- speedup;
- parallel efficiency;
- GPU-seconds/output.

Calculation:

```python
speedup = t1 / t2
efficiency = t1 / (2 * t2)
gpu_seconds_1 = t1
gpu_seconds_2 = 2 * t2
```

---

## 14. Quantization test — hard no-regression gate

Do not start this test until the exact baseline is already optimized.

Store exact quantization metadata:

```yaml
quantizer: ...
version: ...
weights: ...
activations: ...
compute_dtype: ...
group_size: ...
excluded_modules: [...]
calibration_hash: ...
```

Run the identical quality corpus.

### Strict acceptance code concept

```python
for metric in REQUIRED_METRICS:
    assert candidate[metric] >= baseline[metric], metric

assert len(candidate_critical_regressions) == 0
```

For noisy metrics, follow the predeclared statistical noise policy from the benchmark chapter. Do not invent a tolerance after seeing the result.

If quality fails, stop. Do not keep the quantized candidate because it is “only slightly worse.”

---

## 15. Paired output regression table

Create:

| Case | Baseline pass | Candidate pass | Regression? | Notes |
|---|---:|---:|---:|---|
| ... | 1 | 1 | 0 | |
| ... | 1 | 0 | **1** | reject if critical |

This is particularly important for typography/editing/video.

---

## 16. Thermal stability

Run long enough to stabilize temperature.

In parallel:

```bash
nvidia-smi \
  --query-gpu=timestamp,temperature.gpu,pstate,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > "$RUN/telemetry.csv"
```

If candidate clocks are materially different from baseline, annotate the result.

---

## 17. Experiment CSV schema

Recommended columns:

```text
run_id
model_revision
case_id
gpu_name
gpu_count
nvlink
pytorch
cuda
attention_backend
compile_mode
quantization
width
height
frames
steps
guidance
latency_s
denoiser_s
vae_s
encoder_s
peak_allocated_bytes
peak_reserved_bytes
quality_pass
quality_score_1
quality_score_2
critical_regressions
sm_clock_avg
power_avg_w
temperature_avg_c
```

Store raw samples, not only aggregates.

---

## 18. Result acceptance checklist

Before accepting any speedup:

- [ ] exact GPU and software recorded;
- [ ] same shape corpus;
- [ ] same prompt/seed corpus;
- [ ] warmup complete;
- [ ] steady state measured;
- [ ] p50 and p95 reported;
- [ ] VRAM allocated and reserved reported;
- [ ] quality suite passed;
- [ ] no critical per-case regression;
- [ ] no thermal/clock confound;
- [ ] profiler supports the claimed mechanism;
- [ ] if multi-GPU, GPU-seconds reported;
- [ ] if quantized, strict no-regression gate passed;
- [ ] result reproduced after process restart.

---

## 19. Recommended experiment naming

Use names that encode the mechanism, not the hoped-for result.

Good:

```text
fa2_vs_sdpa_1024_fp16
compile_regional_reduceoverhead_1024
vae_tile_1024_256tile
model_offload_group4_pinned
cp2_nvlink_49frames
int8_weightonly_mlp_g128_qualitygate
```

Bad:

```text
FAST_FINAL
super_optimized
best_config_v7_really_final
```

---

## 20. Decision summary

Every experiment ends with:

```markdown
## Decision
ACCEPT / REJECT / NEEDS_MORE_DATA

## Mechanism observed
...

## Performance delta
...

## Quality result
...

## Memory delta
...

## Production risk
...

## Next experiment
...
```

This creates a real optimization history instead of a pile of benchmark screenshots.
