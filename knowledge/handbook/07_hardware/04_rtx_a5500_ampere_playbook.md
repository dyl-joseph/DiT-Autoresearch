# NVIDIA RTX A5500 / Ampere Playbook for Diffusion Transformer Inference

This is the hardware-specific tuning document for the repository. Generic DiT advice should be interpreted through this file.

## 1. Hardware facts that matter

NVIDIA’s RTX A5500 datasheet lists:

| Property | RTX A5500 |
|---|---:|
| Architecture | Ampere |
| VRAM | 24 GB GDDR6 ECC |
| Memory bandwidth | 768 GB/s |
| Memory interface | 384-bit |
| CUDA cores | 10,240 |
| Tensor Cores | 320, 3rd generation |
| Advertised FP32 | 34.1 TFLOPS |
| PCIe | Gen4 x16 |
| Board power | 230 W |
| NVLink | 2-card low-profile bridge |
| NVLink bandwidth | 112.5 GB/s bidirectional |

Primary source: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/proviz-nvidia-rtx-a5500-datasheet-2130578-r3-us-web.pdf

Treat advertised peaks as orientation only. The Wafer GPU-performance repository explicitly warns that vendor peak numbers are not performance measurements. The workload, precision, shape, implementation, and correctness method have to accompany any claim.

---

## 2. Confirm the real device

Before tuning:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv
nvidia-smi topo -m
```

From PyTorch:

```python
import torch

idx = 0
print(torch.cuda.get_device_name(idx))
print("capability", torch.cuda.get_device_capability(idx))
print(torch.cuda.get_device_properties(idx))
```

The target build architecture for this workstation Ampere class is normally `sm_86`; verify it at runtime rather than relying on a table.

---

## 3. cc 8.6 limits and why they matter

NVIDIA’s current Ampere tuning guide documents for compute capability 8.6:

- architectural maximum of 48 concurrent warps per SM;
- up to 16 thread blocks per SM;
- 64K 32-bit registers per SM;
- up to 255 registers per thread;
- 100 KB shared-memory capacity per SM;
- up to 99 KB shared memory per block with the required dynamic-memory opt-in;
- 128 KB combined L1/texture/shared-memory capacity;
- hardware-accelerated asynchronous global-to-shared copies;
- Tensor Core FP16/BF16 and TF32 support;
- recommendation to compile explicitly for cc 8.6.

Source: https://docs.nvidia.com/cuda/ampere-tuning-guide/

### Implication: register pressure can matter quickly

A fused transformer kernel can use many registers. If register pressure reduces active warps too far, latency hiding suffers. But do not optimize occupancy as an end in itself. A lower-occupancy kernel can still be faster if it saves global-memory traffic and instructions.

### Implication: shared-memory tiling has a hard budget

Attention/GEMM/custom fused kernels that use large shared-memory tiles must fit the cc 8.6 budget. A kernel configuration copied from A100/H100 can fail to launch, trigger a different schedule, or reduce residency.

### Implication: compile for `sm_86`

Generic binaries can work while missing architecture-specific code generation.

---

## 4. Precision map

### Recommended primary modes

- FP16
- BF16

Use whichever is supported by the model and passes the benchmark suite.

### Conditional mode

- TF32 for FP32 matrix math, only if the quality benchmark permits the numerical change.

### Not native target modes

- FP8
- MXFP8
- FP4
- MXFP4
- NVFP4

Those are associated with newer accelerator generations. Low-bit **storage** can still be implemented, but that is not equivalent to native low-bit Tensor Core compute.

---

## 5. Why 24 GB changes everything

At 24 GB, the model’s persistent weights are only one part of the budget. You also need:

- text encoder weights;
- VAE weights;
- latent tensors;
- Q/K/V and attention workspaces;
- MLP activations;
- compiler/CUDA Graph static pools;
- allocator fragmentation;
- library workspaces;
- control/reference inputs;
- cache state;
- safety margin.

A model that “has 20 GB of weights” may not fit comfortably.

Target a persistent safety reserve rather than filling VRAM to 99%. The exact reserve should be measured per workload; compiler and shape changes can alter it.

---

## 6. Single-GPU performance ladder

### Stage 0 — eliminate accidental slow paths

Check:

- GPU actually used for all hot modules;
- no CPU fallback ops;
- correct FP16/BF16 reference dtype;
- no debug synchronization;
- no frequent `empty_cache()`;
- no tensor printing or `.item()` in denoising loop;
- no model component moving devices every step.

### Stage 1 — attention backend

Benchmark:

- PyTorch SDPA;
- FlashAttention-2;
- xFormers if integrated.

FA3/FA4 are not the A5500 path.

### Stage 2 — compile

Compile the repeated transformer block/region. Use stable shape buckets.

### Stage 3 — fuse remaining hot pointwise chains

After compile, only custom-fuse operators still visible as material timeline cost.

### Stage 4 — fix component residency

Keep the denoiser hot. Move cold encoders out rather than shuttling hot transformer layers every step.

### Stage 5 — exact caches

Prompt embeddings, static masks, position tables, shape-specific compiled artifacts.

### Stage 6 — two-GPU if needed

Only after one-GPU path is healthy.

### Stage 7 — approximation research

Quantization/caching/step changes only behind no-regression gates.

---

## 7. Memory hierarchy mindset

The useful hierarchy is:

`registers -> shared memory/L1 -> L2 -> GDDR6 -> PCIe host memory -> storage`

Performance falls as the hot loop moves farther down that hierarchy.

For DiTs:

- keep repeated transformer weights in VRAM if possible;
- keep frequently reused conditioning tensors in VRAM;
- tile attention to avoid giant global intermediates;
- fuse pointwise operations to avoid write/read cycles to GDDR6;
- if offloading is necessary, offload cold components first;
- prefetch coarse groups, not individual tensors one by one.

---

## 8. PCIe 4.0 offload realities

The card has PCIe Gen4 x16. Theoretical link bandwidth is not the same as sustained PyTorch transfer throughput.

Measure:

```python
# Conceptual transfer benchmark; use tensors sized like real groups.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
y = x.to("cuda", non_blocking=True)
end.record()
end.synchronize()
print(start.elapsed_time(end))
```

Use pinned host memory for asynchronous H2D where appropriate.

A group-offload plan succeeds when transfer of group `k+1` overlaps compute of group `k`. It fails when the GPU waits on every copy.

---

## 9. Pinned memory

Pinned memory can improve/enable asynchronous CPU↔GPU transfers, but it is a scarce host resource.

Rules:

- preallocate bounded pinned pools;
- reuse buffers;
- do not pin the entire model multiple times;
- monitor system RAM and swap;
- separate pinned-memory footprint from ordinary RSS in experiment notes.

If host memory pressure causes swapping, the optimization has failed.

---

## 10. Text encoder strategy

Large prompt encoders can consume several GB that the denoiser needs.

Best exact options:

1. compute embeddings once;
2. cache embeddings if prompt/encoder state repeats;
3. move/free the encoder before denoising;
4. keep only the embeddings resident.

For repeated prompt corpora or batch generation, prompt-embedding preprocessing can be a major win.

Quantizing the text encoder is lower priority because its work is often one-time and quality-sensitive.

---

## 11. VAE strategy

VAE decode can spike activation memory at high resolution.

Test:

- decode after freeing large denoiser temporaries;
- VAE tiling;
- VAE slicing/chunking;
- CPU or second-GPU VAE placement if transfer is acceptable.

The fastest VAE configuration depends on output size. Tiling trades memory for extra launches/border logic.

---

## 12. Attention on Ampere

The A5500 is a good match for FA2-style exact attention because Ampere is in the supported architecture family.

Benchmark attention using actual DiT shapes. Key variables:

- sequence length;
- head dimension;
- number of heads;
- dtype;
- joint/cross/self attention;
- mask;
- batch;
- layout.

Do not benchmark an LLM causal shape and assume the result applies to noncausal visual attention.

---

## 13. GEMMs

The DiT MLP and projection layers are dense matrix-multiply heavy.

Practical hierarchy:

1. let PyTorch/cuBLASLt/Inductor choose strong algorithms;
2. compile/static-shape specialize;
3. profile repeated GEMM shapes;
4. only then consider custom Triton/CUTLASS kernels.

The Wafer resource map’s Volkov and CUDA matmul references are valuable here because they teach tiling/data reuse rather than architecture-specific magic flags.

---

## 14. Fused pointwise kernels

On a GDDR6 device, repeated norm/modulation/residual kernels can become bandwidth/launch-heavy.

Potential fusion group:

`norm -> scale/shift from timestep conditioning -> activation/gate -> residual`

If Inductor already fuses it, do not rewrite it manually.

Use Nsight Systems to verify launch-count reduction and Nsight Compute to verify lower memory traffic.

---

## 15. CUDA Graphs

A DiT’s repeated shape makes graph replay appealing.

But 24 GB VRAM makes graph memory cost important.

Test per shape bucket:

- eager;
- compiled no graph;
- compiled/graph path.

Record:

- reserved memory;
- latency;
- first-request cost;
- recompile/recapture behavior.

Do not capture arbitrary shapes dynamically and let the worker accumulate unbounded graph pools.

---

## 16. Power and thermal stability

Long denoising/video workloads can run near sustained power limits.

Monitor:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,pstate,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory --format=csv -l 1
```

Important experimental control:

- warm both baseline/candidate to similar thermal state;
- ensure chassis airflow is constant;
- avoid comparing one card in a cool first slot with another in a thermally constrained slot without noting it;
- report sustained rather than one-second burst behavior.

---

## 17. ECC

A5500 includes ECC support according to NVIDIA. If ECC configuration changes usable memory/performance on the deployed system, keep it constant across comparisons and record it.

Do not disable reliability features merely to win a benchmark unless the deployment requirement explicitly permits it.

---

## 18. Two-A5500 NVLink path

NVIDIA states that the low-profile bridge can connect two A5500s and provide 112.5 GB/s bidirectional bandwidth.

### What NVLink does

- provides a faster peer link than ordinary PCIe paths for supported peer communication;
- can support application-aware scaling of memory/performance.

### What NVLink does not do automatically

- turn two 24 GB allocations into one transparent 48 GB PyTorch tensor address space;
- make every collective free;
- guarantee a distributed strategy scales;
- connect more than the supported bridge topology without additional system architecture.

### Validate

```bash
nvidia-smi topo -m
```

Then run a peer-access/bandwidth test or NCCL test appropriate to the software stack.

### Best two-GPU uses

- model sharding for fit;
- context/sequence parallel attention;
- independent data-parallel requests for throughput;
- dedicating a second GPU to VAE/encoder only if transfer/queue analysis proves it useful.

Fine-grained tensor parallelism can spend too much time synchronizing on only two workstation GPUs; measure.

---

## 19. Multi-GPU performance math

If one GPU takes `T1` and two take `T2`:

`speedup = T1 / T2`

`parallel_efficiency = T1 / (2*T2)`

`GPU_seconds_1 = T1`

`GPU_seconds_2 = 2*T2`

Report all of them.

A 1.5× latency speedup on two GPUs has 75% parallel efficiency and consumes more GPU-seconds. It may still be worth it for an interactive SLO.

---

## 20. If there is no NVLink bridge

Prefer strategies with low synchronization frequency:

- independent request/data parallelism;
- coarse pipeline/model sharding;
- component split where transfers happen once per request, not every transformer block.

Context/tensor parallelism over PCIe can still be necessary for fit, but likely costs more latency.

---

## 21. Custom kernel workflow

The Wafer repository’s resource ordering is exactly right:

1. profile;
2. identify hot kernel;
3. understand roofline/memory behavior;
4. inspect existing cuBLAS/cuDNN/SDPA/FA2/Triton implementations;
5. write custom kernel only when the existing stack leaves material performance on the table;
6. verify correctness with reference tensors and full model benchmarks.

For CUDA kernels:

```bash
compute-sanitizer ./your_kernel_test
```

Then profile with Nsight Compute.

---

## 22. Toolchain sanity checks

Capture:

```bash
python -m torch.utils.collect_env
nvcc --version
nvidia-smi
```

For source-built extensions:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

Use `ninja` for build parallelism but cap jobs if host RAM is limited.

A stale binary compiled only for a different SM can create silent JIT or suboptimal code paths.

---

## 23. Recommended workstation configuration discipline

For reproducibility:

- fixed driver/CUDA/PyTorch environment;
- fixed power policy;
- no background GPU workloads;
- enough host RAM for model offload without swap;
- high-bandwidth PCIe slot running at expected generation/width;
- NVLink bridge verified when used;
- stable cooling;
- local model files on fast storage for cold-start testing;
- cache directories on persistent fast storage.

---

## 24. What to ignore in generic DiT optimization posts

Ignore or heavily discount claims based on:

- H100/H200/B200 without an A5500 reproduction;
- FP8/MXFP8/NVFP4 speedups;
- FA3/FA4;
- 8-GPU NVSwitch scaling;
- huge HBM capacity that eliminates offload;
- undocumented quality evaluation;
- benchmark screenshots without software/shape details;
- speedups against an uncompiled/unoptimized baseline.

The mechanism may still be useful, but the number is not portable.

---

## 25. A5500 success criteria

A strong final configuration should be able to answer:

- Which exact shape buckets are supported?
- Which attention backend runs for each?
- Is the transformer compiled?
- How many graph breaks/recompiles occur?
- What is peak allocated/reserved VRAM?
- Which components reside on GPU during denoising?
- How many bytes move over PCIe per request?
- If two GPUs are used, what fraction of time is communication?
- What are p50/p95 latency and GPU-seconds/output?
- Does every required quality benchmark pass?
- Is quantization disabled, or if enabled, what evidence proves no accuracy regression?

If those questions have answers, the system is being engineered rather than merely tuned by folklore.
