# Interconnect and Topology for Multiple RTX A5500 GPUs

Multi-GPU optimization on A5500 is not the same as an H100/B200 NVSwitch node. The physical topology is part of the algorithm.

## 1. A5500 NVLink facts

NVIDIA’s A5500 datasheet states:

- a low-profile NVLink bridge can connect **two** RTX A5500 cards;
- NVLink bandwidth is **112.5 GB/s bidirectional** for this product/bridge;
- application support is required to scale memory/performance.

Do not interpret that as transparent VRAM pooling.

---

## 2. Always inspect topology

```bash
nvidia-smi topo -m
```

Record the output with every multi-GPU benchmark.

Look for:

- NVLink connection between the intended pair;
- CPU/NUMA affinity;
- PCIe switches/root complexes;
- peer-access route.

If the pair is not connected as expected, change rank placement or hardware layout before tuning collectives.

---

## 3. Verify peer access

At CUDA/PyTorch level, verify that peer access is available and enabled by the framework/runtime.

A topology label is necessary but not sufficient; measure a peer copy or NCCL test.

---

## 4. Local VRAM versus link bandwidth

Each A5500 has 768 GB/s local GDDR6 bandwidth. The two-card NVLink bandwidth is far lower than local memory bandwidth.

Therefore:

- repeated remote access is expensive relative to local access;
- fine-grained sharding can become link-bound;
- keep tensors local for as long as possible;
- communicate compact activations rather than repeatedly fetching remote weights when possible.

This ratio explains many poor scaling results.

---

## 5. Best topology-aware parallelism choices

### Data parallelism

Each GPU holds a full model and processes independent requests.

Pros:

- almost no inter-GPU communication in inference;
- near-linear throughput if each model fits;
- simplest and most robust.

Cons:

- does not reduce one-request latency;
- does not solve model fit if one copy exceeds 24 GB.

### Pipeline/model sharding

Split layer groups across GPUs.

Pros:

- solves weight fit;
- only activations cross between stages.

Cons:

- one request walks stages serially unless pipelined across requests/steps;
- stage imbalance hurts utilization.

### Tensor parallelism

Split GEMMs across GPUs.

Pros:

- can reduce one-request compute time;
- divides weight residency.

Cons:

- collectives inside every block;
- can be communication-bound on workstation topology.

### Context/sequence parallelism

Split visual tokens/attention context.

Pros:

- lowers per-GPU activation/attention footprint;
- valuable for video/high-resolution.

Cons:

- attention exchanges/collectives;
- synchronization each layer/step depending algorithm.

---

## 6. If NVLink is absent

Prefer:

1. data parallel independent requests;
2. coarse pipeline/model split;
3. component split;
4. fine-grained TP/CP only when required for fit.

PCIe-only fine-grained communication can make two GPUs slower than one.

---

## 7. If NVLink is present

Still profile. NVLink improves the communication roof but does not eliminate synchronization.

Good uses:

- sharded transformer weights;
- context parallel high-resolution/video attention;
- peer-resident component handoff.

Questionable use:

- dozens of tiny all-reduces per block when one GPU already fits and is compute-efficient.

---

## 8. NCCL measurement

Wafer’s resource list includes NCCL as the canonical collective implementation reference.

Use NCCL tests or framework microbenchmarks to measure:

- all-reduce bandwidth;
- all-gather bandwidth;
- reduce-scatter bandwidth;
- point-to-point latency;

at message sizes matching actual DiT activations.

A 1 MB collective and a 500 MB collective live in different latency/bandwidth regimes.

---

## 9. Communication accounting

For each denoising step:

```text
step time
local GEMM/attention time
collective time
copy time
synchronization/gap time
```

If communication exceeds 20–30% of step time, optimize the split before tuning local kernels further.

This threshold is a heuristic, not a universal law; the exact acceptable share depends on latency goals.

---

## 10. GPU-seconds

A two-GPU speedup can reduce wall latency while increasing total compute cost.

Always report:

`GPU-seconds/output = wall service time × active GPU count`

This prevents misleading “1.4× faster” results that use 2× the hardware.

---

## 11. Topology can change thermal behavior

Two 230 W workstation cards in adjacent slots can run hotter than one. Sustained clocks may drop depending on chassis airflow.

Record per-GPU:

- temperature;
- power;
- clocks;
- utilization.

The second GPU can slow the first thermally even before communication is considered.

---

## 12. Practical two-A5500 sequence

1. optimize one GPU;
2. capture `nvidia-smi topo -m`;
3. verify bridge/P2P;
4. measure NCCL/peer bandwidth;
5. choose DP for throughput if model fits;
6. choose coarse sharding for fit;
7. test CP/TP for one-request latency only with profiler evidence;
8. report communication share and GPU-seconds;
9. rerun quality benchmark because distributed reduction order can change numerics.
