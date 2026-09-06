# CFG, Data, Tensor, and Pipeline Parallelism for DiTs

## Parallelism solves different problems

Do not ask “How do I use multiple GPUs?” Ask which resource you want to improve:

- **data parallelism (DP):** more independent requests / throughput
- **CFG parallelism:** lower latency for two guidance branches
- **context/sequence parallelism:** split long token contexts / activations and attention compute
- **tensor parallelism (TP):** split weight matrices and compute
- **pipeline parallelism (PP):** split layers/stages

Each has a different communication pattern.

## Data parallelism

Replicate the full pipeline on each GPU and route different prompts to each.

Best when:

- the model fits on one GPU
- single-request latency is already acceptable
- traffic is high

Advantages:

- near-linear throughput scaling
- no inter-GPU communication on hot path
- simplest failure isolation

It does not reduce single-request latency or memory.

## CFG parallelism

Conditional and unconditional predictions are independent until the final guidance combine.

With two GPUs:

- GPU 0 runs conditional branch
- GPU 1 runs unconditional branch
- exchange/merge small output tensors

This can be an excellent low-latency strategy when classic CFG is truly two branches and model weights fit on both GPUs.

Cost: two complete weight replicas and GPUs per request.

## Tensor parallelism

Shard linear/tensor dimensions within each transformer block. Large GEMMs execute across ranks and synchronize results.

Best when:

- weights do not fit on one GPU
- a single request needs lower latency
- GPUs have high-bandwidth interconnect
- the engine has mature TP kernels for the architecture

Downside: collectives happen frequently—often every block—so PCIe can erase gains.

TP is more natural for very large DiTs/MoE/hybrid transformers whose weights are a major part of the problem. For video models whose activations dominate while weights fit, CP is generally the more direct tool.

## Pipeline parallelism

Split layers across devices. A single microbatch flows through stages sequentially.

For batch-1 interactive DiT inference, naive PP has poor utilization because only one stage is busy at a time. It is useful when:

- weights cannot fit otherwise
- many microbatches/requests can fill the pipeline
- a PipeFusion-like algorithm can overlap denoising patches/stages

## Hybrid layouts

### DP x CP

Multiple CP groups, each serving independent requests. Excellent for high-volume video serving.

Example: 16 GPUs -> 4 groups of CP4.

### CFG x CP

Two CP groups execute guidance branches in parallel. Example: 8 GPUs -> 2 x CP4.

### TP x CP

Weights sharded in one mesh dimension, sequence in another. Useful when both model size and activation context are too large. Complex and communication-heavy.

### DP x TP

Multiple TP replicas. Common for large weight-heavy models.

## Communication frequency comparison

| Method | Communication frequency | What is replicated | Good interconnect needed? |
|---|---|---|---|
| DP | none per forward | all weights | no |
| CFG parallel | once/small merge per step | all weights | modest |
| CP | within each attention | weights | yes |
| TP | multiple collectives per block | portions of weights | strongly yes |
| PP | between stage boundaries | stage weights | medium; latency sensitive |

## Selecting by the bottleneck

### Model fits, one image too slow, classic CFG enabled
Try CFG parallelism before complex TP.

### Model fits, video activations/attention huge
Use CP/SP.

### Model weights do not fit
Do **not** use quantization first on the A5500 target. First try exact two-GPU placement: component sharding/device maps, CFG parallelism when applicable, then TP/PP only when their communication cost is justified. Treat quantization as a last-resort capacity experiment that must pass the no-regression benchmark gate.

### Latency is fine, throughput insufficient
Use DP. It is usually the highest-efficiency scaling strategy.

### Batch/offline generation
DP plus batching can maximize GPU utilization; PP may become more feasible because there are enough microbatches.

## Topology awareness

The uploaded PDF emphasizes that inter-GPU links are much slower than VRAM and that parallelism must be topology-aware. Practical placement:

- keep a TP/CP group inside one NVSwitch/NVLink domain
- avoid crossing nodes for per-block latency-sensitive collectives unless necessary
- use multi-node DP before multi-node TP for throughput workloads

## Memory interactions

- DP/CFG parallelism replicate everything; no weight savings
- TP reduces per-rank weights and some activations
- CP reduces sequence activations but replicates weights
- PP reduces per-rank weights but may need activation staging

Choose based on the memory bucket that actually OOMs.

## Benchmark matrix

For `P` GPUs, compare at least:

- DP(P) throughput
- CP(P) latency
- TP(P) latency if supported
- hybrid groups that divide P (e.g., 2 x CP4 vs CP8)

Record both latency and GPU-seconds/output:

`GPU_s_per_output = wall_time * GPU_count`

A 2x latency speedup using 8x GPUs is expensive unless latency is the product objective.

## Source basis

**PDF-derived:** model parallelism tradeoffs and communication overhead; TP is latency-oriented within a node, PP has utilization disadvantages, topology matters; video favors context parallelism.  
**Expansion:** CFG parallelism, hybrid DiT layouts, GPU-seconds cost framework.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
