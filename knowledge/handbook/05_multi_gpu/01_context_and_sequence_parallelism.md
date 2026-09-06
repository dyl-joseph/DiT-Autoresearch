# Context and Sequence Parallelism for Diffusion Transformers

## Why it fits visual DiTs

Context/sequence parallelism (CP/SP) partitions the visual token sequence across GPUs while typically replicating model weights. This is especially well matched to video and high-resolution DiTs because:

- weights may fit on each GPU
- activations/attention over the huge latent sequence may not
- attention is often the dominant compute

The uploaded PDF describes video inference using context parallelism rather than ordinary tensor parallelism: weights are copied to each GPU and the attention/latent context is split, often using a ring-like mechanism.

## What gets sharded

For sequence length `N` and `P` ranks, each GPU owns roughly `N/P` query tokens. The challenge is that each query still needs information from keys/values across the global sequence.

Different algorithms organize that communication differently.

## Ring attention

Each rank starts with a shard of K/V and circulates K/V blocks around a ring. For each received shard, the rank updates online softmax statistics for its local queries.

### Strengths

- memory-efficient for very long sequences
- point-to-point/structured communication
- does not require each rank to materialize the complete K/V sequence

### Weaknesses

- multiple communication stages
- latency can be high for moderate sequences
- implementation/numerical stability is more complex

Best for very long contexts where memory is the binding constraint.

## Ulysses-style sequence parallelism

Ulysses uses all-to-all communication to redistribute tokens/heads so each rank computes full-sequence attention for a subset of heads, then reverses the redistribution.

### Strengths

- high local compute efficiency
- often lower latency on good interconnects for moderate contexts

### Weaknesses

- all-to-all is demanding
- parallel degree is traditionally constrained by head count/divisibility
- can hold more K/V locally than ring

Recent Diffusers includes Ulysses, “Anything” variants for arbitrary shapes/head counts, ring, and unified configurations.

## Unified / hybrid sequence parallelism

Combine Ulysses and ring across a 2D device mesh:

- one dimension redistributes heads/sequence
- the other processes long context via ring

Useful when:

- you have 4+ GPUs
- sequence is too large for pure Ulysses memory behavior
- head count limits Ulysses degree

## Shape divisibility

Classic implementations may require:

- sequence length divisible by rank count
- head count divisible by Ulysses degree

Modern “Anything” variants may pad or gather irregular partitions. Padding has compute/communication cost, so shape buckets aligned to parallel degree can still be faster.

## CP memory model

If weights are replicated, per-rank memory is roughly:

`M_rank ~= M_weights + M_activations/P + M_attention_workspace/P-ish + M_fixed`

Exact scaling depends on backend. CP does **not** solve a model whose weights alone exceed one GPU.

## CP latency model

Simplified:

`T(P) ~= T_compute/P + T_comm(P) + T_sync + imbalance`

Scaling is good only while reduced compute exceeds added communication.

Measure:

`speedup = T(1) / T(P)`

`efficiency = speedup / P`

A 4-GPU speedup of 3.2x = 80% efficiency. A 4-GPU speedup of 1.6x = 40% and may be unjustifiable unless memory requires it.

## Choosing ring vs Ulysses

Start empirically:

### Ulysses first when

- sequence is moderate
- head count permits degree
- NVLink/NVSwitch is strong
- lowest latency is goal

### Ring first when

- sequence is extremely long
- per-rank activation memory is tight
- all-to-all scales poorly

### Unified when

- enough GPUs for a 2D mesh
- pure Ulysses degree is constrained
- ring alone leaves too much communication latency

Use current framework benchmarks as hints only; the crossover depends on model/head/shape/topology.

## CP and Flash/Sage attention

The local attention kernel still matters. A CP implementation may support only a subset of backends. Combinations can change:

- local sequence length
- head count per rank
- kernel-selected tiling

A backend that was fastest on one GPU may lose after sharding because local shapes are smaller.

## CP and caching

Caches must be sharded consistently with the hidden state. FirstBlockCache/PAB can reduce local compute; however collective cost may remain, making communication a larger fraction. Retune parallel degree after caching.

## CP and quantization

For the A5500 target, **do not plan CP around quantization**. Native Hopper/Blackwell FP8 assumptions do not apply, and reducing communication precision changes numerics. First optimize exact FP16/BF16 local compute and the communication schedule.

If a compressed/quantized path is later investigated, treat local compute precision and communication precision as separate experiments. Each must pass the full no-regression benchmark gate, and the parallel degree must be retuned because faster local kernels can make communication dominate sooner.

## CP and CFG

Possible layouts on 8 GPUs:

- CP8: one guidance branch at a time across 8 GPUs
- 2 x CP4: conditional and unconditional branches in parallel
- CP4 + data parallel 2: two requests at once, each on 4 GPUs

The best layout depends on whether the objective is single-request latency or throughput.

## Current Diffusers pattern

Recent Diffusers exposes `ContextParallelConfig` with ring and Ulysses degrees and model `enable_parallelism(...)` APIs. Details are version-sensitive. Also note that some compile/CLI combinations may not yet compose cleanly with CP.

## Production requirements

- rank placement follows NVLink/NVSwitch topology
- NCCL/Gloo initialization configured correctly
- all ranks use identical model revision/precision
- per-rank memory tracked
- failures tear down/restart the whole group cleanly
- request scheduler reserves an entire CP group atomically

## Current reference

- Diffusers Distributed inference: https://huggingface.co/docs/diffusers/main/training/distributed_inference

## Source basis

**PDF-derived:** video models use context parallelism, replicate weights, split attention/latent context, and can use ring attention; VAE can also be parallelized.  
**Expansion:** Ulysses/unified strategies, scaling formulas, CP+CFG/cache/quantization composition.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
