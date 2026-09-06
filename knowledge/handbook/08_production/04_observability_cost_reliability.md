# Observability, Cost, Reliability, and Quality in Production DiT Serving

## Performance that cannot be observed cannot be maintained

A DiT engine is a moving target: driver updates, kernel changes, model revisions, traffic shapes, cache thresholds, and quantization all change behavior. Production needs metrics at four levels.

## 1. Request metrics

- end-to-end latency
- queue time
- service time
- model/profile
- shape bucket
- steps
- guidance policy
- seed (or request trace ID; protect privacy)
- success/cancel/OOM

## 2. Model pipeline metrics

- text encode ms
- denoiser total ms
- ms/step distribution
- cache hit/skip ratio
- VAE decode ms
- postprocess/codec ms

## 3. GPU metrics

- utilization
- Tensor Core/SM utilization when available
- HBM usage and bandwidth
- power/clock/throttling
- ECC/health
- per-rank NCCL time

## 4. Engine lifecycle metrics

- compile count/time
- graph/shape cache hit rate
- cold start time
- attention backend selected/fallbacks
- model load time
- quantization time

## Cost metrics

The most useful unit is often:

`cost_per_accepted_output`

not cost per request. Include:

- GPU-seconds
- retries
- rejected/canceled outputs
- moderation/safety overhead
- decoder/encoding workers

A cache/quantization change that creates more bad generations can reduce raw latency and increase cost per accepted output.

## GPU-seconds

`GPU_s = wall_time * number_of_GPUs`

Use it to compare parallel configurations. Example:

- 1 GPU x 8 s = 8 GPU-s
- 4 GPUs x 3 s = 12 GPU-s

The second is faster for user latency but 50% more accelerator time.

## Quality observability

Not all quality can be scored online, but track proxies and sampled evaluation:

### Image

- prompt/image similarity
- OCR success for text prompts
- artifact/NSFW/safety classifiers
- human preference sample

### Video

- temporal consistency/flicker
- motion adherence
- identity consistency
- sampled human review

Tie quality metrics to engine configuration so regressions can be attributed to FP8/cache/backend changes.

## Canary deployment

For a new optimization:

1. offline quality suite
2. shadow/canary 1-5% traffic
3. compare latency, memory, failure, quality proxies
4. expand gradually
5. automatic rollback on SLO/quality threshold

Do not roll out quantization/cache globally based only on benchmark speed.

## Tail latency

Track p50/p90/p95/p99. Sources of DiT tails:

- cold compile
- rare shape recompilation
- allocator fragmentation
- queueing behind long video
- multi-GPU slow rank
- offload page faults/NUMA
- thermal throttling

A faster p50 with worse p99 can be a product regression.

## OOM reliability

OOM should be a controlled admission error, not an unpredictable crash. Monitor:

- per-profile peak memory
- active request dynamic memory
- allocator reserved/allocated gap
- OOM count
- worker restarts

Feed new observed peaks back into the admission table.

## Engine fingerprint

Every response/log should be attributable to a fingerprint:

`model_rev + runtime + GPU_arch + dtype + attention + compile + cache + shape_profile`

Without this, “performance regressed yesterday” is difficult to diagnose.

## Regression testing

On every environment upgrade run a fixed matrix:

- 3-5 shapes
- 2 step counts
- cache on/off
- quantization profile
- attention backend
- image/video quality suite

Store historical results and alert on >X% changes.

## Source basis

**PDF-derived:** observability, cost estimation, percentile latency, reliability, testing/deployment and end-to-end metrics.  
**Expansion:** engine fingerprints, accepted-output economics, optimization canary/quality monitoring.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
