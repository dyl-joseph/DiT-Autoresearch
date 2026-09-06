# RTX A5500 Optimization Checklists

## Baseline

- [ ] GPU is confirmed as RTX A5500.
- [ ] `torch.cuda.get_device_capability()` recorded.
- [ ] Driver, CUDA, PyTorch, Diffusers versions recorded.
- [ ] Model/component revisions pinned.
- [ ] Scheduler, steps, guidance, shapes pinned.
- [ ] Quantization disabled in reference.
- [ ] Benchmark corpus hash recorded.
- [ ] Reference quality outputs/metrics saved.
- [ ] Warm and cold latency separated.
- [ ] p50/p90/p95 measured.
- [ ] Peak allocated and reserved VRAM measured.
- [ ] Sustained clocks/temperature checked.

## Profiling

- [ ] Nsight Systems trace captured.
- [ ] CPU launch gaps inspected.
- [ ] H2D/D2H copies inside denoising loop inspected.
- [ ] Top 5 GPU kernels identified.
- [ ] Hot kernels profiled with Nsight Compute.
- [ ] Roofline/SM/memory/Tensor Core behavior classified.
- [ ] No optimization selected only because it is popular.

## Attention

- [ ] PyTorch SDPA benchmarked.
- [ ] Actual SDPA backend/fallback verified in profiler.
- [ ] FlashAttention-2 benchmarked for real shapes if supported.
- [ ] xFormers benchmarked only if useful.
- [ ] FA3/FA4 excluded from A5500 implementation plan.
- [ ] Whole transformer block benchmarked, not only isolated attention.
- [ ] Quality suite passes after backend change.

## Compile

- [ ] Transformer/repeated region compiled before whole pipeline.
- [ ] Compile time recorded.
- [ ] First-call and warm-call latency separated.
- [ ] Graph breaks inspected.
- [ ] Recompiles inspected.
- [ ] Shape buckets defined.
- [ ] Peak reserved VRAM recorded.
- [ ] `sm_86` native target verified for custom extensions.
- [ ] CUDA Graph memory cost measured.

## Memory

- [ ] Persistent versus phase-local component memory identified.
- [ ] Text encoder removed/offloaded after embeddings when possible.
- [ ] Prompt embedding cache evaluated.
- [ ] VAE moved/loaded at correct phase.
- [ ] Activation peak phase identified.
- [ ] Compiler/static-pool memory included.
- [ ] Safety headroom reserved.
- [ ] Quantization not used as first response to OOM.

## Offload

- [ ] Offload starts with cold components.
- [ ] Transformer group offload used only if necessary.
- [ ] Pinned-memory pool bounded.
- [ ] Async prefetch overlap verified in Nsight Systems.
- [ ] H2D/D2H bytes/request recorded.
- [ ] Host RAM and swap checked.
- [ ] PCIe topology/NUMA considered.
- [ ] Quality suite passes.

## Two A5500s

- [ ] Physical NVLink bridge presence recorded.
- [ ] `nvidia-smi topo -m` saved.
- [ ] Peer access verified.
- [ ] NCCL/peer bandwidth measured for relevant sizes.
- [ ] Strategy goal stated: fit, latency, or throughput.
- [ ] Per-GPU VRAM recorded.
- [ ] Communication share measured.
- [ ] Speedup and parallel efficiency computed.
- [ ] GPU-seconds/output reported.
- [ ] Distributed numerical/quality benchmark passes.

## Quantization — hard gate

- [ ] Exact optimized FP16/BF16 baseline exists.
- [ ] Quantization reason stated: capacity and/or speed.
- [ ] Exact modules quantized recorded.
- [ ] Storage and compute dtype recorded separately.
- [ ] Calibration dataset/revision recorded.
- [ ] Dequantization/fused-kernel path understood.
- [ ] Quantized eager and compiled compared when feasible.
- [ ] Exact offload/sharding alternatives compared.
- [ ] All required aggregate benchmark metrics are non-decreasing under project policy.
- [ ] Zero critical baseline-pass -> candidate-fail regressions.
- [ ] If any required accuracy drops: **REJECT**.

## Approximate caching / step shortcuts

- [ ] Exact baseline retained.
- [ ] Changed mechanism isolated.
- [ ] Hit/skip/step behavior logged.
- [ ] Full image/video quality suite run.
- [ ] Temporal artifacts checked for video.
- [ ] Candidate rejected on benchmark regression.

## Production

- [ ] Shape-aware routing.
- [ ] Prewarm common compile profiles.
- [ ] Compile/cache footprint bounded.
- [ ] Queue time and service time separated.
- [ ] Memory-aware admission control.
- [ ] Engine/config fingerprint logged.
- [ ] GPU telemetry logged.
- [ ] Canary + rollback.
- [ ] Benchmark-quality regression alarm.
- [ ] Quantization state visible in telemetry/config.

## Final sign-off

- [ ] Primary performance objective improved.
- [ ] p95/p99 acceptable.
- [ ] GPU-seconds understood.
- [ ] Peak VRAM safe.
- [ ] Sustained thermals stable.
- [ ] Full required quality benchmark passes.
- [ ] Critical regressions = 0.
- [ ] Result reproduces after process restart.
- [ ] Configuration and source versions documented.
