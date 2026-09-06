# Quantization and Precision on RTX A5500 — Strict No-Regression Policy

This chapter intentionally changes the usual quantization narrative.

For this project:

> **Do not deploy quantization if it reduces required benchmark accuracy.**

And for RTX A5500 specifically:

> **Do not assume quantization will be a speed optimization.** The card does not have native Hopper-style FP8 Tensor Cores, so low-bit storage often means dequantizing into FP16/BF16 compute.

The default inference plan is therefore **unquantized FP16/BF16** plus kernel, compile, memory-management, and multi-GPU optimization.

---

## 1. Separate four different meanings of “precision”

A model can use different formats for:

1. **checkpoint storage** — bytes on disk;
2. **resident weight storage** — bytes in GPU/CPU memory;
3. **activation storage** — intermediate tensors;
4. **compute** — Tensor Core / ALU input and accumulation formats.

A 4-bit checkpoint does not imply 4-bit Tensor Core compute. On A5500, many low-bit paths unpack or dequantize to FP16/BF16 before GEMM.

Therefore record all four.

---

## 2. Native A5500-friendly floating-point modes

NVIDIA’s Ampere tuning guide documents third-generation Tensor Core support for:

- FP16;
- BF16;
- TF32 for FP32-style matrix operations.

For DiTs, the default is the model’s reference FP16/BF16 inference dtype.

### FP16 versus BF16

Benchmark both only when the model/runtime supports both as valid inference modes.

BF16 has FP32-like exponent range and can be numerically safer for some networks; FP16 has more mantissa bits but smaller dynamic range. Performance can be similar, but kernel support and model numerics decide the result.

Do not switch purely on theory. Use the benchmark suite.

### TF32

TF32 can accelerate FP32 matmul on Ampere, but it reduces mantissa precision relative to full FP32. If your baseline actually uses FP32 matrix math and quality cannot move, TF32 belongs in the same quality-gated category as other reduced-precision changes.

It is not relevant to a pipeline already executing FP16/BF16 GEMMs.

---

## 3. Why FP8 advice from modern inference guides does not transfer

Hopper introduced native FP8 Tensor Core paths; Blackwell adds further low-precision formats such as MXFP8/MXFP4/NVFP4. A5500 predates those features.

Consequences:

- no native FP8 throughput multiplier should be assumed;
- FA3 FP8 examples do not apply;
- Blackwell microscaling recommendations do not apply as hardware-native compute;
- storing FP8-like data can still reduce residency, but conversion cost must be measured;
- low-bit kernels may use integer Tensor Core paths or custom unpacking, which have different performance/accuracy characteristics.

Any file in this handbook that mentions FP8 in an architecture-general context should be read through this override.

---

## 4. Quantization is a capacity experiment first

On a 24 GB card, quantization’s strongest possible value is often **fit**, not speed.

Example problem:

- unquantized transformer + text encoder + VAE exceeds 24 GB;
- low-bit weights make it fit.

That is useful only if:

- quality remains benchmark-equal;
- runtime is acceptable;
- an exact offload/sharding alternative is not better.

Compare quantization against:

- text-encoder offload;
- cached prompt embeddings;
- group/model CPU offload;
- VAE tiling;
- two-A5500 sharding;
- lower concurrency/batch size.

Do not compare quantized “fits” only against unquantized “OOM” and call the result a speedup.

---

## 5. The hard acceptance gate

For every quantized candidate:

### Required metadata

- quantizer/tool/version;
- exact modules quantized;
- storage dtype;
- compute dtype;
- group/block size;
- symmetric/asymmetric scheme;
- per-channel/per-group scaling;
- calibration dataset/revision;
- excluded layers;
- dequantization/fused-kernel path;
- model revision;
- benchmark suite revision.

### Required result

Candidate is accepted only if:

- all required aggregate benchmark scores are non-decreasing under the project’s acceptance rule;
- no critical per-case regression appears;
- latency or capacity improves meaningfully;
- peak VRAM and temporary conversion memory are measured;
- cold-start conversion cost is accounted for.

### Zero-observed-drop policy

If the requirement is literal, enforce:

```text
for every required aggregate metric:
    candidate >= baseline

for every critical deterministic case:
    if baseline == PASS:
        candidate must == PASS
```

Do not average away a regression.

---

## 6. What to quantize first if experimentation is allowed

Because the project rejects quality loss, start with the least semantically sensitive targets and test one class at a time.

### Candidate A — cold text encoder weights

If the text encoder must remain resident, low-bit weight storage can save VRAM. But a better exact strategy may be:

1. run the encoder once;
2. cache embeddings;
3. move or free the encoder.

This often provides memory relief with no model arithmetic change in the denoiser.

### Candidate B — large MLP/linear weights

These dominate parameter count in many DiTs and are conventional weight-only quantization targets.

Caveats on A5500:

- dequantization may dominate at batch size 1;
- unsupported/fallback kernels can be slower than FP16;
- quality can still change across dozens of denoising steps.

### Candidate C — attention projections, not attention arithmetic

Quantizing projection weights is different from quantizing QK/softmax/PV arithmetic. Test separately.

### Candidate D — attention arithmetic

Highest risk. Do not combine with other quantization in the first experiment.

### Candidate E — VAE

Usually low priority. Pixel-space output is directly exposed to VAE numerical changes, and VAE compute may be only a fraction of total runtime.

---

## 7. Layers to preserve initially

When testing layerwise weight quantization, preserve full reference precision for high-sensitivity boundaries until data proves otherwise:

- input projection;
- output projection;
- normalization parameters;
- timestep/conditioning embeddings;
- modulation/gating layers;
- first transformer block;
- last transformer block;
- any small layer whose quantization saves negligible memory.

This is not a guarantee that middle layers are safe. It is a conservative search order.

---

## 8. Calibration must resemble real generation

Calibration data for post-training quantization should cover the activation regimes produced by:

- diverse prompts;
- short/long prompts;
- typography;
- unusual object counts;
- extreme aspect ratios;
- high/low guidance;
- early/middle/late denoising steps;
- video motion cases;
- control/editing conditions;
- negative prompts.

A calibration set of generic text prompts can miss large activation outliers from editing/control/video paths.

---

## 9. Quantization across denoising time

The uploaded book discusses selective precision by timestep/layer for video models: early steps can be more sensitive because they establish global structure.

On A5500, this idea is mainly a **quality-search method**, not a native FP8 speed path.

Possible experiment:

- keep early steps fully unquantized;
- if a static quantized candidate has already passed the strict full-suite quality gate, optionally test whether restricting that same candidate to later steps preserves parity while reducing its scope. This is a research experiment, not a default speed path.

But beware:

- runtime dtype/weight switching can add overhead;
- duplicated weight copies can destroy the memory benefit;
- compile/graph capture becomes harder;
- benchmark parity is still mandatory.

Do not add this complexity unless static quantization already demonstrates a valuable capacity/performance result.

---

## 10. Quantization and `torch.compile`

Possible outcomes:

- compile fuses dequantization with GEMM and the candidate becomes faster;
- graph breaks around custom quantized ops eliminate compile benefits;
- the quantizer uses opaque kernels that compile cannot optimize;
- dynamic scale computation creates extra launches;
- CUDA Graph capture fails due dynamic allocations.

Therefore test four states when feasible:

1. FP16/BF16 eager;
2. FP16/BF16 compiled;
3. quantized eager;
4. quantized compiled.

The fair production comparison is usually #2 versus #4, not eager baseline versus compiled quantized candidate.

---

## 11. Quantization and offload

Low-bit storage can reduce PCIe traffic if weights stay compressed while transferred, but only if the runtime architecture actually supports that flow.

Questions:

- are weights stored quantized in host RAM?
- are they copied quantized over PCIe?
- where are they dequantized?
- is dequantization overlapped with compute?
- does the backend allocate a second full-precision copy?
- is pinned memory used?

Measure bytes transferred. Do not infer from checkpoint size.

---

## 12. Quantization and caching

Approximate caches often decide whether to reuse hidden states based on residual differences or thresholds. Quantization changes those values.

Therefore:

- never carry a cache threshold from full precision into a quantized model without retuning;
- benchmark the combination separately;
- if the project is strict about accuracy, first establish quantization parity with caching disabled, then cache parity on top.

Stacked approximations make attribution difficult.

---

## 13. Quantization and multi-GPU

Potential benefit:

- smaller shards;
- lower peer-transfer bytes if the quantized representation is communicated directly.

Potential failure:

- collectives operate on dequantized activations anyway;
- custom quantized kernels prevent efficient sharding;
- communication becomes dominant once local compute shrinks;
- duplicated scales/metadata erode memory savings.

For two A5500s, exact sharding over NVLink can be a cleaner way to preserve quality.

---

## 14. Benchmark matrix

At minimum:

| Config | Precision/storage | Compile | Fits? | p50 | p95 | Peak VRAM | Quality | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | FP16/BF16 | no | | | | | baseline | |
| exact optimized | FP16/BF16 | yes | | | | | must pass | |
| quantized eager | exact config | no | | | | | **no regression** | |
| quantized compiled | exact config | yes | | | | | **no regression** | |
| exact offload | FP16/BF16 | yes/no | | | | | must pass | |
| exact 2-GPU | FP16/BF16 | yes/no | | | | | must pass | |

Quantization only wins if it beats the best quality-preserving alternative for the relevant objective.

---

## 15. Common failure patterns on A5500

### “4-bit uses less VRAM but is slower”

Likely causes:

- dequantization overhead;
- unfused weight unpack;
- poor kernel support for the exact matrix shape;
- memory savings do not matter because compute is dominant.

### “8-bit benchmark passes but text rendering is worse”

Reject under the project quality contract.

### “Quantized model fits but load OOMs”

The conversion path may temporarily hold both original and quantized weights. Quantize/load on CPU or use a serialized prequantized format if supported.

### “Quantized attention looks fine on images but video flickers”

Reject or expand the temporal benchmark. Single-frame metrics are insufficient.

### “Quantized candidate wins only against eager FP16”

Compile/optimize the FP16 baseline and compare again.

---

## 16. Recommended project policy

### Production default

- FP16 or BF16 reference inference;
- no quantization;
- exact attention backend;
- compile/fusion;
- exact memory/offload techniques;
- multi-GPU if required.

### Quantization research branch

Run only when:

- 24 GB fit remains a major problem after exact memory tactics;
- the benchmark suite is mature;
- a supported A5500 quantized kernel path exists;
- the team is willing to reject most candidates if accuracy moves.

### Deployment rule

No benchmark parity, no quantization deployment.

That rule is more important than any theoretical compression ratio.
