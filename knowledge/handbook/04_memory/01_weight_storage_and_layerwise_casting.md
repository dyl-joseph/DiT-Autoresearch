# Weight Residency and Storage on a 24 GB RTX A5500

The original version of this chapter emphasized low-bit and layerwise-casting options. For the target hardware and quality policy, the priority changes:

1. preserve the reference FP16/BF16 model;
2. keep the repeated denoiser weights resident if possible;
3. remove cold components from VRAM;
4. shard across a second A5500 if available;
5. use CPU/group offload if transfer cost is acceptable;
6. test quantized storage only as a final capacity experiment and reject it if benchmark accuracy decreases.

---

## 1. Start with exact weight accounting

For a parameter tensor:

`weight_bytes = numel × bytes_per_element`

Typical storage:

- FP32: 4 bytes/parameter
- FP16/BF16: 2 bytes/parameter
- INT8-like packed storage: ~1 byte/parameter plus scales/metadata
- 4-bit packed storage: ~0.5 byte/parameter plus scales/metadata

But the model’s actual resident footprint includes:

- scale metadata;
- padding/alignment;
- duplicated tied/converted tensors;
- runtime packed formats;
- optimizer state only if accidentally loaded;
- temporary conversion buffers;
- compiler copies/workspaces.

Measure process VRAM instead of multiplying parameter count and stopping.

---

## 2. Persistent versus phase-local components

A DiT pipeline often contains:

- transformer/denoiser — repeated every step;
- text encoder(s) — usually once per prompt;
- VAE — usually encode/decode outside the denoising loop;
- control encoders — sometimes once, sometimes repeated;
- safety/postprocessing — usually after generation.

The repeated denoiser has the strongest claim on scarce VRAM.

### Exact residency strategy

- load/run text encoder;
- keep prompt embeddings;
- free/offload text encoder;
- keep transformer resident through all denoising steps;
- free transformer temporaries as possible;
- load/run VAE decode.

This serial component lifecycle can fit models that appear impossible if all modules are kept resident simultaneously.

---

## 3. Prompt-embedding caching is a memory optimization

Caching prompt embeddings allows workers to omit a large text encoder from GPU residency for repeated prompts or preprocessed workloads.

Cache key should include:

- tokenizer revision;
- text encoder revision;
- prompt;
- negative prompt;
- max sequence length/truncation behavior;
- dtype;
- any prompt-weighting scheme.

If any of these changes, invalidate the cache.

This is exact reuse, unlike hidden-state caches across denoising steps.

---

## 4. Low-bit storage is not native low-bit compute on A5500

A5500 has Ampere FP16/BF16 Tensor Core paths, not Hopper FP8 Tensor Cores.

Therefore low-bit weight storage may:

- reduce VRAM;
- reduce PCIe transfer bytes if copied compressed;
- add unpack/dequantization kernels;
- still execute GEMMs in FP16/BF16;
- run slower at batch 1.

Do not label a checkpoint “INT8” and assume a throughput benefit.

---

## 5. Layerwise casting

Some frameworks can store cold/inactive weights in a lower precision and cast around execution.

On A5500 this is mostly a **residency/transfer tradeoff**.

Questions:

- is each block cast every denoising step?
- is there a persistent full-precision copy?
- are casts fused with loading/GEMM?
- are weights moved from CPU or merely converted in VRAM?
- how many bytes are written to GDDR6 per cast?

Repeated per-step casting can be worse than keeping FP16/BF16 resident.

---

## 6. State-dict loading without transient duplication

A model can OOM during load even if final residency fits because loading may temporarily hold:

- CPU state dict;
- model initialized weights;
- converted weights;
- GPU destination copy.

Prefer framework mechanisms that support:

- meta-device initialization;
- low-memory loading;
- memory mapping;
- streaming shard load;
- direct target dtype load.

Avoid “load FP32 -> move GPU -> cast FP16” if a direct FP16/BF16 load is supported and reference semantics allow it.

---

## 7. LoRA/adapters

If serving a fixed LoRA configuration, fusing adapter weights into base weights can reduce per-layer adapter GEMMs and runtime objects.

Caveats:

- fusion changes the stored weights and may introduce rounding differences;
- dynamic per-request adapters make fusion impractical;
- multiple fused variants multiply storage/cache requirements.

Benchmark quality and memory after fusion.

---

## 8. Shard before quantizing when quality cannot move

With two A5500s, exact sharding can preserve FP16/BF16 weights while solving the 24 GB limit.

Potential splits:

- transformer layer groups across GPUs;
- tensor parallel projection shards;
- context/sequence parallel activations;
- component placement.

The cost is communication. Compare exact sharding versus quantized single-GPU using both latency and quality.

---

## 9. Recommended memory decision sequence

```text
Does full pipeline fit?
  yes -> keep full precision; optimize kernels/compile
  no  -> can text encoder be removed after prompt encoding?
          yes -> do that
          no/insufficient -> can VAE be loaded only for decode?
          insufficient -> exact CPU/group offload
          insufficient/too slow -> second A5500 sharding
          still constrained -> quantization experiment behind strict quality gate
```

---

## 10. Quantization acceptance reminder

A low-bit candidate is not accepted because it fits. It must also:

- preserve every required benchmark;
- avoid critical per-case regressions;
- beat or justify itself against exact offload/sharding alternatives;
- have measured conversion/workspace overhead.

On this project, quality decides whether quantization exists at all.
