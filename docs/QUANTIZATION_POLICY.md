# Quantization Policy

Quantization is **not** a default optimization in this research program.

## Reason

The target is RTX A5500/Ampere. Native Hopper/Blackwell FP8/FP4 execution paths are unavailable as target-native speed mechanisms. Low-bit storage may still help residency/capacity, but it can add dequantization overhead and can change model quality.

The project requirement is explicit: **do not use quantization if it reduces benchmark accuracy/quality.**

## Enforcement

Every quantization experiment must be classified:

```bash
--classification quantization
```

The harness then forces zero allowed regression on every required quality metric and preserves critical case passes.

A quantized candidate that is 2x faster but lowers a required benchmark metric by any measured amount fails under the strict contract.

## Research order before quantization

For 24 GB capacity problems, investigate first:

1. text-encoder eviction after prompt embeddings;
2. prompt-embedding reuse where semantically valid;
3. VAE lifecycle separation;
4. VAE tiling/slicing/chunking;
5. group/model CPU offload;
6. transfer overlap with pinned memory where measured;
7. attention memory reduction through exact kernels;
8. allocator/fragmentation fixes;
9. two-A5500 model/context/sequence sharding if available.

For speed, investigate exact attention, compilation/fusion, graphs, GEMM/layout, and launch/transfer elimination before low-bit experiments.

## If a quantized candidate fails quality

Discard it. Do not:

- weaken the evaluator;
- remove hard prompts from the benchmark;
- change seeds;
- round the score more coarsely;
- claim the difference is visually irrelevant without a predeclared benchmark policy.

You may later test a more selective quantization scheme, but it starts from the same baseline quality contract.
