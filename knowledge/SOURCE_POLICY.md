# Source and Evidence Policy

This research system inherits the evidence discipline used in the compiled handbook and the Wafer AI GPU performance resource map.

## Preferred evidence

Prefer, in order:

1. the paper introducing a mechanism;
2. official NVIDIA/CUDA/PyTorch/Diffusers/runtime documentation;
3. the repository implementing the mechanism;
4. a direct implementer report with code and enough measurement detail to reproduce it.

Avoid turning generic summaries, marketing claims, or unsupported leaderboard numbers into experiment expectations.

## Performance-claim minimum

A claimed win must identify:

- hardware;
- software/runtime versions;
- workload shapes/distribution;
- precision and algorithm;
- baseline;
- correctness/quality method.

For this package, that minimum is extended with model revision, denoising schedule/steps, prompt/seed corpus, and A5500 topology.

## Hardware portability

Transfer **mechanisms**, not multipliers.

For example, FlashAttention's IO-aware exact-attention principle is relevant across GPU generations, but an H100 speedup number is not an A5500 result. Re-measure on the A5500 and exact DiT shapes.

## Adding new sources

A new source should answer:

1. What mechanism does it teach?
2. Why is it primary or directly reproducible evidence?
3. Which current handbook section does it refine?
4. Does it apply to Ampere/A5500, or is it architecture context only?
5. Does it introduce a quality-sensitive approximation?

Do not modify the fixed benchmark contract to make a new technique look favorable.
