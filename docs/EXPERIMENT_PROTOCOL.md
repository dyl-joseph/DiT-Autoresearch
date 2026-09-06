# Experiment Protocol

## Experiment unit

One experiment is one committed hypothesis evaluated against one fixed benchmark/evaluation suite.

Do not combine unrelated changes. If two changes are individually good, combine them in a later explicit interaction experiment.

## Baseline

The baseline is the exact unmodified code at session start. It supplies:

- permanent quality reference;
- initial performance best;
- environment snapshot;
- benchmark/evaluator version.

## Candidate decision

A candidate can be kept only when all three are true:

1. immutable harness still matches the initialized hashes;
2. quality gate passes against the original baseline;
3. branch-local primary performance metric improves enough against current best and all performance guardrails pass.

## Why quality compares to original baseline

Suppose each of ten candidates loses 0.1% quality but is compared only to the previous candidate. The final system can lose ~1% while every local change looks small. Original-baseline comparison prevents this ratchet.

## Why performance compares to current best

The branch should monotonically improve its chosen objective. Comparing to the original baseline would allow a worse-than-current candidate to be kept just because it is still better than day zero.

## Measurement noise

Do not use a microscopic performance win as evidence if run-to-run variance is larger than the claimed improvement. Set `min_improvement_fraction` based on actual baseline variance.

For quality, the stated policy is stricter. If the benchmark is noisy, improve the benchmark's sampling and statistical stability before running autonomous research. Do not quietly introduce an accuracy-loss allowance.

## Failed experiments are valuable

Keep `.autoresearch/` outside the git reset path. It stores:

- command logs;
- perf/quality JSON;
- gate decisions;
- environment snapshot;
- results TSV.

This prevents the agent from cycling through the same failed mechanism without memory.

## Interaction experiments

After two independent exact optimizations are kept, they are already combined because the branch advances. For two discarded near-misses that may be synergistic, create a deliberate combined experiment only when you can state a plausible interaction mechanism.

Examples:

- compile may make a manual pointwise fusion redundant;
- CUDA Graphs may matter more after shape bucketing;
- offload may become less expensive after text-encoder eviction;
- a different attention backend may change the best compile boundary.
