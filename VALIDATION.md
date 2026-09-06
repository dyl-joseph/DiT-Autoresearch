# Validation Report

The package was validated before delivery with standard-library tests and a synthetic end-to-end git research session.

## Unit tests

`./scripts/run_tests.sh` checks:

- strict zero-regression quality gate rejects any lower required metric;
- critical case regressions are rejected even when aggregate quality improves;
- performance objective + relative guardrails behave as configured;
- the Markdown knowledge index/search returns relevant documents.

## End-to-end harness test

A temporary git repository was created with:

- one mutable runtime configuration file;
- fixed smoke/performance/quality scripts;
- a fixed benchmark/eval directory;
- a quality metric named `accuracy`;
- latency as the active objective.

The test sequence verified:

1. `dit-ar init` locked fixed-file hashes and indexed knowledge;
2. `dit-ar baseline` stored the original performance and quality reference;
3. an exact candidate improving latency from 10 to 9 while preserving quality at 0.90 was **KEPT**;
4. a quantized candidate improving latency from 9 to 8 and VRAM from ~20 GB to 15 GB but lowering accuracy from 0.90 to 0.89 was **DISCARDED-QUALITY**;
5. the quantized failure also created a new critical-case failure and was rejected even though its performance gate passed;
6. results were retained in `.autoresearch/results.tsv`.

This synthetic test does not validate real A5500 performance; it validates the research control logic and, specifically, the no-quality-regression firewall.

## What must still be validated in the user's target repo

The package cannot validate these without the actual model repository and GPU:

- benchmark/evaluation correctness;
- A5500 kernel support for the model's exact shapes;
- real latency/VRAM/throughput variance;
- Nsight profiler conclusions;
- multi-GPU topology and scaling;
- model-specific benchmark accuracy.

`dit-ar doctor`, `baseline`, and a first profiler pass should therefore be run on the actual A5500 host before autonomous research begins.
