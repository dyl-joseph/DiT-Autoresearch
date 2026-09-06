# Integration Guide

The AutoResearch harness must live close enough to the target model repository that it can enforce git scope, run the real benchmark commands, and search the bundled handbook.

## Recommended integration

Copy these into the target repository:

- `program.md`
- `SKILL.md`
- `config/`
- `knowledge/`
- optionally `prompts/`

Install the package code editable from this directory or vendor `dit_autoresearch/` and `pyproject.toml` into the target repo.

## Project-specific work you must do

### Performance command

Implement a fixed script that writes `<run_dir>/perf.json`.

It must measure the same production-representative cases every run. Use warmup, CUDA synchronization, repeated samples, and raw sample retention where possible.

### Quality command

Implement a fixed script that writes `<run_dir>/quality.json`.

It must run the benchmark suite that determines whether the model is acceptable. This is the most important project-specific component.

### Mutable paths

Narrow them aggressively. A good first session might allow only:

```json
[
  "src/inference.py",
  "src/attention.py",
  "src/kernels"
]
```

Do not list the entire repository unless truly necessary.

### Fixed paths

Include all benchmark and evaluator logic. Also include fixed model-wrapper code if changing it would make the benchmark incomparable.

## Separate research lanes

Use one branch/config per primary goal.

Example latency config:

```json
"objective": {
  "metric": "latency.median_s",
  "direction": "lower",
  "min_improvement_fraction": 0.005
}
```

Example VRAM config:

```json
"objective": {
  "metric": "peak_vram_mb",
  "direction": "lower",
  "min_improvement_fraction": 0.01
}
```

For the VRAM lane, latency should normally become a guardrail so the agent cannot "win" by moving the whole model to CPU.

## Benchmark-harness lifecycle

Once a session is initialized, fixed harness files, the active config, and the quality contract are hashed. If you need to improve the evaluator:

1. stop the current research session;
2. change/review the evaluator deliberately;
3. commit it;
4. create a fresh research branch;
5. run `dit-ar init` again;
6. establish a new baseline.

This is analogous to keeping the evaluator outside the agent-editable file in the original autoresearch pattern.
