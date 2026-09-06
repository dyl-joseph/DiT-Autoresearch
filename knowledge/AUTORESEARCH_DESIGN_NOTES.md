# AutoResearch Design Notes

Source: `karpathy/autoresearch`, read from the repository's `README.md` and `program.md` in September 2026.

## Mechanisms adopted

The original project deliberately makes the autonomous research environment small:

- one fixed preparation/evaluation file;
- one agent-editable implementation file;
- one human-authored Markdown program that acts as the research skill;
- a mandatory unmodified baseline;
- one experiment per commit;
- a fixed evaluator;
- a results TSV outside the commit/revert chain;
- keep the commit when the metric improves, reset when it does not;
- autonomous continuation without asking the human between experiments.

The original training project uses a fixed five-minute training budget and `val_bpb` as the scalar objective. Those exact choices do **not** transfer to inference optimization.

## Adaptation for DiT inference

For inference research:

- fixed **workload and benchmark corpus** replace fixed training wall-clock as the comparability anchor;
- quality is a hard constraint;
- performance has a branch-local primary objective;
- peak VRAM/p95/other properties can be guardrails;
- fixed evaluation files are hashed so an agent cannot accidentally benchmark-hack;
- quality is compared to the original baseline for the full branch;
- performance is compared to the current best;
- knowledge search is part of hypothesis generation;
- hardware constraints are explicit because optimization results are architecture-sensitive.

## Why no embedded LLM client

The original repo's key idea is that `program.md` can be a lightweight skill consumed by whatever coding agent is operating in the repository. This package keeps that separation. The `dit-ar` CLI is the experiment judge and memory system; the external coding agent is the researcher.

This prevents coupling the research harness to a single model provider or agent SDK and keeps the benchmark/evaluation logic auditable.

## Inference-specific anti-patterns

Do not directly copy these training-loop assumptions:

- "fixed time budget implies comparability" — for inference, use the same requests/shapes/steps and stable warmup;
- "one scalar quality metric is enough" — generative media often needs multiple benchmark metrics plus case-level failures;
- "VRAM is a soft constraint" — on a 24 GB A5500, VRAM can be a hard feasibility constraint;
- "all architecture changes are fair game" — inference optimization must preserve the chosen model's benchmark capability unless a model change is explicitly in scope.
