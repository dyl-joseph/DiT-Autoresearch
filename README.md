# DiT AutoResearch — RTX A5500 / Ampere

A Karpathy-style autonomous research loop for **Diffusion Transformer inference engineering**, specialized for **NVIDIA RTX A5500 (Ampere)** and a **hard no-quality-regression contract**.

This package turns the previously compiled *Diffusion Transformer Inference Engineering* handbook into an agent-usable research system rather than a static reference. It bundles the full handbook under `knowledge/handbook/`, indexes it locally, locks the benchmark/evaluation harness, establishes a reference baseline, and gives an autonomous coding agent a repeatable loop for proposing, implementing, measuring, keeping, or reverting performance changes.

The design is inspired by `karpathy/autoresearch`, but the objective is different. Training autoresearch can use a fixed wall-clock budget and a scalar validation metric. Inference optimization needs a **fixed workload**, **fixed quality benchmark**, **hardware-local measurement**, and usually several performance dimensions. Here, quality is a constraint; speed, latency, throughput, memory, cold start, or another chosen metric is the optimization objective.

## Core contract

1. **Target hardware is RTX A5500 / Ampere.** Hopper- and Blackwell-only techniques are references, not execution plans.
2. **Quality is non-negotiable.** A candidate is never kept if it fails the configured quality contract.
3. **Quantization is allowed only as an experiment and gets the strongest gate.** If any required benchmark accuracy/quality metric is lower than the baseline, the quantized candidate is rejected.
4. **The evaluator is immutable.** Benchmark cases, quality evaluation, performance measurement, and fixed harness files are hashed at initialization. A candidate that changes them is invalid.
5. **The first run is the baseline.** Performance candidates are compared with the current best; quality is always compared with the original reference baseline.
6. **One hypothesis per commit.** Keep diffs reviewable and causal.
7. **Measure on the target GPU.** Performance multipliers from H100/B200 or unrelated workloads do not count as evidence for A5500.
8. **Prefer exact optimizations first.** FA2/SDPA backend selection, compile/fusion, CUDA Graphs, layout work, component lifecycle management, memory placement, kernel selection, and exact sharding precede quality-sensitive approximations.

## Why this structure

The package preserves the useful parts of the autoresearch pattern:

- a human-authored `program.md` that defines the autonomous research organization;
- a small explicit set of files the agent may modify;
- a fixed evaluator the agent may not modify;
- baseline-first experimentation;
- experiment logs that survive discarded commits;
- commit/benchmark/keep-or-revert iteration;
- autonomous continuation until the operator stops the agent.

For DiT inference, the package adds:

- an A5500 hardware contract;
- a strict quality firewall;
- a local Markdown knowledge-search tool over the entire handbook;
- performance guardrails;
- immutable-harness hashing;
- experiment classification (`exact`, `numerical`, `approximate`, `quantization`);
- hardware/environment capture;
- profiler-oriented research prompts;
- explicit rules for multi-objective optimization.

## Package layout

```text
DiT_AutoResearch_A5500/
├── README.md
├── SKILL.md                       # compact agent skill entry point
├── program.md                     # autonomous research loop instructions
├── pyproject.toml                 # installs the `dit-ar` CLI
├── config/
│   ├── autoresearch.example.json
│   └── quality_contract.example.json
├── dit_autoresearch/
│   ├── cli.py                     # doctor/init/index/search/baseline/experiment/summary
│   ├── config.py
│   ├── gates.py                   # strict quality + performance decisions
│   ├── gitutils.py                # scope and immutable-harness checks
│   ├── hardware.py                # A5500/environment capture
│   ├── knowledge.py               # SQLite FTS5/fallback Markdown search
│   ├── results.py
│   ├── runner.py
│   └── schemas.py
├── docs/
│   ├── INTEGRATION.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── METRIC_SCHEMA.md
│   ├── QUANTIZATION_POLICY.md
│   ├── A5500_RULES.md
│   └── TROUBLESHOOTING.md
├── prompts/
│   ├── hypothesis_generator.md
│   ├── profiler.md
│   ├── experiment_critic.md
│   └── quality_judge.md
├── templates/
│   ├── bench_perf.example.py
│   ├── eval_quality.example.py
│   ├── smoke_test.example.py
│   ├── perf.example.json
│   ├── quality.example.json
│   └── benchmark_cases.example.jsonl
└── knowledge/
    ├── INDEX.md
    ├── AUTORESEARCH_DESIGN_NOTES.md
    ├── SOURCE_POLICY.md
    └── handbook/                   # full compiled DiT inference handbook
```

## Quick start

The package is a research harness, not a model implementation. Integrate it into the repository that actually runs your DiT.

### 1. Install the CLI

From this folder:

```bash
python -m pip install -e .
```

No runtime Python dependencies are required beyond the standard library. Your target project still needs PyTorch/Diffusers/etc.

### 2. Copy the control files into the target repo

Recommended target layout:

```text
my-dit-project/
├── program.md
├── SKILL.md
├── config/
│   ├── autoresearch.json
│   └── quality_contract.json
├── knowledge/
│   └── handbook/ ...
├── benchmarks/ ...                # fixed
├── eval/ ...                      # fixed
├── scripts/
│   ├── bench_perf.py              # fixed
│   ├── eval_quality.py            # fixed
│   └── smoke_test.py              # fixed
└── src/ ...                       # only listed mutable paths may change
```

You can copy this entire package into the target repo, or keep it beside the repo and point the config at the target root.

### 3. Implement the two required fixed commands

`commands.perf` must write:

```text
<run_dir>/perf.json
```

`commands.quality` must write:

```text
<run_dir>/quality.json
```

See `docs/METRIC_SCHEMA.md` and the templates.

The loop **refuses to treat an experiment as valid without a quality output**.

### 4. Define the quality contract

Start from `config/quality_contract.example.json`.

For your stated requirement, keep:

```json
{
  "strict_zero_regression": true
}
```

and set every required metric to zero allowed regression. Quantization also forces zero regression regardless of a looser general contract.

### 5. Define mutable and fixed paths

In `config/autoresearch.json`:

- `mutable_paths`: only code/config the research agent may change;
- `fixed_paths`: benchmark corpus, quality evaluator, performance harness, and any immutable reference code.

After `dit-ar init`, hashes of fixed files are locked for the session. The active autoresearch config and quality contract are also locked automatically.

### 6. Create a research branch

```bash
git checkout -b autoresearch/a5500-latency-sep5
```

Use a separate branch per primary objective lane when practical:

- `...-latency`
- `...-vram`
- `...-throughput`
- `...-cold-start`

A single scalar objective is much easier for the keep/discard loop than trying to collapse all dimensions into one arbitrary score.

### 7. Initialize

```bash
dit-ar --config config/autoresearch.json doctor
dit-ar --config config/autoresearch.json init
```

`doctor` captures the GPU/software environment and warns if an A5500 is not detected. `init` locks the fixed harness and indexes every handbook Markdown file.

### 8. Establish baseline

```bash
dit-ar --config config/autoresearch.json baseline
```

This runs the fixed smoke/performance/quality suite. The baseline quality data remains the reference for the entire branch.

### 9. Launch your coding agent with `program.md`

For Codex/Claude/another repo-capable coding agent, use a prompt like:

```text
Read SKILL.md and program.md completely. Follow the setup checks, then begin the autonomous DiT inference research loop. Do not modify fixed evaluation files. Continue experimenting until I stop you.
```

The agent should query the compiled handbook before or during hypothesis generation:

```bash
dit-ar --config config/autoresearch.json search "A5500 attention backend long sequence FA2 SDPA"
dit-ar --config config/autoresearch.json search "torch.compile regional compilation denoising transformer"
dit-ar --config config/autoresearch.json search "24 GB VRAM offload text encoder VAE"
```

### 10. Evaluate each committed candidate

```bash
dit-ar --config config/autoresearch.json experiment \
  --classification exact \
  --description "switch transformer attention backend from eager math to FA2"
```

Exit status:

- `0`: **KEEP**
- `3`: **DISCARD / CRASH / GATE FAILURE**
- `2`: harness/config/setup error

If `KEEP`, advance from that commit. If discarded, the agent resets to the parent and tries another idea.

## The quality firewall

Quality is always compared with the original baseline. This prevents a sequence of tiny degradations from accumulating while each candidate only compares against the immediately previous candidate.

For quantization, the CLI sets `strict_zero_regression=True`. That means:

- a required higher-is-better metric must be **at least** the baseline value;
- a required lower-is-better metric must be **no worse** than the baseline value;
- every baseline-passing critical case must remain passing;
- a missing required metric/case is a failure.

This is intentionally stricter than most performance projects.

If a benchmark is noisy, solve the **measurement problem** instead of silently adding quality tolerance. Increase samples, fix seeds, stabilize evaluation, and define a statistically defensible contract before the research session begins.

## A5500-specific research priority

The bundled handbook gives the complete reasoning. The short default queue is:

1. establish stable shapes and a trustworthy baseline;
2. profile end-to-end with Nsight Systems;
3. identify top GPU-time families and CPU launch gaps;
4. select exact attention implementation for actual shapes (FA2/SDPA on Ampere);
5. test `torch.compile` / regional compile;
6. remove graph breaks and fuse repeated pointwise/norm/modulation work;
7. test CUDA Graphs when shapes are stable and memory permits;
8. optimize repeated GEMM shapes and layouts;
9. eliminate H2D/D2H movement inside denoising steps;
10. optimize text encoder / denoiser / VAE residency lifecycle for 24 GB;
11. use a second A5500 only when topology/communication measurements justify it;
12. investigate approximations only after exact paths are exhausted;
13. investigate quantization only if it passes the full no-regression quality suite.

Do **not** spend research cycles trying to port Hopper TMA, FlashAttention-3-only paths, Blackwell FP4/MX formats, or B200-specific kernels to an A5500 as if they were native performance options.

## Knowledge system

The full compiled documentation is not merely included; it is searchable from the loop.

```bash
dit-ar search "activation memory attention sequence length"
```

The index uses SQLite FTS5 when available and falls back to a dependency-free lexical scorer otherwise. Experiment prompts instruct the agent to use knowledge search to support hypotheses and to read the relevant source files before making architecture-specific changes.

Recommended mandatory documents are listed in `knowledge/INDEX.md`.

## What the loop does *not* automate

The CLI intentionally does not contain an embedded LLM API or vendor-specific agent SDK. This follows the spirit of the original autoresearch project: the **Markdown program is the agent skill**, and your coding agent is the autonomous researcher.

This makes the harness usable with:

- Codex-style coding agents;
- Claude Code-style agents;
- local agents;
- custom multi-agent supervisors;
- a human researcher using the exact same experiment protocol.

The only project-specific work you must supply is the real model integration and the benchmark/evaluation commands.

## Safety against benchmark hacking

The loop rejects candidate commits that modify files outside `mutable_paths`, and it independently hashes `fixed_paths`. Put all of the following under fixed paths:

- benchmark prompts/seeds/cases;
- evaluation metric code;
- performance measurement code;
- reference output manifests;
- any scripts that select easier cases;
- quality thresholds.

If the agent needs to improve the benchmark harness, end the current research session, review that change as a human, deliberately establish a new baseline, and start a new branch/session.

## Source lineage

This package combines three layers:

1. the compiled DiT inference handbook created from the uploaded *Inference Engineering* material plus later refinement;
2. the performance-engineering evidence discipline and primary-source orientation from `wafer-ai/gpu-perf-engineering-resources`;
3. the autonomous experiment-loop design from `karpathy/autoresearch`.

The handbook itself contains its own source map and reading list. See `knowledge/SOURCE_POLICY.md` for how new evidence should enter the research system.

## Recommended first session

Use a **latency lane** first, with a strict peak-VRAM guardrail. The first 10–20 hypotheses should be exact implementation changes only. Run Nsight before custom kernel work. Keep a separate memory lane if 24 GB is the immediate blocker.

The loop is designed to keep learning from failures: discarded experiments stay in `.autoresearch/results.tsv` and their logs remain in `.autoresearch/runs/`, even when the code commit is reset.
