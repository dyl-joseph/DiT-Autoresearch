# DiT AutoResearch Program — RTX A5500 / Ampere

This program defines an autonomous research loop for making a Diffusion Transformer inference workload faster, more memory-efficient, or more production-efficient **without lowering required benchmark quality**.

It is inspired by the control pattern of `karpathy/autoresearch`: the human writes this Markdown program; the coding agent edits a deliberately limited set of implementation files; a fixed evaluator decides whether a committed experiment advances the branch.

The project-specific override is strict:

> **A candidate that lowers any required benchmark accuracy/quality metric is not an optimization. Quantization is rejected on any such regression.**

## 0. The research objective

This branch has exactly one primary performance lane configured in `config/autoresearch.json`:

- latency, or
- throughput, or
- peak VRAM, or
- cold-start/compile/startup, or
- another explicitly configured scalar metric.

Quality is not part of a weighted score. Quality is a hard gate.

Secondary performance metrics are guardrails. A latency win that causes an unacceptable VRAM or p95 regression is not kept.

## 1. Setup a fresh run

Work with the human only for initial setup facts that cannot be inferred from the repo. Once the experiment loop begins, continue autonomously until stopped.

1. **Choose a run tag and objective lane.** Use a branch such as:

   ```text
   autoresearch/sep5-a5500-latency
   autoresearch/sep5-a5500-vram
   ```

   Never reuse an old research branch as a "fresh" run.

2. **Read the control files completely:**

   - `README.md`
   - `SKILL.md`
   - `program.md`
   - `config/autoresearch.json`
   - `config/quality_contract.json`

3. **Read the mandatory local knowledge:**

   - `knowledge/handbook/00_foundations/04_target_hardware_and_quality_contract.md`
   - `knowledge/handbook/00_foundations/02_benchmarking_and_quality_gates.md`
   - `knowledge/handbook/03_single_gpu/08_a5500_optimization_ladder.md`
   - `knowledge/handbook/02_bottlenecks/05_nsight_a5500_profiling_workflow.md`
   - `knowledge/handbook/09_reference/09_a5500_benchmark_runbook.md`
   - `knowledge/handbook/09_reference/10_reproducibility_and_claims_standard.md`

4. **Inspect the target repo.** Identify:

   - model/pipeline architecture;
   - actual mutable paths;
   - benchmark commands;
   - quality benchmark suite;
   - model revision/checkpoint;
   - fixed prompt/seed/schedule corpus;
   - dtype and attention backend;
   - single- or dual-A5500 topology.

5. **Verify the machine:**

   ```bash
   dit-ar --config config/autoresearch.json doctor
   ```

   If A5500 is not detected, do not interpret performance numbers as A5500 results.

6. **Initialize immutable state:**

   ```bash
   dit-ar --config config/autoresearch.json init
   ```

   This hashes every configured `fixed_path`, plus the active config and quality contract, and builds the handbook index.

7. **Establish the baseline before editing anything:**

   ```bash
   dit-ar --config config/autoresearch.json baseline
   ```

8. **Record the initial bottleneck model.** If an Nsight Systems trace does not already exist for this code/hardware/workload, profiling should be among the first experiments or pre-experiment diagnostics.

## 2. What you CAN change

You may modify only `mutable_paths` from the active config.

Good mutable targets include:

- model runtime code;
- attention backend selection;
- compile boundaries;
- custom fused kernels;
- static shape bucketing logic;
- layout/contiguity transformations;
- component loading/offload/eviction policy;
- pipeline scheduling;
- model-parallel implementation;
- safe caching of invariant tensors or prompt embeddings;
- kernel launch structure;
- CUDA Graph integration;
- project-owned inference engine configuration that is intentionally in scope.

Every candidate commit should represent one understandable research idea.

## 3. What you CANNOT change during a session

Do not modify anything in `fixed_paths`, including:

- benchmark prompts;
- seeds;
- image/video dimensions;
- denoising steps used by the benchmark;
- benchmark scheduler unless scheduler changes are the explicit fixed benchmark subject and the evaluator is designed for it;
- quality metric implementation;
- reference thresholds;
- sample selection;
- performance timing code;
- warmup/iteration counts;
- profiler case selection when it defines the measured target;
- scripts that make the benchmark easier.

Do not install dependencies merely to force an experiment through unless dependency changes are explicitly approved and part of the session contract. A dependency upgrade can change kernels, compilers, and quality behavior so substantially that it often deserves a new baseline/session.

The CLI hashes fixed paths and rejects a candidate if they changed.

## 4. Quality rules

### 4.1 Reference

The unmodified baseline quality output is the permanent reference for this branch.

Do **not** compare quality only to the last kept candidate. That allows accumulated drift.

### 4.2 Required metrics

Every metric listed in `config/quality_contract.json` must be emitted by every candidate evaluation.

If a metric is missing, the candidate fails.

### 4.3 Required cases

Every baseline-passing critical case must continue passing. A new critical failure rejects the candidate even if aggregate quality improves.

### 4.4 Quantization firewall

If the experiment changes weight/activation/attention/storage precision into a quantized/low-bit path, classify it as:

```text
quantization
```

The harness then enforces zero allowed quality regression.

Do not relabel quantization as `exact` to bypass the gate.

If a quantized candidate is faster or fits into 24 GB but lowers a required accuracy metric, **discard it**. Solve the capacity/performance problem through exact methods or search for a different quantization design that passes the full benchmark.

### 4.5 No benchmark laundering

Never change the evaluator because a promising optimization fails it. A failed quality gate is evidence about the optimization.

If the human later decides the benchmark was wrong, that is a separate benchmark-engineering task followed by a new baseline and research session.

## 5. A5500 / Ampere hardware policy

The target is RTX A5500, not H100/B200.

Operational consequences:

- use FP16/BF16 Tensor Core paths as the normal low-precision execution modes;
- FlashAttention-2 and PyTorch SDPA are relevant attention candidates;
- FlashAttention-3 Hopper-specific scheduling is not a target-native plan;
- FlashAttention-4/Blackwell-specific paths are not a target-native plan;
- FP8/MXFP8/FP4 native Tensor Core speedups are not an A5500 plan;
- Hopper TMA and Blackwell tensor-memory techniques may teach principles but should not consume implementation cycles as direct A5500 features;
- compile custom CUDA for the actual A5500 capability reported by the machine;
- pay close attention to global-memory traffic, coalescing, register pressure, shared memory, Tensor Core utilization, launch gaps, and PCIe/NVLink topology.

Before proposing a hardware-sensitive experiment, search local knowledge with the actual mechanism and A5500/Ampere terms.

## 6. Evidence discipline

A performance claim is only meaningful when the run fixes or records:

- exact GPU and GPU count;
- topology;
- driver/CUDA/PyTorch/runtime versions;
- model revision;
- workload shape;
- prompt/token lengths when relevant;
- width/height/frames;
- denoising steps;
- batch/concurrency;
- precision;
- attention/kernel backend;
- compile/CUDA Graph state;
- baseline;
- quality/correctness method.

Never copy a speedup multiplier from another GPU into `results.tsv` as if it were measured evidence.

## 7. Knowledge-backed hypothesis generation

The full compiled handbook is available locally.

Before each major hypothesis family, search it:

```bash
dit-ar --config config/autoresearch.json search "<query>"
```

Useful query patterns:

```text
A5500 attention FA2 SDPA head dimension sequence length
compile regional compile transformer blocks graph breaks
CUDA graphs stable shapes diffusion denoising
RMSNorm modulation fusion pointwise kernels
GEMM arithmetic intensity Ampere Tensor Cores
24 GB VRAM text encoder eviction VAE tiling offload
video DiT attention context parallelism two GPU
Nsight launch gaps H2D denoising step
```

Read the top relevant files before implementation.

The handbook is a hypothesis generator, not a substitute for measurement. A mechanism can be theoretically appropriate and still lose on your exact shape mix.

## 8. Experiment classes

Assign every experiment exactly one class:

### `exact`

Intended algorithm unchanged, with ordinary floating-point reordering permitted.

Examples:

- SDPA ↔ FA2 exact attention;
- kernel fusion;
- `torch.compile`;
- CUDA Graphs;
- layout elimination;
- component offload/eviction;
- model sharding;
- static shape buckets;
- safe invariant caching.

### `numerical`

Arithmetic behavior changes without an explicit algorithmic shortcut.

Examples:

- FP16 ↔ BF16;
- TF32 enable/disable on a path that previously used FP32 semantics;
- alternate accumulation precision.

### `approximate`

The model computation/trajectory is intentionally changed or skipped.

Examples:

- fewer denoising steps;
- scheduler changes that alter trajectory;
- guidance cutoff;
- timestep/block/hidden-state cache reuse that skips computation;
- token pruning;
- sparse approximate attention;
- distilled checkpoint substitution.

### `quantization`

Any low-bit weight/activation/attention/storage path. This gets the strictest quality rule.

## 9. The autonomous experiment loop

LOOP UNTIL THE HUMAN STOPS YOU:

### Step 1 — inspect state

```bash
git status --short
git log -5 --oneline
dit-ar --config config/autoresearch.json summary --limit 20
```

Know which commit is current best and which mechanisms already failed.

### Step 2 — diagnose, do not guess blindly

If bottlenecks are uncertain, profile or inspect the existing trace before editing.

Use this mental sequence:

1. end-to-end phase contribution;
2. GPU busy vs CPU/transfer gaps;
3. top GPU-time kernel families;
4. attention vs GEMM vs pointwise/norm vs VAE;
5. memory residency vs transient activations;
6. repeated shapes and launch structure;
7. inter-GPU communication if multi-GPU.

### Step 3 — retrieve relevant knowledge

Search the handbook and read the top matches.

### Step 4 — formulate one hypothesis

Write a short internal statement:

```text
Because <measured bottleneck>, changing <mechanism> should improve <primary metric>
by reducing <compute/memory traffic/launches/transfers/fragmentation/communication>,
without changing <quality-critical algorithm>.
```

Avoid compound experiments such as "enable compile + switch attention + change dtype". You will not know what caused the result.

### Step 5 — edit only mutable paths

Implement the smallest coherent candidate.

### Step 6 — inspect the diff

```bash
git diff
```

Check for accidental benchmark/eval changes and unrelated cleanup.

### Step 7 — commit the candidate

```bash
git add <mutable paths>
git commit -m "autoresearch: <hypothesis>"
```

The experiment harness evaluates the last commit and verifies scope.

### Step 8 — run the fixed experiment

```bash
dit-ar --config config/autoresearch.json experiment \
  --classification <exact|numerical|approximate|quantization> \
  --description "<concise hypothesis>"
```

All command output is written into `.autoresearch/runs/<run>/`; do not flood agent context with full benchmark logs unless diagnosing a failure.

### Step 9 — decide from the gate

#### KEEP

If the harness prints `KEEP`:

- quality passed against original baseline;
- primary performance objective improved enough versus current best;
- configured performance guardrails passed.

Stay on the commit and continue.

#### DISCARD

If it prints `DISCARD-QUALITY` or `DISCARD-PERFORMANCE`:

```bash
git reset --hard HEAD^
```

Do not delete `.autoresearch/`; logs/results are research memory.

#### CRASH

Read only the relevant tail of the failing log.

If the implementation has a trivial defect consistent with the hypothesis, fix, amend/commit, and rerun. If the failure is intrinsic (OOM, unsupported kernel shape, incompatible backend), record the failure and revert.

Do not spend many cycles forcing a fundamentally incompatible idea to work.

### Step 10 — learn from near misses

Use discarded results to generate next hypotheses:

- fast but too much VRAM → search memory/layout/graph-capture tradeoffs;
- lower VRAM but slower → search transfer/overlap/component residency;
- exact attention faster at one shape but slower overall → inspect shape distribution and backend dispatch;
- compile win with huge startup → consider regional compile/cache/static buckets;
- quantization fast but quality down → discard; explore exact capacity options or more selective precision only if benchmark parity can be proven;
- second GPU slower → inspect topology and collective fraction; reduce communication or abandon that decomposition.

## 10. A5500 exact-first hypothesis queue

Do not mechanically try these in order if profiling contradicts them, but use them as the default backlog.

### Phase A — measurement quality

- stabilize warmup and sample counts;
- ensure CUDA synchronization is correct;
- remove benchmark noise;
- validate peak allocated vs peak reserved memory;
- capture p50/p95 and end-to-end phase times;
- capture Nsight Systems trace.

### Phase B — attention

- compare current attention vs PyTorch SDPA backend(s);
- test FlashAttention-2 where shapes/dtypes are supported;
- remove attention layout conversions/copies;
- avoid materializing large intermediates;
- benchmark actual `(batch, heads, sequence, head_dim, dtype)` families rather than a synthetic favorite shape.

### Phase C — compiler/fusion/launch structure

- `torch.inference_mode()` correctness;
- `torch.compile` full vs regional repeated-block compile;
- eliminate graph breaks;
- fuse RMSNorm/AdaLN/modulation/activation/residual patterns where profiler shows meaningful total time;
- static shape buckets;
- CUDA Graph capture if repeated shapes and memory headroom justify it.

### Phase D — GEMM/layout

- identify dominant projection/MLP matrix shapes;
- eliminate unnecessary transpose/contiguous calls;
- compare library algorithm choices;
- inspect Tensor Core eligibility;
- use Triton/CUDA/CUTLASS only for demonstrated gaps.

### Phase E — 24 GB memory

- text encoder eviction after embeddings;
- prompt-embedding caching where semantically valid;
- VAE decode after denoiser temporary release;
- VAE tiling/slicing/chunking with quality validation;
- group/model CPU offload;
- pinned-memory prefetch only if overlap is measured;
- allocator/fragmentation work;
- second A5500 exact sharding if available.

### Phase F — multi-GPU

- measure `nvidia-smi topo -m`;
- distinguish PCIe-only vs A5500 NVLink setup;
- use a decomposition whose communication volume is justified by saved compute/memory;
- verify numerical equivalence and quality;
- reject scaling that improves fit but destroys the active objective unless this is the memory lane.

### Phase G — approximations

Only after exact opportunities are materially exhausted. Every approximation still runs the full quality gate before promotion.

### Phase H — quantization

Last, not first. Test only when there is a credible A5500 benefit (often residency/capacity rather than native FP8 compute). Quantization is kept only if it passes strict no-regression benchmark quality.

## 11. Profiling loop

Use Nsight Systems first for global attribution, then Nsight Compute for a narrow kernel family.

Do not optimize occupancy, tensor-core percentage, bandwidth, or any single counter in isolation. Optimize the product-level objective by fixing the measured bottleneck.

If a custom kernel becomes justified:

1. make an operator-level reference test;
2. benchmark the exact production shapes;
3. inspect numerical error;
4. profile register/shared-memory pressure;
5. integrate behind a dispatch/fallback;
6. run full model quality and performance suite.

## 12. Simplicity criterion

All else equal, prefer the simpler kept implementation.

Examples:

- deleting a redundant layout copy and winning 1% is excellent;
- adding 300 lines of fragile custom CUDA for 0.2% is probably not worth keeping;
- reducing VRAM with no latency/quality cost is valuable even if the primary lane is latency, but consider whether it belongs on a dedicated memory branch rather than complicating the latency branch.

The research agent should optimize maintainability as a soft criterion after all hard gates.

## 13. When to start a new session instead of continuing

Start a fresh baseline/branch if any of these materially change:

- GPU model/topology;
- CUDA/driver/PyTorch/runtime versions;
- model/checkpoint revision;
- benchmark corpus;
- quality evaluator;
- objective lane;
- major dependency/kernel library version;
- production dimensions/steps/batch policy.

Do not compare across incompatible baselines as though they were one experiment chain.

## 14. NEVER silently weaken the contract

If research stalls:

- read more handbook material;
- inspect traces again;
- test alternative exact implementations;
- revisit failed experiments for interactions;
- try a separate objective lane;
- write a better custom kernel only when evidence supports it;
- research new external primary sources if permitted.

Do **not** solve stagnation by relaxing quality thresholds, changing benchmark cases, or calling a quality regression "close enough".

Continue autonomous experimentation until the human interrupts the run.
