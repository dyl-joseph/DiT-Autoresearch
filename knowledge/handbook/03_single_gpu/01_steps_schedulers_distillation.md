# Steps, Schedulers, Solvers, and Distillation: The Largest Speed Lever

## Why step count dominates

A diffusion/flow pipeline repeatedly evaluates the expensive denoiser. If one transformer evaluation costs `t`, then denoiser time is approximately:

`T_denoise ~= evaluations * t`

This is why reducing evaluations often beats low-level optimization. A 20% faster kernel cannot compete with cutting 50 transformer calls to 10 if quality remains acceptable.

The uploaded PDF explicitly notes that image generation time tracks roughly linearly with denoising step count and discusses few-step models that generate with single-digit steps. This is a first-order lever, but it trades speed against output quality and sometimes diversity/prompt adherence.

## Count model evaluations, not UI “steps”

Different pipelines can perform different numbers of transformer forwards per nominal step:

- classic classifier-free guidance may execute conditional and unconditional branches separately
- some implementations concatenate those branches into one larger batch
- some distilled models are guidance-free
- predictor-corrector or special solvers may evaluate more than once per step
- control branches may add extra model work

Benchmark **number of denoiser forwards** and total GPU time, not only `num_inference_steps`.

## Three ways to need fewer steps

### 1. Better numerical scheduler/solver

For the same trained model, a stronger scheduler may achieve similar perceptual quality with fewer evaluations. The best scheduler is checkpoint-dependent. A scheduler optimized for one prediction parameterization or noise schedule can degrade another model.

Test a grid such as:

- model default/reference scheduler
- Euler-family variants
- DPM-style multistep methods when supported
- flow-matching-specific schedulers for flow models
- model-author recommended shifts or dynamic schedules

Do not change scheduler and step count simultaneously when establishing causality.

### 2. Training/distillation for few-step inference

A model can be trained or distilled specifically for low-step generation. Common families include:

- latent consistency / consistency-style models
- progressive distillation
- adversarial diffusion distillation
- trajectory/flow distillation
- student models trained to match multi-step teacher trajectories

These alter the quality-speed Pareto frontier rather than merely walking along it.

### 3. Adaptive step allocation

Not all requests require identical compute. Potential production policies include:

- preview mode: 4-8 steps
- interactive mode: 8-16 steps
- final render: 20-50 steps
- difficult prompts or edit tasks: more steps only after a quality predictor detects risk

A service can expose quality tiers rather than forcing every request to pay the worst-case cost.

## Step sensitivity is nonuniform

Early denoising steps tend to establish broad structure and semantics; later steps refine details. This observation supports several optimizations:

- keep the whole trajectory in FP16/BF16 by default; if precision reduction is ever tested, restrict it to candidate late-step regions and reject it on any required benchmark regression
- disable or weaken CFG later
- cache more aggressively in redundant middle/late regions
- use step-dependent thresholds

Do not assume “later is always less important.” Fine text, faces, motion consistency, and edit boundaries can remain sensitive late in the trajectory.

## Scheduler overhead can become visible after compilation

When the denoiser is slow, Python scheduler overhead looks negligible. After a successful `torch.compile` + attention optimization stack, tiny CPU/GPU synchronizations in the scheduler can become measurable.

Watch for:

- tensor indexing that synchronizes with CPU
- `.item()` calls
- repeated dtype/device conversions of sigma/timestep arrays
- dynamic branches that break the compiled graph

A healthy highly optimized pipeline keeps scheduler work small and avoids per-step synchronization.

## A step-count experiment

For a fixed seed/prompt set, sweep:

`[4, 6, 8, 12, 16, 20, 28, 40, 50]`

For each point, record:

- end-to-end latency
- denoiser latency
- quality score(s)
- human preference rate against baseline
- prompt adherence
- text rendering accuracy if relevant
- temporal consistency for video
- failure rate

Plot quality vs GPU-seconds, not just quality vs steps.

The knee of the curve is often the best production point.

## Distillation decision tree

### Use a few-step checkpoint when:

- interactive latency is a primary requirement
- the task tolerates modest quality/style change
- the few-step variant has model-specific ecosystem support
- the quality gate passes on your prompts

### Keep the full-step model when:

- fidelity or prompt adherence is the differentiator
- control/editing strength depends on the full trajectory
- few-step variants alter style or diversity too much
- you can hide latency through offline/asynchronous execution

### Serve both when:

- users need previews and final renders
- your workload naturally has “draft” and “final” stages
- accepting a draft can cancel an expensive full render

## Resolution-dependent step count

Do not assume the same optimal step count at all resolutions. Higher resolution can expose detail/refinement defects that are invisible at 512px. Conversely, a heavily downscaled thumbnail may not benefit from the last half of the trajectory.

Create a step policy indexed by:

- model
- resolution bucket
- task (T2I, I2I, edit, control)
- guidance strength
- quality tier

## Video-specific considerations

For video, each denoiser evaluation is much more expensive, so step reduction is especially valuable. But fewer steps can amplify:

- temporal flicker
- motion discontinuity
- identity drift
- weak physical consistency

Use video quality gates, not framewise image metrics only. A method that looks good on sampled frames may fail in motion.

## Interaction with caching

Step reduction and cache acceleration are not automatically additive. If you cut 50 steps to 12:

- neighboring timesteps are farther apart
- residual similarity may decrease
- a cache threshold tuned for 50 steps may miss more often or create larger approximation error

Retune cache intervals/thresholds after changing scheduler or step count.

## Interaction with compilation

Fewer steps increase the relative importance of cold-start and compile cost. A 5-second compilation amortizes easily over thousands of 50-step generations but can dominate a short-lived 4-step interactive worker. Use regional compilation and engine caches for low-step models with frequent cold starts.

## Practical recommendation

Before hand-picking CUDA kernels, answer this:

> Is the chosen checkpoint/scheduler doing the minimum number of transformer evaluations required by the product's quality bar?

If the answer is unknown, low-level tuning is premature.

## Source basis

**PDF-derived:** linear relationship between image generation time and step count; few-step/latent-consistency/distillation approaches; early steps establish broad structure.  
**Expansion:** evaluation protocol, quality tiers, scheduler overhead, interactions with cache/compile, video-specific gates.

## RTX A5500 scope override

For this project, interpret every optimization in this chapter through these fixed constraints:

- target hardware is RTX A5500 / Ampere with 24 GB GDDR6;
- prefer FP16/BF16, PyTorch SDPA/FlashAttention-2, `sm_86` compilation, fusion, static shapes, and exact offload/sharding;
- Hopper/Blackwell FP8/MXFP8/FP4 and FlashAttention-3/4 are not target execution paths;
- quantization and other approximate changes are accepted only if the required benchmark suite shows no accuracy regression.

See [../00_foundations/04_target_hardware_and_quality_contract.md](../00_foundations/04_target_hardware_and_quality_contract.md).
