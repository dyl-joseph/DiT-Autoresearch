# Metric Schema

The harness is deliberately model-agnostic. Your project emits JSON files with a small stable contract.

## `perf.json`

Any nested numeric fields are allowed. The configured objective and guardrails address them with dotted paths.

Recommended schema:

```json
{
  "latency": {
    "median_s": 7.832,
    "p95_s": 7.991,
    "mean_s": 7.851,
    "samples_s": [7.81, 7.85]
  },
  "throughput_items_s": 0.1277,
  "peak_vram_mb": 21984.0,
  "persistent_vram_mb": 18720.0,
  "cold_start_s": 14.7,
  "compile_s": 0.0,
  "phases": {
    "text_encoder_s": 0.21,
    "denoiser_s": 7.11,
    "vae_s": 0.49
  },
  "metadata": {}
}
```

Only numeric fields can be objectives/guardrails.

### Recommended required performance fields

For A5500 DiT work, emit at least:

- `latency.median_s`
- `latency.p95_s`
- `peak_vram_mb`

Add throughput and cold-start if they matter to the product.

## `quality.json`

Recommended schema:

```json
{
  "metrics": {
    "benchmark_accuracy": 0.8732,
    "prompt_adherence": 0.8121
  },
  "cases": {
    "case_001": {"passed": true},
    "case_002": {"passed": true}
  },
  "metadata": {}
}
```

The quality contract names the metrics it cares about. Extra metrics are retained but do not decide the gate unless listed.

### Metric direction

For higher-is-better metrics:

```json
"benchmark_accuracy": {"direction": "higher"}
```

For lower-is-better metrics:

```json
"error_rate": {"direction": "lower"}
```

### No-regression rule

Set both:

```json
"max_abs_regression": 0.0,
"max_rel_regression": 0.0
```

Quantization forces this behavior even if the general contract is looser.

## Case-level passes

Use cases for benchmark properties that should never regress even if an aggregate metric increases.

Examples:

- text rendering critical prompt passes;
- identity preservation case passes;
- control/pose adherence passes;
- safety/format invariants pass;
- video temporal consistency threshold passes.

If a case passed in the baseline and fails in the candidate, the candidate is rejected when `require_case_passes=true`.
