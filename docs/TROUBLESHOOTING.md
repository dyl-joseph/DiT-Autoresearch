# Troubleshooting

## `doctor` says A5500 not detected

Run on the target host. The package can be prepared elsewhere, but performance conclusions are valid only on the target machine.

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## `baseline` says fixed hash manifest missing

Run:

```bash
dit-ar --config config/autoresearch.json init
```

from a clean committed state.

## Candidate rejected for out-of-scope path

The last commit changed a path not under `mutable_paths`. Split the commit, revert unrelated files, or deliberately start a new session with a reviewed scope change.

## Candidate rejected because fixed harness changed

This is intentional. Stop and revert the benchmark/evaluator change. If the harness itself needs improvement, review it separately and start a new baseline/session.

## Perf/quality command succeeds but CLI says JSON missing

The command must write exactly:

- `{run_dir}/perf.json`
- `{run_dir}/quality.json`

Use the `{run_dir}` placeholder in config commands.

## Quantization never gets kept

If it improves speed/memory but quality is lower, this is expected under the project's contract. Try exact capacity/performance techniques or a different quantization design. Do not relax the benchmark automatically.

## Tiny performance wins oscillate keep/discard

Your performance benchmark is likely too noisy or `min_improvement_fraction` is too small. Increase samples, stabilize clocks/temperature/load, and use a meaningful minimum improvement.

## Nsight tools missing

Install Nsight Systems/Compute on the target workstation or profile with available PyTorch tools first. The loop still works without Nsight, but custom kernel research is much less grounded.

## SQLite FTS5 unavailable

`dit-ar search` automatically falls back to a dependency-free lexical scorer.
