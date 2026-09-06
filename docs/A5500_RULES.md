# RTX A5500 / Ampere Rules

The detailed hardware material is in the handbook. These are the loop-level rules.

## Treat as target-native

- FP16 Tensor Cores;
- BF16 Tensor Cores when supported by the model/runtime;
- TF32 only when the reference arithmetic policy permits it;
- FlashAttention-2 on supported Ampere shapes;
- PyTorch SDPA;
- CUDA 11/12-era Ampere kernels;
- `torch.compile`/Inductor/Triton paths that support sm_86;
- CUDA Graphs;
- cuBLAS/cuDNN/CUTLASS kernels appropriate to Ampere;
- async global-to-shared copy mechanisms supported on Ampere;
- PCIe 4.0 and optional A5500 NVLink topology.

## Treat as architecture context, not direct target plan

- Hopper TMA-specific kernels;
- FlashAttention-3 Hopper-only scheduling;
- Blackwell FlashAttention-4-specific scheduling;
- native FP8 Tensor Core claims;
- MXFP8/MXFP4/NVFP4 execution claims;
- Blackwell tensor memory;
- NVSwitch topology assumptions.

## Profiling priority

On A5500, ask:

- Is the GPU actually busy?
- Is attention or GEMM dominating?
- Are pointwise/norm kernels fragmented?
- Are there H2D copies inside denoising steps?
- Are layout conversions consuming time?
- Are kernels Tensor-Core eligible?
- Is register pressure causing spills or suppressing useful concurrency?
- Does CUDA Graph capture fit within 24 GB?
- Is a second GPU communication-bound?

Never infer the answer from a generic architecture article when Nsight can measure the actual workload.
