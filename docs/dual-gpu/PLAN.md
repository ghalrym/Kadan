# Two-GPU execution plan and first slice

## Current state (2026-10-08)

The live MR119 service is unchanged at dea7b0f: Docker exposes only physical GPU1,
logical cuda:0, with a 22 GiB device budget. GPU0 is free (2 MiB baseline). The
single-card milestone is not two-GPU execution.

ResourceManager can atomically reserve a map of device budgets and maintains one
FIFO owner, but native_resident.py selects one device and reserves one arena.
model_worker.cpp accepts one device index, calls cudaSetDevice once, and constructs
one CUDA Model. The image adapter likewise chooses one CUDA device; component
CPU offload moves text_encoder → transformer → VAE on that device. The generic
packed placement policy does not make either executor tensor-parallel.

The pinned QwenImage21Transformer2DModel has neither _cp_plan nor _tp_plan.
Attention processors accepting a parallel_config do not supply the missing model
partition, masks, RoPE, prefix cache and residual routing. Older Qwen plans are not
interchangeable. No supported Qwen2.1 parallel flag is claimed.

Topology: nvidia-smi reports PHB between the two RTX3090s, not NVLink.
cudaDeviceCanAccessPeer(0,1) and (1,0) both returned success with accessible=0.
This was a capability query only, without tensor allocations or kernels. Actual
host-staged/NCCL bandwidth and collective latency are still unmeasured. They must
be measured in an agreed GPU1 maintenance window before selecting a production
communication pattern. Do not infer performance from combined VRAM capacity.

## Designs to compare

| Design | Single-image work on both cards | Tradeoff for this model |
| --- | --- | --- |
| Component/device-map placement | Different sequential stages | Can retain encoder on another card and avoid transfers, but does not divide the ~264 s cached denoising work. Useful capacity baseline, not the main latency solution. |
| Layer/pipeline placement | Sequential layer stages for one sample | Primarily capacity; without sufficient microbatch/patch overlap one card waits on the other. Approximate/stale patch methods are not an exact-numerics baseline. |
| TP2 (Megatron-style) | Shard QKV/MLP projections and heads, sum output projections | Roughly half transformer weights per card; two full-hidden activation all-reduces per block. Valid native LLM direction and image baseline, but potentially more communication here. |
| Target-K/V all-gather context parallel | Replicated weights, half target rows for QKV/MLP and local-Q/global-KV attention | Simple exact reference; gathers only target K/V and prepends the cached prefix once. Each card retains global all-head K/V. |
| Ulysses sequence/head exchange | Replicated weights, half rows for GEMMs; full target sequence/half heads for attention | Fused QKV all-to-all plus inverse output exchange; avoids all-head global K/V replication. Established scalable design, not automatically faster than gather at two ranks. |
| Ring/USP | Local sequence with streamed K/V or a hybrid exchange grid | Requires online-softmax/attention kernel integration and measured overlap. Keep as a later measured alternative; do not copy another model's plan. |

For the measured batch1 square case: target tokens16,384, hidden4096, heads32,
head width128, BF16. Each two-rank local hidden is [1,8192,4096], local Q/K/V
[1,8192,32,128]. Ulysses yields [1,16384,16,128]; prefix K/V are [1,27,16,128]
per rank and are prepended once. The inverse exchange restores local rows/all
heads before projection/residual/MLP. Per-rank off-rank payload is approximately
128 MiB/block for BOTH gather and Ulysses, or 4 GiB per32-block denoising step.
TP2's two summed [1,16384,4096] activations are approximately256 MiB/rank/block
under a two-rank bandwidth-optimal all-reduce, or8 GiB/step. These are logical
payload estimates, not measured PCIe traffic or latency; staged copies add costs.

Replicating the transformer requires14,230,249,472 parameter bytes on EACH card,
plus per-rank activations, collective buffers, prefix, allocator/workspace and
CUDA contexts. It is not a single13.25 GiB charge against48 GiB pooled VRAM.
The encoder cannot stay beside that replica on a24 GiB card; phase ownership must
park it before denoising and reserve all transition peaks. CPU construction,
replicas and transfer buffers need explicit host accounting too.

## First reviewable implementation

api/inference/image/parallel.py implements original fused QKV sequence/head
all-to-all and inverse exchange, plus target-K/V all-gather, against the pinned
block's existing projections, norms and RoPE operator. No external implementation
was copied. Both paths execute local QKV and MLP GEMMs on every rank. It is an
experimental block adapter, NOT a full pipeline or API option.

Two Gloo ranks compared complete cached-block outputs against the pinned eager
block in FP32/BF16: all3 tests passed in14.380s, covering12 mode/dtype/shape cases
on each rank. Maximum absolute error was2.384185791015625e-07 inFP32 and0 inBF16
on CPU. Exact case records are in cpu-numerics.json; this is not CUDA parity. Tests cover redistribution row/head order, odd cached lengths
with padded keys masked out, global RoPE positions, prefix t=0 extraction versus
target-time modulation, invalid text/target keys, immutable compact cache storage,
and a prefix containing an eager condition-image block. The recorded27+16384
joint prefill is odd. This slice leaves prefill eager; it does not assert that
joint block-causal prefill, editing or arbitrary reference-image layouts are
already distributed correctly. Full pipeline editing remains an acceptance gate.

## Small MR sequence

1. This foundation: topology/architecture audit, exact cached-block Ulysses and
   gather implementations, two-rank numerical tests. No live configuration change.
2. Bounded communication and denoiser harness: NCCL two-rank setup/teardown,
   same-shape gather/all-to-all/all-reduce and staged-copy bandwidth, bounded
   allocation, per-rank synchronization, cancellation/failure propagation. Compare
   isolated32-block cached passes and memory before choosing Ulysses/gather/TP.
3. Pipeline adapter: eager global prefill and compact prefix distribution initially;
   target sharding maintained through the complete cached transformer; final target
   gather once per step; exact text/reference-image masks and cache reset. Retain
   original2048²/BF16/40step/seed42 settings. Compare denoiser tensors and final
   pixels; repeated/changed prompt and edit tests, park/unpark and failure cleanup.
4. Shared ownership/API integration in a separate MR: expose both UUIDs; reserve
   a single image owner with device_bytes for both GPUs atomically under the
   existing FIFO; park text before dual-card image execution, release each device
   only after confirmed rank cleanup, quarantine both on uncertain failure. Keep
   caches in bounded host storage and do not bypass global order with per-rank jobs.
5. Native LLM TP2 foundation separately: versioned multi-device planner/protocol,
   packed projection/head/MLP sharding and output reductions, per-device arenas,
   quantization-aware numerical tests, then real-token latency. CUDA visibility
   alone is not multi-GPU text execution. Build only affected native targets and
   required communication dependency, not full images routinely.

## Test window and acceptance

GPU1 is in use by the home API. Do not allocate a second model or run bandwidth
kernels there while live testing continues. A deliberate API-only maintenance
window is required for both-card measurements; stop/release the live owner first,
verify both cards free, run isolated bounded tests, then restore the exact existing
API configuration. Agree the window after this source review and harness bounds
are concrete. Read-only preparation and CPU numerics do not require interruption.

Report end-to-end and cached-step wall times, transfer/collective/kernel times,
rank utilization, cold/warm costs, actual parameter/cache/allocator/physical peaks,
aggregate host RAM, and errors. Correct numerics AND a measured single-request
latency improvement are required; two busy GPUs alone are not success. No matched
Qwen2.1/dual3090 benchmark is assumed. Keep API changes unmerged for review.

## Primary design references

- NVIDIA Megatron column/row tensor parallel primitives:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html
- NVIDIA parallelism guide:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- TensorRT-LLM parallel execution strategies:
  https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html
- Diffusers component placement versus context parallelism:
  https://huggingface.co/docs/diffusers/main/en/training/distributed_inference
- DeepSpeed-Ulysses sequence/head exchange:
  https://www.deepspeed.ai/tutorials/ds-sequence/
- xDiT USP/ring/Ulysses and pipeline design comparison:
  https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md
  https://arxiv.org/html/2411.01738v1
