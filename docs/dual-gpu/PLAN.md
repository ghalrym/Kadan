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

## Review follow-up: preflight and bounded probe (not yet executed on CUDA)

`cached_block` now requires a request identity and a bounded Gloo control group
with the same rank membership/order as the data group. Local input validation
precedes projections/data collectives; both ranks agree request, step, block,
mode, global/local shape, dtype and key-mask digest. A bad mask on one rank is
reported to both before either enters the exchange. The mask digest includes a
CPU copy/synchronization; measure it as validation cost, not communication-only
or pure Python overhead. Post-gate allocation/runtime errors still require
fail-stop supervision; a failed NCCL communicator is never reused.

The CPU tests assert malformed one-rank mask/dtype/shape and mismatched
mode/step/request fail before projection or data exchange. A parent watchdog
also tests peer exit and injected CUDA-OOM exception without GPU allocations,
stopping and reaping both children. This is not proof of NCCL failure behavior.

`launch_probe.py` defaults to a read-only plan. After exact-head source review,
green CI and coordination, `--execute-reviewed <full SHA> --evidence <new dir>`
stops only the API after an idle-queue check, verifies physical GPU ownership
release, then runs the pinned image with both explicit UUIDs. Each case has one
exclusive lease on the existing model-volume inference.lock, and torchrun owns
both ranks. No model weights are loaded and no database is touched. Container
exit plus empty CUDA process inventory and return to physical baseline are the
release fences between cases. The API stays stopped throughout the sweep.

The host watchdog allows150 seconds/case; the rank supervisor allows135 seconds
and kills/reaps the process group. NCCL timeout is45 seconds; control timeout60, allowing CPU initialization skew for the real-width block.
The allocator hard limit is2 GiB/rank (CUDA/NCCL allocations outside PyTorch are
additionally watched physically). Container RAM/no-swap is8 GiB,4 CPU,1 GiB SHM.
Guards: CPU80 C, GPU90 C, host available16 GiB, physical GPU free256 MiB; require
4 GiB/card before start. Normal probe is followed by peer-exit and injected-OOM
cases. OOM is injected, not induced by exhausting hardware. Captured NCCL logs
show transport selection; capability flags alone do not prove transport.

On success or test failure the launcher starts the exact captured MR119 container ID with `docker start` (no Compose
reevaluation) and verifies unchanged ID, image, Config, HostConfig, Mounts, Path
and Args, Docker/HTTP health, model ready and unchanged unrelated container IDs.
Unconfirmed CUDA-process cleanup quarantines the cards and leaves API stopped.
Expected interruption is5–8 minutes including recovery, superseding the earlier
3–5 minute estimate. The launcher itself remains subject to source review; no
CUDA parity/bandwidth/failure result is claimed yet.

### Temporary allocation budget (per rank, real BF16 communication shape)

| Operation | Input retained | Pack/send | Receive | Reorder/result | Conservative tensor live bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target K/V gather | QKV192 MiB + TP buffer128 MiB | stack128 MiB | two receives256 MiB | concatenation256 MiB | 960 MiB |
| Fused Ulysses forward | same320 MiB | stack192 MiB + contiguous send up to192 MiB |192 MiB | reorder up to192 MiB | 1088 MiB |
| Ulysses inverse | inputs320 MiB + exchanged192 MiB | send up to64 MiB |64 MiB | reorder up to64 MiB | 704 MiB |
| TP two reductions | same320 MiB | in place | backend scratch not counted here | in place | 320 MiB + backend scratch |

These conservative live tensor bounds count simultaneous references and copies,
not just the128 MiB off-rank logical payload. Allocator caching, backend scratch,
CUDA contexts, host staging and the parity block's weights are additional;
record actual allocated/reserved/physical peaks. Communication timing includes
packing/reordering and synchronization; it is not raw link bandwidth. This probe
is synthetic projection parity/communication, not a complete 32-block denoiser
or an end-to-end speedup result.

Follow-up local validation: all5 CPU tests passed in55.511 seconds in the pinned
image with no GPU devices exposed (4 CPU,4 GiB RAM/no swap). Probe `--plan`
imports passed in the same runtime with no GPU exposure. Python compilation and
`git diff --check` passed. Exact-head CI is recorded separately after publication.

Launcher review correction: stop/start uses the captured immutable container ID.
Changes to `.env`/Compose cannot cause recreation during restoration. Identity and
configuration are compared before start, after start and after readiness. The
Gloo control timeout is60 seconds to tolerate independent real-width CPU block
initialization under the CPU quota; per-case initialization duration is recorded.
NCCL45 seconds, probe120 seconds, supervisor135 seconds and host150 seconds remain
bounded and unchanged. No GPU execution was needed to make these corrections.
