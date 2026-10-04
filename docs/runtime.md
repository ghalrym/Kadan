# Kadan inference and shared resources (draft)

Kadan owns inference and resource lifecycles. It does not launch FreeToken or
use its runtime as a dependency. FreeToken was studied as an architectural/weight-
layout reference; the implementation lives in `api/inference/`.

## Ownership and execution

- `resources.py` admits named workloads (`llm`, `image`, `video`, `speech`,
  `decision`) against separate host-RAM and per-GPU byte budgets. Active leases
  prevent eviction. An inactive owner's cleanup callback must release tensors
  before its reservation is reclaimed. An exclusive workload can evict inactive
  GPU residents while retaining separately reserved CPU backing. Future modality
  adapters will use this same API; no image/video/speech inference is implemented.
- `checkpoint.py` reads only local indexed safetensors. Model skeletons start on
  the meta device so constructing a model does not allocate all dense experts.
  The three adapters validate their expected checkpoint tensor mappings and
  substitute Kadan's expert execution while preserving family-specific routing,
  attention, recurrent state, shared experts and activations.
- `offload.py` owns CPU packed expert banks and a byte-bounded GPU LRU cache.
  It transfers selected experts, evicts before allocation, and waits for CUDA
  completion before recycling storage. CPU banks are safetensors-backed mappings;
  they are not a promise that the OS pins every model byte in physical RAM.
  Only requested transfer staging is pinned. Disk/page-fault latency is possible.
- `quantization.py` implements NVFP4/MXFP4 and supported FP8 reference math with
  bounded row-wise decode workspace. It does not expand all experts into BF16.
  These PyTorch reference paths prioritize inspectable correctness, not throughput;
  optimized fused kernels and performance tuning remain work to do.
- `generation.py` owns bounded greedy autoregressive generation and cancellation.
GPU residency is restored when chat follows an inactive model's eviction. Full
  unload releases host backing too. The API reports `offloaded` rather than
  pretending an evicted model is still in VRAM.

Production selects one CUDA GPU using `KADAN_GPU` (default `0`). Two 24 GiB cards
are not pooled into 48 GiB and tensor parallelism is not implemented. The initial
resource capacity is 80% of probed available host/CUDA memory, with another live
availability check at admission. Adapters reserve resident parameters, cache and
working space. Reservations cannot prevent other processes consuming memory or
predict every PyTorch allocator peak: allocation errors remain possible and are
reported. Only one Kadan API process/worker may own a model store/GPU.

## Models and validation boundaries

| Catalog | Direct implementation | Evidence and remaining work |
| --- | --- | --- |
| Small: NVIDIA Qwen3.6-35B-A3B-NVFP4 | Qwen3.5-MoE text architecture with hybrid GatedDeltaNet/attention, shared expert and mixed FP8/NVFP4 projections | Tiny synthetic mixed-quantization checkpoint and architecture parity tests; actual 35B checkpoint loading and GPU parity pending |
| Medium: OpenAI GPT-OSS-120B | Native MXFP4 experts, GPT-specific interleaving/bias/clamped activation, preserved router/attention | Tiny full-model prefill/cached-decode parity with resident reference; actual 120B checkpoint loading and GPU parity pending |
| Large: RedHatAI GLM-5.3-Flash-NVFP4 | GLM5-next text architecture, KDA/DSA/mHC, dense-prefix/shared experts and mixed packed weights | Tiny architecture/quantization tests; actual 320B tensor mapping, memory fit and GPU parity pending |

Unknown tensor layouts, missing tensors, unsupported scale shapes and inadequate
budgets fail explicitly. A checkpoint download is not inference validation. Vision
and MTP execution are outside this text-chat task even when those files are in a
checkpoint. The original claim that a FreeToken integration completed Kadan's
model-management system is superseded by this design.

GLM's ordinary dense linear weights also remain in host RAM and stream in bounded
tiles; keeping them all on GPU exceeded the target budget. A full-default meta
geometry check calculates about 1.431 GiB of remaining resident tensors plus an
8 GiB cache/workspace reservation, rather than treating all dense weights as
resident. This resolves the known admission-policy failure; it does not measure
real peak GPU usage or establish actual checkpoint compatibility.

No real catalog weights were downloaded or loaded in cloud, and no CUDA inference
was run. Tests use real CPU tensor arithmetic and tiny synthetic checkpoints for
mathematical checks; lifecycle/HTTP/browser tests use controlled doubles. CUDA tests
are skipped without hardware. Target validation must inspect actual checkpoint
headers, compare logits/generation with a reference, measure RAM/VRAM peaks,
exercise expert churn, cancellation and cross-workload eviction/restoration, and
verify all reservations/allocations are released. No speed or OOM guarantees.

## Install and run on the target Linux GPU host

The API-only Compose image does not install CUDA inference dependencies. Create a
separate environment; do not replace an existing working CUDA environment. Choose
a PyTorch 2.11.0 CUDA wheel compatible with the host driver using PyTorch's official
installation instructions, then install the pinned remaining runtime requirements:

```sh
python3 -m venv .venv-inference
# Install the appropriate torch==2.11.0 CUDA wheel in this environment first.
.venv-inference/bin/pip install -r api/requirements.txt -r api/requirements-runtime.txt
export KADAN_GPU=0
export KADAN_MODEL_DIR=/path/to/model/storage
.venv-inference/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not use `--reload` or multiple API workers. In another terminal, run
`npm ci --prefix frontend` and `npm --prefix frontend run dev -- --host 127.0.0.1`.
Download/select a model in Settings, load it, and use Chat after readiness. If the
load fails, inspect the error; do not treat synthetic tests as a successful model
installation. Do not run Compose's API on the same port simultaneously.

The Transformers dependency is pinned to source commit
`469230357aab0f2b303b0d638c1f8d06edb14184`: release 5.16.0 lacks GLM5-next definitions.
PyTorch 2.11.0, Accelerate 1.10.1 and Safetensors 0.8.0 are pinned too. Transitive
packages and CUDA drivers are not a complete hardware-validated lock.

Load/generation execute in an owned thread. Cancellation sets an event checked
between loading/generation/cache operations and waits for actual work to stop
before freeing memory; it cannot preempt an in-flight CUDA kernel. Timeout requests
also wait for cooperative cleanup. A hung driver/kernel can therefore delay
shutdown. The server never releases a reservation merely because an HTTP request
was abandoned. Load has a 30-minute deadline; generation has a five-minute deadline,
4096-token context bound for Qwen/GPT-OSS (512 for GLM's reference path), and a
256-token output budget. Prefill is chunked to 32 tokens to bound eager attention
query workspace; native optimized attention/FP4 kernels remain future work.

## Research and licensing

PyTorch (BSD-style), Transformers (Apache-2.0), Accelerate (Apache-2.0) and Safetensors
(Apache-2.0) remain third-party dependencies. Checkpoint terms are linked from
Settings: Qwen/GPT-OSS Apache-2.0; GLM MIT. Preserve upstream notices when redistributing.
FreeToken's Apache-2.0 source at `d3512b43affe981465e03ee28cbd88f49c39b9aa` informed
layout research only; it is neither installed nor invoked. Source references are
recorded in the relevant modules. No upstream FreeToken implementation was vendored.
