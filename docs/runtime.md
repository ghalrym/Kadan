# Local FreeToken runtime (draft; GPU validation pending)

Kadan owns one local `ft serve` subprocess. Settings downloads the selected,
pinned checkpoint first; Load acquires a selection lease until unload or failure.
There is no implicit download or mock fallback during load or chat. One model and
one chat request run at a time; overlapping generations return HTTP 429.

Run the API natively on the Linux GPU host with **one worker and without reload**.
The existing API Docker image does not contain CUDA or FreeToken and is not a GPU
runtime installation. Do not launch multiple Kadan processes against the same
model directory/GPU. Bind the API to localhost; this is a trusted local service,
not an authenticated public deployment.

## Separate GPU environment

The optional `api/requirements-runtime.txt` pins FreeToken to
`d3512b43affe981465e03ee28cbd88f49c39b9aa` (Apache-2.0). Its transitive dependencies
follow upstream constraints; this is not a fully locked or hardware-validated
environment. Upstream currently requires Linux x86_64, NVIDIA driver r580+, a
CUDA 13 toolkit with nvcc for first-use JIT, torch >=2.11,<2.12 and
transformers >=5.16,<5.17. Do not modify a working GPU environment in place.

On the target host, create a separate environment and install the pinned file:

```sh
python3 -m venv .venv-freetoken
.venv-freetoken/bin/pip install -r api/requirements-runtime.txt
.venv-freetoken/bin/ft --version
export KADAN_FT_EXECUTABLE="$PWD/.venv-freetoken/bin/ft"
export KADAN_FT_GPU=0
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.txt
.venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal run `npm ci --prefix frontend` and
`npm --prefix frontend run dev -- --host 127.0.0.1`. Open the printed URL, download
and select a model in Settings, click Load selected model, wait for `ready`, then
open Chat. Do not run the Compose API on the same port at the same time.
Chat/model storage do not access Postgres; retain the existing migration/database
setup for other database work. No demo data is inserted.

The API environment also needs its normal requirements, including httpx; the GPU
worker environment is separate. FreeToken output inherits API logs. Startup errors
are surfaced in Runtime status; inspect those logs for detailed kernel/weight errors.

NVFP4 explicitly uses the native Triton kernel, **not** the vLLM Marlin donor.
Upstream acknowledges that donor's transformers<5 dependency conflicts with its
current transformers>=5 requirements. We do not suggest forcing incompatible wheels
with --no-deps. The native Triton path has pre-sm89 e4m3 emulation for Ampere;
this is source compatibility evidence, not an RTX 3090 inference test.

## Memory and lifecycle

`--moe-strategy offload` puts expert banks in host RAM and uses FreeToken's LRU GPU
expert slots. `--moe-cache-auto --memory-ratio 0.8 --kv-reserve-tokens 4096` asks its
budget policy to size slots from available VRAM after weights/KV reserves, rather
than allocating a fixed fraction of a 198GB checkpoint. `--text-model-only` skips
vision towers. Context is capped at 4096 and output at 1024 tokens. The model's
tokenizer still determines whether a supplied prompt fits; token-limit errors are
reported as generation failures. These are conservative starting limits, not an
OOM guarantee. Host pinned-memory/working-set needs can exceed checkpoint bytes.

Only GPU 0 is used by default; `KADAN_FT_GPU` selects another single index or UUID.
The two 24GB 3090 cards are **not** treated as pooled 48GB memory; there is no tensor
parallelism or CPU expert execution. RAM is expert storage and GPU kernels compute
selected experts. No guaranteed fit or performance is claimed for the catalog.

FreeToken binds a dynamically chosen localhost port. Readiness requires its health
status and matching served model ID, plus a live owned process. Startup has a
30-minute timeout for initial JIT/loading. Kadan monitors the worker after startup,
terminates its entire process group on unload/shutdown/failure, and escalates to
SIGKILL after ten seconds. Cancelling a chat HTTP connection unloads the worker
because dropping a proxy request alone does not prove GPU cancellation. Generation
transport/response errors also unload; reload explicitly after reviewing the error.
Selection is blocked during the runtime lease. Unload then select/load to switch.

## Reviewed upstream contracts

All links are pinned to the reviewed source, not moving main:

- [CLI flags](https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/docs/cli.md)
- [Host banks and LRU strategy](https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/docs/models.md)
- [Native NVFP4 kernel and Marlin donor distinction](https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/python/freetoken/layers/quantization/moe/nvfp4.py)
- [Ampere e4m3 compatibility](https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/python/freetoken/kernel/triton/e4m3_compat.py)
- [Dependency constraints and donor conflict](https://github.com/FlashML-org/FreeToken/blob/d3512b43affe981465e03ee28cbd88f49c39b9aa/pyproject.toml)

Tests use controlled subprocess doubles and HTTP responses: they validate lifecycle,
request translation, concurrency rejection and failure behavior. No model weights,
GPU inference, performance, or peak RAM/VRAM behavior were tested in cloud. Target-host
validation must check install/JIT, each checkpoint's actual load, measured memory,
chat, cancellation and unload before treating this draft as production-ready.

Cloud validation also exercised the actual API's unloaded-model HTTP 503 response
through Chromium and the Vite proxy. Controlled browser routes covered chat
retry without duplicate turns, cancellation suppressing late responses, download
failure/retry/progress/cancel/selection, and runtime load/readiness/unload. These
browser responses were fixtures, not claims of successful GPU inference or downloads.
