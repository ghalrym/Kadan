# Kadan

## Overview

Kadan aims to give local agents chat, decisions, image and video generation, and
speech tools on one machine without an expensive home lab.

The server will manage model loading for each request. Its Kadan-owned
mixture-of-experts loading shares GPU VRAM and system RAM across models. When an
image, video, or speech request needs the GPU, Kadan can move the LLM entirely out
of VRAM and reload it afterward. Agents just call the API. Kadan handles the memory.

The goal is to fit all these tools in one box, accepting slower model switches
to keep hardware requirements down.

## Chat development flow

Settings can download and select the three pinned language-model checkpoints.
The chat page sends real API requests. Kadan's inference code owns checkpoint
loading, packed expert storage, a bounded GPU expert cache, memory reservations,
generation and model eviction/restoration. It uses PyTorch and pinned Transformers
architecture definitions; it does not run FreeToken. Settings shows loading,
readiness, offloaded state or errors. Chat has no mock fallback. Decisions use the
same loaded runtime and validate structured answers. Model selection is persisted;
chat conversations stay in the browser. Request history and dashboard metrics now
report bounded process-local HTTP observations and measured memory.

Image generation/editing, video, speech generation/cloning and transcription forms
send API requests, but their model providers are not implemented. They return
explicit unavailable errors and empty media history, never fabricated results.
These are connected forms, not working media generation. The API access page reads
the running OpenAPI reference and does not create keys or change authentication.
Media checkpoints/licenses, input ingestion, artifacts and resource-managed
adapters remain to be selected and implemented.

The inference service uses host-RAM expert offload and a GPU cache on one selected
GPU; it does not pool two cards' VRAM. Source modules in `api/inference/` describe
ownership and execution invariants alongside the implementation.
GPU installation, memory fit and inference remain unvalidated on target hardware.
The Compose API image below supports downloads but does not include inference dependencies.

![Kadan request dashboard showing AI requests, latency, and GPU and system memory usage](docs/images/requests.png)

![Kadan settings showing model selections for language, images, video, and speech](docs/images/settings.png)

## Setup with Docker Compose

Install Docker with Docker Compose and run these commands from the repository
root.

1. Build and start the services:

   ```sh
   docker compose up --build -d
   ```

2. Open the app at [http://localhost:5173](http://localhost:5173).
   API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

3. Check service status or follow startup logs:

   ```sh
   docker compose ps
   docker compose logs -f
   ```

4. Stop the services when finished:

   ```sh
   docker compose down
   ```

## Native model service

The API-only Compose image does not include inference dependencies. On a target
Linux GPU host, create a separate environment and install a PyTorch 2.11.0 CUDA
wheel compatible with the host driver, then the pinned remaining requirements:

```sh
python3 -m venv .venv-inference
# Install the appropriate torch==2.11.0 CUDA wheel first.
.venv-inference/bin/pip install -r api/requirements.txt -r api/requirements-runtime.txt
export KADAN_GPU=0
export KADAN_MODEL_DIR=/path/to/model/storage
.venv-inference/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not run multiple API workers or use `--reload` with a loaded model. The
`/model-lifecycle` controls are Kadan management endpoints, separate from `/v1`
inference calls. Settings downloads/selects a checkpoint and explicitly loads it.
The status endpoint reports loading, ready, offloaded, unloading or errors;
selection/download alone does not mean inference is ready. Do not run Compose's
API on the same port simultaneously. Use `API_PROXY_TARGET` for another API port.

Inference tests use tiny synthetic CPU checkpoints. Full catalog loading,
CUDA correctness, memory peaks and throughput are not hardware-validated.
Reservations cannot prevent another process from consuming memory. Validate
repeated load/generate/cancel/evict/restore/unload cycles on the target GPU before
relying on resource fit. Reference tensor kernels favor correctness over speed.

Runtime requirements pin Transformers source because the released version lacks
GLM5-next architecture definitions. Dependencies include PyTorch (BSD-style),
Transformers/Accelerate/Safetensors (Apache-2.0); preserve upstream notices when
redistributing. FreeToken Apache-2.0 source was layout research only, never an
installed engine or vendored runtime. Source references are kept in code.

## Frontend checks

From the repository root, run:

```sh
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

Open `/chat` after starting the development services. Vite proxies `/v1` to
`http://127.0.0.1:8000`; set `API_PROXY_TARGET` to use another development backend.
To check recovery manually, stop the API, send a message, restart the API and retry.
Cancel a pending request and verify that a late reply is not appended.

## Checkpoint storage

Settings uses `/v1/models` to download and select the three pinned catalog
checkpoints. Set `KADAN_MODEL_DIR` to a writable disk with sufficient space
(default `~/.local/share/kadan/models`). Docker Compose persists its `model_data`
volume at `/var/lib/kadan/models`; `docker compose down -v` deletes that volume.
Run one API worker and do not share its store between independent servers.

Downloads verify pinned file sizes and hashes before publishing completion.
Cancel is cooperative and retry restarts from scratch; do not edit completed
checkpoint files externally. Selection persists, but download progress is
process-local. Selection does not load a model or establish GPU compatibility.
Review catalog model cards/licenses before downloading: Qwen and GPT-OSS are
Apache 2.0; GLM is MIT. Downloader ownership and integrity details live alongside
its implementation in `api/services/model_downloads.py`.

Chat forwards the full nonblank message text and conversation history without fixed
character or turn limits. Capacity is determined by the backend’s loaded model and
configured token context; API context errors are shown in the chat page.

Set a per-model context limit in Settings, or leave it blank to use that
checkpoint's architecture maximum. Configuration persists across restarts.
Unload the active model before changing context. A saved context is not a memory
allocation or a guarantee that a request of that size fits on the target hardware.

Context memory is admitted from each request's actual prompt and output allowance,
not preallocated at the configured ceiling. Packed GPU expert entries share that
budget and can be evicted to make room while CPU weights remain available. RAM
pressure can also evict an idle model's host backing; its next request reconstructs
it from the local checkpoint. Active work is protected from either eviction.
Linux admission respects visible cgroup limits. Estimates and working-expert
preflight are conservative checks, not guarantees against external allocations;
actual entry allocation is admitted again before copying. Target-GPU validation
is still required.

## Request monitoring

Request history retains the latest 1,000 completed inference HTTP requests in the
API process and resets on restart. Run one worker. It stores status, elapsed time,
byte counts and structural summaries, not prompt/output text or credentials.
Metrics cover the last 60 seconds of retained requests and flag truncated windows.
Memory meters include other processes; missing GPU measurements are explicit.
These are observations of HTTP handling, not proof of model quality or performance.

For the optional browser smoke, start a disposable API with empty history and
its Vite proxy, then run `node frontend/tests/monitoring.browser.cjs`. The script
accepts `MONITORING_TEST_URL`, `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` to use existing
local browser tooling; it does not install dependencies or load a model.
