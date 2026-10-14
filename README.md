# Kadan

Kadan serves chat, Laya decisions, image generation, H3 video with audio,
CustomVoice speech and Whisper transcription from one machine.

## Runtime ownership

One long-lived `kadan-inference-worker` C++ process owns the bounded FIFO,
model lifecycle, placement, memory reservations and all six inference engines.
Each engine runs in that process and shares its resource ledger. Switching
engines releases the previous engine before measuring available memory and
admitting the next request. Temporary resource pressure retains the queue head;
cancellation waits for owned work and cleanup before advancing it.

The Python API validates requests, serializes a typed Pydantic inference task,
and submits it to the worker. It manages waiting responses, SSE streams, job
status and cancellation. The worker acquires checkpoints, prepares inputs and
model assets, executes the task, packages results and removes its workspace.
Preparation starts only when a task reaches the FIFO head. Checkpoint conversion, downloads, media serialization and execution all run
in the C++ worker. It never invokes Python. Video encoding uses FFmpeg as a codec.

The API starts the inference worker without loading a model or running inference.
Settings can save a model and context and explicitly load or unload it. Chat can
also load the selected model on its first request. Checkpoint selection persists;
chat conversations stay in the browser. Native prefix reuse is currently unavailable.
HTTP request observations and native reservations are distinct from measured
system memory.

## Supported native capabilities

- Chat: pinned Qwen3.5 small and GLM5-next large checkpoints, text messages,
  JSON or streaming SSE responses. The medium checkpoint remains downloadable
  but has no enabled native chat engine. Context admission uses the configured
  token capacity; it never silently truncates history.
- Decisions: pinned Laya on CPU, typed choice, score and Noul answers.
- Images: Qwen Image generation. Editing is unavailable.
- Video: H3 FL2VA INT8 + Turbo, 480p/768p, 16:9/9:16/1:1,
  24 fps and integer durations from 1–15 seconds. Distilled H3 does not support
  negative prompts.
- Speech: Qwen3-TTS 1.7B CustomVoice, named speakers, supported languages and
  instructions. Describe and clone modes have no enabled native integration.
- Transcription: native Whisper, English, up to 30 seconds of mono 16-bit
  16-kHz PCM WAV. Transcript rewriting is unavailable.

Empty history and unavailable errors are explicit; the server never fabricates
media results. The API access page reads OpenAPI and does not create credentials.
Hardware power and thermal policy belongs to the machine, not the application.

## Chat API scope

`POST /v1/chat/completions` accepts nonempty text messages with `system`, `user`
or `assistant` roles. Each message uses either Kadan's `text` field or the
OpenAI `content` string. JSON responses include `choices` and Kadan's `message`;
SSE responses use content deltas, a `stop` or `length` terminal chunk and `[DONE]`.

This is a text-only subset, not full OpenAI API compatibility. Tool calls,
`developer`/`tool` roles, content-part arrays, reasoning controls, sampling controls
and streaming usage options are unsupported. Unknown request and message fields
are rejected instead of silently dropping their meaning. Before streaming starts,
errors use FastAPI's HTTP status and `detail` body; stream failures use an `error`
event followed by `[DONE]`, without a successful terminal choice.

## Docker Compose

Use a Linux NVIDIA host with a compatible driver, Docker Compose and NVIDIA
Container Toolkit. The production image builds and installs the one native
inference executable and its runtime libraries.

```sh
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

Open [the playground](http://localhost:5173) or
[API documentation](http://localhost:8000/docs). Download/select a supported
checkpoint in Settings, then load it or submit a request. No startup generation
or automatic model load occurs.

```sh
docker compose down
```

Run one API process. The inference worker takes an exclusive lock in the model
store. Development hot reload shuts down and joins its owned process before the
replacement starts; reloading cancels in-flight work.

For a separately built native installation, use the build dependencies and CMake
options in `api/Dockerfile`, install the `inference-runtime` component, and set
`KADAN_INFERENCE_WORKER` to its installed `bin/kadan-inference-worker` wrapper.
Install `api/requirements.txt` in a virtual environment and set `KADAN_MODEL_DIR`
to writable checkpoint storage before starting one Uvicorn worker.
`API_PROXY_TARGET` selects the frontend's API destination.

### Development

Install the tooling dependencies and use the central command from the repository root:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python bin/kadan.py --help
```

`bin/kadan.py` is the entry point for repeatable developer operations. Each command
group and command supports `--help` with its arguments and options.

| Command | Purpose |
| --- | --- |
| `generate api` | Export OpenAPI and regenerate the frontend client. |
| `generate schema` / `generate client` | Run either generation step separately. |
| `assets whisper` / `assets whisper-assets` / `assets tts-tokenizer` | Export local model assets using the API's serializers. |
| `native build` | Compile CPU-only checkpoint inspectors and metadata planners. |
| `native --help` | List checkpoint/model inspection and memory-planning commands. |
| `infer chat` / `infer image` / `infer video` | Submit a JSON request file through the live API to its worker queue. |
| `infer speech` / `infer transcribe` / `infer decisions` | Submit speech, transcription, or decision requests to the same queue. |
| `infer video-status` / `infer video-cancel` / `infer video-download` | Inspect, cancel, or download a server-owned video job. |
| `profile build` / `profile phase` | Build the profiler and mark phases on an already instrumented live worker. |
| `profile symbols` / `profile summarize` | Extract executable symbols and summarize captured phase measurements. |
| `check --help` | List formatting, frontend builds, and unit-test commands. |
| `pr preflight` / `pr publish` | Run the restricted owner-dispatched GitHub publisher with workflow inputs. |

Native inspection tool builds require CMake and a C++20 compiler. They only read
checkpoint metadata and plan memory; they do not load or execute models. Use
`native --build-dir PATH` before the subcommand to select another build directory.
The optional profiler additionally requires the CUDA toolkit. It can analyze an
already instrumented live worker; the CLI never launches an inference process.

All generation runs in the API-owned inference worker. CLI commands send the same
JSON request bodies as the frontend to the existing API endpoints. The API applies
validation, checkpoint preparation, queueing, memory admission, and cancellation.
The CLI has no local inference fallback. Select a server with
`infer --url http://127.0.0.1:8000` or `KADAN_API_URL`; `--timeout` defaults to 3600
seconds and includes time waiting in the queue. Submissions are never retried
automatically because a lost response may already have been queued.

For example, save this request as `build/video-request.json`:

```json
{"prompt": "Rain falling on a quiet lake", "duration": 1, "resolution": "480p", "aspect": "16:9", "seed": 42}
```

```sh
python bin/kadan.py infer video build/video-request.json
python bin/kadan.py infer video-status JOB_ID
python bin/kadan.py infer video-download JOB_ID build/video.mp4
```

Video submission returns the server's job ID immediately; `Queued` means accepted,
not completed. Download after its status is `Done`; `video-cancel JOB_ID` asks the
worker to cancel it. Other inference commands wait for the API response. Use
`--output PATH` to save the response instead of printing it. Image and speech
responses contain base64 media; transcription expects a WAV data URL in its JSON
request. Chat streaming responses, if requested, are saved/printed as SSE after
the stream finishes. The running server's `/docs` describes all request fields.

Python implementations and C++ inspection sources live under
`bin/`. Native runtime builds exclude developer executables unless
`KADAN_BUILD_TOOLS=ON`. Skills remain in their existing directories. Schema
generation requires the API dependencies. Asset export uses the C++
`kadan-prepare-assets` tool built by `native build`; CLI help needs only
`bin/requirements.txt`.

Use the development override for Python hot reload. The frontend already uses
Vite with a source bind mount. Explicit file selection avoids loading an unrelated
local `compose.override.yaml`.

```sh
docker compose -f compose.yaml -f compose.development.yaml build api
docker compose -f compose.yaml -f compose.development.yaml up -d --no-build api frontend
```

Python and frontend edits reload automatically. Edit Python only when inference
is idle or cancellation is acceptable: reloading also replaces its owned inference
worker. C++ changes need an image rebuild and replacement of only the API:

```sh
docker compose -f compose.yaml -f compose.development.yaml build api
docker compose -f compose.yaml -f compose.development.yaml up -d --no-deps --no-build --force-recreate --wait api
```

The API and migration service share the built image. The API keeps the worker
installed in that image; no separately deployed native binary is needed. Existing
model and database volumes remain attached. On a recovered server, include its
reviewed storage override in every command and preserve its mount dependencies
before switching service supervision. These commands alone do not migrate an
existing systemd-managed installation.

Browser checks exercise generation manually. Use one second for video checks and
cancel a check if it has not completed after three minutes. These are testing
limits, not application timeouts or duration restrictions.

## Checkpoint storage

The catalog pins revisions and verifies downloaded files before publishing them.
Docker Compose persists models in `model_data` at `/var/lib/kadan/models`.
`docker compose down -v` deletes named volumes, including stored data.
Do not modify completed checkpoints externally or share a store between
independent API servers. Review the catalog model cards and licenses before use.

Cancellation joins checkpoint preparation and native cleanup. Memory reservations
are conservative admission estimates, not physical hard limits; other programs
can change available memory. A cleanup failure stops admission rather than
reusing uncertain allocations. Recovered model directories can remain on their
existing storage; copying the entire collection is unnecessary.

## Checks

Install the pinned Python development tools and format Python files from the
repository root:

```sh
python -m pip install -r requirements-dev.txt
python bin/kadan.py check format --write
```

CI runs `python bin/kadan.py check format` on every push and pull request and fails
if any Python file needs formatting.

The same requirements pin clang-format and clang-tidy for C++ and CUDA sources.
`.clang-format` owns style and `.clang-tidy` lists the bug-focused checks that gate CI:

```sh
python bin/kadan.py check cpp-format --write
python bin/kadan.py check cpp-lint
```

`cpp-lint` configures every CPU component, so it needs the nlohmann_json, ICU,
OpenBLAS and PCRE2 development headers. CUDA-only sources are formatted but not linted.

Automated tests use pure logic, synthetic checkpoint metadata or fake boundaries.
They do not start the application, execute inference or benchmark GPUs.
The workflows list the unit targets and API tests.

```sh
python bin/kadan.py check format
python bin/kadan.py check cpp-format
python bin/kadan.py check cpp-lint
python bin/kadan.py check api
python bin/kadan.py check inference-worker
python bin/kadan.py check frontend-lint
python bin/kadan.py check frontend-unit
python bin/kadan.py check frontend-build
```

Manual end-to-end testing uses the playground. Check generation, cancellation and
model switching there; a successful compile or unit test does not establish
model accuracy, throughput or target-machine memory fit.

### Qwen-Image-2.1-Turbo (draft)

`POST /v1/images/generations` accepts `model_id: "qwen-image-2.1-turbo"`.
Omitting it uses the image model selected on the Settings page (stored by the
worker in `image-selection.json`), which defaults to Turbo. Selecting a model
in Settings requires a completed download; download the chosen checkpoint
through the model catalog before submitting generation.
Turbo always uses the eight sigmas saved in its `model_index.json`, followed by
Euler's terminal zero; the request's `steps` does not replace that grid. Its
scheduler uses shift 1 without dynamic shifting or terminal stretching. The
ATen implementation reuses CFG=1 and per-image prefix KV caching, automatic
component placement, and the existing queue and cancellation path.

The pinned [official checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b)
uses one text-encoder safetensors file and the same transformer/VAE architecture.
Its [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE)
permits research and evaluation only; commercial use requires a separate license.
The catalog retains the existing explicit license acknowledgment before download.
This change adds text-to-image code support; owner review and manual image-quality
validation remain outstanding.

### C++ ATen image execution (draft)

`KADAN_ENABLE_ATEN_IMAGE=ON` selects the complete C++ ATen implementation for
CUDA image requests in the existing serial worker. The default `OFF` retains the
existing implementation. API, queue, cancellation, image count, prompt validation,
seed progression, PNG publication and frontend remain unchanged. There is no
Python inference executor or secondary service. This draft is for architecture
review; it has not passed the owner’s image-quality or performance gate.

The supported dependency pair is PyTorch/LibTorch **2.14.1+cu130** and CUDA
**13.0**. The ATen operator calls are linked through the official Torch CMake
package. Build the existing API image with:

```sh
docker build -f api/Dockerfile \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg KADAN_ENABLE_ATEN_IMAGE=ON \
  -t kadan-aten:review .
```

For an existing LibTorch installation, use `-DCMAKE_PREFIX_PATH=<torch-prefix>`
and `-DKADAN_ENABLE_ATEN_IMAGE=ON` alongside the existing full-worker CUDA build
options. CMake rejects a mismatched toolkit and fixes the architecture to SM86.
The build does not probe a GPU. The optional dependency is substantial; distribute
its accompanying license notices with the binary. The Docker build retains the
PyTorch and NVIDIA dependency notices under `/opt/kadan/share/kadan/licenses`.
Kadan’s license is unchanged; no ComfyUI GPL source is copied. The operator reuse
is PyTorch’s BSD-style library, including its distributed dependency notices.

Each image component is preloaded completely on the admitted GPU with the most
available memory. Text stores BF16 weights and performs FP32 forward operations;
denoising keeps BF16 weights/hidden states and uses library SDPA, normalization,
SwiGLU and linear operations; VAE uses FP32 convolution and normalization. Text
prefix K/V is cached for the current sampling request only. The FP32 Euler latent
and Kadan noise/seed contract are retained. The 1152-channel VAE attention uses
256-query SDPA slices to bound fallback scratch. All component capacities are preflighted before the queue commits to running.
Stage workspace admission caps ATen’s allocator, and cancellation is checked at weight-read chunks, layers,
blocks and VAE attention slices. Insufficient whole-component capacity fails
admission instead of silently falling back to a different arithmetic path.

ATen caches CUDA handles for the process lifetime. In this opt-in build **all
workloads** synchronize and free their owned allocations on retirement without
`cudaDeviceReset`, so switching workloads cannot invalidate ATen handle pools.
Those library/context pools survive until worker exit. Fresh physical memory
probes account for that baseline before the next engine is admitted. Component
weights, activations, cuBLAS workspaces and allocator blocks are released before
ledger acknowledgment; failed cleanup retains admission and blocks worker idle.
Owner review must cover this lifecycle change and repeated model switching.

`KADAN_ATEN_IMAGE_TIMING` lines on stderr report synchronized `load`, `generation`
and `unload` seconds per component and the automatically chosen device. Sum only
the three `generation` records for the loading-free comparison. Transfers,
operator setup and conversion to owned CPU outputs are included; tokenizer/noise
preparation and PNG/API publication are outside these timers. The existing
`KADAN_IMAGE_TIMING` resident-backend counters describe the old path only.

After separately approving deployment, benchmark through the existing API with
cached Qwen-Image2.1 weights, one worker, 2048×2048, two Euler steps, the saved
prompt and seed. Compare against the synchronized 20.424118 s Comfy generation
reference (text 0.649512 s, denoiser 10.504174 s, VAE 9.270431 s), excluding
load/unload and publication on both sides. Record component placement, dependency
versions, arithmetic, peak memory, cancellation and output quality. Equal seed
does not match Comfy’s PyTorch CPU CHW noise: Kadan retains point-major
MT19937_64/Box–Muller noise. Use an explicitly shared replay tensor for numerical
parity work. Compile/unit/CI success does not establish speed or output quality.
No generation, live deployment or default adoption is part of this draft.
