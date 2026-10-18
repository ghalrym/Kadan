# Hikari inference service

Hikari provides inference for Kadan: chat, decisions, images, video, speech and
transcription. Its FastAPI server lives in `api/`, the native inference worker in
`inference_worker/`, the operator frontend in `frontend/` and developer tooling in
`bin/`. Start it with the rest of the stack using the root
[Docker Compose](../../README.md#docker-compose) instructions.

Commands below run from the repository root.

## Runtime ownership

One long-lived `kadan-inference-worker` C++ process owns bounded workload queues,
model lifecycle, placement, and one shared host/GPU memory ledger. Idle models
remain resident when budgets fit. Distinct engines can execute concurrently on
disjoint GPU assignments; a single GPU executes sequentially, and image batches
remain exclusive to preserve their calibration. Execution allowances protect
weights, KV caches, activations, and workspaces before dispatch. Resource waits
retain each workload's FIFO head, with bounded bypass across workloads. Cancellation
waits for owned execution and cleanup before terminal delivery. See the native
worker's [scheduling contract](inference_worker/README.md#shared-scheduling-and-residency).

The Python API validates requests, serializes a typed Pydantic inference task,
and submits it to the worker. It manages waiting responses, SSE streams, job
status and cancellation. The worker acquires checkpoints, prepares inputs and
model assets, executes the task, packages results and removes its workspace.
Preparation starts at each workload queue head; compatible adjacent image requests may be
prepared together for one memory-admitted batch. Checkpoint conversion, downloads, media serialization and execution all run
in the C++ worker. It never invokes Python. Video encoding uses FFmpeg as a codec.

The API starts the inference worker without loading a model or running inference.
Settings saves a hardware tier, which selects its language model and queues its
downloads; requests load the tier's model on demand. Changing tiers unloads a
resident language model of the previous tier. The tier persists;
chat conversations stay in the browser. Native prefix reuse is currently unavailable.
HTTP request observations and native reservations are distinct from measured
system memory.

## Supported native capabilities

- Chat: one pinned GGUF mixture-of-experts checkpoint per hardware tier
  (Qwen3.6 35B-A3B and Qwen3.8 Flash Next at two precisions), with routed
  experts tiered across VRAM, RAM and disk; text messages, JSON or streaming
  SSE responses. Context admission uses the configured token capacity; it never
  silently truncates history.
- Decisions: pinned Laya on CPU, typed choice, score and Noul answers.
- Images: Qwen Image generation. Editing is unavailable.
- Video: H3 FL2VA INT8 + Turbo with audio across all three hardware tiers,
  480p/768p, 16:9/9:16/1:1, 24 fps and integer durations from 1–15 seconds.
  Omitted clip settings use the selected tier's defaults; explicit settings
  remain subject to actual memory admission. See [video defaults and memory
  placement](inference_worker/README.md#hardware-tier-video). Distilled H3 does
  not support negative prompts.
- Speech: Qwen3-TTS 1.7B CustomVoice, named speakers, supported languages and
  instructions. Describe and clone modes have no enabled native integration.
- Transcription: native Whisper, English, up to 30 seconds of mono 16-bit
  16-kHz PCM WAV. The saved formatting switch runs S1-mini by Superwhisper
  (`superwhisper/s1-mini`) after Whisper through the same worker. Disable it to
  return Whisper's text unchanged; each API request can also override the setting.
  S1-mini uses semi-formal prose/general context, English input up to 1,000 tokens,
  and a 1,024-token output limit. It executes on CPU with shared memory accounting.
  Responses retain `raw_text`, identify `formatting_model`, and report
  `formatting_status`; a formatter failure returns raw text with `formatting_error`.
  Download the pinned formatter from Settings before use. Model output quality and
  latency still require validation on the installed model.

Empty history and unavailable errors are explicit; the server never fabricates
media results. The API access page reads OpenAPI and does not create credentials.
Hardware power and thermal policy belongs to the machine, not the application.

## Chat API scope

`POST /v1/chat/completions` accepts nonempty text messages with `system`, `user`
or `assistant` roles. Each message uses either Hikari's `text` field or the
OpenAI `content` string. JSON responses include `choices` and Hikari's `message`;
SSE responses use content deltas, a `stop` or `length` terminal chunk and `[DONE]`.

This is a text-only subset, not full OpenAI API compatibility. Tool calls,
`developer`/`tool` roles, content-part arrays, reasoning controls, sampling controls
and streaming usage options are unsupported. Unknown request and message fields
are rejected instead of silently dropping their meaning. Before streaming starts,
errors use FastAPI's HTTP status and `detail` body; stream failures use an `error`
event followed by `[DONE]`, without a successful terminal choice.

## Native installation

For a separately built native installation, use the build dependencies and CMake
options in `api/Dockerfile`, install the `inference-runtime` component, and set
`HIKARI_INFERENCE_WORKER` to its installed `bin/kadan-inference-worker` wrapper.
Install `api/requirements.txt` in a virtual environment and set `HIKARI_MODEL_DIR`
to writable checkpoint storage before starting one Uvicorn worker.
`API_PROXY_TARGET` selects the Vite development proxy destination; the production
frontend reaches the Compose `hikari-api` service.

## Development

Install the tooling dependencies and use the central command from the repository root:

```sh
.venv/bin/python -m pip install -r services/hikari/requirements-dev.txt
.venv/bin/python services/hikari/bin/hikari.py --help
```

`bin/hikari.py` is the entry point for repeatable developer operations. Each command
group and command supports `--help` with its arguments and options.

| Command | Purpose |
| --- | --- |
| `generate api` | Export OpenAPI and regenerate the frontend client. |
| `generate schema` / `generate client` | Run either generation step separately. |
| `assets whisper` / `assets whisper-assets` / `assets tts-tokenizer` | Export local model assets using the API's serializers. |
| `native build` | Compile CPU-only checkpoint inspectors and metadata planners. |
| `native --help` | List checkpoint inspection and weight-cache replay commands. |
| `infer chat` / `infer image` / `infer video` | Submit a JSON request file through the live API to its worker queue. |
| `infer speech` / `infer transcribe` / `infer decisions` | Submit speech, transcription, or decision requests to the same worker. |
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
`infer --url http://127.0.0.1:8000` or `HIKARI_API_URL`; `--timeout` defaults to 3600
seconds and includes time waiting in the queue. Submissions are never retried
automatically because a lost response may already have been queued.

For example, save this request as `build/video-request.json`:

```json
{"prompt": "Rain falling on a quiet lake", "duration": 1, "resolution": "480p", "aspect": "16:9", "seed": 42}
```

```sh
python services/hikari/bin/hikari.py infer video build/video-request.json
python services/hikari/bin/hikari.py infer video-status JOB_ID
python services/hikari/bin/hikari.py infer video-download JOB_ID build/video.mp4
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

Use the development override for API, inference worker and frontend hot reload
with source bind mounts. The frontend keeps the same local port 8088. Explicit
file selection avoids loading an unrelated local `compose.override.yaml`.

```sh
docker compose -f compose.yaml -f compose.development.yaml build hikari-migrate hikari-api hikari-frontend
docker compose -f compose.yaml -f compose.development.yaml up -d --no-build hikari-api hikari-frontend
```

The development API runs `bin/hikari.py dev serve`. Python and frontend edits
reload automatically. Inference worker C++ edits rebuild the worker incrementally
in the container (`docker compose logs -f hikari-api` shows the build); a
successful build, or a catalog or asset edit, restarts the API with the new
worker. A failed build keeps the previous worker running. Edit only when
inference is idle or cancellation is acceptable: every restart replaces the
owned inference worker. The `hikari_worker_build` volume keeps the build tree;
remove it after changing the worker's toolchain or build options.

The migration service uses the release image. Release images install the worker
at build time; no separately deployed native binary is needed. Existing model and
database volumes remain attached. On a recovered server, include its reviewed
storage override in every command and preserve its mount dependencies before
switching service supervision. These commands alone do not migrate an existing
systemd-managed installation.

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
python -m pip install -r services/hikari/requirements-dev.txt
python services/hikari/bin/hikari.py check format --write
```

CI runs `python services/hikari/bin/hikari.py check format` on every push and pull request and fails
if any Python file needs formatting.

The same requirements pin clang-format and clang-tidy for C++ and CUDA sources.
`.clang-format` owns style and `.clang-tidy` lists the bug-focused checks that gate CI:

```sh
python services/hikari/bin/hikari.py check cpp-format --write
python services/hikari/bin/hikari.py check cpp-lint
```

`cpp-lint` configures every CPU component, so it needs the nlohmann_json, ICU,
OpenBLAS and PCRE2 development headers. CUDA-only sources are formatted but not linted.

Automated tests use pure logic, synthetic checkpoint metadata or fake boundaries.
They do not start the application, execute inference or benchmark GPUs.
The workflows list the unit targets and API tests.

```sh
python services/hikari/bin/hikari.py check format
python services/hikari/bin/hikari.py check cpp-format
python services/hikari/bin/hikari.py check cpp-lint
python services/hikari/bin/hikari.py check api
python services/hikari/bin/hikari.py check inference-worker
python services/hikari/bin/hikari.py check frontend-lint
python services/hikari/bin/hikari.py check frontend-unit
python services/hikari/bin/hikari.py check frontend-build
```

Manual end-to-end testing uses the playground. Check generation, cancellation and
model switching there; a successful compile or unit test does not establish
model accuracy, throughput or target-machine memory fit.

### Qwen-Image-2.1-Turbo (draft)

`POST /v1/images/generations` uses Qwen-Image-2.1-Turbo, the image model of
every hardware tier; `model_id`, when given, must be `"qwen-image-2.1-turbo"`.
Choosing a tier in Settings downloads it with the tier's other checkpoints.
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
Turbo text-to-image uses the accepted ATen execution path described below.

### C++ ATen image execution

C++ ATen is the supported image execution path in the shared worker with exclusive image execution.
Image generation requires CUDA; there is no CPU or custom resident image backend
fallback. API queueing, cancellation, image count, prompt validation, seed
progression and PNG publication remain owned by the same worker and memory ledger.
There is no Python inference executor or secondary service.

The supported dependency pair is PyTorch/LibTorch **2.14.1+cu130** and CUDA
**13.0**. The API Dockerfile installs this pair by default:

```sh
docker build -f services/hikari/api/Dockerfile -t hikari-api services/hikari
```

For an existing LibTorch installation, use `-DCMAKE_PREFIX_PATH=<torch-prefix>`
and `-DKADAN_ENABLE_IMAGE_GENERATION=ON` alongside the full-worker CUDA build
options in `api/Dockerfile`. CMake requires the pinned Torch package, rejects a
mismatched toolkit and fixes the architecture to SM86 without probing a GPU.
CPU unit/tool builds leave image generation disabled. Distribute the dependency
license notices with the binary; Docker retains the PyTorch and NVIDIA notices
under `/opt/kadan/share/kadan/licenses`. Hikari's license is unchanged.

The accepted October 13 fox comparison used build
`bdf012f034158c27b08ec13ff78c8cd39aa87a2a`, ATen enabled and TF32 disabled,
2048×2048 Turbo, eight Euler steps and CFG 1. Generation took **52.339790 s** in
Kadan and **53.031068 s** in ComfyUI. Kadan loading and unloading took 6.402629 s
and 0.580019 s separately. This was one sample per engine, with different noise
implementations; it establishes the accepted path for that workload, not broad
performance or numerical parity. The evidence is the October 13 task-26
`comparison.json`, `kadan-turbo.png` and `comfy-turbo.png`. Later arithmetic
corrections are documented below; the measured timings belong to the stated build.

The execution comparison uses official ComfyUI commit
[`8010b48d2c1be986613f8e8c761527d205bd756f`](https://github.com/Comfy-Org/ComfyUI/tree/8010b48d2c1be986613f8e8c761527d205bd756f),
specifically `comfy/text_encoders/qwen_image21.py` and
`comfy/ldm/qwen_image21/model.py`. This is an independently composed ATen model
implementation, not a direct ComfyUI source port. Its modulation keeps
normalization and `(1 + scale)` in FP32 until the final BF16 cast, following the
arithmetic boundary in Apache-2.0 Comfy Kitchen
[`be003b7c23c5b01328657955b8bc5d3f073d868e`, `comfy_kitchen/backends/cuda/ops/adaln.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/cuda/ops/adaln.cu)
(Copyright 2025 Comfy Org). Shared scales and tanh gates are prepared once per
step; residuals use ATen `addcmul_` with one output rounding. CPU arithmetic unit
fixtures verify these boundaries without a model or CUDA execution.

This composition does not reproduce Comfy Kitchen’s CUDA reduction tree or
kernel fusion. ComfyUI also fuses gate/up projections and forms rotary tables in
FP64 before casting; Hikari retains separate projections and FP32 rotary tables.
Matching those kernels, equal-seed noise, or full numerical output is not
established by this correction. No GPL source or upstream kernel source is
copied, and no license is added to Hikari.

Image stages reuse component weight banks retained on an admitted GPU when their
workspace fits. Each bank keeps a shared-memory-manager reservation, is pinned
while active, and can be evicted after synchronization when another stage needs
that device. Workspace is request-local. Model changes, competing workloads,
cancellation and execution failures release the image caches through the existing
worker owner. Physical cleanup is confirmed before releasing reservations.

Serialized checkpoint weights also remain in the existing bounded host cache,
using at most half the RAM left after request and stage staging budgets. GPU
uploads still occur on a device-cache miss. If whole-component residency does not
fit even after reclaiming inactive banks, the existing operator-streaming path
remains available. Weights keep their checkpoint precision.

Denoiser placement compares measured loading plus concurrent denoising time and
the reload cost of component banks it would displace. It starts conservatively,
permits one successful parallel calibration per comparable shape/cache regime,
and retains both-GPU execution when the measured cost or memory constraints favor
it. This is a bounded estimator, not a guarantee of optimal placement. In
particular, 24 GiB cards cannot keep both the approximately 14.1 GiB text bank
and 13.3 GiB denoiser bank on one device. Keeping them on separate GPUs can avoid
reloads that two simultaneous denoiser replicas would cause.

Text stores BF16 weights and performs FP32 forward operations; denoising keeps
BF16 weights/hidden states and uses library SDPA, normalization, SwiGLU and linear
operations. Text prefix K/V is cached for the current sampling request only. The
FP32 Euler latent and Hikari noise/seed contract are retained.

VAE computation stays on the GPU in FP32. If the full-resolution workspace does
not fit, the decoder keeps global attention on the full latent grid, then runs
its convolutional up blocks in 16-cell latent tiles with a 13-cell halo. Only
the tile interiors are published. The encoder tiles its convolutional down blocks
with a nine-cell halo and assembles the complete bottleneck before global attention.
Halos cover convolution dependencies and preserve stride alignment; no blending or
independent per-tile attention is used. The 1152-channel VAE attention still uses
256-query SDPA slices to bound scratch. Different convolution shapes may select
different library kernels; numerical parity and lower-tier performance require
manual validation on the installed model.

All stage plans are preflighted before the queue commits to running. Preparation
and execution use the same planner, and stage admission caps ATen's allocator.
Cancellation is checked during weight reads, layers, blocks and tile loops. If
neither resident nor streamed/tiled execution fits the available RAM and GPU
budgets, admission fails; there is no CPU execution fallback.

ATen caches CUDA handles for the process lifetime. In the full worker, all
workloads synchronize and free their owned allocations on retirement without
`cudaDeviceReset`, so switching workloads cannot invalidate ATen handle pools.
Those library/context pools survive until worker exit. Fresh physical memory
probes account for that baseline before the next engine is admitted. Component
weights are retained only under explicit cache reservations. Activations and
unused allocator blocks are released at stage boundaries; model retirement also
releases all cached banks before ledger acknowledgment. Failed cleanup retains
admission and blocks worker idle.
Repeated model switching remains a manual lifecycle check.

`KADAN_ATEN_IMAGE_TIMING` lines on stderr report synchronized `load`, `generation`
and `unload` seconds per component and the automatically chosen device. Sum the
`generation` records for the loading-free comparison (three for generation,
four when editing also encodes a source image). Transfers,
operator setup and conversion to owned CPU outputs are included; tokenizer/noise
preparation and PNG/API publication are outside these timers. The former resident
image timing option no longer drives worker execution; component timing utilities
remain available for unit checks and diagnostic tooling.

TF32 remains disabled for FP32 matrix multiplication and cuDNN operations during
ATen generation, as does BF16 reduced-precision reduction. This consolidation does not change the accepted
precision policy, sampling or model operators. New deployments and GPU validation
remain separate owner-coordinated actions.

### Image editing

The Images page accepts a PNG, JPEG or WebP source and an optional opaque mask.
Uploads are normalized to PNG on the model's 32-pixel grid, with at most 3072
pixels per side and 4,718,592 pixels total. A mask must match the original source
dimensions; both receive the same resize. Black preserves source pixels, white
allows editing, and gray blends. Each upload is limited to 20 MiB.

`POST /v1/images/edits` accepts PNG data URLs in `image` and optional `mask`, plus
the generation settings and `strength` from zero to one (default 0.65). API
clients must supply already normalized dimensions; output follows the source
size. Describe the desired result in `prompt`. Zero strength returns source
pixels; one starts the editable area from noise. Intermediate strengths encode
the source with the installed checkpoint's VAE and use the corresponding suffix
of its sampling schedule, including Turbo's saved sigma grid.

This is source-conditioned image-to-image denoising (SDEdit), with latent mask
restoration and a final source-pixel composite. It does not implement Qwen's
separate vision/reference-image attention pipeline. Encoding, text conditioning,
denoising and decoding run sequentially through the existing worker and shared
resource admission. Real-model editing quality and GPU memory behavior still
require owner-coordinated manual validation; unit checks and compilation do not
establish those results.
