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

All generation runs in the API-owned native worker. CLI commands send the same
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
is idle or cancellation is acceptable: reloading also replaces its owned native
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
python bin/kadan.py check native
python bin/kadan.py check frontend-lint
python bin/kadan.py check frontend-unit
python bin/kadan.py check frontend-build
```

Manual end-to-end testing uses the playground. Check generation, cancellation and
model switching there; a successful compile or unit test does not establish
model accuracy, throughput or target-machine memory fit.
