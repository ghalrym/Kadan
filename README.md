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

The Python API validates requests, acquires pinned checkpoints, prepares transport
files and presents results. It does not execute numerical models, schedule
inference, keep a memory ledger or launch modality-specific inference children.
First-use Whisper/checkpoint and TTS-tokenizer serialization can use CPU export
dependencies. Video encoding uses FFmpeg as a codec, not an inference engine.

The API starts the native process without loading a model or running inference.
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
  24 fps and integer durations from 4–15 seconds. Distilled H3 does not support
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

Run one API process. The native process takes an exclusive lock in the model
store. Development hot reload shuts down and joins its owned process before the
replacement starts; reloading cancels in-flight work.

For a separately built native installation, use the build dependencies and CMake
options in `api/Dockerfile`, install the `inference-runtime` component, and set
`KADAN_INFERENCE_WORKER` to its installed `bin/kadan-inference-worker` wrapper.
Install `api/requirements.txt` in a virtual environment and set `KADAN_MODEL_DIR`
to writable checkpoint storage before starting one Uvicorn worker.
`API_PROXY_TARGET` selects the frontend's API destination.

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

Automated tests use pure logic, synthetic checkpoint metadata or fake boundaries.
They do not start the application, execute inference or benchmark GPUs.
The workflows list the unit targets and API tests.

```sh
python -m unittest discover -s api/tests -v
npm --prefix frontend run lint
npm --prefix frontend run build
```

Manual end-to-end testing uses the playground. Check generation, cancellation and
model switching there; a successful compile or unit test does not establish
model accuracy, throughput or target-machine memory fit.
