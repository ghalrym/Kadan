# Kadan

Kadan is one local desktop application (`apps/kadan/`) that runs directly on your
computer with no Docker or external database required. It either hosts all Kadan
services locally or connects to another computer running Kadan.

It ships with two supporting services in this repository: **Hikari** provides
native local inference (and optional subscription passthrough), and **Tsuchi**
stores agent memory, documents, files and conversations. One web app serves
Kadan at its home page and Kadan Server's management pages (Home, AI models,
Your agent, Activity and Settings) under `/server`.

| Service | Responsibility | Local interface |
| --- | --- | --- |
| Kadan | Agent conversations, coding threads and actions | Embedded desktop app or <http://127.0.0.1:8000> |
| Kadan Server pages | AI models, the agent's stored data, activity and settings | <http://127.0.0.1:8000/server> |
| Tsuchi | Durable per-agent storage, search and history | API at `/tsuchi` |
| Hikari | Chat, decisions, images, speech and transcription | API at `/hikari` |
| Combined API | One FastAPI application serving `/kadan`, `/tsuchi` and `/hikari` | <http://127.0.0.1:8000> |

Kadan owns agent decisions; Tsuchi owns storage and retrieval; Hikari owns
inference. All four local inference capabilities ship compiled into one native
C++/CUDA `kadan-worker` binary (`llm`, `speech`, `tts`, `image`) with zero
LibTorch or container dependencies.

[User expectations](docs/user-expectations/README.md) describe current behavior,
limits and recovery by feature. [Open proposals](docs/user-expectations/proposals.md)
are separate from behavior available on master.

## Architecture and services

- **Combined API (`services/api/combined/main.py`):** Mounts Tsuchi (`/tsuchi`),
  Hikari (`/hikari`), and Kadan (`/kadan`), and serves the built frontend SPA
  when `KADAN_FRONTEND_DIR` is set. Uses SQLite WAL mode (`HIKARI_DATABASE_URL`)
  out of the box.
- **Agent runner (`python -m kadan.runner`):** Runs agent conversation loops,
  background memory extraction, compaction, and image/pet queues against the
  combined API.
- **Native inference worker (`kadan-worker`):** Single C++20/CUDA executable
  launched once per capability (`--workload=llm|speech|tts|image`). Every worker
  connects over WebSocket to `/hikari/v1/workers` and shares one per-computer
  RAM/VRAM ledger (`WorkerHub` + `memory_scheduler`). Models load lazily on
  demand, queue by priority (`llm` → `tts` → `speech` → `image`), and reclaim
  idle VRAM automatically.
  - **Language, Vision & Decisions (`llm`):** Inkling 2 Mini on Hikari's native
    Strata engine, with exact in-place hot-swapping between standard (`base`)
    and `abliterated` weights when both checkpoints of the same precision are
    downloaded.
  - **Image generation & editing (`image`):** FLUX.2 Klein 4B Distilled on pure
    native C++/CUDA (cuBLAS + custom FlashAttention/Conv/GroupNorm kernels, no
    LibTorch).
  - **Speech to text (`speech`):** Qwen3-ASR 1.7B.
  - **Text to speech (`tts`):** Qwen3-TTS 1.7B CustomVoice.

## Building and bundling natively

### Stage the desktop runtime (web app + Python venv + all 4 CUDA workers)

From the repository root on a Linux machine with the NVIDIA CUDA Toolkit
installed:

```sh
bash scripts/bundle-kadan-runtime.sh
```

This builds the frontend (`services/frontend`), stages the Python services and a
relocatable `venv`, compiles `kadan-worker` with `KADAN_ENABLE_CUDA=ON` (all 4
workloads: `llm`, `speech`, `tts`, `image`), and **verifies** via
`kadan-worker --list-workloads` that all 4 workloads are present before
finishing.

To build the desktop app (`apps/kadan`):

```sh
cd apps/kadan && cargo tauri build
```

### Remote GPU workers

To run inference workers on a separate GPU machine pointing at an existing Kadan
server:

```sh
kadan-worker --workload=llm --api-url=ws://API_HOST:8000/hikari/v1/workers --token=insecure-api-key
kadan-worker --workload=image --api-url=ws://API_HOST:8000/hikari/v1/workers --token=insecure-api-key
kadan-worker --workload=speech --api-url=ws://API_HOST:8000/hikari/v1/workers --token=insecure-api-key
kadan-worker --workload=tts --api-url=ws://API_HOST:8000/hikari/v1/workers --token=insecure-api-key
```

## Development and unit tests

Install the shared tracing package into `.venv` first:

```sh
.venv/bin/python -m pip install -e services/tracing
```

Run unit tests (automated tests are unit tests only; no inference runs):

```sh
.venv/bin/python services/hikari/bin/hikari.py check inference-worker
.venv/bin/python services/hikari/bin/hikari.py check api
.venv/bin/pytest services/kadan/kadan/api/tests services/tsuchi/tests services/api/tests services/tracing/tests
cd apps/kadan/src-tauri && cargo test
cd services/frontend && npm test
```

## Local tracing

[Observability architecture](docs/architecture/observability.md) defines the
shared span-first tracing model and privacy boundary. Tracing stays disabled
until `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is configured (with
`OTEL_SDK_DISABLED=false`).

## Security

Kadan is a single-owner local prototype with simple bearer API-key checks
(default `insecure-api-key`). See [security and operator responsibility](docs/user-expectations/security.md).
