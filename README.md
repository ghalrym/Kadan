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
See [endpoint status and remaining decisions](docs/endpoint-status.md) and
[monitoring limits](docs/monitoring.md).

See [checkpoint storage](docs/model-downloads.md), [GPU runtime setup](docs/runtime.md),
and [web chat testing](docs/chat-web-testing.md). The runtime uses host-RAM expert
offload and a GPU cache on one selected GPU; it does not pool two cards' VRAM.
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
