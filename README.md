# Kadan

## Overview

Kadan aims to give local agents chat, decisions, image and video generation, and
speech tools on one machine without an expensive home lab.

The server will manage model loading for each request. Its planned FreeToken-style
mixture-of-experts loading shares GPU VRAM and system RAM across models. When an
image, video, or speech request needs the GPU, Kadan can move the LLM entirely out
of VRAM and reload it afterward. Agents just call the API. Kadan handles the memory.

The goal is to fit all these tools in one box, accepting slower model switches
to keep hardware requirements down.

## Chat development flow

Settings can download and select the three pinned language-model checkpoints.
The chat page sends real API requests. With a separately installed FreeToken
worker, Settings can load/unload the selected model and show its readiness or
failure. Chat has no mock fallback when a model is unavailable. Other modalities,
history and dashboard metrics still use fixtures.

See [checkpoint storage](docs/model-downloads.md), [GPU runtime setup](docs/runtime.md),
and [web chat testing](docs/chat-web-testing.md). The runtime uses host-RAM expert
offload and a GPU cache on one selected GPU; it does not pool two cards' VRAM.
GPU installation, memory fit and inference remain unvalidated on target hardware.
The Compose API image below supports downloads but does not include the GPU worker.

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
