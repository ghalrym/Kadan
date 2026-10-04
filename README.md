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
