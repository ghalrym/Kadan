# Kadan

Kadan is the local agent application and the primary place to work with your agents.
It ships with two supporting services in this repository: Hikari provides
inference, and Tsuchi stores agent memory, documents, files and conversations.
Their management interfaces are for inspecting and operating those services.

| Service | Responsibility | Local interface |
| --- | --- | --- |
| [Kadan](services/kadan/README.md) | Agent conversations and actions | <http://127.0.0.1:8630> |
| [Tsuchi](services/tsuchi/README.md) | Durable per-agent storage, search and history | <http://127.0.0.1:8620> |
| [Hikari](services/hikari/README.md) | Chat, decisions, images, video, speech and transcription | <http://127.0.0.1:8088> |

[Start all services with Docker](#docker-compose). Kadan owns agent decisions;
Tsuchi owns storage and retrieval; Hikari owns inference. Kadan and Tsuchi run
without a GPU. Hikari's full inference image needs a supported NVIDIA host.

## Hikari rename and existing installations

Hikari is the inference service formerly named Kadan. The user-facing agent app
now owns the Kadan name. All three services ship together in `ghalrym/Kadan`;
the repository, webhooks and publishing configuration keep that name. The
inference service, agent application and storage service live in
`services/hikari/`, `services/kadan/` and `services/tsuchi/`.

Existing installations keep their data and operational identifiers:

| Surface | Hikari behavior |
| --- | --- |
| Developer CLI | Use `python services/hikari/bin/hikari.py`; `python services/hikari/bin/kadan.py` still runs the same commands. |
| API configuration | `HIKARI_MODEL_DIR`, `HIKARI_VIDEO_DIR` and `HIKARI_INFERENCE_WORKER` take precedence over their `KADAN_*` counterparts. Unset Hikari variables retain the existing configuration and defaults. |
| CLI API URL | `HIKARI_API_URL` takes precedence over `KADAN_API_URL`; an explicit `infer --url` takes precedence over both. |
| Persistent storage | Keep `~/.local/share/kadan/models`, `~/.local/share/kadan/videos`, `/var/lib/kadan/models`, database credentials and named volumes. No automatic moves or data rewriting occur. |
| Compose | The default project name stays `kadan`, even from a checkout named Hikari. Keep any existing `-p` / `COMPOSE_PROJECT_NAME` and storage overrides. |
| HTTP clients | Endpoint paths, request/response fields, model IDs and `X-Kadan-Request-Id` are unchanged. |
| Native tools | `kadan-inference-worker`, `kadan-*` tool names, C++ namespaces/includes, `KADAN_*` CMake options, native environment variables, timing records and prepared-asset formats stay compatible. |

The API forwards `HIKARI_VIDEO_DIR` as `KADAN_VIDEO_DIR` to its child worker,
so the installed native executable and API use the same video directory.

Keep the old `/kadan.svg` asset URL for already-open clients; the application now
uses `/hikari.svg`. Historical benchmark names and evidence remain unchanged.

Existing repository integrations, the `kadan-pr-publishing` environment and its
secret references stay in place. The combined Compose project keeps the existing
Hikari storage identifiers and adds Tsuchi's dedicated store. Kadan has no
separate server-side data volume.

## Docker Compose

Use a Linux NVIDIA host with a compatible driver, Docker Compose and NVIDIA
Container Toolkit. The Hikari image builds its native inference executable and
runtime libraries. The other images build their browser applications and need
no model downloads.

For a new installation, run from the repository root:

```sh
docker compose up --build -d
docker compose ps
```

Open **[Kadan](http://127.0.0.1:8630)** to use your agents. Each app runs as an
API container and a frontend container, named after the app:

| App | Containers |
| --- | --- |
| Kadan | `kadan-api`, `kadan-frontend` |
| Tsuchi | `tsuchi-api`, `tsuchi-frontend` |
| Hikari | `hikari-api`, `hikari-frontend`, plus `hikari-postgres` and the one-shot `hikari-migrate` |

The supporting interfaces are [Tsuchi](http://127.0.0.1:8620),
[Hikari Settings](http://127.0.0.1:8088/settings), and
[Hikari API documentation](http://127.0.0.1:8000/docs).
Select/download the required checkpoints in Hikari Settings before asking an agent
to generate a response. Startup does not generate content or load a model.

All published ports bind to `127.0.0.1`. This is a local, single-owner prototype;
namespace keys are storage selectors, not credentials. Keep these services off
the public internet. Within Compose, Kadan uses `http://tsuchi-api:8000` for
storage and `http://hikari-api:8000` for inference. Browser links use the loopback
URLs above. Each frontend container serves its built assets with nginx and proxies
its API paths, including unbuffered streaming responses, through the same browser
origin. Only the frontends and Hikari's API publish ports.

Copy `.env.example` to `.env` to change browser ports or the Tsuchi upload limit.
`KADAN_PORT`, `TSUCHI_PORT` and `HIKARI_FRONTEND_PORT` keep Kadan's operator links
in sync with the published ports. The inference API retains its local port 8000.

```sh
docker compose logs --tail=100 kadan-api kadan-frontend tsuchi-api tsuchi-frontend hikari-api hikari-frontend
docker compose stop
```

### Existing installations and data

Keep your existing Compose project name, database settings and reviewed storage
overrides. The `model_data` and `postgres_data` declarations are unchanged.
Tsuchi uses the named volume `tsuchi_tsuchi_data` by default, matching its initial
standalone deployment. Set `TSUCHI_DATA_VOLUME` when using a different existing
store or an isolated installation. Kadan stores its durable state through Tsuchi.

Switching a running installation is an owner-coordinated cutover. Stop the
standalone Tsuchi service using its original Compose project, then back up its
volume using the original project's stopped-volume backup procedure. Stop the
standalone Kadan service and the host Hikari frontend before starting their
Compose replacements on the same ports. Do not run two
Tsuchi containers against the same data volume. Preserve Hikari's current model
and database mounts, and do not start a second API against its model store.
A volume created by the standalone Tsuchi project may produce a Compose ownership
warning when reused; keep that volume and its contents.

`docker compose stop` and `docker compose down` preserve stored data. **Do not use
`docker compose down -v`**: it removes the database, model and Tsuchi volumes.
This source change does not switch or restart an existing installation.

Services were renamed to `hikari-*`, `kadan-*` and `tsuchi-*`. When upgrading,
run `docker compose down` (never `-v`) before `docker compose up --build -d`, or
pass `--remove-orphans` to `up`. Otherwise the old `api`, `frontend`, `kadan` and
`tsuchi` containers keep running, holding the published ports and the model-store
lock. Named volumes belong to the project, not the service, so data is kept.

Run one API process. The inference worker takes an exclusive lock in the model
store. Development hot reload shuts down and joins its owned process before the
replacement starts; reloading cancels in-flight work.

### Development hot reload

`compose.development.yaml` bind-mounts source into every service on the same
published ports. The frontends run Vite dev servers, and the APIs run uvicorn
with `--reload`, so local edits apply without rebuilding images. Copy
`.env.example` to `.env`: its `COMPOSE_FILE` line makes plain `docker compose`
commands include the override.

```sh
docker compose up --build -d
```

To pick up only frontend or Kadan/Tsuchi API edits, replace just those services
and leave `hikari-api` running:

```sh
docker compose up -d --no-deps --build --force-recreate --wait hikari-frontend kadan-frontend tsuchi-frontend kadan-api tsuchi-api
```

Frontend dependencies install into per-frontend `node_modules` volumes on start,
so `package.json` changes need only a container restart. Python dependency
changes need an image rebuild. Remove `COMPOSE_FILE` from `.env` to run the
nginx frontends and baked API images.
