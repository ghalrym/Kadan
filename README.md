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
without a GPU. Hikari's API does too; its GPU worker containers need a supported
NVIDIA host.

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
| Native tools | `kadan-*` tool names, C++ namespaces/includes, `KADAN_*` CMake options, native environment variables, timing records and prepared-asset formats stay compatible. The single `kadan-inference-worker` is replaced by one `kadan-worker-<type>` program per inference type. |

Workers return finished videos to the API, which writes them to
`HIKARI_VIDEO_DIR`.

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

For a new installation, copy `.env.example` to `.env`, set `HIKARI_WORKER_TOKEN`
(for example `openssl rand -hex 32`), then run from the repository root:

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

Run one API process. Each Hikari worker container runs one worker program and
locks its own workspace in the shared model volume. Reloading the API drops its
worker connections, which cancels in-flight work; the workers reconnect.

### Local tracing with Jaeger

Jaeger is optional infrastructure for inspecting trace waterfalls. No application
exports traces yet; service instrumentation is a separate change. The `tracing`
profile adds one official Jaeger 2.22.0 container, pinned to its multi-platform
image digest, with no additional database or collector.

After configuring `.env` as above, start only Jaeger in the existing Compose
project (keep your existing `-p` / `COMPOSE_PROJECT_NAME`):

```sh
docker compose --profile tracing up -d --no-deps jaeger
docker compose logs --tail=100 jaeger
```

Open <http://127.0.0.1:16686>. The UI and query API have no authentication; keep
this loopback binding and do not proxy it to the public internet. OTLP collection
ports 4317 and 4318 are reachable only on the existing Compose network and are
not published to the host. The query gRPC server binds to container loopback.
Legacy Jaeger/Zipkin receivers, remote sampling, debug endpoints, AI/MCP and
external telemetry exporters are not configured.

Storage is **temporary RAM only**, capped at 10,000 traces. This is a count-based
capacity limit, not a time-based retention promise: traces can be evicted as the
store fills, and all traces disappear whenever the Jaeger process exits,
including on stop, restart, recreation, upgrade or an out-of-memory kill. There
is no persistent volume, disk archive or backup. Export a selected trace from
the UI before stopping if you need to keep it; treat exported files as sensitive.
Durable tracing with a time-based deletion policy needs a separately reviewed
storage configuration; Kadan's Postgres and Tsuchi stores are not used for traces.

The container is limited to one CPU, 512 MiB RAM and 128 processes. Go has a
400 MiB soft memory limit. The ingestion memory limiter starts applying pressure
at 288 MiB with a 384 MiB hard threshold, checked once per second; batches hold
at most 1,024 spans and OTLP requests are limited to 4 MiB. Trace count is not a
byte limit. Oversized traces or load can cause rejected/dropped spans or a
container restart; this local diagnostic store does not guarantee delivery.

```sh
docker compose stop jaeger
docker compose rm -f jaeger
```

These commands affect only Jaeger and clear its temporary traces. To start it
again, repeat the `up` command. After changing its config or pinned image, use
`docker compose --profile tracing up -d --no-deps --force-recreate jaeger`.
Do not use `down -v` to clear traces: it would remove application data volumes.

#### Contract for service instrumentation

The follow-up should use OpenTelemetry OTLP with `service.name` identifying each
service/worker, sending gRPC to `http://jaeger:4317` or HTTP/protobuf to
`http://jaeger:4318` (trace-specific HTTP URL: `/v1/traces`). These are plaintext
endpoints for containers in the same Compose project. Host processes, remote
workers and browsers cannot use these endpoints directly; their transport and
access require a separate design. Do not publish collector ports to support them.

Propagate W3C `traceparent` / `tracestate` across HTTP, queued work and worker
messages so meaningful operation and timing spans share one trace waterfall.
Keep tracing optional with bounded SDK queues, sampling and export timeouts;
Jaeger availability must not block application startup or requests. Use an
attribute allowlist for timings, operation names, status and opaque correlation
IDs. Do not collect prompts, completions, tool arguments/results, message bodies,
documents, file contents, credentials or arbitrary headers, query strings or
exception text. No payload logging or automatic payload capture is enabled by
this configuration. Instrumentation and its privacy controls belong in that
follow-up, not in this infrastructure change.

Configuration references: [official deployment guide](https://www.jaegertracing.io/docs/2.22/deployment/),
[memory storage](https://www.jaegertracing.io/docs/2.22/storage/memory/), and
[v2.22.0 configuration](https://www.jaegertracing.io/docs/2.22/deployment/configuration/).

### Development hot reload

`compose.development.yaml` bind-mounts source into every service on the same
published ports. The frontends run Vite dev servers, and the APIs run uvicorn
with `--reload`, so local edits apply without rebuilding images. Copy
`.env.example` to `.env`: its `COMPOSE_FILE` line makes plain `docker compose`
commands include the override.

```sh
docker compose up --build -d
```

Each Hikari worker's development container rebuilds only its own program from
the source mount and restarts it when its binary changes, so other workers keep
running. To pick up only frontend or Kadan/Tsuchi API edits, replace just those
services and leave `hikari-api` running:

```sh
docker compose up -d --no-deps --build --force-recreate --wait hikari-frontend kadan-frontend tsuchi-frontend kadan-api tsuchi-api
```

Frontend dependencies install into per-frontend `node_modules` volumes on start,
so `package.json` changes need only a container restart. Python dependency
changes need an image rebuild. Remove `COMPOSE_FILE` from `.env` to run the
nginx frontends and baked API images.
