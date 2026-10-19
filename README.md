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

Jaeger is optional infrastructure for inspecting trace waterfalls. Application
tracing remains disabled until an OTLP trace endpoint is configured. The `tracing`
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

#### Service tracing and waterfall coverage

After an owner-authorized rollout of the application images, set these values in
`.env` to opt in. This is configuration guidance; rebuilding or restarting the
application services is a separate rollout decision.

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4318/v1/traces
KADAN_TRACE_SAMPLE_RATIO=0.1
OTEL_SDK_DISABLED=false
```

Use `KADAN_TRACE_SAMPLE_RATIO=1` temporarily when inspecting a particular ordinary
request. Sampling is parent based: a downstream service follows its parent’s
sample decision. The ratio controls new root traces, not an incoming sampled
parent. Leave the endpoint empty, or set `OTEL_SDK_DISABLED=true`, to disable the
SDK, exporter thread, and automatic HTTP client instrumentation. A missing Jaeger
container does not prevent startup or request completion: export uses a bounded
512-span queue, batches of 64, a one-second exporter timeout and a one-second
batch schedule. A full queue drops telemetry; it never backpressures requests.
Shutdown drains that bounded queue on a background thread. Trace loss is expected
when collectors or workers disconnect or processes terminate abruptly.

| Service / boundary | Waterfall coverage | Limits |
| --- | --- | --- |
| Kadan frontend | Its API request appears at Kadan ingress; `x-kadan-trace-id` on the response identifies the trace | Browser rendering, browser-to-server network time, and nginx serving time are not measured |
| Kadan API | HTTP ingress through the last response body; outgoing HTTPX calls; graph context, inference, tool invocation and execution; image generation, recovery, saved-asset verification and completion; pet/frame generation | No LangChain/LangGraph payload callbacks, per-token spans, or prompt capture |
| Durable conversation, image and pet queues | Optional W3C parent persisted in the existing JSON job record; worker execution restores it; queue wait uses the persisted preclaim enqueue timestamp; image completion events retain causality | Existing records without a parent start a new trace. Recovery attempts are marked `kadan.recovery=true`; their full job age is not called queue wait |
| Tsuchi API | HTTP ingress, durable run/image mutations and message delivery, asset upload and export | SQLite statement text, document contents and individual filesystem operations are not traced |
| Tsuchi frontend | API ingress traces in `tsuchi-api` | Browser/nginx timing is not measured |
| Hikari API | HTTP ingress, inference lifetime, result presentation/delivery after the native result, worker FIFO wait | Result presentation is distinct from HTTP response serialization/stream delivery; transport delay is visible between worker and API intervals, not presented as a measured network metric |
| Hikari frontend | API ingress traces in `hikari-api` | Browser/nginx timing is not measured |
| All seven native workload workers: llm, decision, tts, speech, formatting, image, video | `hikari-worker-<kind>` resources; actual native request preparation, model load, request admission and batch execution intervals | Shared batch intervals overlap across participating requests; they are wall time, not exclusive GPU time |
| Language and formatting workers | Separate native prefill and decode intervals | Decode includes token selection/text streaming work; no per-token, kernel or inner-loop tracing |
| Speech recognition worker | Separate audio frontend, acoustic encode and decode intervals | No per-layer tracing |
| Image, video, decision and TTS engine internals | Native execution interval plus separate preparation/load/admission | Finer conditioning, denoise, VAE, codec, decision-stage and TTS subphases are not split in this change |
| Postgres, migration job, Jaeger and nginx containers | Application requests around those boundaries only | No database-query, migration, collector-self, or proxy spans |

Native workers receive W3C `traceparent` in the existing job envelope and retain it
through the FIFO and engine calls. Each sampled request buffers at most 16 fixed
phase records in memory and attaches them to its existing terminal result over
the authenticated WebSocket. No new worker thread, port, collector dependency,
or telemetry send is added during inference. The API validates the worker/job
association and parent, then exports those measured intervals using the originating
worker’s service resource. This is an API export relay, not an OTLP SDK running in
the native process. Native intervals appear after the terminal result, and can be
lost if the worker disconnects first. Retries can exceed the 16-record cap; extra
phase records are dropped. Queue intervals use native timestamps; clock skew
between machines can affect their placement in the waterfall.

HTTP and job propagation carries only `traceparent`; arbitrary baggage and vendor
`tracestate` are deliberately omitted. Context copying into asyncio tasks and
threadpool calls follows OpenTelemetry/Python contextvars. Queue workers restore
the persisted context before claiming work instead of inheriting the polling loop’s trace. Duplicate
submissions return the original durable record and parent; tracing does not
resubmit inference or alter cancellation, leases, image saving or completion
transactions.

The exporter has an attribute allowlist for route templates, HTTP method/status,
fixed operation/stage names, worker kind and recovery state. Span names use fixed
operations and server route templates. It strips all exception descriptions,
events, links and unapproved attributes, including automatic HTTPX URLs. No prompt,
completion, image/audio payload, tool arguments/results, file content, auth header,
credential, query string, arbitrary header, conversation ID or personal content is
exported. Error spans retain status without exception text. No payload callback or
SQL instrumentation is installed.

`kadan.kv_cache.cross_request_reuse=false` on language prefill describes the current
engine behavior: `text::Engine::begin_request()` resets the request’s recurrent/KV
state. It is not a measured cache hit/miss count. The native engine does not expose
measured cross-request KV hit/miss counters, so none are invented from the API’s
compatibility cache fields. Weight-cache reuse is a separate mechanism and is not
reported as KV reuse.

In Jaeger, select `kadan-api` and a recent trace, or copy `x-kadan-trace-id` from the
frontend request’s response in developer tools and search that trace ID. Expand
the Kadan graph/tool spans, HTTP child calls to Tsuchi/Hikari, native worker queue,
load/admission/execute spans, and language prefill/decode siblings. Compare the
intervals to identify where wall time is spent. Other entry points start under
`tsuchi-api` or `hikari-api`. Absence of a sampled trace is not evidence of a fast
request. Jaeger’s existing RAM-only retention and loopback UI limits still apply.

For local Python development, install the shared package before service dependencies:
`python -m pip install -e services/tracing`. Compose supplies it as the named
`tracing` build context; standalone API image builds need the equivalent
`--build-context tracing=<path-to-services/tracing>` option. No local collector is
needed for unit tests: `python -m pytest services/tracing/tests` uses an in-memory
exporter and fake boundaries. The implementation follows the official
[Python exporter guidance](https://opentelemetry.io/docs/languages/python/exporters/),
[SDK processor contract](https://opentelemetry.io/docs/specs/otel/trace/sdk/), and
[Python batch processor API](https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.export.html).

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
