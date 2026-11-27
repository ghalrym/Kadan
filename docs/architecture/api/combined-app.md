# Combined API

`services/api` serves Kadan, Tsuchi and Hikari from one FastAPI application in
one `api` container. Each service keeps its own app, routes, settings, database
and documentation; the combined app only mounts them and runs their startup and
shutdown. Running one process and one image is simpler to deploy and operate
than three, at the cost of the shared restarts described below.

| Prefix | Service | App |
| --- | --- | --- |
| `/kadan` | Kadan | `kadan.api.server:app` |
| `/tsuchi` | Tsuchi | `tsuchi.main:app` |
| `/hikari` | Hikari | `api.server:app` |

Each service's architecture pages describe its paths relative to its own app, so
Hikari's `/v1/chat/completions` is served at `/hikari/v1/chat/completions`. Each
reference is at `<prefix>/docs` and `<prefix>/openapi.json`; the root has none.
There are no unprefixed aliases, so direct Hikari clients and remote workers use
`/hikari/v1/...` and `ws://API_HOST:8000/hikari/v1/workers`.

## Startup and shutdown

Starlette does not run a mounted app's lifespan, so `combined/mounting.py` enters
each service's lifespan in order (Tsuchi, Hikari, then Kadan, which calls the
other two) and leaves them in reverse. If one fails to start, those already
started are shut down. Kadan's runner stays a separate container: it is a
long-running worker, not an API.

## Calls between services

Kadan still calls Tsuchi and Hikari over HTTP, at
`http://127.0.0.1:8000/tsuchi` and `/hikari` inside the container (the runner uses
`http://api:8000/...`). Keeping HTTP preserves the service boundaries and lets the
services be separated again; nothing imports another service's code. Tsuchi's
handlers are synchronous and run in the thread pool, so its SQLite work does not
block Hikari's worker connections on the shared event loop.

Code that builds a URL to its own service uses the request's `root_path` (the
mount prefix) rather than assuming it is served at `/`, such as Kadan's MCP OAuth
callback. Hikari's request telemetry and the
shared request tracing strip or keep that prefix explicitly.

## Shared process

- **Restarts.** Uvicorn reloads the whole process when any service's Python source
  changes. Reloading drops Hikari's worker connections, which cancels inference in
  progress; the workers reconnect.
- **Tracing.** OpenTelemetry allows one tracer provider per process. The combined
  app configures it as `kadan-services`; each service's own `configure` call then
  returns nothing and its shutdown is a no-op. Request spans keep the prefix in
  `http.route`, so the service remains identifiable.
- **Dependencies.** One Python 3.14 environment (`services/api/requirements.txt`)
  holds every service's pins; keep it in step with each service's own
  requirements file. The image also carries the Claude Code and Codex CLIs for
  Hikari's subscription providers.
- **Migrations.** `hikari-migrate` runs Hikari's Alembic migrations from the same
  image before `api` starts.

The image is built from `services/api` with the `hikari`, `tsuchi`, `kadan` and
`tracing` named build contexts, so each service's code stays in its own folder.
Compose bind-mounts those folders for hot reload.
