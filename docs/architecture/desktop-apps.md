# Desktop Applications: Kadan Server & Kadan

Kadan delivers two standalone Tauri v2 desktop applications built as versioned GitHub Release artifacts (`.github/workflows/release-apps.yml`) with zero Docker or external Postgres dependency:

1. **Kadan Server** (`apps/kadan-server/`) — Hosts the full combined backend stack on a single port and embeds the Hikari and Tsuchi management interfaces in a system-tray desktop app.
2. **Kadan** (`apps/kadan-desktop/`) — Lightweight native client wrapping the unified `services/frontend/` UI with runtime server host configuration.

## Kadan Server (`apps/kadan-server/`)

### Process Supervision
`src-tauri/src/supervisor.rs` manages three service groups in dependency order with health checks, isolated restarts, graceful teardown on quit, and ring-buffered per-service logs:

1. **`api`** (`uvicorn combined.main:app` in `services/api/`) — Serves `/tsuchi`, `/hikari` (SQLite via `HIKARI_DATABASE_URL=sqlite:///<data_dir>/hikari.db`, WAL mode, automatic Alembic migrations via `HIKARI_AUTO_MIGRATE=true`), `/kadan`, and the built unified SPA (`KADAN_FRONTEND_DIR`) on a single configured port (default `8000`).
2. **`runner`** (`python -m kadan.runner`) — Executes LangGraph conversation turns and image/pet queues against the local combined API.
3. **`worker`** (native C++20/CUDA `kadan-worker` binary) — Inspects compiled workloads via `kadan-worker --list-workloads` and launches one isolated worker process per supported capability (`llm` Inkling 2 Mini, `lfm3_audio`, `lfm3_decision`, `formatting`, `speech`, `tts`, `image`, `video`; skipping GPU-only workloads on CPU-only machines). All workers connect to `ws://127.0.0.1:<port>/hikari/v1/workers` and share `<data_dir>/models` and Hikari's physical RAM/VRAM ledger. Hikari owns all model downloads, licenses, and provider settings; Kadan Server exposes only host/network/data-folder settings inside its **Overview** tab.

### Bundling
Because all inference engines (including Inkling 2 Mini, LFM3 Audio, and d1) are pure C++20/CUDA, the Python APIs require only lightweight packages (`fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `httpx`, `langgraph`, `Pillow`). `scripts/bundle-kadan-server.sh` (Linux/macOS) and `scripts/bundle-kadan-server.ps1` (Windows) stage:
- A standalone relocatable Python `venv`
- Minimal Python service sources (`api`, `tracing`, `tsuchi`, `hikari/api`, `kadan`)
- Built unified static frontend (`web/`)
- The installed `kadan-worker` runtime (`bin/kadan-worker`, `libexec/kadan-worker`, `lib/`, `share/kadan/`)

### Hikari SQLite Support
`services/hikari/api/database.py` honors `HIKARI_DATABASE_URL` when set (enabling SQLite WAL mode and foreign keys) while preserving `POSTGRES_*` configuration for existing Postgres deployments. ORM columns (`orm_models/base.py`) use SQLAlchemy `.with_variant(...)` so PostgreSQL keeps native `JSONB`/`ARRAY` and SQLite uses portable `JSON`.

## Kadan Desktop (`apps/kadan-desktop/`)

`apps/kadan-desktop/` wraps the built `services/frontend/dist` assets behind an embedded Rust loopback reverse proxy (`axum` + `reqwest` + `tokio-tungstenite`) bound to `127.0.0.1:<ephemeral>`:

- Injects `window.__KADAN_DESKTOP__ = true` into `index.html`, activating the **Server settings** dialog (`services/frontend/src/kadan/ServerSettings.tsx`) in the sidebar without adding Tauri packages to the web build.
- Exposes local control endpoints `/__kadan_desktop__/config` (`GET`/`PUT`) and `/__kadan_desktop__/test` (`POST`) backed by `servers.json` in the OS config directory.
- Relays `/kadan/*`, `/tsuchi/*`, and `/hikari/*` (REST, SSE streams, uploads/downloads, and WebSockets such as voice calls) to the active configured `Kadan Server` host URL.
- **Why loopback proxying matters:** Serving the UI from `http://127.0.0.1` keeps WebView in a secure context even when connecting to a LAN `Kadan Server` over plain `http://192.168.x.x:8000`, ensuring `getUserMedia` (microphone) and `AudioWorklet` continue working for voice input and voice calls without CORS or mixed-content errors.

See [Install and use](../user-expectations/install-and-use.md) for the user-facing expectations.
