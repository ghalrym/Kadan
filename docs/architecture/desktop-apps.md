# Desktop Applications: Kadan Server & Kadan

Kadan delivers two standalone Tauri v2 desktop applications built as versioned GitHub Release artifacts (`.github/workflows/release-apps.yml`) with zero Docker or external Postgres dependency:

1. **Kadan Server** (`apps/kadan-server/`) — Hosts the full combined backend stack on a single port and embeds the Hikari and Tsuchi management interfaces in a system-tray desktop app.
2. **Kadan** (`apps/kadan-desktop/`) — Lightweight native client wrapping the unified `services/frontend/` UI with runtime server host configuration.

## Kadan Server (`apps/kadan-server/`)

### Process Supervision
`src-tauri/src/supervisor.rs` manages three service groups in dependency order with health checks, isolated restarts, graceful teardown on quit, and ring-buffered per-service logs:

1. **`api`** (`uvicorn combined.main:app` in `services/api/`) — Serves `/tsuchi`, `/hikari` (SQLite via `HIKARI_DATABASE_URL=sqlite:///<data_dir>/hikari.db`, WAL mode, automatic Alembic migrations via `HIKARI_AUTO_MIGRATE=true`), `/kadan`, and the built unified SPA (`KADAN_FRONTEND_DIR`) on a single configured port (default `8000`).
2. **`runner`** (`python -m kadan.runner`) — Executes LangGraph conversation turns and image/pet queues against the local combined API.
3. **`worker`** (native C++20/CUDA `kadan-worker` binary) — Inspects compiled workloads via `kadan-worker --list-workloads` and launches one isolated worker process per supported capability (`llm` Inkling 2 Mini for language, vision, and single-pass decisions, `lfm3_audio`, `formatting`, `speech`, `tts`, `image`, `video`; skipping GPU-only workloads on CPU-only machines). All workers connect to `ws://127.0.0.1:<port>/hikari/v1/workers` and share `<data_dir>/models` and Hikari's physical RAM/VRAM ledger. Hikari owns all model downloads, licenses, and provider settings; Kadan Server exposes only host/network/data-folder settings inside its **Overview** tab.

### Bundling
Because all inference engines (including Inkling 2 Mini and LFM3 Audio) are pure C++20/CUDA, the Python APIs require only lightweight packages (`fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `httpx`, `langgraph`, `Pillow`). `scripts/bundle-kadan-server.sh` (Linux/macOS) and `scripts/bundle-kadan-server.ps1` (Windows) stage:
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

## Self-Updating & Agent Update Tools

`.github/workflows/release-apps.yml` stamps the computed release version into both apps' `tauri.conf.json` and `Cargo.toml`, archives macOS `.app.tar.gz` bundles alongside `.dmg`, and publishes a `release-manifest.json` with SHA-256 checksums for every release artifact.

- **Kadan Server (`src/updater.rs`, `src/control.rs`):** Runs a per-launch Bearer-authenticated loopback control server (`127.0.0.1:<ephemeral>`) passed to `api` and `runner` via `KADAN_SERVER_CONTROL_URL`, `KADAN_SERVER_CONTROL_TOKEN`, and `KADAN_APP_VERSION`. It checks GitHub Releases, downloads and SHA-256-verifies staged installers into `<data_dir>/updates/staging/`, gracefully stops supervised children before replacing and relaunching (Linux AppImage atomic swap, Windows NSIS silent `/S` + relaunch script, macOS `.app.tar.gz` swap), and tracks connected Kadan Desktop clients.
- **Kadan Desktop (`src/updater.rs`, `src/proxy.rs`):** Attaches `X-Kadan-Desktop-Version` and `X-Kadan-Desktop-Can-Update` headers to proxied API requests and exposes `/__kadan_desktop__/updates` and `/__kadan_desktop__/updates/install`. When Kadan Server records a pending desktop update, the frontend in Kadan Desktop acknowledges it and triggers the local desktop self-updater.
- **Two-phase agent update flow (`services/kadan/kadan/runner/tools/updates.py`):** `check_app_update` inspects server and connected desktop update availability. `install_app_update` stages and verifies the download first, delivers the agent's restart announcement to Tsuchi, durably completes the run (`end_run=True`), and triggers `/updates/apply` strictly after the run save completes—preventing restart replay loops or mid-inference kills. On replay after a crash, `install_app_update` refuses to re-run and directs the agent to call `check_app_update`.

See [Install and use](../user-expectations/install-and-use.md) for the user-facing expectations.
