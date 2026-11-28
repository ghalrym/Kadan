# Desktop application: Kadan

Kadan ships as one Tauri v2 desktop app (`apps/kadan/`), built as versioned GitHub Release artifacts by `.github/workflows/release-apps.yml` with no Docker or external Postgres dependency. The same app either runs Kadan's services on its own computer or connects to another computer running them. It replaced the separate Kadan Server and Kadan Desktop apps; see [Install and use](../user-expectations/install-and-use.md) for what people see.

## Two modes, one proxy

`config.rs` stores a `mode` beside the saved servers in the client's `servers.json`:

- **`host`** ("This computer"): the app supervises the combined stack itself and every request goes to `http://127.0.0.1:<server_port>`.
- **`connect`** ("Another computer"): nothing heavy starts, and requests go to the chosen `active_host`.

Settings saved before the merge have no `mode`. A client pointed at a loopback address hosts; one pointed elsewhere keeps connecting. A computer that only ran Kadan Server had no client settings, so it gets the default, `host`. Its server settings (`server-config.json` under the old Kadan Server config directory) and data folder are reused unchanged, so existing installs keep their data.

Both modes share the embedded loopback reverse proxy (`proxy.rs`, `axum` + `reqwest` + `tokio-tungstenite`, bound to `127.0.0.1:<ephemeral>`). It serves the built `services/frontend/dist`, injects `window.__KADAN_DESKTOP__ = true`, and relays `/kadan/*`, `/tsuchi/*` and `/hikari/*` (REST, SSE, uploads, downloads and WebSockets) to `ProxyState::upstream()`, which resolves the mode on every request. Serving from `http://127.0.0.1` keeps the webview in a secure context even when the other computer is plain `http://192.168.x.x`, so microphone capture and `AudioWorklet` work for voice input and calls.

Changing the mode through `PUT /__kadan_desktop__/config` starts or stops the local services immediately and repoints the local Work sidecar, so no restart is needed.

## Hosting: supervising Kadan's services

`supervisor.rs` manages three service groups in dependency order, with health checks, isolated restarts, teardown on quit and ring-buffered per-service logs:

1. **`api`** (`uvicorn combined.main:app` in `services/api/`): serves `/tsuchi`, `/hikari` (SQLite via `HIKARI_DATABASE_URL=sqlite:///<data_dir>/hikari.db`, WAL mode, automatic Alembic migrations) and `/kadan` on one port (default `8000`).
2. **`runner`** (`python -m kadan.runner`): runs conversation turns and image and pet queues against the local API.
3. **`worker`** (native C++20/CUDA `kadan-worker`): lists its compiled workloads and launches one process per supported capability, skipping GPU-only workloads on computers without a GPU. Workers connect to `ws://127.0.0.1:<port>/hikari/v1/workers` and share `<data_dir>/models` and Hikari's memory ledger.

The web pages control hosting through native endpoints on the same proxy, which keeps every flow in the web UI: `GET /__kadan_desktop__/host` (mode, service status, settings, hardware), `PUT /__kadan_desktop__/host/settings` (network access, port, data folder, start with the app, launch at login through `tauri-plugin-autostart`), `POST …/host/start`, `…/host/stop`, `…/host/restart` and `GET …/host/logs`. Kadan Server's [Settings](../user-expectations/kadan-server/settings.md) page shows them only inside the app.

While hosting, closing the window hides it and the tray icon (`tray.rs`: **Open Kadan**, **Quit Kadan**) keeps Kadan reachable; quitting stops the services. In `connect` mode, closing the window quits.

## Bundling

Because the inference engines are native C++, the Python services need only light packages. `scripts/bundle-kadan-runtime.sh` (Linux and macOS) and `scripts/bundle-kadan-runtime.ps1` (Windows) stage `apps/kadan/src-tauri/resources/runtime/`:

- a relocatable Python `venv`
- the Python service sources (`api`, `tracing`, `tsuchi`, `hikari/api`, `kadan`)
- the built web app (`web/`)
- the installed `kadan-worker` runtime (`bin/`, `libexec/`, `lib/`, `share/kadan/`)

The release job bundles that runtime into every Linux and Windows installer, so one download can host or connect. The app is larger for people who only connect; that is accepted for now. The local Work sidecar (`work_sidecar.rs`) also finds its Python and services in that runtime. During development, `supervisor.rs` and `work_sidecar.rs` fall back to the repository's `.venv` and `services/`.

### Hikari SQLite support

`services/hikari/api/database.py` honors `HIKARI_DATABASE_URL` when set (enabling SQLite WAL mode and foreign keys) while keeping `POSTGRES_*` configuration for Postgres deployments. ORM columns use SQLAlchemy `.with_variant(...)` so PostgreSQL keeps native `JSONB`/`ARRAY` and SQLite uses portable `JSON`.

## Updates

The release workflow stamps the version into `apps/kadan`, archives macOS `.app.tar.gz` bundles beside `.dmg`, and publishes `release-manifest.json` with SHA-256 checksums under one app key, `kadan`.

- **Hosting (`host_updater.rs`, `control.rs`):** a per-launch, bearer-authenticated loopback control server is passed to `api` and `runner` as `KADAN_SERVER_CONTROL_URL` and `KADAN_SERVER_CONTROL_TOKEN`. It checks GitHub Releases, downloads and verifies the `Kadan` installer into `<data_dir>/updates/staging/`, stops the supervised services, replaces the app (Linux AppImage swap, Windows NSIS silent install and relaunch, macOS `.app.tar.gz` swap), and tracks Kadan apps connected to it so they update too.
- **Connected apps (`updater.rs`):** only in `connect` mode does the proxy send `X-Kadan-Desktop-Version` and `X-Kadan-Desktop-Can-Update`, so a hosting app never counts itself as a remote client. When the computer it connects to records a pending update, the web app acknowledges it and runs the local self-updater.
- **Agent update flow (`services/kadan/kadan/runner/tools/updates.py`):** `check_app_update` inspects availability for this computer and connected apps. `install_app_update` stages and verifies first, delivers the agent's restart announcement, completes the run, and only then triggers `/updates/apply`, so a restart cannot replay mid-inference.
