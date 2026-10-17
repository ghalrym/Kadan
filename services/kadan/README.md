# Kadan agent app

Kadan is the everyday agent interface in this repository. This service coordinates conversations and workspace browsing; Tsuchi stores the records and Hikari supplies inference. The React interface and FastAPI coordinator ship together in one image.

## Available in this prototype

- Create and select agents, with each agent's records kept in its own Tsuchi namespace.
- Create and edit personas, select one for a message, and reopen persisted conversations.
- Save a note without inference, or stream a response through Hikari's `/v1/chat/completions` API when the selected language model is available.
- Search memories and Spaces by text, browse Space children, and search file metadata or download a file.
- See actionable storage, tier, download and inference availability states, with links to the supporting operator interfaces.

A user message is saved before generation starts. An assistant response is saved only after a valid terminal event and stream completion; interrupted text remains visible as unsaved. Version checks prevent silently overwriting concurrent changes. Pending message IDs and drafts stay in browser session storage so an uncertain save can be reconciled after reload.

Selecting a tier and managing model downloads remain explicit actions in Hikari Settings. Kadan does not perform them automatically. It does not execute tools or shell commands, retrieve workspace content into prompts automatically, manage authentication, edit memories or files, render slide decks, or run autonomous tasks. Workspace browsing is read-only. Files are searched by metadata, not their contents. Chat requests are limited to 100 messages and 48,000 characters; longer conversations require a new thread, and the UI accepts 16,000 characters per message. Persona selection applies to the next request and is not a saved agent default.

## Docker integration

Build from the repository root:

```sh
docker build -t kadan-agent:local services/kadan
```

The root Compose integration supplies the following environment. The upstream names are specific to the shared Compose network; browser URLs point at the loopback ports published on the host.

| Variable | Compose value | Purpose |
| --- | --- | --- |
| `KADAN_TSUCHI_URL` | `http://tsuchi:8000` | Storage API inside Docker |
| `KADAN_HIKARI_URL` | `http://api:8000` | Inference API inside Docker |
| `KADAN_TSUCHI_BROWSER_URL` | `http://127.0.0.1:8620` | Storage operator interface |
| `KADAN_HIKARI_BROWSER_URL` | `http://127.0.0.1:8088` | Hikari operator interface; Settings is `/settings` |
| `KADAN_FRONTEND_DIRECTORY` | `/app/web/dist` | Compiled interface |
| `KADAN_ALLOWED_HOSTS` | `["127.0.0.1","localhost","::1"]` (default) | Exact HTTP request hosts and additional browser-link hosts |

The image listens on `0.0.0.0:8630` inside its container. Root Compose publishes it as `127.0.0.1:8630:8630` by default. It runs as UID/GID `10001:10001`, supports a read-only filesystem, and requires no volume: durable application records live in Tsuchi. Use `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]` and `restart: unless-stopped`. The image healthcheck requests `http://127.0.0.1:8630/api/health`; this checks the coordinator process. `/api/status` separately reports upstream availability without starting inference.

Upstream URLs accept loopback HTTP or their respective exact Compose endpoint above. Browser URLs accept loopback HTTP and hosts explicitly listed in `KADAN_ALLOWED_HOSTS`. The service rejects other HTTP Host headers and cross-origin writes; these guards do not replace authentication.

For owner-authorized access from another computer on a trusted LAN, use a machine-specific Compose override to publish port 8630 on the server's exact LAN address. Add that address to `KADAN_ALLOWED_HOSTS`, retaining loopback for the image healthcheck. Set both browser URLs to that server address and the published Tsuchi/Hikari ports; keep the upstream URLs on Docker DNS. Tsuchi also needs that address in `TSUCHI_ALLOWED_HOSTS` and a matching LAN port binding. The frontend APIs use their current page origin, so no CORS relaxation is needed. Keep the services off public interfaces and do not configure router forwarding. Root Compose defaults and durable volumes remain unchanged.

On plain HTTP LAN pages, Hikari microphone recording and clipboard actions remain subject to browser secure-context requirements. Those features need trusted HTTPS; ordinary API access and audio-file upload do not require enabling microphone access. This change does not configure HTTPS.

## Local development

Requires Python 3.12 or newer and Node.js 24. From `services/kadan`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm ci --prefix web
npm run build --prefix web
.venv/bin/python -m uvicorn kadan.api.server:app --host 127.0.0.1 --port 8630
```

Open `http://127.0.0.1:8630`. Local upstream defaults are Tsuchi on port 8620 and Hikari on port 8088. `.env.example` lists the available settings. For frontend development, `npm run dev --prefix web` starts Vite on loopback and proxies `/api` to port 8630.

## API boundary

| Route | Behavior |
| --- | --- |
| `GET /api/health` | Coordinator liveness |
| `GET /api/status` | Storage and inference readiness, including browser links |
| `GET, POST /api/agents` | List or create agents |
| `GET, POST /api/agents/{key}/resources` | Browse/search resources; create a persona or conversation |
| `GET, PUT /api/agents/{key}/resources/{id}` | Read a resource; update a persona with an expected version |
| `GET /api/agents/{key}/assets` | Search active file metadata |
| `GET /api/agents/{key}/assets/{id}` | Read file metadata |
| `GET /api/agents/{key}/assets/{id}/content` | Stream content; `download=true` requests an attachment |
| `POST /api/agents/{key}/conversations/{id}/notes` | Append a user note with an expected version and message ID |
| `POST /api/agents/{key}/conversations/{id}/chat` | Save a user message, then stream inference and save a completed response |

Chat requests include `content`, `expected_version`, an optional `message_id`, and an explicit `persona_id` UUID or `null`. Streaming events are `user_saved`, `delta`, `done`, and `error`. `done` includes the persisted conversation and `finish_reason`; a missing `done` must not be presented as a saved response. The service's `/docs` exposes the request schemas.

## Validation

From this directory:

```sh
.venv/bin/python -m pytest
npm test --prefix web
npm run build --prefix web
```

Automated checks are isolated unit tests: Python uses an in-memory mocked HTTP transport; frontend tests exercise SSE parsing with synthetic streams. They never persist fixtures or call a model. Root `python bin/hikari.py check format` also covers this service.

The standalone prototype was manually checked on desktop against temporary isolated storage for agent/persona creation, persona editing, conversation persistence, memory search, Space navigation and file metadata browsing. This monorepo packaging still needs owner verification with the combined Compose stack. Mobile rendering, browser download completion and real inference have not been verified. No model was downloaded or tier selected during prototype work.
