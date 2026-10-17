# Kadan agent app

Kadan is the everyday agent interface in this repository. This service coordinates conversations and workspace browsing; Tsuchi stores the records and Hikari supplies inference. The React interface and FastAPI coordinator ship as separate frontend and API images.

## Available in this prototype

- Create and select agents, with each agent's records kept in its own Tsuchi namespace.
- Choose a saved image as an agent avatar, or generate a candidate with Hikari and save it in Tsuchi before selecting it.
- Create and edit personas, select one for a message, and reopen persisted conversations.
- Save a note without inference, or stream a response through Hikari's `/v1/chat/completions` API when the selected language model is available.
- Search memories and Spaces by text, browse Space children, and search file metadata or download a file.
- See actionable storage, tier, download and inference availability states, with links to the supporting operator interfaces.

A user message is saved before generation starts. An assistant response is saved only after a valid terminal event and stream completion; interrupted text remains visible as unsaved. Version checks prevent silently overwriting concurrent changes. Pending message IDs and drafts stay in browser session storage so an uncertain save can be reconciled after reload.

Selecting a tier and managing model downloads remain explicit actions in Hikari Settings. Kadan does not perform them automatically. It does not execute tools or shell commands, retrieve workspace content into prompts automatically, manage authentication, edit memories or files, render slide decks, or run autonomous tasks. Workspace browsing is read-only. Files are searched by metadata, not their contents. Chat requests are limited to 100 messages and 48,000 characters; longer conversations require a new thread, and the UI accepts 16,000 characters per message. Persona selection applies to the next request and is not a saved agent default.

## Agent avatars

Click an agent’s portrait in the header to select a saved image, return to initials, or generate a candidate. Upload existing images in Tsuchi, then refresh the picker. **Save avatar** persists the selection for that agent in Tsuchi; generation never replaces the selected avatar automatically. The picker can close while generation continues in the current page, and retains its candidate when reopened. Reloading the page loses the in-flight UI state; check the agent’s Tsuchi image library before generating again.

The asset contract is one existing verified raster asset ID, or `null` for initials. Avatars and picker thumbnails decode and draw a still frame into a square canvas; animated uploads therefore remain still too. The original asset stays unchanged in Tsuchi. This is whole-image motion, not an articulated character or sprite sheet: idle portraits move at most 4 pixels and tilt at most 2 degrees toward a mouse pointer. Movement resets on blur, pointer exit, touch, and activity changes. Reduced-motion and non-hover devices disable motion. Waiting for Hikari uses a dashed ring; actual response deltas use the working ring. No voice or speaking state is inferred.

Generation calls Hikari’s existing `/v1/images/generations` endpoint with one square image and its selected local model, then uploads the PNG to Tsuchi. It does not select/download a model or use a hosted generator. Hikari must already be configured for image inference. The request permits up to 30 minutes and a 20 MiB PNG; upstream failures are shown without changing the selected avatar. Closing the picker does not cancel inference. Generated composition, cropping, and likeness depend on the selected model and prompt; preview the result before saving. Trashing the selected asset in Tsuchi clears its avatar selection.

## Docker integration

Build from the repository root:

```sh
docker build -t kadan-api:local services/kadan
docker build -t kadan-frontend:local services/kadan/web
```

Compose runs these as `kadan-api` and `kadan-frontend`. The frontend image serves the compiled interface with nginx and proxies `/api`, `/docs` and `/openapi.json` to `kadan-api`.

The root Compose integration supplies the following environment. The upstream names are specific to the shared Compose network; browser URLs point at the loopback ports published on the host.

| Variable | Compose value | Purpose |
| --- | --- | --- |
| `KADAN_TSUCHI_URL` | `http://tsuchi-api:8000` | Storage API inside Docker |
| `KADAN_HIKARI_URL` | `http://hikari-api:8000` | Inference API inside Docker |
| `KADAN_TSUCHI_BROWSER_URL` | `http://127.0.0.1:8620` | Storage operator interface |
| `KADAN_HIKARI_BROWSER_URL` | `http://127.0.0.1:8088` | Hikari operator interface; Settings is `/settings` |
| `KADAN_ALLOWED_HOSTS` | `["127.0.0.1","localhost","::1"]` (default) | Exact HTTP request hosts and additional browser-link hosts |

The API image listens on `0.0.0.0:8630` inside its container and is not published; the frontend publishes `127.0.0.1:8630:8080`. It runs as UID/GID `10001:10001`, supports a read-only filesystem, and requires no volume: durable application records live in Tsuchi. Use `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]` and `restart: unless-stopped`. The image healthcheck requests `http://127.0.0.1:8630/api/health`; this checks the coordinator process. `/api/status` separately reports upstream availability without starting inference.

Upstream URLs accept loopback HTTP or their respective exact Compose endpoint above. Browser URLs accept loopback HTTP and hosts explicitly listed in `KADAN_ALLOWED_HOSTS`. The service rejects other HTTP Host headers and cross-origin writes; these guards do not replace authentication.

For owner-authorized access from another computer on a trusted LAN, use a deployment-specific Compose override to publish `kadan-frontend` on the server's exact LAN address, mapping host port 8630 to container port 8080. Publish `tsuchi-frontend` and `hikari-frontend` on that address at their respective host ports, also targeting container port 8080. Keep the API services internal. Add the address to `KADAN_ALLOWED_HOSTS` on `kadan-api`, retaining loopback for the image healthcheck. Set both browser URLs there to that server address and the published Tsuchi/Hikari ports; keep the upstream URLs on Docker DNS. Add the address to `TSUCHI_ALLOWED_HOSTS` on `tsuchi-api`, retaining its existing loopback and `tsuchi-api` entries. The frontend proxies preserve the browser's Host header, and the frontend APIs use their current page origin, so no CORS relaxation is needed. Keep the services off public interfaces and do not configure router forwarding. Root Compose defaults and durable volumes remain unchanged.

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
| `GET, POST /api/agents` | List or create agents, including the saved avatar asset ID |
| `PUT /api/agents/{key}/avatar` | Select an active image by `asset_id`, or clear with `null` |
| `POST /api/agents/{key}/avatar/generations` | Generate a local image from `prompt`, store it in Tsuchi, and return the unselected asset |
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

Automated checks are isolated unit tests: Python uses an in-memory mocked HTTP transport; frontend tests exercise SSE parsing, avatar movement bounds, activity mapping, and identifier generation. They never persist fixtures or call a model. Root `python services/hikari/bin/hikari.py check format` also covers this service.

The standalone prototype was manually checked on desktop against temporary isolated storage for agent/persona creation, persona editing, conversation persistence, memory search, Space navigation and file metadata browsing. This monorepo packaging still needs owner verification with the combined Compose stack. Mobile rendering, browser download completion and real inference have not been verified. No model was downloaded or tier selected during prototype work.
