# Kadan agent app

Kadan is the everyday agent interface in this repository. This service coordinates conversations and workspace browsing; Tsuchi stores the records and Hikari supplies inference. The React interface and FastAPI coordinator ship as separate frontend and API images.

## Available in this prototype

- Create and select agents, with each agent's records kept in its own Tsuchi namespace.
- Text each agent in one continuous message chain. There are no threads: the chain is the agent's whole history with you.
- Create and edit personas and select one for a message.
- Save a note without inference, or stream a response through Hikari's `/v1/chat/completions` API when the selected language model is available.
- Search memories and Spaces by text, browse Space children, and search file metadata or download a file.
- See actionable storage, tier, download and inference availability states, with links to the supporting operator interfaces.
- Each agent has an animated pet that reacts to chat activity. The player reads the Codex pet atlas (8 columns of 192x208 cells; v1 9 rows, v2 11 rows with look directions). Until an agent has its own pet, Kadan draws a pixel hatchling coloured from the agent key.

A user message is saved before generation starts. An assistant response is saved only after a valid terminal event and stream completion; interrupted text remains visible as unsaved. Version checks prevent silently overwriting concurrent changes. Pending message IDs and drafts stay in browser session storage so an uncertain save can be reconciled after reload.

Selecting a tier and managing model downloads remain explicit actions in Hikari Settings. Kadan does not perform them automatically. The model has no tools: chat sends only the persona and conversation text to Hikari. Kadan does not install servers, execute shell commands, manage authentication, edit arbitrary binary files, render slide decks, or run unrelated autonomous background tasks. Pet creation is a durable background job started explicitly through the UI. The workspace browser remains read-only. Files are searched by metadata, not their contents. Each request replays at most the latest 100 messages and 48,000 characters of the chain, and the UI accepts 16,000 characters per message. Persona selection applies to the next request and is not a saved agent default.

## Pet creation

Open **Edit pet**. The left column lists **New pet**, the default **Hatchling** and
every candidate with its progress; the right side previews the chosen one. Under
**New pet**, **Describe** the character and art style, then choose **Create pet**.
Creation runs independently of the dialog. The existing pet stays in
use until the user previews the candidate and chooses **Use pet**. The preview
plays all nine animation states and exposes all sixteen look directions. **Use
hatchling** resets selection. Selected pets appear in the roster, home and chat.

To bring over a pet made for ChatGPT or Codex, choose **New pet → Import** and drop
or choose its folder (`pet.json` plus the spritesheet), a zip of that folder, or the
spritesheet alone. The browser previews the sheet before upload. Kadan follows
`spritesheetPath`, takes the name and description from `pet.json` unless you rename
it, checks the atlas is 1536x1872 (v1) or 1536x2288 (v2), and stores it re-encoded
as lossless WebP. The import becomes a ready candidate; choose **Use pet** to switch.
v1 sheets have no look directions, so they do not follow the cursor. Imports run no
inference and cannot be repaired frame by frame.

Kadan generates a canonical base through Hikari, then generates each articulated
pose on its own 384x416 canvas using Hikari's native reference-image conditioning.
Up to four independent poses are submitted together so Hikari can batch them.
Each request ID and seed is persisted before submission, and each completed frame
is saved independently. Cancellation and resume fence every pending request.
Update the Kadan and Tsuchi backends together for this additional pending-frame
state; existing saved jobs load without a data migration.
The two neutral idle endpoints reuse the canonical artwork with explicit provenance.
Kadan removes the chroma background and packs validated cells into a 1536x2288 v2
atlas without fitting each silhouette independently. Structural checks do not prove
identity, articulation or gaze quality; the preview is the final quality decision.

For a completed inconsistent frame, choose it under **Repair a frame**, describe
what needs fixing and choose **Regenerate frame**. This creates a revision retaining
every other saved frame, the rejected artwork and its provenance. The correction
reaches that frame's generation prompt and the retry uses a different seed. The
previous candidate and active selection remain unchanged. Neutral idle repairs use
the canonical base; already canonical endpoints do not need regeneration.

Tsuchi persists the base, individual raw/prepared frames, generation settings and seeds,
atlas, typed manifest, and versioned job records. A worker lease prevents another
coordinator from taking live work. Cancellation waits for Hikari cleanup;
unconfirmed cleanup remains visible. Expired work becomes interrupted and needs
explicit resume. Resume reuses completed steps and reconciles saved output before
submitting another request. A failed job never changes the current pet. Moving
referenced artwork to trash clears its selection in Tsuchi.

API: under `/api/agents/{key}`, `GET/POST /pets`, `GET /pets/{id}`, and
`POST /pets/{id}/cancel` or `/pets/{id}/resume` manage candidates. `POST /pets/imports`
takes multipart `id` (caller UUID), optional `name` and `files` (one zip, or a
spritesheet with an optional `pet.json`) and returns the ready candidate with HTTP 201. Creation accepts a caller
UUID and prompt and returns immediately with HTTP 202. `POST /pets/{id}/revisions` accepts `{id, expected_version, frame_names, reason}` and returns a queued revision, preserving unrejected frames. `GET/PUT /api/agents/{key}/pet` reads or explicitly selects a candidate with
`{ "pet_id": "candidate-uuid" }`; `{ "pet_id": null }` resets selection. Generation settings use the existing
Hikari model; all inference stays in its native worker. No additional model
runtime is installed by this feature.

## Chat

Each agent has exactly one conversation, the chain, which `GET /api/agents/{key}/chain`
returns and creates on first use. The resources API does not create conversations.
The UI reads like texting: a timestamp separates messages more than four hours
apart, the newest message you sent shows **Sending…**, **Delivered** once saved,
then **Read** once Hikari is working on it or has replied. While it works the agent
shows as typing, and its reply appears whole when it is saved. A note is delivered
but stays unread until the next reply.

Each message is one streamed completion: Kadan sends the optional persona as a
system message, then the chain's recent user and assistant text. The chain never
ends, so older messages drop out of the prompt: the replayed window opens on a
multiple of 20 messages, so its start, and the prompt prefix, stays put for many
turns. Saved history is never trimmed. There is no
planning step and the model cannot call tools. Conversations saved while the
retired plugin system existed may contain `tool` messages; Kadan keeps them in
Tsuchi but neither shows them nor sends them to the model.

Hikari serves one request at a time and loads the selected model on demand, so
the first reply can wait for a model load. Kadan sets no read deadline on chat
requests; while it waits, the chat shows when Hikari reports the model loading.
Stop disconnects, which cancels the request in Hikari.

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
| `GET /api/status` | Storage and inference readiness, including browser links and whether the selected model is loaded (`hikari.model_state`) |
| `GET, POST /api/agents` | List or create agents |
| `GET, POST /api/agents/{key}/resources` | Browse/search resources; create a persona or conversation |
| `GET, PUT /api/agents/{key}/resources/{id}` | Read a resource; update a persona with an expected version |
| `GET /api/agents/{key}/assets` | Search active file metadata |
| `GET /api/agents/{key}/assets/{id}` | Read file metadata |
| `GET /api/agents/{key}/assets/{id}/content` | Stream content; `download=true` requests an attachment |
| `POST /api/agents/{key}/conversations/{id}/notes` | Append a user note with an expected version and message ID |
| `POST /api/agents/{key}/conversations/{id}/chat` | Save a user message, then stream inference and save a completed response |

Chat requests include `content`, `expected_version`, an optional `message_id`, and an explicit `persona_id` UUID or `null`. Streaming events are `user_saved`, `status`, `delta`, `done`, and `error`; `status` carries a progress message. `done` includes the persisted conversation and `finish_reason`; a missing `done` must not be presented as a saved response. The service's `/docs` exposes the request schemas.

## Validation

From this directory:

```sh
.venv/bin/python -m pytest
npm test --prefix web
npm run build --prefix web
```

Automated checks are isolated unit tests: Python uses an in-memory mocked HTTP transport; frontend tests exercise SSE parsing with synthetic streams. They never persist fixtures or call a model. Root `python services/hikari/bin/hikari.py check format` also covers this service.

The standalone prototype was manually checked on desktop against temporary isolated storage for agent/persona creation, persona editing, conversation persistence, memory search, Space navigation and file metadata browsing. This monorepo packaging still needs owner verification with the combined Compose stack. Mobile rendering, browser download completion and real inference have not been verified. No model was downloaded or tier selected during prototype work.
