# Kadan agent app

Kadan is the everyday agent interface in this repository. This service coordinates conversations and workspace browsing; Tsuchi stores the records and Hikari supplies inference. The React interface and FastAPI coordinator ship as separate frontend and API images.

## Available in this prototype

- Create and select agents, with each agent's records kept in its own Tsuchi namespace.
- Create and edit personas, select one for a message, and reopen persisted conversations.
- Save a note without inference, or stream a response through Hikari's `/v1/chat/completions` API when the selected language model is available.
- Search memories and Spaces by text, browse Space children, and search file metadata or download a file.
- Enable configured plugins per thread, inspect their skills/setup, and watch tool calls and results in the conversation.
- Let the agent search/read/create/update memories and Space documents through the built-in MCP server.
- See actionable storage, tier, download and inference availability states, with links to the supporting operator interfaces.
- Each agent has an animated pet that reacts to thread activity. The player reads the Codex pet atlas (8 columns of 192x208 cells; v1 9 rows, v2 11 rows with look directions). Until an agent has its own pet, Kadan draws a pixel hatchling coloured from the agent key.

A user message is saved before generation starts. An assistant response is saved only after a valid terminal event and stream completion; interrupted text remains visible as unsaved. Version checks prevent silently overwriting concurrent changes. Pending message IDs and drafts stay in browser session storage so an uncertain save can be reconciled after reload.

Selecting a tier and managing model downloads remain explicit actions in Hikari Settings. Kadan does not perform them automatically. The agent can use enabled MCP tools to retrieve context and edit memories or Space documents. Kadan does not install servers, execute shell commands, manage authentication, edit arbitrary binary files, render slide decks, or run unrelated autonomous background tasks. Pet creation is a durable background job started explicitly through the UI or enabled pet tool. The workspace browser remains read-only. Files are searched by metadata, not their contents. Chat requests are limited to 100 messages and 48,000 characters; longer conversations require a new thread, and the UI accepts 16,000 characters per message. Persona selection applies to the next request and is not a saved agent default.

## Pet creation

Open **Edit pet**. The left column lists **New pet**, the default **Hatchling** and
every candidate with its progress; the right side previews the chosen one. Under
**New pet**, **Describe** the character and art style, then choose **Create pet**.
Creation runs independently of the dialog or MCP turn. The existing pet stays in
use until the user previews the candidate and chooses **Use pet**. The preview
plays all nine animation states and exposes all sixteen look directions. **Use
hatchling** resets selection. Selected pets appear in the roster, home and chat.

The built-in plugin exposes `pet_create`, `pet_status`, `pet_cancel` and
`pet_resume`, scoped to the current agent. It has no activation tool. Retain the
creation UUID to check or resume the same job after an ambiguous response.

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

## Plugin execution

Plugins are explicit server configuration, with a stable ID, display metadata,
setup instructions, skill instructions, and an MCP server. The default `kadan`
plugin exposes twelve tools: `memory_search`, `memory_read`, `memory_create`,
`memory_update`, the corresponding `workspace_*` tools, and the four pet tools
listed above. Names are qualified
as `kadan__memory_search` when presented to the model. Workspace tools operate on
Tsuchi Space Markdown documents, not host filesystem paths. The selected agent
key is bound when the built-in server is created; models cannot supply another
namespace. Updates require a fresh `expected_version` and retain all metadata
outside the title/Markdown edit. Searches return excerpts and pagination;
reads page through Markdown by character offset.

The chat composer lists configured plugins and saves per-thread deselections in
this browser. **All enabled tools run with full access and no per-call approval.**
Each MCP server owns its own restrictions. Disabling a plugin affects later turns;
Stop cancels the current turn. Cancellation cannot undo completed writes.

`KADAN_PLUGINS` accepts a JSON array of manifests. Leaving it unset uses the
built-in plugin; `[]` disables all plugins. To add an explicitly configured server,
provide the full desired list (maximum 16). For example, the following configuration
keeps the built-in tools and adds an already-running local Streamable HTTP server:

```json
[
  {
    "id": "kadan",
    "name": "Memory and workspace",
    "description": "Agent memories and Space documents in Tsuchi",
    "enabled": true,
    "server": {"transport": "builtin"},
    "setup": {"instructions": "Requires Tsuchi."},
    "skills": [{"name": "Workspace", "description": "Use saved context", "instructions": "Search and read before updating. Use the returned version and preserve concurrent changes."}]
  },
  {
    "id": "local-tools",
    "name": "Local tools",
    "description": "An owner-configured MCP server",
    "enabled": true,
    "server": {"transport": "streamable_http", "url": "http://127.0.0.1:9000/mcp"},
    "setup": {"instructions": "Start the reviewed MCP server before enabling it."},
    "skills": []
  }
]
```

Set this on the Kadan API process (or in an owner-managed Compose override).
Container loopback refers to that container, so use a reachable endpoint there.
Kadan never installs or launches the configured server. This slice has no stdio
subprocesses, OAuth, credential fields, account connections or marketplace.
The catalog is metadata-only: opening it does not contact configured servers.
Connections open for the selected turn and close before the final reply.

The implementation uses the [official Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
1.30 client, server and transports, including an in-memory client/server connection
for built-in tools. The built-in MCP server is internal to Kadan; this change does
not publish a general-purpose MCP HTTP endpoint. Tool discovery follows
[MCP tools/list and tools/call](https://modelcontextprotocol.io/specification/2025-11-25/server/tools),
with paginated discovery, local JSON-schema validation and bounded calls. Remote
schema references are rejected; redirects and environment proxy inheritance are
disabled. No external MCP server was connected while implementing this feature.

Hikari currently exposes text completions without native tool-call fields. Kadan
uses a separate, bounded planning completion that must return exactly
`{"type":"tool","name":"...","arguments":{...}}` or `{"type":"reply"}`.
Hikari's complete GLM reasoning envelope is removed before validating the final JSON; unfinished reasoning is rejected. Persona, skill and planner instructions are combined into one leading system message for Hikari's non-GLM templates. Malformed, incomplete or oversized decisions never dispatch a tool. Each result
feeds the next planning step; a reply decision starts ordinary streamed text
completion. Skill instructions accompany the turn. This adds at least one
planning completion when tools are selected and depends on the selected model
following the JSON instruction; it is not grammar-constrained generation.
Model quality and latency require owner review with real inference.

A turn permits eight calls, 64 discovered tools, a 15-second discovery timeout
per plugin, and a 30-second timeout per call. Planner and final context stay within
the existing 100-message/48,000-character budget. Tool results are limited to
32,000 serialized characters. Errors and version conflicts return to the model
as tool results; malformed decisions and exhausted budgets end the turn visibly.
A call timeout ends the turn and closes its MCP sessions before any further planning or execution; its saved intent remains an unknown outcome. There are no automatic write retries.

Tool intent is appended to the Tsuchi conversation before dispatch, then its
outcome is appended before the next decision. Both use the conversation's version
check. An interrupted call, uncertain storage response or failed result save can
leave a started record with no outcome. The UI shows **Outcome unknown** after
reload; inspect current storage before repeating a write. The final assistant
reply is saved only after its complete stream. Tool traces remain as `tool`
messages with JSON execution records and are replayed as explicitly labeled data
because Hikari only accepts system/user/assistant text. API clients can observe
progress and durable activity through the new SSE events.

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
| `GET /api/plugins` | Configured plugin catalog, skills, setup and MCP transport metadata |
| `GET /api/status` | Storage and inference readiness, including browser links |
| `GET, POST /api/agents` | List or create agents |
| `GET, POST /api/agents/{key}/resources` | Browse/search resources; create a persona or conversation |
| `GET, PUT /api/agents/{key}/resources/{id}` | Read a resource; update a persona with an expected version |
| `GET /api/agents/{key}/assets` | Search active file metadata |
| `GET /api/agents/{key}/assets/{id}` | Read file metadata |
| `GET /api/agents/{key}/assets/{id}/content` | Stream content; `download=true` requests an attachment |
| `POST /api/agents/{key}/conversations/{id}/notes` | Append a user note with an expected version and message ID |
| `POST /api/agents/{key}/conversations/{id}/chat` | Save a user message, then stream inference and save a completed response |

Chat requests include `content`, `expected_version`, an optional `message_id`, an explicit `persona_id` UUID or `null`, and optional `plugin_ids`. Omitting `plugin_ids` enables all server-enabled plugins; `[]` disables tools for that turn. Unknown, duplicate or server-disabled plugin IDs are rejected before saving the user message. Streaming events are `user_saved`, `status`, `tool_started`, `tool_finished`, `delta`, `done`, and `error`. Tool events carry the persisted `conversation` and an `execution` record; `status` carries a progress message. `done` includes the persisted conversation and `finish_reason`; a missing `done` must not be presented as a saved response. The service's `/docs` exposes the request schemas.

## Validation

From this directory:

```sh
.venv/bin/python -m pytest
npm test --prefix web
npm run build --prefix web
```

Automated checks are isolated unit tests: Python uses an in-memory mocked HTTP transport; frontend tests exercise SSE parsing with synthetic streams. They never persist fixtures or call a model. Root `python services/hikari/bin/hikari.py check format` also covers this service.

The plugin change was verified with isolated unit fixtures and frontend builds. A temporary static preview with in-memory HTTP fixtures was inspected in Chrome for the plugin chooser and saved tool result cards; it never contacted the live services. It still needs owner UI/API review and real-model evaluation; no inference, runtime rollout, GPU load or live storage writes were performed.

The standalone prototype was manually checked on desktop against temporary isolated storage for agent/persona creation, persona editing, conversation persistence, memory search, Space navigation and file metadata browsing. This monorepo packaging still needs owner verification with the combined Compose stack. Mobile rendering, browser download completion and real inference have not been verified. No model was downloaded or tier selected during prototype work.
