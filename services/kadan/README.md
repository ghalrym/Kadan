# Kadan agent app

Kadan is the everyday agent interface in this repository. This service coordinates conversations and workspace browsing; Tsuchi stores the records and Hikari supplies inference. The React interface and FastAPI coordinator ship as separate frontend and API images. A separate runner process uses the API image to execute durable work.

## Available in this prototype

- Create and select agents, with each agent's records kept in its own Tsuchi namespace.
- Text each agent in one continuous message chain. There are no threads: the chain is the agent's whole history with you.
- Create and edit personas and select one for a message.
- Save a note without inference, or enqueue a durable agent run through Kadan.
- Continue the same conversation while an image is being generated.
- Receive text and images through the agent's explicit messaging tool.
- Browse existing memories, Spaces and file metadata without changing them.
- Give the agent its own computer, an [AIO Sandbox](https://github.com/agent-infra/sandbox),
  and watch or control it live from the agent's card.

Kadan keeps API management separate from runner execution. The API queues durable
work and exposes saved state; the runner owns the continuation loop. Hikari supplies text and image
inference through its existing API and workers; Tsuchi stores conversations,
runs, image jobs and assets. No new model runtime is installed. Tier selection,
model downloads and settings remain explicit Hikari operations.

[AGENT.md](AGENT.md) is the directly editable, human-maintained instruction file.
Kadan reads it once when each run starts; resumed runs keep their saved snapshot.
The Docker image includes `/app/AGENT.md`; a read-only bind mount there permits
edits without rebuilding. `KADAN_AGENT_INSTRUCTION_FILE` can select another file.
The agent cannot modify it. This file is separate from memories, which this flow
does not write. Tool schemas, limits, leases and delivery rules remain in code.

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

Each agent has one continuous conversation, returned by `GET /api/agents/{key}/chain`.
Sending a message atomically saves it and queues its run. Closing the browser does
not cancel work. The interface polls saved run and conversation state, keeps the
composer available, and shows text as soon as the messaging tool delivers it.
Run completion never appends an answer.

LangGraph routes each claimed run through context preparation, model inference,
and tool execution. LangChain Core supplies the chat-model, message and structured
tool interfaces. Tsuchi remains the only durable checkpoint and receipt store:
commands are saved before tool execution, and retries reuse their action IDs.
The graph ends at the existing tool boundaries; image events enqueue a new run in
the same conversation. No graph runs or inference are scheduled while idle.

Hikari currently exposes text completions. Its request schema does not accept
native tool binding; the async `HikariChatModel` adapter streams through it.
Kadan binds its tools with `bind_tools()`. Hikari accepts only text, so the
adapter renders the tool definitions into the system message in the Qwen
checkpoint's tool format and parses `<tool_call>` blocks in the reply into
LangChain tool calls. Kadan takes one call per inference, validates it, persists
it, executes it, saves its result, and continues. `send_message` delivers text
(optionally a completed image job), `generate_image` submits an image job, and
`finish` ends the current run without a user message.
A completed reply without a tool call also ends the run silently. Invalid tool
calls fail the run without a retry. Only `send_message` delivers text to the user.
A run is bounded to 24 commands and to the selected model's context window,
less the bound tool definitions (48,000 characters when Hikari does not report one).
The recent conversation window contains at most 97 messages. Saved history remains
intact. Only user and assistant chat messages enter this window.

Image submission returns a durable job ID immediately. A separate runner queue task waits
on Hikari, stores its output in Tsuchi, then atomically records completion or
failure with a new run event in the same conversation. That event wakes the agent,
which delivers the image using `send_message`. Idle queues perform no inference.
Image events and delivery receipts are idempotent. Uncertain interrupted image
requests are cancelled and checked for saved artwork; they are never silently
submitted again. If no saved output is recoverable, the agent receives a failure.

One run at a time owns each conversation. Leases and persisted commands support
restart recovery, and message/job IDs prevent duplicate side effects on replay.
Stop cancels queued/running conversation runs; image controls cancel their jobs
separately. Cancellation fences delivery immediately and closes inference at the
next ownership check (within the infrastructure lease renewal interval). Image cleanup must
be acknowledged by Hikari before cancellation completes. Failures remain visible.
A queued run uses the current selected Hikari model when it starts; no model setting
is changed. Model loading may delay the first reply.

## Computer

Choose the agent's pet at the top of the chat (or on the home stage) to open its
card: its activity, a **Call** button (not available yet; it reports that) and its
computers. **{Agent}'s computer** shows the sandbox's state and a screenshot.
Choose it for a live VNC view. The view starts watch-only; **Take control** passes
your mouse and keyboard to the computer.

The computer is the `kadan-sandbox` Compose service, the pinned
`ghcr.io/agent-infra/sandbox` image. `kadan/runner/tools/` defines wrappers for its
`/v1` routes as LangChain tools: one module per area (`browser`, `shell`, `bash`,
`file`, `code`, `jupyter`, `nodejs`, `mcp`, `sandbox`, `proxy`, `skills`,
`display`, `util`), each tool a Pydantic argument model and an async function
whose docstring is its description, named `<area>_<action>` (for example
`browser_navigate`, `shell_exec_command`). `conversation.py` holds
`send_message`, `generate_image` and `finish`. Each module's `TOOLS` list
selects the enabled computer tools. When a computer is configured, `ALL_TOOLS`
is bound to inference with `HikariChatModel.bind_tools()`; otherwise only the
conversation tools are bound.
Results are truncated to 4000 characters; binary content is reported by size
only. Sandbox errors are returned to the agent as tool results so it can correct
itself, and they do not fail the run. The sandbox has no idempotency keys, so
after a runner restart an unconfirmed non-GET operation is not repeated; the
agent is told it may or may not have run.

The sandbox joins only the `computer` network with `kadan-api` and
`kadan-runner`; it has no direct access to the default service network.
In Compose, Kadan's API binds only
to its `kadan-api-internal` address on the default network, leaving no control
API listener on the computer-facing interface. Kadan can still make outbound
HTTP and VNC connections to the sandbox. This separation does not block access
through separately reachable LAN addresses. The sandbox publishes no port.
Kadan relays the screenshot and VNC stream, so the browser never holds
`KADAN_SANDBOX_API_KEY`. The VNC WebSocket accepts only the page's own origin.

## Docker integration

Build from the repository root:

```sh
docker build --build-context tracing=services/tracing -t kadan-api:local services/kadan
docker build -t kadan-frontend:local services/kadan/web
```

Compose runs these as `kadan-api`, `kadan-runner` and `kadan-frontend`. The runner uses the API image with watchfiles restarting it after source edits, has no HTTP listener or published port, and disables the image’s API healthcheck. The frontend image runs Vite with hot reload and proxies `/api`, `/docs` and `/openapi.json` to `kadan-api`.

The root Compose integration supplies the following environment. The upstream names are specific to the shared Compose network; browser URLs supply the published operator ports. Status responses replace their loopback hostname with the validated browser hostname. Explicit non-loopback browser URLs remain unchanged.

| Variable | Compose value | Purpose |
| --- | --- | --- |
| `KADAN_RUNNER_ID` | `runner-1` | Stable runner identity, unique among live runner processes |
| `KADAN_RUNNER_CAPACITY` | `5` | Strict positive integer; maximum assigned agent keys, including idle agents |
| `KADAN_TSUCHI_URL` | `http://tsuchi-api:8000` | Storage API inside Docker |
| `KADAN_HIKARI_URL` | `http://hikari-api:8000` | Inference API inside Docker |
| `KADAN_TSUCHI_BROWSER_URL` | `http://127.0.0.1:8620` | Storage operator interface |
| `KADAN_HIKARI_BROWSER_URL` | `http://127.0.0.1:8088` | Hikari operator interface; Settings is `/settings` |
| `KADAN_ALLOWED_HOSTS` | `["127.0.0.1","localhost","::1"]` (default) | Explicit DNS request hosts and additional browser-link hosts; IP request hosts are accepted |
| `KADAN_SANDBOX_URL` | `http://kadan-sandbox:8080` | The agent's computer; unset leaves the agent without one |
| `KADAN_SANDBOX_API_KEY` | from `.env` | Shared with `kadan-sandbox` as `SANDBOX_API_KEY` |

Compose binds the API to `kadan-api-internal:8630` on the default network and
does not publish it; the frontend publishes `0.0.0.0:8630:8080`. The API runs
as UID/GID `10001:10001`, supports a read-only filesystem, and requires no
volume: durable application records live in Tsuchi. Use `cap_drop: [ALL]`,
`security_opt: [no-new-privileges:true]` and `restart: unless-stopped`.
The Compose healthcheck requests `/api/health` at that bound address with
`Host: localhost`, which the existing host guard accepts. This checks only the
coordinator process. `/api/status` separately reports upstream availability
without starting inference. Outside Compose, the image retains its
`0.0.0.0:8630` listener and loopback healthcheck.

Upstream URLs accept loopback HTTP or their respective exact Compose endpoint above. Configured browser URLs accept loopback HTTP and hosts explicitly listed in `KADAN_ALLOWED_HOSTS`. Requests accept IP literals and configured DNS hosts. Cross-origin writes and VNC WebSockets from another origin remain rejected; these guards do not replace authentication.

Root Compose exposes all three frontend ports on `0.0.0.0`; open `http://HOST_IP:8630` from another computer on a trusted network. No Compose override or fixed LAN IP is needed. Browser links follow that host and the configured Tsuchi/Hikari ports. For a DNS name, set the optional host lists documented in the root README and `.env.example`, retaining localhost and Tsuchi's internal `tsuchi-api` entry. The frontend proxies preserve the browser's Host header, so no CORS relaxation is needed. The direct Hikari model API stays loopback-only and the sandbox has no published host port, but Hikari's frontend exposes its proxied API controls. Interfaces are reachable wherever routing and firewall rules permit; keep them off the public internet. This does not add authentication or change firewall rules.

On plain HTTP LAN pages, Hikari microphone recording and clipboard actions remain subject to browser secure-context requirements. Those features need trusted HTTPS; ordinary API access and audio-file upload do not require enabling microphone access. This change does not configure HTTPS.

## Running locally

Requires Python 3.12 or newer and Node.js 24. From `services/kadan`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ../tracing
.venv/bin/python -m pip install -r requirements-dev.txt
npm ci --prefix web
npm run build --prefix web
.venv/bin/python -m uvicorn kadan.api.server:app --host 127.0.0.1 --port 8630
```

In another terminal from `services/kadan`, start execution separately:

```sh
.venv/bin/python -m kadan.runner
```

Open `http://127.0.0.1:8630`. Local upstream defaults are Tsuchi on port 8620 and Hikari on port 8088. `.env.example` lists the available settings. To run the frontend separately, `npm run dev --prefix web` starts Vite on loopback and proxies `/api` to port 8630.

## Runner assignment and recovery

Tsuchi atomically assigns registered agent keys to stable runner identities, up to
`KADAN_RUNNER_CAPACITY` per runner (default five). All assigned agents count,
including agents with no queued work. Agents are logical records, not permanent OS
threads. One runner serves multiple agents; Hikari still schedules inference across
its existing workload workers. Image and pet queue loops remain independent of
LangGraph turns and do not consume additional agent slots.

When every runner is full, excess registered agents and their queued work wait in
durable storage. Idle owners keep their slots; stopping a process does not free them.
Each process registers a fresh session UUID and renews a 60-second infrastructure
ownership lease. After expiry, a replacement with the same stable runner ID can
claim ownership and recover expired work. A concurrent live process with that ID
cannot take over. Ownership and run claims fence stale sessions before durable
state-changing work. Lease renewal performs no agent inference or Decisions API
calls; periodic agent wakes and the future agent heartbeat are deferred.

A different runner does not automatically inherit a stopped runner’s agents.
Inspect `GET /api/runners`, then explicitly move an agent with
`POST /api/agents/{key}/runner`, supplying both `runner_id` and
`expected_runner_id` (the current owner, or `null` for an unassigned agent).
`runner_id: null` releases the assignment; an available runner can subsequently
claim it. Reassignment compares the expected owner atomically, checks destination
capacity and fences earlier ownership. A capacity reduction below the number of
existing assignments is rejected; explicitly move agents before reducing it.
Assignments are durable across process restarts; they are not queue leases.

Keep IDs unique and stable. The root `compose.yaml` provides a second explicit service behind the
`runners` profile with ID `runner-2`; do not scale identical copies of `runner-1`.
When mounting a custom instruction file, mount it in every runner: snapshots are
read at run start by the executing process, and resumed runs retain their snapshot.
Tracing uses the existing durable W3C parent across the API/runner boundary, stays
disabled without an OTLP endpoint and excludes prompt/tool payloads.

## API boundary

| Route | Behavior |
| --- | --- |
| `GET /api/runners` | Inspect durable runner ownership and assignments |
| `POST /api/agents/{key}/runner` | Explicitly compare and reassign an agent owner |
| `GET /api/health` | Coordinator liveness |
| `GET /api/status` | Storage and inference readiness, including browser links and whether the selected model is loaded (`hikari.model_state`) |
| `GET, POST /api/agents` | List or create agents |
| `GET, POST /api/agents/{key}/resources` | Browse/search resources; create a persona or conversation |
| `GET, PUT /api/agents/{key}/resources/{id}` | Read a resource; update a persona with an expected version |
| `GET /api/agents/{key}/assets` | Search active file metadata |
| `GET /api/agents/{key}/assets/{id}` | Read file metadata |
| `GET /api/agents/{key}/assets/{id}/content` | Stream content; `download=true` requests an attachment |
| `POST /api/agents/{key}/conversations/{id}/notes` | Append a user note with an expected version and message ID |
| `POST /api/agents/{key}/conversations/{id}/runs` | Atomically save the user message and queue a run; HTTP 202 |
| `GET /api/agents/{key}/conversations/{id}/runs` | Read runs, image jobs and saved conversation |
| `POST /api/agents/{key}/runs/{id}/cancel` | Cancel a queued/running run |
| `POST /api/agents/{key}/image-jobs/{id}/cancel` | Request image cancellation |
| `GET /api/computer` | Whether the agent's computer is `ready`, `unavailable` or `not_configured` |
| `GET /api/computer/screenshot` | Current screen as JPEG, not cached |
| `WS /api/computer/vnc` | VNC relay to the computer; same-origin only |

Run requests include `content`, `expected_version`, `message_id`, and an explicit
`persona_id` UUID or `null`. Repeating the same message ID and payload returns the
original run, even if the conversation version has advanced. The response contains
`run` and `conversation`; polling returns `runs`, `images` and `conversation`.
The service's `/docs` exposes the request and response schemas.

## Run boundaries

`generate_image` persists the job ID and completes its submitting run immediately.
`send_message` requires an explicit `end_run` boolean: true saves its delivery result
and completes the run without another inference call; false continues immediate tool
work. `finish` ends a run that needs no visible message. Completion itself never
creates a chat message. Tool result and run completion are saved together; interrupted
side effects replay with the same IDs. New turns see pending image jobs in context.

User messages and image events retain creation order within a conversation. A pending
image owns no agent turn; its terminal event wakes the idle scheduler. Events never
bypass earlier turns. Application logs record run claim, inference start/end and yield
with run IDs and UTC times, without prompts or responses. Inference elapsed time includes
Hikari queueing/loading/processing; it is separate from Kadan scheduling delay.

This command schema requires `end_run`; deploy at an idle boundary with no queued or
running runs. Existing completed transcripts remain stored without reinterpretation.

## Validation

From this directory:

```sh
.venv/bin/python -m pytest
npm test --prefix web
npm run build --prefix web
```

Automated checks are isolated unit tests: Python uses an in-memory mocked HTTP transport; frontend tests exercise outbox and run-state helpers. They never persist fixtures or call a model. Root `python services/hikari/bin/hikari.py check format` also covers this service.

The standalone prototype was manually checked on desktop against temporary isolated storage for agent/persona creation, persona editing, conversation persistence, memory search, Space navigation and file metadata browsing. This monorepo packaging still needs owner verification with the combined Compose stack. Mobile rendering, browser download completion and real inference have not been verified. No model was downloaded or tier selected during prototype work.
