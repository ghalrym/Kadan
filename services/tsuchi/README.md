# Tsuchi

Tsuchi stores agent personas, memories, Spaces, slide decks, files, and conversations on one local machine. Its management website shows the stored content, usage, revision history, trash, and activity. Kadan can use the same HTTP API. Inference belongs in Hikari; agent execution belongs in the Kadan runner service.

## Run with Docker

Tsuchi ships with Kadan and Hikari in the repository's root Compose file. From the repository root:

```sh
docker compose up --build -d
docker compose ps
```

Kadan at <http://127.0.0.1:8630> is the everyday agent interface. Tsuchi's site is the storage management interface for the owner. See the [root startup guide](../../README.md#docker-compose) for prerequisites and existing-installation handling.

Open <http://127.0.0.1:8620>. Create an agent namespace using a name and a key such as `research`. No accounts, passwords, API credentials, or sample data are created. Interactive API documentation is at <http://127.0.0.1:8620/docs> and the schema is at `/openapi.json`.

Compose publishes **127.0.0.1 only**. This is a single-owner local prototype with no authentication. Agent keys select separate namespaces; they do not grant or restrict access to the owner. Keep the service on loopback or a private container network. Do not publish it to the internet. Compose runs `tsuchi-api`, whose `0.0.0.0:8000` listener stays inside Docker, and `tsuchi-frontend`, a Vite container that serves the site with hot reload and proxies `/api`, `/docs` and `/openapi.json` to it. Only the frontend publishes a port, on loopback.

The service runs as UID 10001 with a read-only application filesystem. The `tsuchi_tsuchi_data` Docker volume holds SQLite and uploaded files. Restarting or rebuilding the containers preserves data. `docker compose down` preserves the volume; **do not use `down -v` with data you want to keep**.

Set `TSUCHI_PORT` or `TSUCHI_MAX_UPLOAD_BYTES` in the repository root `.env` to change the local port or upload limit. The default per-file limit is 100 MiB. There is no GPU requirement or model download.

```sh
docker compose logs --tail=100 tsuchi-api tsuchi-frontend
curl http://127.0.0.1:8620/api/health
docker compose stop
```

## What is stored

| Content | Representation and controls |
| --- | --- |
| Personas | Markdown, tags, sources, attachments, revisions |
| Memories | Markdown with full-text search |
| Spaces | Markdown documents with same-namespace parent/child links |
| Slide decks | Versioned JSON with slide IDs, layouts, text, notes, and image asset IDs |
| Images | Verified PNG/JPEG/WebP/GIF files, original bytes and metadata |
| Attachments | Audio, video, PDF, and general files, with original filenames and SHA-256 checksums |
| Conversations | Ordered messages with roles, timestamps, attachments, sources, and stored-resource links |
| Pets | Creation jobs, generation provenance, imported v1/v2 atlases, atlas manifests, and explicit agent selection |
| Agent runs | Conversation triggers, persisted inference transcripts, worker leases, status, and message delivery receipts |
| Image jobs | Conversation-scoped generation requests, worker leases, saved image assets, and completion events |

The image library stores generated images supplied by Hikari, Kadan, or the owner. Tsuchi does not generate them. Images referenced by decks stay separate assets. Slide layouts are `title`, `text`, `split`, and `image`; decks remain editable in the UI and JSON export. This is a defined v1 format, not full ChatGPT feature parity or PowerPoint round-trip support.

Search uses SQLite FTS5 across resource titles, Markdown, tags, slide text, and message content. It is lexical search, without embeddings or GPU usage. File contents such as PDF text and audio are not automatically extracted or transcribed. Tags and asset titles can be searched in the file library.

Edits use an expected version to prevent silent overwrites. Resource history can restore an earlier snapshot as a new revision. Trash keeps content and bytes; there is no permanent-delete action. A Space with live children and an asset referenced by live content cannot be trashed until those links are removed. The UI reports these conflicts. Namespace operations validate all parent, attachment, slide image, and message resource links against the selected agent.

### Editable slides and Spaces

Decks support up to 100 slides on a fixed 1920 × 1080 canvas. Select a slide, edit its title, Markdown body and speaker notes, choose a title/text/split/image layout, and select or remove one image from the namespace's image library. The preview updates while editing. Add and delete slides or reorder them with Move up and Move down, then save the deck. Deck metadata, revisions, trash/restore, and namespace export use the same resource controls. The editor does not provide freeform element positioning, animations, a slideshow player, or PPTX import/export.

Spaces are Markdown documents that can also be parents. Create a Space, edit its title and Markdown, preview it, and select its Parent Space under Details to nest or move it. The Spaces view shows the hierarchy and opens each document for editing. Other stored resources can also choose a Parent Space. Tags, sources, attachments, full-text search, revision restore, and trash/restore are supported. Parent links must stay in the same namespace and cannot form cycles. A parent with active children cannot be trashed; a child cannot be restored until its parent is active. Moving uses the Parent Space selector, without drag-and-drop or automatic subtree deletion.

## Runner ownership

Tsuchi stores runner sessions and logical agent assignments separately from agent content. A runner defaults to five assigned agents; idle agents keep their slots. The capacity is a positive integer set by runner registration. Assignment scans atomically select the oldest unassigned agents, ordered by creation time and key, up to remaining capacity. Excess agents and their queued runs remain durable until a slot becomes available. New containers do not steal existing assignments.

`PUT /api/runners/{runner_id}` registers a stable runner identity with a fresh process `worker_id` UUID and `capacity` (default 5). The response includes `lease_expires_at` and `agent_keys`. Repeating the request with the same worker renews a server-calculated 60-second infrastructure ownership lease. Another process can take over the same stable identity after expiry, or immediately with `expected_worker_id` matching the current owner. This renewal protects ownership; it does not wake agents or perform inference. Assignments survive session expiry and process restarts.

`POST /api/runners/{runner_id}/assign` with `worker_id` fills that runner's free slots. `GET /api/runners` lists sessions and assignments. `POST /api/agents/{key}/runner` with `runner_id` and `expected_runner_id` explicitly moves an agent using compare-and-set ownership and destination capacity checks. A null destination releases the slot and makes that agent eligible for the next assignment scan; it does not pause the agent. To retire a runner, stop its process and explicitly move its assignments to other runners with capacity.

Changing a session owner or moving an agent expires its running run leases and advances their versions without discarding transcripts. Claims, renewals, message delivery and image submission require both a live runner session and the current agent assignment, fencing stale processes. Image and pet workload workers retain their separate job leases. Runner sessions and assignments are infrastructure state and are excluded from namespace exports; imported agents enter the unassigned queue.

## Kadan API

All content routes include the agent key: `/api/agents/{key}/...`. Keys match `[a-z0-9][a-z0-9_-]{0,63}`. IDs are UUIDs. Errors use HTTP status codes and a `detail` field. Version conflicts return 409. The API contract below lists operations and fields; `/docs` shows the generated request schemas.

```sh
curl -X POST http://127.0.0.1:8620/api/agents \
  -H 'Content-Type: application/json' \
  -d '{"key":"research","name":"Research"}'

curl -X POST http://127.0.0.1:8620/api/agents/research/resources \
  -H 'Content-Type: application/json' \
  -d '{"kind":"memory","title":"Storage decision","markdown":"Use Tsuchi for durable agent memory.","tags":["architecture"]}'

curl 'http://127.0.0.1:8620/api/agents/research/resources?q=durable'

curl -X POST http://127.0.0.1:8620/api/agents/research/assets \
  -F 'file=@image.png' -F 'title=Generated concept'

curl -o research.zip http://127.0.0.1:8620/api/agents/research/export
```

Update a resource with `PUT /resources/{id}`, its complete editable fields, and `expected_version` from the last read. Append a conversation message using `POST /resources/{id}/messages` with `{expected_version,message}`. Each message needs a stable UUID and UTC timestamp. Message IDs cannot repeat. Sources accept HTTP(S) links only. Binary uploads are immutable; upload a new asset to replace a file.

### Durable agent work

Kadan owns agent decisions and inference; Tsuchi stores their state. `POST /api/agents/{key}/resources/{conversation_id}/runs` accepts `{expected_version,message,persona_id?}`. It appends the user message and enqueues its run in one transaction. The message UUID is also the run UUID. Retrying the same input returns the existing run before checking the conversation version; timestamps supplied by a retry do not replace the original message timestamp.

Kadan claims or updates a run with `PUT /api/agents/{key}/runs/{run_id}`, supplying its complete mutable state, `expected_version`, and `worker_id`. Only one unexpired run lease can own a conversation. A new owner can reclaim an expired lease while retaining the persisted transcript. Kadan persists each tool command before executing it and derives stable operation IDs from that checkpoint. Finishing a run changes its status without appending a chat answer.

`POST /api/agents/{key}/runs/{run_id}/messages` takes `{worker_id,message,image_job_id?}`. It appends an assistant message and its delivery receipt atomically. With an image job ID, storage attaches that job's completed image and records delivery. Retries return the existing message; subsequent runs cannot deliver the same image job a second time. Lease checks fence cancelled or superseded workers before new delivery.

`POST /api/agents/{key}/runs/{run_id}/image-jobs` accepts `{id,prompt,request_id,worker_id}` and durably queues work. It does not deliver an image. Kadan updates the image state through `PUT /api/agents/{key}/image-jobs/{job_id}` with `expected_version` and `worker_id`. A terminal image result and its deterministic event run are saved together. Kadan resumes on that event to decide what to communicate. An expired image lease preserves the inference request ID; Kadan must reconcile saved output and confirm inference cleanup before ending uncertain work.

`GET /api/agent-runs` and `/api/image-jobs` list active work for reconciliation without running inference. Conversation-scoped `GET /api/agents/{key}/resources/{conversation_id}/runs` and `/image-jobs` include terminal records for UI polling. `POST /api/agents/{key}/runs/{run_id}/cancel` immediately fences that run. Image cancellation at `POST /api/agents/{key}/image-jobs/{job_id}/cancel` finishes queued work immediately; running work enters `cancelling` until its owner confirms cleanup. Runs, image jobs, and delivery receipts are included in namespace exports.

## Export and backup

**Export** in the website downloads a ZIP for the selected namespace. It includes a JSON manifest with resources, revisions, metadata, trash, activity, and original files under `files/{asset-id}`. This is a portable export; there is no ZIP import endpoint in v1.

For a complete backup of all namespaces, stop the service and archive the whole volume. Stop it first so the SQLite database and files cannot change during the copy. The commands below apply to Tsuchi managed by this repository's root Compose project. If you still use the standalone deployment, stop and back up through that original Compose project first; root Compose cannot stop a container belonging to another project. Run these commands from the repository root, writing the backup somewhere with enough free disk space:

```sh
docker compose stop tsuchi-api tsuchi-frontend
docker compose run --rm -T --no-deps --entrypoint tar tsuchi-api \
  -czf - -C /data . > tsuchi-backup.tar.gz
docker compose up -d
```

Check that the archive is readable before relying on it:

```sh
tar -tzf tsuchi-backup.tar.gz
```

Restore your own trusted backup into a **new empty volume**, preserving the original. The following uses a separate Compose project and refuses to extract into a nonempty data directory:

```sh
TSUCHI_DATA_VOLUME=tsuchi-restored-data docker compose -p tsuchi-restored run --rm -T --no-deps --entrypoint sh tsuchi-api \
  -c 'test -z "$(ls -A /data)" && tar -xzf - -C /data --no-same-owner' \
  < tsuchi-backup.tar.gz
TSUCHI_DATA_VOLUME=tsuchi-restored-data TSUCHI_PORT=8621 docker compose -p tsuchi-restored up -d --no-deps tsuchi-api tsuchi-frontend
```

Inspect the restored instance at <http://127.0.0.1:8621> before switching anything over. Coordinate this additional service and port with the server owner. The original volume is untouched.

## Development

Python 3.14 and Node 24 or later are used for the verified build. Outside Docker, the backend serves the built frontend from `web/dist`, so one process and one local port are enough. In Compose the frontend runs in its own Vite container.

Run from `services/tsuchi`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ../tracing
.venv/bin/python -m pip install -r requirements-dev.txt
cd web
npm ci
npm run build
cd ..
.venv/bin/python -m pytest
.venv/bin/uvicorn tsuchi.main:app --host 127.0.0.1 --port 8620
```

To run the frontend separately, run `npm run dev` in `web` after starting the API. Vite proxies `/api` to `127.0.0.1:8620`. Keep listeners local. Local data defaults to `./data`; set `TSUCHI_DATA_DIR` for another dedicated directory. Do not point tests or temporary checks at a real data volume.

The backend has explicit Pydantic models, SQLite transactions, and namespace-qualified lookups. Uploaded filenames never become disk paths. Raster images are verified before inline display. HTML, SVG, and general files download as attachments. The server checks allowed hosts and unsafe browser origins. These safeguards support a local app; they are not an authentication system.

## Manual check

1. Create an agent, save a persona and memory, then search for a word in the memory.
2. Create a Space and child document. Edit the child and restore its earlier revision.
3. Upload an image, place it in a two-slide deck, change a layout, and reload the page.
4. Append a conversation message with an attachment and a source link.
5. Switch to another agent and check that the first agent's content is absent.
6. Trash and restore a memory, download an export, and inspect Overview and Activity.
7. Restart the container and confirm the saved content remains.

## API contract

Local single-owner storage service. FastAPI + Pydantic, SQLite FTS5, file blobs, React + TypeScript + Vite. No inference and no authentication. All content operations live under `/api/agents/{key}`. An agent key is a human-chosen namespace slug, not a secret. Keys match `[a-z0-9][a-z0-9_-]{0,63}`. IDs are UUIDs. Times are UTC ISO 8601 strings. Lists return plain arrays. Errors use FastAPI `{detail: string}`.

### Agents and system

- `GET /api/health` => `{status:"ok", version:"0.1.0"}`
- `GET /api/agents` => Agent[]
- `POST /api/agents` body `{key, name}` => Agent (201)
- Agent: `{key,name,created_at,selected_pet_id:string|null}`. No agent delete in v1.
- `GET /api/agents/{key}/stats` => `{resources:{persona:number,memory:number,space:number,deck:number,conversation:number},assets:number,images:number,trashed_resources:number,trashed_assets:number,storage_bytes:number,database_bytes:number,revisions:number}`. Counts exclude trash unless named otherwise. Storage bytes include retained files in trash. Database bytes are global, not agent-specific.
- `GET /api/agents/{key}/activity?limit=50` => `{id,action,entity_type,entity_id,title,created_at}`[]. Newest first.

### Resources

- `GET /api/agents/{key}/resources?kind=memory&q=term&trash=false&limit=100&offset=0` => Resource[]. Omitted kind means all; q uses safe tokenized FTS across titles, Markdown, tags and structured text. Limit max 200.
- `POST /api/agents/{key}/resources` ResourceInput => Resource (201)
- `GET /api/agents/{key}/resources/{id}` => Resource including trash.
- `PUT /api/agents/{key}/resources/{id}` ResourceInput plus required `expected_version` => Resource. Stale version => 409.
- `DELETE /api/agents/{key}/resources/{id}?expected_version=N` => Resource marked trashed. Refuse trash of a Space with active children. Do not purge files or records.
- `POST /api/agents/{key}/resources/{id}/restore` `{expected_version}` => Resource. Parent must be active.
- `GET /api/agents/{key}/resources/{id}/versions` => `{version,created_at,snapshot:Resource}`[] newest first.
- `POST /api/agents/{key}/resources/{id}/versions/{version}/restore` `{expected_version}` => Resource with new version.
- `POST /api/agents/{key}/resources/{id}/messages` `{expected_version,message:Message}` => Resource. Append to a conversation.

ResourceInput: `{kind:"persona"|"memory"|"space"|"deck"|"conversation",title:string,markdown:string="",parent_id:string|null=null,tags:string[]=[],sources:SourceLink[]=[],attachment_ids:string[]=[],deck:Deck|null=null,messages:Message[]=[]}`.

Resource adds `{id,agent_key,version,created_at,updated_at,deleted_at:string|null}`. Kind cannot change on update. Parent IDs must identify an active Space in the same namespace. No cycles. Resource refs and attachments must belong to this namespace. Sources are HTTP(S) URLs; no arbitrary file paths. For decks, deck must be present; non-decks cannot carry deck data. Messages only on conversations.

SourceLink: `{label:string,url:string}`. Message: `{id:string,role:"system"|"user"|"assistant"|"tool",content:string,created_at:string,attachment_ids:string[]=[],resource_ids:string[]=[],sources:SourceLink[]=[]}`. Caller supplies stable UUID message IDs and UTC times. Duplicate message IDs forbidden.

Deck: `{schema_version:1,width:1920,height:1080,slides:Slide[]}`. Slide: `{id:string,layout:"title"|"text"|"split"|"image",title:string,body:string,notes:string,image_asset_id:string|null}`. Max 100 slides. Slide IDs stable UUIDs. This is an editable structured v1 format, not ChatGPT parity or PowerPoint round-trip.

### Assets

- `GET /api/agents/{key}/assets?images=false&trash=false&q=term` => Asset[]. `images=true` restricts to safe raster image formats.
- `POST /api/agents/{key}/assets` multipart `file`, optional `asset_id` UUID, `title`, `tags` JSON string array, `sources` JSON SourceLink array => Asset (201). 100 MiB upload limit by default. Immutable UUID disk path; original filename retained as metadata only. Replaying an upload with the same `asset_id`, bytes and metadata returns the original asset. Conflicting content, another namespace or a trashed asset returns 409. A retry can finish registration if a crash left the immutable blob before its database record.
- `GET /api/agents/{key}/assets/{id}` => Asset.
- `GET /api/agents/{key}/assets/{id}/content` => file. Verified PNG/JPEG/GIF/WebP images and parsed WAV audio may be inline; all others force attachment, including other audio/video, SVG, HTML, and PDF. `download=true` forces attachment.
- `PUT /api/agents/{key}/assets/{id}` `{title,tags,sources,expected_version}` => Asset. Binary bytes immutable; upload a new asset to replace.
- `DELETE /api/agents/{key}/assets/{id}?expected_version=N` => Asset in trash. Refuse if active resources reference it.
- `POST /api/agents/{key}/assets/{id}/restore` `{expected_version}` => Asset.

Asset: `{id,agent_key,title,filename,media_type,size_bytes,sha256,is_image:boolean,tags,sources,version,created_at,updated_at,deleted_at}`.

### Pets

- `POST /api/agents/{key}/pets` `{id,prompt}` => queued PetJob. Same ID and prompt are idempotent; another namespace or prompt conflicts.
- `POST /api/agents/{key}/pets/imports` `{id,prompt,manifest}` => ready PetJob with `origin: "imported"`. The manifest references an active raster atlas in the namespace and has no base artwork; v1 atlases have 9 rows and v2 atlases 11. Imported pets have no steps, cannot be resumed or revised, and are never selected automatically. Same ID, prompt and manifest are idempotent.
- `GET /api/agents/{key}/pets` => all candidates/jobs, newest first.
- `GET /api/agents/{key}/pets/{id}` => PetJob.
- `PUT /api/agents/{key}/pets/{id}` => PetJob. Supply full mutable state, `expected_version`, and `worker_id` for worker updates. Job identity and prompt cannot change.
- `GET /api/pet-jobs` => queued, running and cancelling jobs across namespaces for the local coordinator.
- `GET /api/agents/{key}/pet` => selected PetJob or null.
- `PUT /api/agents/{key}/pet` `{pet_id:UUID|null}` => selected PetJob or null. Only ready candidates with active same-namespace raster assets may be selected. Creation and updates never select a candidate.

Jobs retain ordered completed steps and their source/output asset IDs, prompts, model IDs, request IDs, seeds, edit strengths and response metadata. A pending step/request pair is persisted before inference. Ready candidates require all 12 outputs (base, nine animation strips and two look strips) and a v2 manifest for an 8-by-11 atlas with 192-by-208 cells. Artwork remains in the asset store; the manifest and provenance live in SQLite and exports.

Queued jobs are claimed atomically with a worker UUID and a lease expiring within five minutes. Worker updates require that live owner and the current version. Heartbeats extend the lease; stale workers cannot publish results. Cancellation is sticky until explicit resume. Expired running jobs can be marked interrupted, preserving their completed and pending steps; resuming a failed, cancelled or interrupted job explicitly returns it to queued. Moving any selected pet dependency to trash clears selection. Restoring artwork does not automatically reselect it.

### Export

- `GET /api/agents/{key}/export` => ZIP with `manifest.json` (agent, resources including trash, versions, assets including trash, pet jobs and selection, activity) and `files/{asset UUID}` binary data. Namespace-scoped, consistent read snapshot. No credentials. All generated exports cleaned up.
- Full-volume backup/restore instructions in README using stopped service and volume copy. No destructive overwrite endpoint.
