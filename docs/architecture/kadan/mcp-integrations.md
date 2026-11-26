# Remote MCP integrations and T3 Code

The [plugin expectations](../../user-expectations/kadan/plugins.md) describe the user experience. Kadan connects to remote MCP servers over Streamable HTTP so the conversation agent can delegate coding work to T3 Code (and call any other remote MCP service) without embedding coding harnesses inside `kadan-sandbox`.

## Data ownership (`tsuchi`)

Tsuchi stores all integration configuration and durable watch state in three global SQLite tables (Kadan is single-agent, so connections are not scoped per agent):

- `mcp_servers`: server identity (`id`, `slug`, `name`, `preset`), `url`, optional `app_url`, `auth_type` (`oauth` | `bearer` | `none`), `auth_state` JSON (OAuth client credentials, tokens, expiry, metadata, or bearer token), `enabled`, `status` (`disconnected` | `authorizing` | `connected` | `error`), `status_error`, `pinned_tools` JSON array, cached `tools` metadata JSON array, `tools_synced_at`, and optimistic-locking `version`.
- `mcp_oauth_states`: short-lived single-use OAuth PKCE state rows (`state`, `server_id`, `code_verifier`, `redirect_uri`, `expires_at`, `consumed_at`), atomically consumed via `POST /api/mcp-oauth-states/{state}/consume`.
- `t3_thread_watches`: durable watch rows (`id`, `server_id`, `thread_id`, `run_id`, `environment_id`, `thread_link`, `thread_title`, `origin_agent_key`, `origin_conversation_id`, `origin_run_id`, `status` (`watching` | `delivered` | `dropped`), `last_pending_request_id`, `version`) so background T3 Code threads survive runner restarts.

Tsuchi's `RunEvent.kind` accepts `"t3"` for T3 Code background event runs (while retaining `"codex"` for historical stored transcripts).

## Protocol and OAuth 2.1 client (`kadan/api/mcp_client.py`)

`McpClient` speaks MCP Streamable HTTP (`POST` JSON-RPC 2.0 accepting both `application/json` and `text/event-stream` responses, managing `Mcp-Session-Id` headers across `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`).

- **Loopback URL rewrite:** When Kadan runs inside Docker (`extra_hosts: host.docker.internal:host-gateway` on `kadan-api` and `kadan-runner`), outbound requests targeting `localhost` / `127.0.0.1` / `::1` rewrite their connect host to `host.docker.internal` while preserving the original `Host` header. Stored URLs and user-facing `app_url` links keep the original host.
- **OAuth 2.1 with PKCE and DCR:** On `connect`, `McpClient` probes the MCP endpoint. If it returns `401`, Kadan discovers authorization metadata via RFC 9728 (`WWW-Authenticate` `resource_metadata` or `/.well-known/oauth-protected-resource`) and RFC 8414 (`/.well-known/oauth-authorization-server`), performs RFC 7591 Dynamic Client Registration when no `client_id` is saved (or when the callback redirect URI changed), persists a PKCE state in Tsuchi, and returns an `authorize_url` for the browser. When Kadan is accessed over plain HTTP on a non-loopback LAN address and a strict OAuth 2.1 server rejects the LAN `redirect_uri` with HTTP `400`, `McpClient` retries registration with `http://localhost:<port>/api/mcp/oauth/callback` and sets `manual_callback: true` so same-machine browsers still complete automatically while remote LAN browsers can finish by pasting the redirected URL into `POST /api/mcp/oauth/complete`.
- **Callback and token refresh:** `GET /api/mcp/oauth/callback` (and `POST /api/mcp/oauth/complete` for pasted callback URLs) consumes the single-use PKCE state, exchanges the authorization code for tokens, immediately runs `tools/list`, saves the server as `connected`, and returns a minimal HTML page that notifies the opener window via `postMessage`. On subsequent `401` responses, `McpClient` refreshes tokens automatically when a `refresh_token` is present; otherwise it marks the server `disconnected` with a reconnect error.
- **Credential redaction:** `kadan/api/routes/mcp.py` projects every browser response through `McpServerPublic`, omitting tokens, refresh tokens, and OAuth client secrets.

## Tool binding, gateway, and replay safety (`kadan/runner/mcp_runtime.py`)

To avoid bloating model context windows with dozens of remote schemas, Kadan splits MCP tools into two tiers:

- **Pinned tools:** Directly bound to the conversation agent as `mcp__<slug>__<tool_name>` with their cached JSON Schema (`input_schema`). For the `t3_code` preset (`kadan/api/mcp_presets.py`), 8 core tools are pinned by default (`t3_project_list`, `t3_thread_list`, `t3_thread_read`, `t3_thread_launch`, `t3_thread_send`, `t3_thread_interrupt`, `t3_pending_request`, `orchestrator_capabilities`), with optional UI toggles for PR, CI, Issue, and Preview/Device groups.
- **On-demand gateway:** Whenever at least one enabled, connected MCP server exists, the runner also binds two small tools (`mcp_search_tools` and `mcp_call_tool`) so the agent can discover and invoke any unpinned remote tool without pre-binding its schema.

Execution rules in `mcp_runtime.py` and `agent_runs.py`:

- **Validation:** Dynamic MCP tool arguments are validated against the tool's cached JSON Schema before persisting or executing, using the same two-correction diagnostic path as static tools.
- **Idempotency & replay safety:** Read-only MCP tools (`annotations.readOnlyHint` or `idempotentHint`, plus known read-only T3 tools) and tools declaring `clientRequestId` in their schema replay safely after a runner restart. For any tool accepting `clientRequestId`, `execute_mcp_tool` deterministically injects the saved Kadan `action_id` when omitted. Non-idempotent state-changing calls without `clientRequestId` are never replayed after a crash; recovery returns an uncertainty notice.
- **Result bounding:** Tool outputs are truncated to 24 KB before entering the agent transcript, and remote errors return structured tool results rather than failing the run.

## Background T3 Code watcher (`kadan/runner/t3_events.py`)

When `execute_mcp_tool` receives a successful result from `t3_thread_launch` or `t3_thread_send` on a `t3_code` server, it extracts `threadId`, optional `runId` / `environmentId` / `link` / `title`, and upserts a `watching` row in Tsuchi's `t3_thread_watches`.

`T3ThreadEvents` runs inside `kadan-runner` on a 5-second poll loop:

1. Lists active `watching` rows from Tsuchi and resolves their `mcp_servers`. Watches for deleted servers are marked `dropped`.
2. Checks `t3_pending_request(action="list")` first. If a pending user question exists, it reads the question once (deduplicated via `last_pending_request_id`), queues an idempotent Tsuchi event run (`kind: "t3"`, `event_id = uuid5(watch.id, f"t3-q-{req_id}")`, `event_type: "pending_question"`), keeps the watch `watching`, and skips terminal status checks while the question remains open.
3. Calls `t3_thread_read(view="messages", limit=5, maxCharsPerItem=12000)`. When the target run or thread enters a terminal status (`completed`, `failed`, `interrupted`, `cancelled`, `rolled_back`, `idle`), it queues an idempotent event run (`event_id = uuid5(watch.id, "t3-done")`, `event_type: "thread_settled"`) with the latest assistant reply and thread link, then marks the watch `delivered`.
4. A 404 or trashed-conversation 409 from Tsuchi drops the watch without retrying; transient errors remain `watching` for the next poll.
