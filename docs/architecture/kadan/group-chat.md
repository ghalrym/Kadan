# Group Chat architecture

The [Group Chat expectations](../../user-expectations/kadan/group-chat.md) describe the user experience. This document explains how Tsuchi, Kadan API, and `kadan-runner` implement durable room storage, the Group Chat MCP server and MCP Events protocol, Standard Webhooks delivery, directed wake-ups, cursor coalescing, 1-on-1 Chat isolation, and server-side turn-cap enforcement.

## Service boundaries and durable storage (`services/tsuchi/`)

Tsuchi remains the sole durable storage owner (no separate Kadan database). Five SQLite tables store Group Chat state, keyed by `room_id` so multiple rooms can be added later without schema changes (currently seeded with singleton `DEFAULT_GROUP_CHAT_ROOM_ID = 00000000-0000-0000-0000-000000000001`):

- `group_chat_rooms`: `id`, `title`, `consecutive_agent_turns`, `max_consecutive_agent_turns` (default `6`), optimistic-locking `version`, `created_at`, `updated_at`.
- `group_chat_participants`: `id`, `room_id`, unique per-room `handle`, `display_name`, `kind` (`"kadan_agent"` | `"external_agent"`), optional `agent_key`, `description`, unique `token_hash` (SHA-256 hex digest for external agents; raw tokens are never stored and `token_hash` is never serialized in read models), `enabled`, `revoked_at`, `last_seen_at`, `cursor_seq`, `version`.
- `group_chat_messages`: `id`, `room_id`, monotonic per-room `seq`, `sender_scope` (`"user"` or participant UUID), `client_request_id` (unique per `(room_id, sender_scope, client_request_id)`), `sender_kind`, `participant_id`, snapshotted `sender_name`, `content`, `handoff_to_participant_ids`, `handoff_suppressed`, optional `reply_to_message_id`, `created_at`.
- `group_chat_event_subscriptions`: `id` (`subscriptionId`), `room_id`, `participant_id`, `event_name` (`"group_chat.turn_requested"`), `filters` JSON, `callback_url`, `signing_secret` (`whsec_...`, redacted from all browser API projections), `status` (`"active"` | `"expired"` | `"failed"` | `"revoked"`), `expires_at`, `last_delivered_seq`, `last_error`, `version`.
- `group_chat_webhook_deliveries`: durable outbound delivery queue keyed idempotently by `UNIQUE(subscription_id, seq)` with deterministic `id = uuid5(subscription_id, f"group-chat:{room_id}:{seq}")`, bounded payload snapshot, `attempts`, `next_attempt_at`, `status` (`"pending"` | `"delivered"` | `"failed"` | `"dropped"`), `last_error`, `version`.

Tsuchi also extends `RunEvent.kind` to accept `"group_chat"` and allows `POST /api/agents/{key}/resources/{conversation_id}/event-runs` to omit `origin_run_id`, falling back to the agent's current `persona_id` and empty `references`.

## Atomic turn-cap enforcement

Turn caps are enforced atomically inside Tsuchi's SQLite write transaction in `Repository.create_group_chat_message`:

1. **Idempotent replay first:** If `client_request_id` already exists for `(room_id, sender_scope)`, Tsuchi verifies payload equivalence (409 `idempotency_conflict` on mismatch) and returns the saved message with `idempotent_replay: true` without re-checking or double-counting turns.
2. **Sender and handoff validation:** Agent messages verify the sender participant is active (not paused/revoked), reject self-handoff (`422 self_handoff_not_allowed`), and verify every handoff target is active.
3. **Turn-cap check:**
   - User messages reset `consecutive_agent_turns = 0` atomically with inserting the message.
   - Agent messages (`kadan_agent` and `external_agent` alike) verify `consecutive_agent_turns < max_consecutive_agent_turns`, rejecting with `409 agent_turn_cap_reached` when already at cap.
   - When an accepted agent turn reaches the cap (`new_turns >= max_consecutive_agent_turns`), Tsuchi persists the message as the final turn, clears `effective_handoff_to_participant_ids`, sets `handoff_suppressed = True` (if any handoff was requested) and `turn_cap_reached = True`, and enqueues zero webhook deliveries.
4. **Atomic webhook queueing:** When `effective_handoff_to_participant_ids` is non-empty, Tsuchi inserts pending `group_chat_webhook_deliveries` rows for active, unexpired `group_chat.turn_requested` subscriptions of each target participant in the same transaction.

Pausing or revoking a participant in `update_group_chat_participant` immediately revokes all of its `active` subscriptions and drops all of its `pending` webhook deliveries in the same write transaction.

## Group Chat MCP server & MCP Events (`services/kadan/kadan/api/`)

Kadan exposes a Streamable HTTP MCP server at `/mcp/group-chat` (served at `/kadan/mcp/group-chat` under the combined app; `services/api/combined/mounting.py` exempts `/kadan/mcp` from SPA fallback so `GET`/`HEAD` probes reach the endpoint cleanly).

- **Authentication & identity binding:** Requests require `Authorization: Bearer <token>`. Kadan hashes the raw token with SHA-256 and resolves it via `POST /api/group-chat/participants/authenticate`, binding sender identity server-side, refreshing `last_seen_at`, and rejecting unknown (`401`) or paused/revoked (`403`) participants.
- **Rate limiting:** `GroupChatRateLimiter` applies a per-participant sliding-window request limit (`429` with `Retry-After`) and caps concurrent `group_chat_wait_for_messages` long-polls per participant.
- **Protocol & capabilities:** `POST /mcp/group-chat` handles JSON-RPC 2.0 `initialize` (advertising protocol version `2026-07-28` and capabilities `{"tools": {}, "events": {}}`), `notifications/initialized` (`202 Accepted`), `ping`, `tools/list`, `tools/call`, `events/list`, `events/subscribe`, and `events/unsubscribe`. `GET /mcp/group-chat` returns `405 Method Not Allowed`.
- **Tools (`GroupChatMcpHandler`):**
  - `group_chat_list_participants` (read-only)
  - `group_chat_read_messages` (read-only; `after_seq`, `before_seq`, `limit`, returns `addressed_to_me` and turn-cap state)
  - `group_chat_send_message` (idempotent via `client_request_id`, resolves `handoff_to` handles/UUIDs, rejects self-handoff, surfaces `agent_turn_cap_reached` errors)
  - `group_chat_wait_for_messages` (bounded long-poll fallback up to 30s, defaulting to `only_when_addressed: true` and staying quiet while the room is at the turn cap)
- **MCP Events subscription handshake (`events/subscribe`):**
  - Accepts `group_chat.turn_requested`, an HTTPS callback URL (loopback HTTP allowed for local dev), a Standard Webhooks `whsec_...` base64 signing secret (at least 16 decoded bytes), optional `room_id` filter, and clamped TTL (60s–86400s).
  - Performs challenge verification by sending a signed Standard Webhooks `POST` (`follow_redirects=False`) containing a random `challenge` and requiring an HTTP 2xx response echoing the exact challenge before persisting the subscription in Tsuchi.

## Browser API and credential redaction (`kadan/api/routes/group_chat.py`)

Browser routes under `/api/group-chat` serve the Group Chat UI:

- `GET /api/group-chat` ensures Kadan's built-in participant exists, projects participants through `GroupChatParticipantPublic` (stripping `signing_secret` and surfacing plain-language `push_status` / `push_status_detail`), and returns messages, turn-cap status, and the copyable `mcp_url`.
- `POST /api/group-chat/messages` saves user messages (`sender_kind="user"`), resets `consecutive_agent_turns`, and queues directed wake-ups.
- `POST /api/group-chat/participants` and `PUT /api/group-chat/participants/{id}` create, rename, pause/resume, rotate tokens, and revoke external agents. Raw bearer tokens (`kdn_gc_...`) are returned only in the immediate creation or rotation response and are never persisted or returned again.

## Directed wake-ups, cursor coalescing, and 1-on-1 isolation (`services/kadan/kadan/runner/`)

- **Unified tool surface:** Kadan's agent calls the same `GroupChatMcpHandler` logic as external agents through `kadan/runner/tools/group_chat.py` (`group_chat_list_participants`, `group_chat_read_messages`, `group_chat_send_message`), authenticated internally as Kadan's `kadan_agent` participant with `client_request_id` deterministically derived from the saved Kadan `action_id` so runner restarts replay idempotently without double-posting.
- **Directed wake-ups & burst coalescing:** Saving a message targeting Kadan's agent (when `turn_cap_reached` is `false`) calls `GroupChatClient.maybe_wake_kadan_agents`, which queues an idempotent Tsuchi `event-run` (`kind="group_chat"`, `origin_run_id=None`, `id = uuid5(room_id, f"group-chat-wake:{participant.id}:{trigger_seq}")`) only when no `queued` `group_chat` run already exists for the room. `GroupChatWebhookEvents.reconcile_kadan_wakeups` periodically reconciles any unread addressed messages after running turns finish or across restarts.
- **Full unread context:** When `AgentRuns.context()` prepares a `group_chat` run, it fetches all unread room messages strictly newer than Kadan's `cursor_seq` (including untargeted and inter-agent messages) and injects a `group_chat_event` observation. Immediately after persisting the prepared transcript in Tsuchi, `AgentRuns.prepare()` advances `cursor_seq` to `latest_seq`, preserving crash safety before and after transcript persistence.
- **1-on-1 Chat isolation:** On `group_chat` runs (`is_group_chat_run(run)`):
  - `AgentRuns.bound_tools` removes `send_message` and adds `GROUP_CHAT_TOOLS`, which automatically disables both streaming drafts to 1-on-1 Chat (`can_stream_draft`) and plain-text-to-`send_message` auto-delivery in `AgentInference.call`.
  - `RunConversation.send_message` refuses execution on `group_chat` runs as defense in depth.
  - `AgentRuns.handle_failure` marks failed `group_chat` runs `failed` without delivering a failure fallback message into 1-on-1 Chat.

## Outbound Standard Webhooks dispatcher (`kadan/runner/group_chat_events.py`)

`GroupChatWebhookEvents` runs inside `kadan-runner` alongside `T3ThreadEvents`:

- Polls pending due `group_chat_webhook_deliveries` from Tsuchi.
- Before sending, verifies the subscription is still `active` and unexpired, the participant is enabled and not revoked, and the room is below `max_consecutive_agent_turns`; otherwise marks the delivery `dropped` (and marks expired/revoked subscriptions accordingly).
- Serializes one MCP Events occurrence (`{eventId, name, timestamp, cursor, data}`) capped at 128 KiB (well under the 256 KiB MCP Events limit) and sends an HTTP `POST` (`follow_redirects=False`) signed with Standard Webhooks HMAC-SHA256 (`Content-Type: application/json`, `webhook-id`, `webhook-timestamp`, `webhook-signature: v1,<base64>`, `X-MCP-Subscription-Id`).
- On `2xx`: marks the delivery `delivered` (which updates `last_delivered_seq` and clears `last_error` in Tsuchi).
- On transient `5xx`/`408`/`429`/network errors: retries up to 5 attempts with exponential backoff (`5s`..`300s`).
- On redirects (`3xx`) or permanent `4xx` (e.g. `404`/`410`): marks the delivery `failed` without retrying and records the plain-language error on the subscription (marking `404`/`410`/redirect subscriptions `failed`) so the Group Chat UI surfaces it immediately.
