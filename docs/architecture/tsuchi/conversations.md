# Conversation storage

Tsuchi stores conversation records without owning agent execution. Kadan queues agent runs through a separate atomic append-and-enqueue operation; storage edits must not replay work. See the user expectations for [stored conversations](../../user-expectations/tsuchi/conversation-records.md).

`POST /api/agents/{key}/resources/{id}/messages` accepts `expected_version` and a message containing a stable UUID, UTC timestamp, role and content. The editor defaults to `user`; the schema also supports `system`, `assistant` and `tool`, with attachment, resource and source references. The conversation must already exist, and resource and asset references must belong to its namespace.

The append checks the resource version and rejects conflicts instead of overwriting concurrent changes. Message identifiers cannot repeat. Ordered stored messages and durable agent-run state are distinct: a conversation edit alone neither invokes Hikari nor delivers an agent result.

Multiple conversations can coexist in a namespace in Tsuchi storage, though Kadan exposes one persistent conversation per agent. Trash and restore add or remove a chat from Kadan's list without other changes. See [Persistent conversation and compaction](../kadan/agent-conversations.md#persistent-conversation-and-compaction).

Each conversation resource also stores optional durable rolling compaction state (`compacted_summary`, `compacted_through_message_id`, and `compacted_at`) and hybrid memory tracking state (`memory_extracted_through_message_id`, `core_memories_snapshot`, and `core_memories_refreshed_at`). `PUT /api/agents/{key}/resources/{id}/compaction` updates compaction fields (and optionally refreshes `core_memories_snapshot` / `core_memories_refreshed_at` atomically) under `expected_version` optimistic locking, rejects missing or backwards message anchors, and optionally enforces `require_idle: true` (rejecting with 409 when any `queued` or `running` run exists for the conversation). `PUT /api/agents/{key}/resources/{id}/memory-state` advances `memory_extracted_through_message_id` (rejecting missing or backwards anchors) and/or sets `core_memories_snapshot` (optionally honoring `only_if_snapshot_unset: true`, `require_idle: true`, and optional `expected_version`) in place without bumping the conversation resource version so background extraction and snapshot initialization never cause 409 conflicts with concurrent user sends. Creating, updating, trashing, restoring, or version-restoring a `memory` resource clears `core_memories_snapshot` on the agent's active conversations unless `preserve_memory_snapshot=true` is passed (used by background memory extraction to preserve the active epoch's KV cache). Compaction never deletes or mutates stored messages, and metadata-only `PUT /resources/{id}` updates that omit compaction or memory-state fields preserve their saved values.

`POST /api/agents/{key}/resources/{id}/event-runs` queues a run woken by an
external event, such as a finished Codex run. It carries an opaque `trigger_event`
(`kind` and Kadan-owned `content`) and copies the persona and reference snapshots
of its origin run, which must belong to the same active conversation. The caller
supplies the run ID, so repeating the request returns the existing run; the same ID
with a different event conflicts. Runs without an event omit the field when stored.

Runs start with an empty transcript and prepare system context after claim. The
saved transcript prefix is immutable on every update, and expired-lease recovery
must retain the whole saved transcript.
