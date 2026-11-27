# Conversation storage

Tsuchi stores conversation records without owning agent execution. Kadan queues agent runs through a separate atomic append-and-enqueue operation; storage edits must not replay work. See the user expectations for [stored conversations](../../user-expectations/tsuchi/conversation-records.md).

`POST /api/agents/{key}/resources/{id}/messages` accepts `expected_version` and a message containing a stable UUID, UTC timestamp, role and content. The editor defaults to `user`; the schema also supports `system`, `assistant` and `tool`, with attachment, resource and source references. The conversation must already exist, and resource and asset references must belong to its namespace.

The append checks the resource version and rejects conflicts instead of overwriting concurrent changes. Message identifiers cannot repeat. Ordered stored messages and durable agent-run state are distinct: a conversation edit alone neither invokes Hikari nor delivers an agent result.

Multiple conversations can coexist in a namespace in Tsuchi storage, though Kadan exposes one persistent conversation per agent. Trash and restore add or remove a chat from Kadan's list without other changes. See [Persistent conversation and compaction](../kadan/agent-conversations.md#persistent-conversation-and-compaction).

Each conversation resource also stores optional durable rolling compaction state (`compacted_summary`, `compacted_through_message_id`, and `compacted_at`). `PUT /api/agents/{key}/resources/{id}/compaction` updates only these fields under `expected_version` optimistic locking, rejects missing or backwards message anchors, and optionally enforces `require_idle: true` (rejecting with 409 when any `queued` or `running` run exists for the conversation). Compaction never deletes or mutates stored messages, and metadata-only `PUT /resources/{id}` updates that omit compaction fields preserve the saved compaction state.

`POST /api/agents/{key}/resources/{id}/event-runs` queues a run woken by an
external event, such as a finished Codex run. It carries an opaque `trigger_event`
(`kind` and Kadan-owned `content`) and copies the persona and reference snapshots
of its origin run, which must belong to the same active conversation. The caller
supplies the run ID, so repeating the request returns the existing run; the same ID
with a different event conflicts. Runs without an event omit the field when stored.

Runs start with an empty transcript and prepare system context after claim. The
saved transcript prefix is immutable on every update, and expired-lease recovery
must retain the whole saved transcript.
