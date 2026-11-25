# Conversation storage

Tsuchi stores conversation records without owning agent execution. Kadan queues agent runs through a separate atomic append-and-enqueue operation; storage edits must not replay work. See the user expectations for [stored conversations](../../user-expectations/tsuchi/conversation-records.md).

`POST /api/agents/{key}/resources/{id}/messages` accepts `expected_version` and a message containing a stable UUID, UTC timestamp, role and content. The editor defaults to `user`; the schema also supports `system`, `assistant` and `tool`, with attachment, resource and source references. The conversation must already exist, and resource and asset references must belong to its namespace.

The append checks the resource version and rejects conflicts instead of overwriting concurrent changes. Message identifiers cannot repeat. Ordered stored messages and durable agent-run state are distinct: a conversation edit alone neither invokes Hikari nor delivers an agent result.

Multiple conversations can coexist in a namespace. A conversation may be parented to a `project`, the resource kind Kadan presents as a Space. A project holds only conversations, has no parent of its own and is never part of a `space` document hierarchy; see [Chats and Spaces](../kadan/spaces.md).

`POST /api/agents/{key}/resources/{id}/event-runs` queues a run woken by an
external event, such as a finished Codex run. It carries an opaque `trigger_event`
(`kind` and Kadan-owned `content`) and copies the persona and reference snapshots
of its origin run, which must belong to the same active conversation. The caller
supplies the run ID, so repeating the request returns the existing run; the same ID
with a different event conflicts. Runs without an event omit the field when stored.

Runs start with an empty transcript and prepare system context after claim. The
saved transcript prefix is immutable on every update, and expired-lease recovery
must retain the whole saved transcript.
