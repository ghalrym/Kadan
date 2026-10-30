# Conversation storage

Tsuchi stores conversation records without owning agent execution. Kadan queues agent runs through a separate atomic append-and-enqueue operation; storage edits must not replay work. See the user expectations for [stored conversations](../../user-expectations/tsuchi/conversation-records.md).

`POST /api/agents/{key}/resources/{id}/messages` accepts `expected_version` and a message containing a stable UUID, UTC timestamp, role and content. The editor defaults to `user`; the schema also supports `system`, `assistant` and `tool`, with attachment, resource and source references. The conversation must already exist, and resource and asset references must belong to its namespace.

The append checks the resource version and rejects conflicts instead of overwriting concurrent changes. Message identifiers cannot repeat. Ordered stored messages and durable agent-run state are distinct: a conversation edit alone neither invokes Hikari nor delivers an agent result.

Multiple conversations can coexist in a namespace. Kadan selects the oldest active conversation, so trash and restore affect selection without adding a thread interface. The storage API does not impose Kadan's single-chat presentation on other clients.

Kadan's append-and-enqueue operation can include an explicit coding continuation
task. Its prompt, resolved directory, deadline and previous job ID are saved in
the run's JSON record in the same transaction as the user message. The task is
immutable across run-state updates and part of submission retry identity; reusing
the message ID with another task is a conflict. Tsuchi stores this intent but does
not contact Codex or decide whether the external session can resume. Kadan's
[coding workflow](../kadan/coding.md) owns execution and result delivery.
