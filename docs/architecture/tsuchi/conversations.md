# Conversation storage

Tsuchi stores conversation records without owning agent execution. Kadan queues agent runs through a separate atomic append-and-enqueue operation; storage edits must not replay work. See the user expectations for [stored conversations](../../user-expectations/tsuchi/conversation-records.md).

`POST /api/agents/{key}/resources/{id}/messages` accepts `expected_version` and a message containing a stable UUID, UTC timestamp, role and content. The editor defaults to `user`; the schema also supports `system`, `assistant` and `tool`, with attachment, resource and source references. The conversation must already exist, and resource and asset references must belong to its namespace.

The append checks the resource version and rejects conflicts instead of overwriting concurrent changes. Message identifiers cannot repeat. Ordered stored messages and durable agent-run state are distinct: a conversation edit alone neither invokes Hikari nor delivers an agent result.

Multiple conversations can coexist in a namespace. Kadan selects the oldest active conversation, so trash and restore affect selection without adding a thread interface. The storage API does not impose Kadan's single-chat presentation on other clients.

The append-and-enqueue input can include an `initial_command`: one assistant
checkpoint saved as the first entry of the existing transcript, atomically with
the user message and queued run. Tsuchi stores its opaque content; Kadan owns the
tool protocol and [coding workflow](../kadan/coding.md). It adds no persisted
fields. Enqueue retries compare the original first assistant checkpoint even
after later actions append results; omitting or changing that seed conflicts.
Ordinary runs start empty and prepare system context after claim.

The transcript prefix is immutable on the first queued claim and every later
update. Expired-lease recovery must retain the whole saved transcript. Older
strict record readers remain compatible, but older enqueue APIs reject the new
input and older storage does not enforce first-claim prefix protection. Coordinate
the API and runner rollout before accepting explicit continuations.
