# Chat attachment uploads and submission

Kadan proxies browser uploads into Tsuchi through `POST /api/agents/{key}/assets`. Multipart input supplies `file`, a required caller-generated `asset_id`, and an optional title. The API measures and hashes the spooled upload in bounded chunks, enforces a 100 MiB cap, rewinds it, and streams it to Tsuchi with the same ID. A returned asset must match the namespace, ID, active state, normalized filename, byte count and hash before Kadan confirms it. The route closes its spool on success, failure and cancellation. Tsuchi remains authoritative for its configured upload limit, immutable file publication and duplicate-ID checks.

The browser keeps a serial upload queue scoped to agent and conversation, with stable IDs and metadata persisted separately from ephemeral File objects. An uncertain attempt is checked through the existing asset metadata read before another POST. Reload can recover an existing asset; otherwise the user must reselect missing local bytes. Removing a draft association does not issue a storage deletion. The picker permits ten files totaling 100 MiB to bound queued local input.

`NoteInput` and `RunInput` accept ordered, unique `attachment_ids` with at most 100 entries. The API builds those references into the user message. Tsuchi validates active same-agent asset references inside the existing write transaction that appends the note or creates its run. Uploading alone creates neither a message nor agent work. Nonblank text remains required.

The conversation-scoped outbox snapshots attachment IDs with the original message ID, content, persona and generation choice. Send retries reuse that snapshot; newer draft uploads stay separate. Confirmed upload references survive failed submissions, and optimistic chat rendering uses the same IDs. Agent history already includes attachment references; `open_workspace_file` provides a checked copy on the computer without injecting binary content into text context.

See [Chat attachment expectations](../../user-expectations/kadan/chat-attachments.md) and [Workspace file delivery](file-delivery.md).
