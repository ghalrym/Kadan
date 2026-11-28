# Memories

Tsuchi stores the agent's Markdown memories. Users search, read, correct and forget them from [Your agent → Memories](../kadan-server/your-agent.md) in Kadan Server. Each memory can have a title, tags, source links, attachments and a parent Space. Linked content must belong to the same agent.

Edits remain drafts until saved; reloading can lose them. A confirmed save persists through service restarts; a failed save must show an error rather than claim success. Conflicting edits must not silently replace a newer version.

Search matches stored words in titles, text and tags. It does not infer meaning or extract text from attached files. A loading or search error must remain distinguishable from having no matching memories.

Saving a memory does not run inference or guarantee that an agent will recall it. Kadan's current response flow does not automatically retrieve memories. See [Agent memory](../kadan/memories.md) for how the agent manages them.

Saved memories support [revisions](revisions.md), [trash](trash.md) and [restore](restore.md).
