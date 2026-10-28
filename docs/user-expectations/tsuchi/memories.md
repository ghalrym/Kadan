# Memories

Tsuchi lets users create, edit and search Markdown memories for the selected agent. Each memory can have a title, tags, source links, attachments and a parent Space. Linked content must belong to the same agent.

Edits and previews remain drafts until saved. Closing the editor with unsaved changes asks whether to discard them. Reloading the page can lose those drafts. A confirmed save persists through service restarts; a failed save must show an error rather than claim success. Conflicting edits must not silently replace a newer version.

Search matches stored words in titles, text and tags. It does not infer meaning or extract text from attached files. A loading or search error must remain distinguishable from having no matching memories.

Saving a memory does not run inference or guarantee that an agent will recall it. Kadan's current response flow does not automatically retrieve memories. See [Workspace browsing](../kadan/workspace.md) for how users inspect them in Kadan.

Saved memories support [revisions](revisions.md), [trash](trash.md) and [restore](restore.md).
