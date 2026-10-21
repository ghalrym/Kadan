# Workspace browsing

Workspace shows the selected agent's active Memories, Spaces and Files from Tsuchi. It starts on Memories. Search filters the current collection; changing agents opens that agent's records. A loading error must be distinguishable from an empty collection, and Retry reloads the requested data.

Select a memory or Space to read its Markdown. A Space also lists its direct children, which can be opened in the same detail view. Select a file to see its original filename, media type and size, then choose Download file to retrieve it. The current file view shows metadata rather than extracting or previewing arbitrary file contents.

This view is read-only. Use Open Tsuchi to create, edit, upload, organize or restore stored content. That link opens the configured Tsuchi interface in another tab. Changes made there may require another search or reopening Workspace to appear; this view does not continuously synchronize external edits.

Search is lexical. Resource search covers stored titles, Markdown, tags and structured text; file search uses metadata, not extracted PDF text or transcribed audio. A missing match does not imply that a phrase is absent from a file's bytes.

Opening a record does not attach it to a chat message or add its text to a response. In the current prototype, preparing a response reads recent chat history and the selected persona; it does not automatically retrieve the displayed memories or Spaces. Storage and browsing alone do not establish long-term agent recall.

See [Tsuchi's storage capabilities](../../../services/tsuchi/README.md#what-is-stored) for editing, search, revision and trash behavior.
