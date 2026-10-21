# Workspace access

Kadan provides a read-only view of Tsuchi resources and asset metadata. Tsuchi owns content, revisions, trash and file bytes; the Workspace view does not maintain a second store. See [Workspace browsing](../../user-expectations/kadan/workspace.md).

Resource search is lexical over stored titles, Markdown, tags and structured text. Asset search uses metadata, not extracted document text or transcribed audio. The Files view downloads the stored asset rather than interpreting arbitrary content. These boundaries keep browsing independent of content ingestion.

The Open Tsuchi link uses the configured storage frontend. Workspace reads on navigation and search rather than continuously synchronizing external edits. Failed reads must remain distinguishable from empty collections.

Browsing does not modify runner context. Current context preparation reads recent chat and the selected persona, without automatically retrieving displayed Memories or Spaces. Adding retrieval later would require an explicit context policy; storage alone does not provide recall.
