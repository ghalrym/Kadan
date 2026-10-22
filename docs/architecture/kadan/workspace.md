# Workspace access

Kadan provides a read-only view of Tsuchi resources and asset metadata. Tsuchi owns content, revisions, trash and file bytes; the Workspace view does not maintain a second store. See [Workspace browsing](../../user-expectations/kadan/workspace.md).

Resource search is lexical over stored titles, Markdown, tags and structured text. Asset search uses metadata, not extracted document text or transcribed audio. The Files view downloads the stored asset rather than interpreting arbitrary content. These boundaries keep browsing independent of content ingestion.

The Open Tsuchi link uses the configured storage frontend. Workspace reads on navigation, search and explicit refresh rather than continuously synchronizing external edits. Selected records refresh independently of search results so filtering cannot imply deletion. A document read uses the unfiltered active resource collection for its text, ancestors and children; memory and file details use their individual reads. Confirmed absence, trash or a changed kind hides obsolete content, while transient failures retain the last loaded record with an error. Each read is invalidated by newer navigation or refreshes so late responses cannot replace the current selection.

Browsing does not modify runner context. Current context preparation reads recent chat and the selected persona, without automatically retrieving displayed Memories or Spaces. Adding retrieval later would require an explicit context policy; storage alone does not provide recall.
