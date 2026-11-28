# Document and memory access

Tsuchi owns content, revisions, trash and file bytes. Kadan exposes separate read contracts for saved documents and agent-managed Memories without maintaining a second store. A Memory is a user-viewable record of content managed by an agent, rather than a Workspace document.

Workspace documents retain Tsuchi's `space` (Markdown) and `deck` kinds. Memories retain `memory`; personas and conversations belong to neither collection. All records remain scoped by `agent_key`. Existing IDs, kinds, revisions, attachments and parent links are unchanged, with no migration or copying of content. In particular, an existing Memory's `parent_id` does not make it a Workspace document.

Users browse and edit saved pages, slides and files in [Space](space.md), which has its own routes. The document read contract below remains because the composer's References picker searches documents and memories together.

## Read contract

- `GET /api/agents/{key}/workspace/documents` returns active `space` and `deck` records as `WorkspaceDocument` values.
- `GET /api/agents/{key}/memories` returns active `memory` records as `Memory` values.
- Each collection has a `/{resource_id}` detail route. A record of another kind or a trashed record returns 404. Tsuchi enforces the agent namespace on lookup.
- Both lists accept `q`, `limit` (1–200, default 100) and `offset` (default 0). Kind, agent, active-state and lexical search filters apply before pagination. Ordering remains most recently updated first, then ID.

These response models retain the existing resource fields, including `kind`, `parent_id` and `version`. A document's parent may be an active `space` document or an active `folder` in the same agent namespace; a deck cannot contain children. Folders are not documents, so a document in a folder has a `parent_id` that this collection does not return. Files remain separately available through the existing asset routes.

Tsuchi's resource list accepts repeated `kinds` query parameters, such as `kinds=space&kinds=deck`, to page the combined document collection in storage. The existing single `kind` filter remains available; when both filters are supplied they intersect. Generic resource routes remain storage-oriented access, not the definition of Workspace membership. Kadan rejects an invalid upstream collection rather than silently dropping records after pagination.

## Ownership

`api/pydantic_models/resources.py`, `api/routes/agents/workspace.py`, `api/routes/agents/memories.py` and `api/services.py` own Kadan's read boundary. Tsuchi's repository and resource routes own filtering and persistence. The References picker owns document and memory selection. Runner document tools create `space` records and use existing Space parent validation; runner memory tools manage `memory` records separately. The read routes do not start runner operations.

Resource search is lexical over stored titles, Markdown, tags and structured text. Asset search uses metadata, not extracted document text or transcribed audio. Failed reads must remain distinguishable from empty collections. Browsing does not modify runner context: current context preparation reads recent chat, automatically includes the agent's owned persona when present, and adds any distinct explicitly selected persona, without automatically retrieving displayed documents or Memories. Storage alone does not provide recall.

## Memory corrections

Kadan provides an explicit correction form for existing Memories. `PUT /api/agents/{key}/memories/{resource_id}` accepts only `title`, `markdown` and `expected_version`. The service checks an active, same-agent Memory and passes its remaining content unchanged to Tsuchi's existing versioned resource update. Tsuchi owns the transaction and revision; no second memory store or inferred provenance is created. Kadan validates the returned identity, kind, next version and corrected text before confirming the save.

The editor reads the current record before editing. A failed save retains the draft and disables another write until an explicit read loads the saved version for comparison. That read advances the expected version without replacing the correction. Matching saved text needs no further write. Tab-local drafts also require that check after reopening; closing the browser can lose them. Corrections do not change already prepared work or automatically add the memory to later model context.
