# Workspace and memory access

Tsuchi owns content, revisions, trash and file bytes. Kadan exposes separate read contracts for Workspace documents and agent-managed Memories without maintaining a second store. A Memory is a user-viewable record of content managed by an agent, rather than a Workspace document.

Workspace documents retain Tsuchi's `space` (Markdown) and `deck` kinds. Memories retain `memory`; personas and conversations belong to neither collection. All records remain scoped by `agent_key`. Existing IDs, kinds, revisions, attachments and parent links are unchanged, with no migration or copying of content. In particular, an existing Memory's `parent_id` does not make it a Workspace document.

## Read contract

- `GET /api/agents/{key}/workspace/documents` returns active `space` and `deck` records as `WorkspaceDocument` values.
- `GET /api/agents/{key}/memories` returns active `memory` records as `Memory` values.
- Each collection has a `/{resource_id}` detail route. A record of another kind or a trashed record returns 404. Tsuchi enforces the agent namespace on lookup.
- Both lists accept `q`, `limit` (1–200, default 100) and `offset` (default 0). Kind, agent, active-state and lexical search filters apply before pagination. Ordering remains most recently updated first, then ID.

These response models retain the existing resource fields, including `kind`, `parent_id` and `version`. Only an active Space in the same agent namespace can parent another document; a deck cannot contain children. Clients build document hierarchies from document records only. Files remain separately available through the existing asset routes.

Tsuchi's resource list accepts repeated `kinds` query parameters, such as `kinds=space&kinds=deck`, to page the combined document collection in storage. The existing single `kind` filter remains available; when both filters are supplied they intersect. Generic resource routes remain storage-oriented access, not the definition of Workspace membership. Kadan rejects an invalid upstream collection rather than silently dropping records after pagination.

## Ownership

`api/pydantic_models/resources.py`, `api/routes/agents/workspace.py`, `api/routes/agents/memories.py` and `api/services.py` own Kadan's read boundary. Tsuchi's repository and resource routes own filtering and persistence. The Workspace frontend owns document navigation and the separate read-only Memories view. Runner document tools create `space` records and use existing Space parent validation; runner memory tools manage `memory` records separately. These read routes do not introduce user memory editing or runner write operations.

Resource search is lexical over stored titles, Markdown, tags and structured text. Asset search uses metadata, not extracted document text or transcribed audio. Failed reads must remain distinguishable from empty collections. Browsing does not modify runner context: current context preparation reads recent chat and the selected persona, without automatically retrieving displayed documents or Memories. Storage alone does not provide recall.
